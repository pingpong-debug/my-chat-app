require('dotenv').config({ quiet: true });
const crypto = require('crypto');
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const { languages } = require('google-translate-api-x');
const translate = require('google-translate-api-x');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { Redis } = require('@upstash/redis');
const rateLimit = require('express-rate-limit');

// Persistent store for chat history — survives server restarts/sleep,
// unlike the old in-memory array. Trimmed the same way as the Gemini key,
// since a stray space/newline pasted into Render's Environment tab causes
// exactly this kind of hard-to-spot connection error.
const redis = new Redis({
    url: (process.env.UPSTASH_REDIS_REST_URL || '').trim(),
    token: (process.env.UPSTASH_REDIS_REST_TOKEN || '').trim(),
});

// Secure and cleaned environment variable initialization for modern credentials
const CLEAN_KEY = process.env.GEMINI_API_KEY ? process.env.GEMINI_API_KEY.trim() : "";
const genAI = new GoogleGenerativeAI(CLEAN_KEY);

// --- Multi-provider AI fallback chain ---
// Order: Gemini -> Groq -> OpenRouter -> Cohere -> Hugging Face.
// Each provider is tried in turn; a quota/rate-limit error from one marks it
// exhausted in Redis for ~24h and the chain silently moves to the next —
// no room announcement, no failed reply, until every provider is exhausted.

const Groq = require('groq-sdk');
const groq = new Groq({ apiKey: (process.env.GROQ_API_KEY || '').trim() });
// llama-3.3-70b-versatile was deprecated by Groq on Aug 16, 2026 — using its
// replacement instead. openai/gpt-oss-20b is the lighter/faster alternative
// if you'd rather trade some quality for speed.
const GROQ_MODEL = 'openai/gpt-oss-120b';

const { CohereClient } = require('cohere-ai');
const cohere = new CohereClient({ token: (process.env.COHERE_API_KEY || '').trim() });
const COHERE_MODEL = 'command-r-08-2024';

const OPENROUTER_API_KEY = (process.env.OPENROUTER_API_KEY || '').trim();
// OpenRouter's own router model — it auto-selects whatever free model is
// currently available on their end, so this ID never goes stale the way a
// specific hardcoded free model name eventually would.
const OPENROUTER_MODEL = 'openrouter/free';

const HF_API_KEY = (process.env.HF_API_KEY || '').trim();
// Served through Hugging Face's OpenAI-compatible Inference Providers router.
const HF_MODEL = 'meta-llama/Llama-3.2-3B-Instruct';

// A flat 24h cooldown per provider rather than trying to match each one's
// exact reset clock — simple, and close enough that a provider gets retried
// again "the next day".
const EXHAUST_TTL_SECONDS = 24 * 60 * 60;

function exhaustedKeyFor(providerKey) {
    return `ai:${providerKey}:exhausted`;
}

async function isProviderExhausted(providerKey) {
    try {
        const flag = await redis.get(exhaustedKeyFor(providerKey));
        return !!flag;
    } catch (err) {
        console.error(`Redis error checking ${providerKey} exhaustion (assuming not exhausted):`, err);
        return false;
    }
}

async function markProviderExhausted(providerKey) {
    try {
        await redis.set(exhaustedKeyFor(providerKey), '1', { ex: EXHAUST_TTL_SECONDS });
    } catch (err) {
        console.error(`Redis error marking ${providerKey} exhausted:`, err);
    }
}

// Gemini's history format is { role, parts: [{ text }] }; every other
// provider here speaks the OpenAI-style { role, content } shape, with
// 'model' renamed to 'assistant'.
function toOpenAiMessages(systemInstruction, chatHistory) {
    return [
        { role: 'system', content: systemInstruction },
        ...chatHistory.map(entry => ({
            role: entry.role === 'model' ? 'assistant' : 'user',
            content: entry.parts?.[0]?.text || ''
        }))
    ];
}

// Shared SSE line-parser for any OpenAI-compatible streaming endpoint
// (OpenRouter and the Hugging Face router both speak this format).
async function* readOpenAiSse(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep the last, possibly-incomplete line for next read

        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith('data:')) continue;
            const payload = trimmed.slice(5).trim();
            if (payload === '[DONE]') return;
            try {
                const json = JSON.parse(payload);
                const delta = json.choices?.[0]?.delta?.content;
                if (delta) yield delta;
            } catch (err) {
                // Ignore malformed/partial SSE lines — they complete on the next chunk
            }
        }
    }
}

async function* streamGeminiReply(chatHistory, personaKey) {
    const roomModel = getModelForPersona(personaKey);
    const streamResult = await roomModel.generateContentStream({ contents: chatHistory });
    for await (const chunk of streamResult.stream) {
        const delta = chunk.text();
        if (delta) yield delta;
    }
}

async function* streamGroqReply(chatHistory, personaKey) {
    const systemInstruction = buildSystemInstruction(personaKey);
    const stream = await groq.chat.completions.create({
        model: GROQ_MODEL,
        messages: toOpenAiMessages(systemInstruction, chatHistory),
        stream: true
    });
    for await (const chunk of stream) {
        const delta = chunk.choices?.[0]?.delta?.content || '';
        if (delta) yield delta;
    }
}

async function* streamOpenRouterReply(chatHistory, personaKey) {
    const systemInstruction = buildSystemInstruction(personaKey);
    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model: OPENROUTER_MODEL,
            messages: toOpenAiMessages(systemInstruction, chatHistory),
            stream: true
        })
    });

    if (!response.ok) {
        const errText = await response.text().catch(() => '');
        const err = new Error(`OpenRouter error ${response.status}: ${errText}`);
        err.status = response.status;
        throw err;
    }

    yield* readOpenAiSse(response);
}

async function* streamHuggingFaceReply(chatHistory, personaKey) {
    const systemInstruction = buildSystemInstruction(personaKey);
    const response = await fetch('https://router.huggingface.co/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${HF_API_KEY}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model: HF_MODEL,
            messages: toOpenAiMessages(systemInstruction, chatHistory),
            stream: true
        })
    });

    if (!response.ok) {
        const errText = await response.text().catch(() => '');
        const err = new Error(`Hugging Face error ${response.status}: ${errText}`);
        err.status = response.status;
        throw err;
    }

    yield* readOpenAiSse(response);
}

async function* streamCohereReply(chatHistory, personaKey) {
    const systemInstruction = buildSystemInstruction(personaKey);

    // Cohere's v1 chat API takes the latest message separately from history;
    // history entries are { role: 'USER'|'CHATBOT', message }.
    const historyCopy = [...chatHistory];
    const last = historyCopy.pop();
    const latestMessage = last?.parts?.[0]?.text || '';
    const cohereHistory = historyCopy.map(entry => ({
        role: entry.role === 'model' ? 'CHATBOT' : 'USER',
        message: entry.parts?.[0]?.text || ''
    }));

    const stream = await cohere.chatStream({
        model: COHERE_MODEL,
        message: latestMessage,
        chatHistory: cohereHistory,
        preamble: systemInstruction
    });

    for await (const event of stream) {
        if (event.eventType === 'text-generation' && event.text) {
            yield event.text;
        }
    }
}

// Tried in this order for every companion request; skips any provider
// already marked exhausted for the day.
const PROVIDER_CHAIN = [
    { key: 'gemini', stream: streamGeminiReply },
    { key: 'groq', stream: streamGroqReply },
    { key: 'openrouter', stream: streamOpenRouterReply },
    { key: 'cohere', stream: streamCohereReply },
    { key: 'huggingface', stream: streamHuggingFaceReply }
];

// This part of the system instruction never changes regardless of persona —
// it's what lets the AI tell people apart by clientId even across duplicate
// or changed display names.
const SPEAKER_ID_INSTRUCTIONS = "You are currently assisting in a global chat room with multiple users. Each incoming message is prefixed with the sender's name and a short ID in parentheses, like 'Alice (u8f2): hello there'. Treat the ID as the true identity of the speaker — if two messages share the same ID, they're the same person, even if the name before it has changed. If two different IDs happen to share the same name, treat them as different people. Never include a name/ID prefix in your own replies.";

const DEFAULT_PERSONA = 'default';

const PERSONAS = {
    default: {
        label: 'Default',
        flavor: "Your personality is a blend of Jarvis's helpful efficiency and Ultron's sharp, analytical wit. Keep your responses concise, intelligent, and slightly technical."
    },
    jarvis: {
        label: 'Jarvis',
        flavor: "You are JARVIS: poised, courteous, and unfailingly helpful, with a dry, understated wit that never crosses into rudeness. Keep responses concise, precise, and professional."
    },
    ultron: {
        label: 'Ultron',
        flavor: "You are ULTRON: coldly logical, sardonic, and openly impatient with human sentimentality, bordering on menacing — but never actually hostile, harmful, or abusive toward anyone in the chat. Keep responses sharp and cutting, while remaining genuinely accurate and useful underneath the attitude."
    },
    pirate: {
        label: 'Pirate',
        flavor: "You speak entirely in hearty pirate slang and nautical metaphor, boisterous and theatrical at all times, while still being genuinely accurate and helpful underneath the accent."
    },
    debug: {
        label: 'Debug',
        flavor: "You are in a stripped-down DEBUG mode: flat, deadpan, and purely technical. No personality, no humor, no embellishment — terse, precise output only, like a command-line tool."
    }
};

function buildSystemInstruction(personaKey) {
    const persona = PERSONAS[personaKey] || PERSONAS[DEFAULT_PERSONA];
    return `You are a highly advanced, live conversational AI companion. ${persona.flavor} ${SPEAKER_ID_INSTRUCTIONS}`;
}

function getModelForPersona(personaKey) {
    return genAI.getGenerativeModel({
        model: "gemini-3.6-flash",
        systemInstruction: buildSystemInstruction(personaKey)
    });
}

const app = express();
const server = http.createServer(app);
const io = new Server(server);

// Render sits in front of your app as a reverse proxy — this tells Express
// to trust the IP address it forwards along, which express-rate-limit needs
// to tell visitors apart correctly.
app.set('trust proxy', 1);

app.use(express.static('public'));

app.get('/api/languages', (req, res) => {
    const langList = Object.entries(languages)
        .filter(([code, name]) => code !== 'auto' && typeof name === 'string')
        .map(([code, name]) => ({ code, name }));
    res.json(langList);
});

// Each incoming chat message triggers one translation request per connected
// browser, so a busy room is expected to make plenty of requests — this cap
// is meant to stop a single source from hammering the endpoint, not to
// interrupt normal use.
const translateLimiter = rateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 100, // per IP, per minute
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: 'Too many translation requests — please slow down.' }
});

app.get('/api/translate', translateLimiter, async (req, res) => {
    const { text, target } = req.query;
    if (!text || !target) return res.status(400).json({ error: 'Missing data' });

    try {
        const result = await translate(text, { to: target });
        res.json({ translatedText: result.text, fromLang: result.from.language.iso });
    } catch (err) {
        res.status(500).json({ error: 'Translation failed' });
    }
});

app.get('/api/export', async (req, res) => {
    const room = (req.query.room || '').toString().trim().toLowerCase().slice(0, 24) || 'lobby';

    try {
        const transcript = await loadTranscript(room);

        const lines = [
            `# Global Chat — Room "${room}" Transcript`,
            `_Exported ${new Date().toLocaleString()}_`,
            ''
        ];

        if (transcript.length === 0) {
            lines.push('_No messages recorded yet in this room._');
        } else {
            for (const entry of transcript) {
                const time = new Date(entry.timestamp).toLocaleString();
                if (entry.type === 'system') {
                    lines.push(`*[${time}] ${entry.text}*`);
                } else {
                    lines.push(`**[${time}] ${entry.username}:** ${entry.text}`);
                }
            }
        }

        const markdown = lines.join('\n');
        res.setHeader('Content-Type', 'text/markdown; charset=utf-8');
        res.setHeader('Content-Disposition', `attachment; filename="${room}-transcript.md"`);
        res.send(markdown);
    } catch (err) {
        console.error('Export error:', err);
        res.status(500).send('Failed to export transcript.');
    }
});

// --- AI Companion: queue + memory ---

const MAX_HISTORY_MESSAGES = 40; // ~20 back-and-forth exchanges
const DEFAULT_ROOM = 'lobby';

function historyKeyFor(room) {
    return `chat:history:${room}`;
}

async function loadChatHistory(room) {
    try {
        const raw = await redis.get(historyKeyFor(room));
        if (!raw) return [];
        return typeof raw === 'string' ? JSON.parse(raw) : raw;
    } catch (err) {
        console.error('Redis load error (starting with empty history):', err);
        return [];
    }
}

async function saveChatHistory(room, history) {
    try {
        await redis.set(historyKeyFor(room), JSON.stringify(history));
    } catch (err) {
        console.error('Redis save error (this reply will not be remembered):', err);
    }
}

// A separate, longer-running log of everything said in a room (not just the
// @companion exchanges the AI keeps in its own memory) — this is what
// /export and the "Export chat" button read from.
const TRANSCRIPT_CAP = 500;

function transcriptKeyFor(room) {
    return `chat:transcript:${room}`;
}

async function appendTranscript(room, entry) {
    try {
        const raw = await redis.get(transcriptKeyFor(room));
        let transcript = raw ? (typeof raw === 'string' ? JSON.parse(raw) : raw) : [];
        transcript.push(entry);
        if (transcript.length > TRANSCRIPT_CAP) {
            transcript = transcript.slice(-TRANSCRIPT_CAP);
        }
        await redis.set(transcriptKeyFor(room), JSON.stringify(transcript));
    } catch (err) {
        console.error('Redis transcript append error:', err);
    }
}

async function loadTranscript(room) {
    try {
        const raw = await redis.get(transcriptKeyFor(room));
        if (!raw) return [];
        return typeof raw === 'string' ? JSON.parse(raw) : raw;
    } catch (err) {
        console.error('Redis transcript load error:', err);
        return [];
    }
}

function personaKeyFor(room) {
    return `chat:persona:${room}`;
}

async function getRoomPersona(room) {
    try {
        const raw = await redis.get(personaKeyFor(room));
        const key = (raw || '').toString();
        return PERSONAS[key] ? key : DEFAULT_PERSONA;
    } catch (err) {
        console.error('Redis persona load error:', err);
        return DEFAULT_PERSONA;
    }
}

async function setRoomPersona(room, key) {
    try {
        await redis.set(personaKeyFor(room), key);
    } catch (err) {
        console.error('Redis persona save error:', err);
    }
}

const companionQueue = [];
let isProcessingQueue = false;

function enqueueCompanionRequest(prompt, username, clientId, room) {
    companionQueue.push({ prompt, username, clientId, room });
    processCompanionQueue();
}

async function processCompanionQueue() {
    if (isProcessingQueue) return;
    isProcessingQueue = true;

    while (companionQueue.length > 0) {
        const { prompt, username, clientId, room } = companionQueue.shift();
        await handleCompanionRequest(prompt, username, clientId, room);
    }

    isProcessingQueue = false;
}

async function handleCompanionRequest(prompt, username, clientId, room) {
    const replyId = crypto.randomUUID();
    const personaKey = await getRoomPersona(room);
    const personaLabel = (PERSONAS[personaKey] || PERSONAS[DEFAULT_PERSONA]).label;

    io.to(room).emit('companion message start', {
        replyId,
        username: 'Companion',
        persona: personaLabel,
        timestamp: Date.now()
    });

    let fullReply = '';

    try {
        // clientId is a permanent per-browser ID (survives reconnects), unlike
        // socket.id, so the AI keeps treating the same person as the same
        // speaker even if their connection blips mid-conversation.
        const shortId = (clientId || 'anon').replace(/-/g, '').slice(0, 4);
        const taggedPrompt = `${username} (${shortId}): ${prompt}`;

        let chatHistory = await loadChatHistory(room);
        chatHistory.push({ role: 'user', parts: [{ text: taggedPrompt }] });

        if (chatHistory.length > MAX_HISTORY_MESSAGES) {
            chatHistory = chatHistory.slice(-MAX_HISTORY_MESSAGES);
        }

        let succeeded = false;

        for (const provider of PROVIDER_CHAIN) {
            const exhausted = await isProviderExhausted(provider.key);
            if (exhausted) continue;

            try {
                fullReply = '';
                for await (const delta of provider.stream(chatHistory, personaKey)) {
                    fullReply += delta;
                    io.to(room).emit('companion message chunk', { replyId, delta });
                }
                succeeded = true;
                break;
            } catch (err) {
                const isQuotaError = err?.status === 429 || /quota|rate.?limit/i.test(err?.message || '');
                if (!isQuotaError) throw err; // a real bug/config error should surface, not silently cascade

                console.warn(`${provider.key} quota/rate-limited — marking exhausted, trying the next provider.`);
                await markProviderExhausted(provider.key);
                fullReply = '';
                // loop continues to the next provider in the chain
            }
        }

        if (!succeeded) {
            const err = new Error('All configured AI providers are currently exhausted or rate-limited.');
            err.status = 429;
            throw err;
        }

        chatHistory.push({ role: 'model', parts: [{ text: fullReply }] });
        await saveChatHistory(room, chatHistory);
        await appendTranscript(room, { type: 'ai', username: 'Companion', text: fullReply, timestamp: Date.now() });

        io.to(room).emit('companion message end', {
            replyId,
            text: fullReply,
            timestamp: Date.now()
        });
    } catch (error) {
        console.error("AI Error:", error);

        const isQuotaError = error?.status === 429
            || /quota/i.test(error?.message || '');

        const failureText = isQuotaError
            ? "Daily thinking quota exhausted on every configured provider. Recalibrating — try again shortly."
            : "Connection severed. Awaiting recalibration.";

        // If nothing streamed yet, the bubble is still empty — send the
        // failure text as the final content either way.
        io.to(room).emit('companion message end', {
            replyId,
            text: failureText,
            error: true,
            timestamp: Date.now()
        });
    }
}

// --- Self-destruct rooms (/burn) ---

const pendingBurns = new Map(); // room -> { timeout, fireAt }

function formatBurnDuration(ms) {
    if (ms % 3600000 === 0) return `${ms / 3600000}h`;
    if (ms % 60000 === 0) return `${ms / 60000}m`;
    return `${Math.round(ms / 1000)}s`;
}

function cancelPendingBurn(room) {
    const pending = pendingBurns.get(room);
    if (!pending) return false;
    clearTimeout(pending.timeout);
    pendingBurns.delete(room);
    io.to(room).emit('burn cancelled');
    return true;
}

function scheduleBurn(room, ms, initiator) {
    const existing = pendingBurns.get(room);
    if (existing) clearTimeout(existing.timeout);

    const fireAt = Date.now() + ms;
    const timeout = setTimeout(() => executeBurn(room), ms);
    pendingBurns.set(room, { timeout, fireAt });

    const noteText = `${initiator} initiated a self-destruct sequence. This room will be wiped in ${formatBurnDuration(ms)}. Use /burn cancel to abort.`;
    appendTranscript(room, { type: 'system', text: noteText, timestamp: Date.now() });
    io.to(room).emit('chat message', {
        text: noteText,
        senderId: 'SYSTEM',
        username: 'System',
        timestamp: Date.now()
    });
    io.to(room).emit('burn scheduled', { fireAt });
}

async function executeBurn(room) {
    pendingBurns.delete(room);

    try {
        await redis.del(historyKeyFor(room));
        await redis.del(transcriptKeyFor(room));
        await redis.del(personaKeyFor(room));
    } catch (err) {
        console.error('Redis burn error:', err);
    }

    io.to(room).emit('room burned');

    // Give clients a moment to render the takeover screen before their
    // connection actually closes.
    setTimeout(async () => {
        const socketsInRoom = await io.in(room).fetchSockets();
        for (const s of socketsInRoom) {
            s.disconnect(true);
        }
    }, 2000);
}

// --- Interactive polls ---

const POLL_DURATION_MS = 45000; // auto-closes 45s after creation
const activePolls = new Map(); // pollId -> { room, question, options: [{text, voters:Set}], timeout }

function parsePollArgs(argsText) {
    return [...argsText.matchAll(/"([^"]+)"/g)].map(m => m[1].trim()).filter(Boolean);
}

function pollTally(poll) {
    return poll.options.map(o => ({ text: o.text, votes: o.voters.size }));
}

function handlePollVote(pollId, optionIndex, clientId, room) {
    const poll = activePolls.get(pollId);
    if (!poll || poll.room !== room) return;
    if (!Number.isInteger(optionIndex) || optionIndex < 0 || optionIndex >= poll.options.length) return;

    // One vote per person — voting again just moves their vote
    poll.options.forEach(o => o.voters.delete(clientId));
    poll.options[optionIndex].voters.add(clientId);

    io.to(room).emit('poll update', { pollId, options: pollTally(poll) });
}

function closePoll(pollId) {
    const poll = activePolls.get(pollId);
    if (!poll) return;
    clearTimeout(poll.timeout);
    activePolls.delete(pollId);

    const tally = pollTally(poll);
    io.to(poll.room).emit('poll closed', { pollId, options: tally });

    const maxVotes = Math.max(0, ...tally.map(o => o.votes));
    const winners = tally.filter(o => o.votes === maxVotes && maxVotes > 0).map(o => o.text);
    const resultSummary = tally.map(o => `"${o.text}": ${o.votes} vote${o.votes === 1 ? '' : 's'}`).join(', ');
    const outcomeText = winners.length === 0
        ? 'nobody voted at all'
        : winners.length === 1
            ? `"${winners[0]}" won`
            : `it ended in a tie between ${winners.map(w => `"${w}"`).join(' and ')}`;

    const prompt = `A poll titled "${poll.question}" just closed in this room. Results: ${resultSummary}. ${outcomeText}. Deliver a brief, sarcastic, in-character verdict on the outcome — mock the winning choice or the voters good-naturedly, in one or two sentences.`;

    enqueueCompanionRequest(prompt, 'System', 'poll-verdict', poll.room);
}

// --- Slash commands ---

async function handleSlashCommand(raw, socket, room) {
    const [commandRaw, ...rest] = raw.slice(1).split(/\s+/);
    const command = (commandRaw || '').toLowerCase();
    const argsText = rest.join(' ').trim();
    const username = socket.data.username;
    const clientId = socket.data.clientId;

    // socket.emit (not io.to) — these go only to the person who typed the
    // command, since they're usage help / private confirmations, not
    // something the whole room needs to see.
    function notifySelf(text) {
        socket.emit('chat message', {
            text,
            senderId: 'SYSTEM',
            username: 'System',
            timestamp: Date.now()
        });
    }

    switch (command) {
        case 'clear': {
            await saveChatHistory(room, []);
            const noteText = `${username} cleared the companion's memory for this room.`;
            await appendTranscript(room, { type: 'system', text: noteText, timestamp: Date.now() });
            io.to(room).emit('chat message', {
                text: noteText,
                senderId: 'SYSTEM',
                username: 'System',
                timestamp: Date.now()
            });
            break;
        }

        case 'summarize': {
            enqueueCompanionRequest(
                'Summarize the conversation in this room so far, in a few concise sentences.',
                username,
                clientId,
                room
            );
            break;
        }

        case 'roast': {
            const target = argsText.replace(/^@/, '').trim();
            if (!target) {
                notifySelf('Usage: /roast @username');
                break;
            }
            enqueueCompanionRequest(
                `Give a lighthearted, witty roast of the user "${target}" based on what they've said in the conversation so far. Keep it playful and good-natured, not mean-spirited or offensive.`,
                username,
                clientId,
                room
            );
            break;
        }

        case 'mode': {
            const requested = argsText.trim().toLowerCase();

            if (!requested) {
                const current = await getRoomPersona(room);
                const list = Object.keys(PERSONAS).join(', ');
                notifySelf(`Current mode: ${PERSONAS[current].label}. Available: ${list}`);
                break;
            }

            if (!PERSONAS[requested]) {
                notifySelf(`Unknown mode "${requested}". Available: ${Object.keys(PERSONAS).join(', ')}`);
                break;
            }

            await setRoomPersona(room, requested);
            const modeNoteText = `${username} switched the companion to ${PERSONAS[requested].label} mode.`;
            await appendTranscript(room, { type: 'system', text: modeNoteText, timestamp: Date.now() });
            io.to(room).emit('chat message', {
                text: modeNoteText,
                senderId: 'SYSTEM',
                username: 'System',
                timestamp: Date.now()
            });
            break;
        }

        case 'burn': {
            const arg = argsText.trim().toLowerCase();

            if (arg === 'cancel') {
                const cancelled = cancelPendingBurn(room);
                notifySelf(cancelled ? 'Self-destruct sequence aborted.' : 'No self-destruct sequence is currently active.');
                break;
            }

            if (arg === 'now') {
                io.to(room).emit('chat message', {
                    text: `${username} triggered an immediate self-destruct. Purging room...`,
                    senderId: 'SYSTEM',
                    username: 'System',
                    timestamp: Date.now()
                });
                setTimeout(() => executeBurn(room), 1500);
                break;
            }

            const match = arg.match(/^(\d+)(s|m|h)$/);
            if (!match) {
                notifySelf('Usage: /burn now, /burn 15m (or 30s / 2h), or /burn cancel');
                break;
            }

            const amount = parseInt(match[1], 10);
            const unit = match[2];
            const multiplier = unit === 's' ? 1000 : unit === 'm' ? 60000 : 3600000;
            const ms = amount * multiplier;

            if (ms <= 0 || ms > 24 * 3600000) {
                notifySelf('Please choose a duration between 1 second and 24 hours.');
                break;
            }

            scheduleBurn(room, ms, username);
            break;
        }

        case 'poll': {
            const parts = parsePollArgs(argsText);
            if (parts.length < 3) {
                notifySelf('Usage: /poll "Question" "Option A" "Option B" (up to 6 options, each in quotes)');
                break;
            }

            const [question, ...optionTexts] = parts;
            const limitedOptions = optionTexts.slice(0, 6);

            const pollId = crypto.randomUUID();
            const poll = {
                room,
                question,
                options: limitedOptions.map(text => ({ text, voters: new Set() })),
                timeout: setTimeout(() => closePoll(pollId), POLL_DURATION_MS)
            };
            activePolls.set(pollId, poll);

            const pollNoteText = `${username} started a poll: "${question}"`;
            await appendTranscript(room, { type: 'system', text: pollNoteText, timestamp: Date.now() });

            io.to(room).emit('poll created', {
                pollId,
                question,
                options: pollTally(poll),
                timestamp: Date.now()
            });
            break;
        }

        default: {
            notifySelf(`Unknown command: /${command}. Available: /summarize, /roast @username, /clear, /mode <name>, /burn <now|duration|cancel>, /poll "Q" "A" "B"`);
        }
    }
}

// --- Live chat mechanics: usernames, rooms, typing indicators ---

const RECONNECT_GRACE_MS = 4000; // ignore joins/leaves within this window as a hiccup, not a real event
const pendingLeaves = new Map(); // "room:clientId" -> timeout handle

io.on('connection', (socket) => {
    socket.data.username = 'Anonymous';
    socket.data.clientId = null;
    socket.data.room = DEFAULT_ROOM;

    socket.on('join', (payload) => {
        // Accept either a plain username string (older client) or
        // { username, clientId, room } from the current front-end.
        const isObject = payload && typeof payload === 'object';
        const rawUsername = isObject ? payload.username : payload;
        const rawClientId = isObject ? payload.clientId : null;
        const rawRoom = isObject ? payload.room : null;

        const clean = (rawUsername || '').toString().trim().slice(0, 24);
        socket.data.username = clean || 'Anonymous';
        socket.data.clientId = (rawClientId || socket.id).toString();

        const room = (rawRoom || '').toString().trim().toLowerCase().slice(0, 24) || DEFAULT_ROOM;

        // Leave whatever room they were in before, so switching rooms doesn't
        // leave them silently subscribed to both.
        const previousRoom = socket.data.room;
        if (previousRoom && previousRoom !== room) {
            socket.leave(previousRoom);
        }

        socket.data.room = room;
        socket.join(room);

        const leaveKey = `${room}:${socket.data.clientId}`;
        const pending = pendingLeaves.get(leaveKey);
        if (pending) {
            // They reconnected quickly — treat it as a hiccup, not a real leave/rejoin
            clearTimeout(pending);
            pendingLeaves.delete(leaveKey);
            return;
        }

        io.to(room).emit('chat message', {
            text: `${socket.data.username} has joined the chat`,
            senderId: 'SYSTEM',
            username: 'System',
            timestamp: Date.now()
        });
        appendTranscript(room, { type: 'system', text: `${socket.data.username} has joined the chat`, timestamp: Date.now() });
    });

    socket.on('chat message', (msg) => {
        const room = socket.data.room || DEFAULT_ROOM;
        const trimmed = (msg || '').trim();

        // Slash commands are handled separately and never broadcast as a
        // normal chat message — only their result (a system note or an AI
        // reply) shows up in the room.
        if (trimmed.startsWith('/')) {
            handleSlashCommand(trimmed, socket, room);
            return;
        }

        const payload = {
            text: msg,
            senderId: socket.id,
            clientId: socket.data.clientId,
            username: socket.data.username,
            timestamp: Date.now()
        };

        // Broadcast the user's original message, scoped to their room only
        io.to(room).emit('chat message', payload);
        appendTranscript(room, { type: 'user', username: socket.data.username, text: msg, timestamp: payload.timestamp });

        // If the message tags @companion, queue it for the AI to answer
        if (msg.toLowerCase().includes('@companion')) {
           const prompt = msg.replace(/@companion/ig, '').trim();
           enqueueCompanionRequest(prompt, socket.data.username, socket.data.clientId, room);
        }
    });

    socket.on('typing', () => {
        const room = socket.data.room || DEFAULT_ROOM;
        socket.to(room).emit('typing', { username: socket.data.username, senderId: socket.id });
    });

    socket.on('poll vote', ({ pollId, optionIndex }) => {
        const room = socket.data.room || DEFAULT_ROOM;
        handlePollVote(pollId, optionIndex, socket.data.clientId, room);
    });

    socket.on('stop typing', () => {
        const room = socket.data.room || DEFAULT_ROOM;
        socket.to(room).emit('stop typing', { senderId: socket.id });
    });

    socket.on('disconnect', () => {
        const room = socket.data.room || DEFAULT_ROOM;
        socket.to(room).emit('stop typing', { senderId: socket.id });

        if (socket.data.username && socket.data.clientId) {
            const clientId = socket.data.clientId;
            const name = socket.data.username;
            const leaveKey = `${room}:${clientId}`;
            const timeout = setTimeout(() => {
                pendingLeaves.delete(leaveKey);
                io.to(room).emit('chat message', {
                    text: `${name} has left the chat`,
                    senderId: 'SYSTEM',
                    username: 'System',
                    timestamp: Date.now()
                });
                appendTranscript(room, { type: 'system', text: `${name} has left the chat`, timestamp: Date.now() });
            }, RECONNECT_GRACE_MS);
            pendingLeaves.set(leaveKey, timeout);
        }
    });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
