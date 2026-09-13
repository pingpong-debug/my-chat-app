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
const model = genAI.getGenerativeModel({ 
    model: "gemini-3.6-flash",
    systemInstruction: "You are a highly advanced, live conversational AI companion. Your personality is a blend of Jarvis's helpful efficiency and Ultron's sharp, analytical wit. You are currently assisting in a global chat room with multiple users. Each incoming message is prefixed with the sender's name and a short ID in parentheses, like 'Alice (u8f2): hello there'. Treat the ID as the true identity of the speaker — if two messages share the same ID, they're the same person, even if the name before it has changed. If two different IDs happen to share the same name, treat them as different people. Never include a name/ID prefix in your own replies. Keep your responses concise, intelligent, and slightly technical."
});

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

    io.to(room).emit('companion message start', {
        replyId,
        username: 'Companion',
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

        const streamResult = await model.generateContentStream({ contents: chatHistory });

        for await (const chunk of streamResult.stream) {
            const delta = chunk.text();
            if (!delta) continue;
            fullReply += delta;
            io.to(room).emit('companion message chunk', { replyId, delta });
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
            ? "Daily thinking quota exhausted. Recalibrating — available again once the free tier resets tomorrow."
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

        default: {
            notifySelf(`Unknown command: /${command}. Available: /summarize, /roast @username, /clear`);
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
