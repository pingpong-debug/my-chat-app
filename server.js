require('dotenv').config({ quiet: true });
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const { languages } = require('google-translate-api-x');
const translate = require('google-translate-api-x');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { Redis } = require('@upstash/redis');
const rateLimit = require('express-rate-limit');

// Persistent store for chat history — survives server restarts/sleep,
// unlike the old in-memory array. Reads UPSTASH_REDIS_REST_URL and
// UPSTASH_REDIS_REST_TOKEN from the environment automatically.
const redis = Redis.fromEnv();
const CHAT_HISTORY_KEY = 'chat:history';

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

// --- AI Companion: queue + memory ---

const MAX_HISTORY_MESSAGES = 40; // ~20 back-and-forth exchanges

async function loadChatHistory() {
    try {
        const raw = await redis.get(CHAT_HISTORY_KEY);
        if (!raw) return [];
        return typeof raw === 'string' ? JSON.parse(raw) : raw;
    } catch (err) {
        console.error('Redis load error (starting with empty history):', err);
        return [];
    }
}

async function saveChatHistory(history) {
    try {
        await redis.set(CHAT_HISTORY_KEY, JSON.stringify(history));
    } catch (err) {
        console.error('Redis save error (this reply will not be remembered):', err);
    }
}

const companionQueue = [];
let isProcessingQueue = false;

function enqueueCompanionRequest(prompt, username, clientId) {
    companionQueue.push({ prompt, username, clientId });
    processCompanionQueue();
}

async function processCompanionQueue() {
    if (isProcessingQueue) return;
    isProcessingQueue = true;

    while (companionQueue.length > 0) {
        const { prompt, username, clientId } = companionQueue.shift();
        await handleCompanionRequest(prompt, username, clientId);
    }

    isProcessingQueue = false;
}

async function handleCompanionRequest(prompt, username, clientId) {
    try {
        // clientId is a permanent per-browser ID (survives reconnects), unlike
        // socket.id, so the AI keeps treating the same person as the same
        // speaker even if their connection blips mid-conversation.
        const shortId = (clientId || 'anon').replace(/-/g, '').slice(0, 4);
        const taggedPrompt = `${username} (${shortId}): ${prompt}`;

        let chatHistory = await loadChatHistory();
        chatHistory.push({ role: 'user', parts: [{ text: taggedPrompt }] });

        if (chatHistory.length > MAX_HISTORY_MESSAGES) {
            chatHistory = chatHistory.slice(-MAX_HISTORY_MESSAGES);
        }

        const result = await model.generateContent({ contents: chatHistory });
        const reply = result.response.text();

        chatHistory.push({ role: 'model', parts: [{ text: reply }] });
        await saveChatHistory(chatHistory);

        io.emit('chat message', {
            text: reply,
            senderId: 'AI_COMPANION',
            clientId: 'AI_COMPANION',
            username: 'Companion',
            timestamp: Date.now()
        });
    } catch (error) {
        console.error("AI Error:", error);
        io.emit('chat message', {
            text: `Connection severed. Awaiting recalibration.`,
            senderId: 'AI_COMPANION',
            clientId: 'AI_COMPANION',
            username: 'Companion',
            timestamp: Date.now()
        });
    }
}

// --- Live chat mechanics: usernames + typing indicators ---

const RECONNECT_GRACE_MS = 4000; // ignore joins/leaves within this window as a hiccup, not a real event
const pendingLeaves = new Map(); // clientId -> timeout handle

io.on('connection', (socket) => {
    socket.data.username = 'Anonymous';
    socket.data.clientId = null;

    socket.on('join', (payload) => {
        // Accept either a plain username string (older client) or
        // { username, clientId } from the current front-end.
        const isObject = payload && typeof payload === 'object';
        const rawUsername = isObject ? payload.username : payload;
        const rawClientId = isObject ? payload.clientId : null;

        const clean = (rawUsername || '').toString().trim().slice(0, 24);
        socket.data.username = clean || 'Anonymous';
        socket.data.clientId = (rawClientId || socket.id).toString();

        const pending = pendingLeaves.get(socket.data.clientId);
        if (pending) {
            // They reconnected quickly — treat it as a hiccup, not a real leave/rejoin
            clearTimeout(pending);
            pendingLeaves.delete(socket.data.clientId);
            return;
        }

        io.emit('chat message', {
            text: `${socket.data.username} has joined the chat`,
            senderId: 'SYSTEM',
            username: 'System',
            timestamp: Date.now()
        });
    });

    socket.on('chat message', (msg) => {
        const payload = {
            text: msg,
            senderId: socket.id,
            clientId: socket.data.clientId,
            username: socket.data.username,
            timestamp: Date.now()
        };

        // Broadcast the user's original message
        io.emit('chat message', payload);

        // If the message tags @companion, queue it for the AI to answer
        if (msg.toLowerCase().includes('@companion')) {
           const prompt = msg.replace(/@companion/ig, '').trim();
           enqueueCompanionRequest(prompt, socket.data.username, socket.data.clientId);
        }
    });

    socket.on('typing', () => {
        socket.broadcast.emit('typing', { username: socket.data.username, senderId: socket.id });
    });

    socket.on('stop typing', () => {
        socket.broadcast.emit('stop typing', { senderId: socket.id });
    });

    socket.on('disconnect', () => {
        socket.broadcast.emit('stop typing', { senderId: socket.id });

        if (socket.data.username && socket.data.clientId) {
            const clientId = socket.data.clientId;
            const name = socket.data.username;
            const timeout = setTimeout(() => {
                pendingLeaves.delete(clientId);
                io.emit('chat message', {
                    text: `${name} has left the chat`,
                    senderId: 'SYSTEM',
                    username: 'System',
                    timestamp: Date.now()
                });
            }, RECONNECT_GRACE_MS);
            pendingLeaves.set(clientId, timeout);
        }
    });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
