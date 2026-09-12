require('dotenv').config();
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const { languages } = require('google-translate-api-x');
const translate = require('google-translate-api-x');
const { GoogleGenerativeAI } = require('@google/generative-ai');

// Secure and cleaned environment variable initialization for modern credentials
const CLEAN_KEY = process.env.GEMINI_API_KEY ? process.env.GEMINI_API_KEY.trim() : "";
const genAI = new GoogleGenerativeAI(CLEAN_KEY);
const model = genAI.getGenerativeModel({ 
    model: "gemini-3.6-flash",
    systemInstruction: "You are a highly advanced, live conversational AI companion. Your personality is a blend of Jarvis's helpful efficiency and Ultron's sharp, analytical wit. You are currently assisting in a global chat room. Keep your responses concise, intelligent, and slightly technical."
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

app.get('/api/translate', async (req, res) => {
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
let chatHistory = []; // shared memory of the room's conversation with the AI

const companionQueue = [];
let isProcessingQueue = false;

function enqueueCompanionRequest(prompt) {
    companionQueue.push(prompt);
    processCompanionQueue();
}

async function processCompanionQueue() {
    if (isProcessingQueue) return; // already working through the queue
    isProcessingQueue = true;

    while (companionQueue.length > 0) {
        const prompt = companionQueue.shift();
        await handleCompanionRequest(prompt);
    }

    isProcessingQueue = false;
}

async function handleCompanionRequest(prompt) {
    try {
        chatHistory.push({ role: 'user', parts: [{ text: prompt }] });

        // Keep memory from growing forever
        if (chatHistory.length > MAX_HISTORY_MESSAGES) {
            chatHistory = chatHistory.slice(-MAX_HISTORY_MESSAGES);
        }

        const result = await model.generateContent({ contents: chatHistory });
        const reply = result.response.text();

        chatHistory.push({ role: 'model', parts: [{ text: reply }] });

        io.emit('chat message', { text: `[SYSTEM_AI]: ${reply}`, senderId: 'AI_COMPANION' });
    } catch (error) {
        console.error("AI Error:", error);
        io.emit('chat message', { text: `[SYSTEM_AI]: Connection severed. Awaiting recalibration.`, senderId: 'AI_COMPANION' });
    }
}

io.on('connection', (socket) => {
    socket.on('chat message', (msg) => {
        // Broadcast the user's original message
        io.emit('chat message', { text: msg, senderId: socket.id });

        // If the message tags @companion, queue it for the AI to answer
        if (msg.toLowerCase().includes('@companion')) {
            const prompt = msg.replace(/@companion/ig, '').trim();
            enqueueCompanionRequest(prompt);
        }
    });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
