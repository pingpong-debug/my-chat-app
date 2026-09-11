const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const { languages } = require('google-translate-api-x');
const translate = require('google-translate-api-x');
const { GoogleGenerativeAI } = require('@google/generative-ai');

// Initialize the AI Companion
const genAI = new GoogleGenerativeAI("process.env.GEMINI_API_KEY");
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

io.on('connection', (socket) => {
    socket.on('chat message', async (msg) => {
        // Broadcast the user's original message
        io.emit('chat message', { text: msg, senderId: socket.id });

        // If the message tags @companion, the AI generates a response
        if (msg.toLowerCase().includes('@companion')) {
            try {
                // Remove the tag so the AI just reads the prompt
                const prompt = msg.replace(/@companion/ig, '').trim(); 
                const aiResponse = await model.generateContent(prompt);
                const reply = aiResponse.response.text();
                
                // Broadcast the AI's response to the chat
                io.emit('chat message', { text: `[SYSTEM_AI]: ${reply}`, senderId: 'AI_COMPANION' });
            } catch (error) {
                console.error("AI Error:", error);
                io.emit('chat message', { text: `[SYSTEM_AI]: Connection severed. Awaiting recalibration.`, senderId: 'AI_COMPANION' });
            }
        }
    });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});