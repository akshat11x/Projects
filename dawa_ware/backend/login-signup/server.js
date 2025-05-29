const express = require('express');
const cors = require('cors');
const session = require('express-session');
const dotenv = require('dotenv');
const connectDB = require('./config/db');

dotenv.config();
connectDB();

const app = express();

// ✅ CORS: allow credentials and origin from frontend
app.use(cors({
  origin: 'http://localhost:5173', // Your React frontend
  credentials: true
}));

// ✅ Body parser
app.use(express.json());

// ✅ Session middleware setup
app.use(session({
  secret: 'your-secret-key', // Use a strong secret, ideally from .env
  resave: false,
  saveUninitialized: false,
  cookie: {
    httpOnly: true,
    secure: false, // Set true if using HTTPS
    maxAge: 24 * 60 * 60 * 1000 // 1 day
  }
}));

// ✅ Routes
app.use('/api/auth', require('./routes/authRoutes'));

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
