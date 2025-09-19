"""Configuration constants for the Deepgram Flux Voice Agent."""

import os

# Flux API Configuration
FLUX_URL = "wss://api.preview.deepgram.com/v2/listen"

# Audio Configuration
SAMPLE_RATE = 16000          # Audio quality (16kHz recommended for Flux)
FLUX_ENCODING = "linear16"   # Audio encoding format required by Flux API

# AI Model Configuration
OPENAI_LLM_MODEL = "gpt-4o-mini"           # LLM model for responses
DEEPGRAM_TTS_MODEL = "aura-2-phoebe-en"    # Voice for agent speech

# Turn Detection Thresholds
EOT_THRESHOLD = 0.8          # When to finalize turn completion (0.5-0.9)
EOT_TIMEOUT_MS = 3000        # Max wait time for turn completion (ms)

# API Keys (from environment variables)
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Server Configuration
HOST = "127.0.0.1"
PORT = 3000
DEBUG = True

# Conversation System Prompts
SYSTEM_PROMPT = """You are a helpful voice assistant powered by Deepgram Flux and OpenAI.
You should:
- Keep responses conversational and natural
- Be concise but helpful
- Respond as if you're having a real-time voice conversation
- Ask follow-up questions when appropriate
- Be friendly and engaging

The user is speaking to you via voice, so respond naturally as if in a live conversation."""

# Available TTS Models
TTS_MODEL_OPTIONS = [
    {"value": "aura-2-phoebe-en", "label": "Phoebe (Female, US English)"},
    {"value": "aura-2-apollo-en", "label": "Apollo (Male, US English)"},
    {"value": "aura-2-stella-en", "label": "Stella (Female, US English)"},
    {"value": "aura-2-luna-en", "label": "Luna (Female, US English)"},
    {"value": "aura-2-mars-en", "label": "Mars (Male, US English)"},
]

# Available LLM Models
LLM_MODEL_OPTIONS = [
    {"value": "gpt-4o-mini", "label": "GPT-4o Mini (Fast & Cost-effective)"},
    {"value": "gpt-4o", "label": "GPT-4o (Best Quality)"},
    {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo (Legacy)"},
]
