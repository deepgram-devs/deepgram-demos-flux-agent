# Deepgram Flux Voice Agent

A real-time voice conversation application powered by Deepgram Flux API, OpenAI, and Deepgram TTS. Features a beautiful web interface built with Deepgram's design system.

## Features

- **Real-time voice conversations** with AI assistant
- **Microphone selection** from available devices
- **Preflighting vs Non-preflighting modes** for optimal performance
- **Configurable AI models** (OpenAI LLM and Deepgram TTS)
- **Advanced settings** for turn detection thresholds
- **Live conversation display** with real-time transcripts
- **A simple UI** using Deepgram design system
- **Debug logging** for development and troubleshooting
- **Conversation export** functionality

## Components

- **Flask + SocketIO**: Web server and real-time communication
- **Deepgram Flux**: Real-time speech-to-text with turn detection
- **OpenAI API**: Language model for generating responses
- **Deepgram TTS**: Text-to-speech for agent voice
- **Web Audio API**: Browser audio capture and playback

## Prerequisites

- Python 3.8 or higher
- Microphone access
- Valid API keys for:
  - Deepgram API (set as DEEPGRAM_API_KEY environment variable)
  - OpenAI API (set as OPENAI_API_KEY environment variable)

## Setup

### 1. Clone and Install

```bash
git clone git@github.com:deepgram-devs/deepgram-flux-agent-demo.git
cd flux-agent-demo
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export DEEPGRAM_API_KEY="your_deepgram_api_key_here"
export OPENAI_API_KEY="your_openai_api_key_here"
```

### 4. Run the Application

```bash
python app.py
```

### 5. Open in Browser

Open your web browser and navigate to: `http://localhost:3000`

## Configuration Options

The application provides configuration options through the web interface:

### Audio Settings
- **Microphone Selection**: Choose from available audio input devices
- **Sample Rate**: Audio quality (default: 16000 Hz, recommended for Flux)

### AI Models
- **Language Model**: Choose from OpenAI models (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
- **Voice Model**: Select Deepgram TTS voice.

### Advanced Settings
- **Preflighting Mode**: Enable/disable preflighting for faster responses
- **Preflight Threshold**: Confidence level for triggering preflighting (0.2-0.9)
- **End-of-Turn Threshold**: Confidence level for finalizing turn completion (0.5-0.9)
- **Turn Timeout**: Maximum wait time for turn completion (1000-10000ms)

## Default Configuration

The application comes with optimized defaults:

```python
SAMPLE_RATE = 16000
OPENAI_LLM_MODEL = "gpt-4o-mini"
DEEPGRAM_TTS_MODEL = "aura-2-phoebe-en"
PREFLIGHT_THRESHOLD = 0.3
EOT_THRESHOLD = 0.8
EOT_TIMEOUT_MS = 3000
```

## How to Use

1. **Configure Settings**: Select your microphone and adjust AI model settings
2. **Start Conversation**: Click "Start Conversation" to begin
3. **Speak Naturally**: The app will transcribe your speech in real-time
4. **Listen to Responses**: The AI assistant will respond with natural speech
5. **Monitor Progress**: Watch the conversation log and debug information
6. **Stop When Done**: Click "Stop Conversation" to end the session

## Preflighting vs Non-Preflighting

### Preflighting Mode (Default)
- **Faster responses**: AI starts generating responses before you finish speaking
- **More LLM calls**: Uses more API credits but provides snappier conversations
- **Best for**: Interactive conversations, demos, real-time use

### Non-Preflighting Mode
- **Conservative approach**: Waits for complete utterances before responding
- **Fewer LLM calls**: More cost-effective for longer conversations
- **Best for**: Cost-conscious usage, longer monologues

## Troubleshooting

### Common Issues

**1. Microphone Access Denied**
- Ensure your browser has microphone permissions
- Check system privacy settings
- Try refreshing the page and allowing microphone access

**2. API Key Errors**
- Verify your Deepgram API key is valid and has credits
- Ensure your OpenAI API key is active
- Make sure environment variables are properly exported in your shell

**3. Connection Issues**
- Ensure port 3000 is available
- Check firewall settings
- Try restarting the application

**4. Audio Playback Issues**
- Check browser audio permissions
- Ensure speakers/headphones are connected
- Try a different browser if issues persist

### Debug Information

The application provides comprehensive debug logging:
- **SOCKET**: WebSocket connection events
- **FLUX**: Deepgram Flux WebSocket events
- **USER**: User speech detection and transcription
- **AGENT**: AI response generation and TTS
- **AUDIO**: Audio capture and playback information
- **CONFIG**: Configuration updates
- **ERROR**: Error messages and stack traces

### Performance Tips

1. **Use headphones** to prevent audio feedback
2. **Adjust thresholds** based on your speaking style
3. **Monitor debug logs** to optimize settings
4. **Choose appropriate models** based on speed vs quality needs





