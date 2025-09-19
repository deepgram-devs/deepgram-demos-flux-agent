"""
Deepgram Flux Voice Agent Web Application
A real-time voice conversation app using Deepgram Flux, OpenAI, and TTS
"""

import asyncio
import json
import logging
import os
import struct
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from enum import Enum, auto

import openai
import websockets
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from deepgram import (
    DeepgramClient,
    SpeakWebSocketEvents,
    SpeakWebSocketMessage,
    SpeakWSOptions,
)

from config import (
    HOST, PORT, DEBUG,
    DEEPGRAM_API_KEY, OPENAI_API_KEY,
    FLUX_URL, FLUX_ENCODING, SAMPLE_RATE,
    OPENAI_LLM_MODEL, DEEPGRAM_TTS_MODEL,
    PREFLIGHT_THRESHOLD, EOT_THRESHOLD, EOT_TIMEOUT_MS,
    EXTRA_LLM_LATENCY_SECONDS, SYSTEM_PROMPT,
    TTS_MODEL_OPTIONS, LLM_MODEL_OPTIONS
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state management
active_sessions: Dict[str, Dict[str, Any]] = {}

class ConversationState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"

class TTSEvent(Enum):
    FLUSHED = auto()

def validate_api_keys():
    """Validate that required API keys are set."""
    missing_keys = []

    if not DEEPGRAM_API_KEY:
        missing_keys.append("DEEPGRAM_API_KEY")
    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")

    if missing_keys:
        logger.error(f"Missing required API keys: {', '.join(missing_keys)}")
        logger.error("Please set them as environment variables or in your .env file")
        raise ValueError(f"Missing API keys: {missing_keys}")

    logger.info("API keys validated successfully")


async def generate_agent_reply_normal(
    messages: List[Dict[str, str]],
    user_speech: str,
    session_id: str,
    config: Dict[str, Any]
) -> Optional[bytes]:
    """Generate agent reply using normal (non-preflighting) approach."""

    logger.info(f"Session {session_id}: *** INSIDE generate_agent_reply_normal() ***")
    logger.info(f"Session {session_id}: User speech: '{user_speech}'")
    logger.info(f"Session {session_id}: Current conversation history: {messages}")
    logger.info(f"Session {session_id}: API Key set: {'Yes' if OPENAI_API_KEY else 'No'}")

    try:
        # Set up OpenAI client
        logger.info(f"Session {session_id}: Creating OpenAI client...")
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

        # Prepare messages for LLM
        llm_messages = messages.copy()
        llm_messages.append({"role": "user", "content": user_speech})

        final_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + llm_messages
        logger.info(f"Session {session_id}: Final messages to send to OpenAI: {final_messages}")
        logger.info(f"Session {session_id}: Calling OpenAI API with model: {config['llm_model']}")

        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model=config['llm_model'],
            messages=final_messages,
            temperature=0.7,
            max_tokens=150  # Keep responses concise for voice
        )

        agent_message = response.choices[0].message.content
        logger.info(f"Session {session_id}: *** OPENAI RESPONSE SUCCESS ***")
        logger.info(f"Session {session_id}: Generated response: '{agent_message}'")

        # Send agent response to UI chat history
        socketio.emit('agent_response', {
            'response': agent_message,
            'timestamp': datetime.now().isoformat()
        }, room=session_id)

        # Generate TTS audio (similar to preflighting but simpler)
        logger.info(f"Session {session_id}: About to generate TTS audio for: '{agent_message}'")
        tts_result = await generate_tts_audio(agent_message, session_id, config)
        logger.info(f"Session {session_id}: TTS generation result: {len(tts_result) if tts_result else 0} bytes")
        return tts_result

    except Exception as e:
        logger.error(f"Session {session_id}: *** ERROR IN generate_agent_reply_normal() ***")
        logger.error(f"Session {session_id}: Error details: {e}")
        import traceback
        logger.error(f"Session {session_id}: Full traceback: {traceback.format_exc()}")
        return None

async def generate_tts_audio(text: str, session_id: str, config: Dict[str, Any]) -> Optional[bytes]:
    """Generate TTS audio for given text."""

    logger.info(f"Session {session_id}: *** STARTING TTS GENERATION ***")
    logger.info(f"Session {session_id}: TTS Text: '{text}'")
    logger.info(f"Session {session_id}: TTS Model: {config['tts_model']}")
    logger.info(f"Session {session_id}: API Key Available: {'Yes' if DEEPGRAM_API_KEY else 'No'}")

    try:
        # Set up TTS WebSocket
        logger.info(f"Session {session_id}: Creating Deepgram TTS WebSocket client...")
        dg_tts_ws = DeepgramClient(api_key=DEEPGRAM_API_KEY).speak.websocket.v("1")
        logger.info(f"Session {session_id}: TTS WebSocket client created")

        audio_queue: asyncio.Queue[Union[bytes, TTSEvent]] = asyncio.Queue()
        audio_chunks_received = 0

        # Get current event loop for thread-safe operations
        loop = asyncio.get_running_loop()

        # TTS event handlers
        def on_binary_data(self, data, **kwargs):
            nonlocal audio_chunks_received
            audio_chunks_received += 1
            logger.info(f"Session {session_id}: *** TTS AUDIO DATA RECEIVED *** Chunk #{audio_chunks_received}, {len(data)} bytes")
            # Use thread-safe method to put data in async queue from sync callback
            asyncio.run_coroutine_threadsafe(audio_queue.put(data), loop)

        def on_flushed(self, **kwargs):
            logger.info(f"Session {session_id}: *** TTS FLUSHED EVENT RECEIVED ***")
            # Use thread-safe method to put event in async queue from sync callback
            asyncio.run_coroutine_threadsafe(audio_queue.put(TTSEvent.FLUSHED), loop)

        def on_open(self, open, **kwargs):
            logger.info(f"Session {session_id}: *** TTS WEBSOCKET OPENED ***")

        def on_error(self, error, **kwargs):
            logger.error(f"Session {session_id}: *** TTS WEBSOCKET ERROR *** {error}")

        def on_warning(self, warning, **kwargs):
            logger.warning(f"Session {session_id}: *** TTS WEBSOCKET WARNING *** {warning}")

        def on_metadata(self, metadata, **kwargs):
            logger.info(f"Session {session_id}: *** TTS METADATA *** {metadata}")

        # Register handlers
        logger.info(f"Session {session_id}: Registering TTS event handlers...")
        dg_tts_ws.on(SpeakWebSocketEvents.AudioData, on_binary_data)
        dg_tts_ws.on(SpeakWebSocketEvents.Flushed, on_flushed)
        dg_tts_ws.on(SpeakWebSocketEvents.Open, on_open)
        dg_tts_ws.on(SpeakWebSocketEvents.Error, on_error)
        dg_tts_ws.on(SpeakWebSocketEvents.Warning, on_warning)
        dg_tts_ws.on(SpeakWebSocketEvents.Metadata, on_metadata)

        # TTS options
        tts_options = SpeakWSOptions(
            model=config['tts_model'],
            encoding="linear16",
            sample_rate=16000  # Match Flux and frontend sample rate
        )
        logger.info(f"Session {session_id}: TTS Options: model={config['tts_model']}, encoding=linear16, sample_rate=16000")

        # Start TTS
        logger.info(f"Session {session_id}: Starting TTS WebSocket connection...")
        start_result = dg_tts_ws.start(tts_options)  # Remove await - this is synchronous
        logger.info(f"Session {session_id}: TTS WebSocket start result: {start_result}")

        if not start_result:
            logger.error(f"Session {session_id}: *** TTS WEBSOCKET START FAILED ***")
            return None

        logger.info(f"Session {session_id}: *** TTS WEBSOCKET STARTED SUCCESSFULLY ***")

        # Send text and flush
        logger.info(f"Session {session_id}: Sending text to TTS: '{text}'")
        dg_tts_ws.send_text(text)  # Remove await - this is synchronous
        logger.info(f"Session {session_id}: Text sent, now flushing...")
        dg_tts_ws.flush()  # Remove await - this is synchronous
        logger.info(f"Session {session_id}: TTS flush complete, waiting for audio...")

        # Collect audio
        audio_chunks = []
        chunk_count = 0
        total_timeout = 10.0  # Increased timeout

        logger.info(f"Session {session_id}: Starting audio collection loop...")
        while True:
            try:
                logger.debug(f"Session {session_id}: Waiting for audio chunk... (timeout: {total_timeout}s)")
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=total_timeout)

                if chunk == TTSEvent.FLUSHED:
                    logger.info(f"Session {session_id}: *** RECEIVED FLUSH EVENT - AUDIO COLLECTION COMPLETE ***")
                    break
                elif isinstance(chunk, bytes):
                    chunk_count += 1
                    audio_chunks.append(chunk)
                    logger.info(f"Session {session_id}: Collected audio chunk #{chunk_count}, {len(chunk)} bytes, total chunks: {len(audio_chunks)}")
                else:
                    logger.warning(f"Session {session_id}: Unexpected chunk type: {type(chunk)}")

            except asyncio.TimeoutError:
                logger.warning(f"Session {session_id}: *** TTS TIMEOUT after {total_timeout}s *** Chunks received: {len(audio_chunks)}")
                break

        # Finalize
        logger.info(f"Session {session_id}: Finishing TTS WebSocket...")
        try:
            dg_tts_ws.finish()  # Remove await - this is synchronous
            logger.info(f"Session {session_id}: TTS WebSocket finished successfully")
        except Exception as finish_error:
            logger.warning(f"Session {session_id}: TTS finish error (non-critical): {finish_error}")

        # Return results
        if audio_chunks:
            combined_audio = b''.join(audio_chunks)
            logger.info(f"Session {session_id}: *** TTS SUCCESS *** Total chunks: {len(audio_chunks)}, Total bytes: {len(combined_audio)}")
            return combined_audio
        else:
            logger.error(f"Session {session_id}: *** TTS FAILED - NO AUDIO CHUNKS RECEIVED ***")
            return None

    except Exception as e:
        logger.error(f"Session {session_id}: *** TTS EXCEPTION *** {e}")
        import traceback
        logger.error(f"Session {session_id}: TTS Full traceback: {traceback.format_exc()}")
        return None

# Flask routes
@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html',
                         tts_models=TTS_MODEL_OPTIONS,
                         llm_models=LLM_MODEL_OPTIONS)

# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    session_id = request.sid
    logger.info(f"Client connected: {session_id}")

    # Initialize session state
    active_sessions[session_id] = {
        'state': ConversationState.IDLE,
        'messages': [],
        'config': {
            'use_preflighting': True,
            'sample_rate': SAMPLE_RATE,
            'llm_model': OPENAI_LLM_MODEL,
            'tts_model': DEEPGRAM_TTS_MODEL,
            'preflight_threshold': PREFLIGHT_THRESHOLD,
            'eot_threshold': EOT_THRESHOLD,
            'eot_timeout_ms': EOT_TIMEOUT_MS,
        },
        'flux_ws': None,
        'conversation_active': False,
    }

    emit('connected', {'session_id': session_id})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    session_id = request.sid
    logger.info(f"Client disconnected: {session_id}")

    # Clean up session
    if session_id in active_sessions:
        # Close any active WebSocket connections
        session = active_sessions[session_id]
        if session.get('flux_ws'):
            # Close Flux WebSocket if active
            pass

        del active_sessions[session_id]

@socketio.on('update_config')
def handle_config_update(data):
    """Handle configuration updates from client."""
    session_id = request.sid
    logger.info(f"Session {session_id}: Config update received: {data}")

    if session_id in active_sessions:
        active_sessions[session_id]['config'].update(data)
        logger.info(f"Session {session_id}: Configuration updated")
        emit('config_updated', {'status': 'success'})
    else:
        emit('config_updated', {'status': 'error', 'message': 'Session not found'})

@socketio.on('start_conversation')
def handle_start_conversation():
    """Start a conversation session."""
    session_id = request.sid
    logger.info(f"Session {session_id}: Starting conversation")

    if session_id not in active_sessions:
        emit('conversation_error', {'error': 'Session not found'})
        return

    session = active_sessions[session_id]
    session['state'] = ConversationState.LISTENING
    session['conversation_active'] = True
    session['messages'] = []  # Reset conversation history

    # Start Flux WebSocket connection in a separate thread
    def start_flux_connection():
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        loop.run_until_complete(connect_to_flux(session_id))

    thread = threading.Thread(target=start_flux_connection)
    thread.daemon = True
    thread.start()

    emit('conversation_started', {
        'timestamp': datetime.now().isoformat(),
        'config': session['config']
    })

@socketio.on('audio_data')
def handle_audio_data(data):
    """Handle incoming audio data from client."""
    session_id = request.sid

    if session_id not in active_sessions:
        return

    session = active_sessions[session_id]

    if not session.get('conversation_active', False):
        return

    # Initialize audio buffer if not exists
    if 'audio_buffer' not in session:
        session['audio_buffer'] = []

    # Convert audio data to bytes and add to buffer
    if isinstance(data, list):
        # Convert JavaScript array to bytes
        audio_bytes = struct.pack(f'{len(data)}h', *data)
        session['audio_buffer'].append(audio_bytes)

@socketio.on('stop_conversation')
def handle_stop_conversation():
    """Stop the current conversation."""
    session_id = request.sid
    logger.info(f"Session {session_id}: Stopping conversation")

    if session_id in active_sessions:
        session = active_sessions[session_id]
        session['state'] = ConversationState.IDLE
        session['conversation_active'] = False

        # Close Flux WebSocket if active
        if session.get('flux_ws'):
            # Signal to close the WebSocket
            session['should_close'] = True

    emit('conversation_stopped', {
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('audio_data')
def handle_audio_data(data):
    """Handle incoming audio data from client."""
    session_id = request.sid

    if session_id not in active_sessions:
        return

    session = active_sessions[session_id]
    if not session['conversation_active']:
        return

    logger.debug(f"Session {session_id}: Received audio data: {len(data)} bytes")

    # Store audio data for Flux WebSocket to consume
    if 'audio_buffer' not in session:
        session['audio_buffer'] = []

    # Convert binary data to bytes if needed
    if isinstance(data, (list, tuple)):
        # If it's still coming as array, convert it
        audio_bytes = struct.pack(f'{len(data)}h', *data)
    else:
        # If it's already binary data, use it directly
        audio_bytes = bytes(data)

    session['audio_buffer'].append(audio_bytes)

async def connect_to_flux(session_id: str):
    """Connect to Deepgram Flux WebSocket and handle the conversation."""

    logger.info(f"Session {session_id}: Connecting to Flux WebSocket")

    if session_id not in active_sessions:
        logger.error(f"Session {session_id}: Session not found")
        return

    session = active_sessions[session_id]
    config = session['config']

    # Build Flux WebSocket URL with parameters (matching reference code)
    flux_url = f"{FLUX_URL}?model=flux-general-en&sample_rate={config['sample_rate']}&encoding={FLUX_ENCODING}"

    headers = {
        'Authorization': f'Token {DEEPGRAM_API_KEY}',
        'User-Agent': 'DeepgramFluxVoiceAgent/1.0'
    }

    try:
        async with websockets.connect(flux_url, additional_headers=headers) as websocket:
            session['flux_ws'] = websocket
            logger.info(f"Session {session_id}: Connected to Flux WebSocket")

            # Send audio and handle responses concurrently
            audio_task = asyncio.create_task(send_audio_to_flux(session_id, websocket))
            response_task = asyncio.create_task(handle_flux_responses(session_id, websocket))

            # Wait for either task to complete (or for shutdown signal)
            try:
                await asyncio.gather(audio_task, response_task)
            except Exception as e:
                logger.error(f"Session {session_id}: Flux connection error: {e}")
            finally:
                audio_task.cancel()
                response_task.cancel()
                logger.info(f"Session {session_id}: Flux connection closed")

                # Clean up session WebSocket reference
                session['flux_ws'] = None

                # Notify client of disconnection
                socketio.emit('flux_disconnected',
                            {'timestamp': datetime.now().isoformat()},
                            room=session_id)

    except Exception as e:
        logger.error(f"Session {session_id}: Failed to connect to Flux: {e}")
        socketio.emit('conversation_error',
                     {'error': f'Failed to connect to Flux: {str(e)}'},
                     room=session_id)

async def send_audio_to_flux(session_id: str, websocket):
    """Send audio data from client to Flux WebSocket."""

    logger.info(f"Session {session_id}: Starting audio transmission to Flux")

    session = active_sessions[session_id]

    try:
        while session['conversation_active'] and not session.get('should_close', False):
            # Check for buffered audio data
            if 'audio_buffer' in session and session['audio_buffer']:
                audio_bytes = session['audio_buffer'].pop(0)

                await websocket.send(audio_bytes)
                logger.debug(f"Session {session_id}: Sent {len(audio_bytes)} bytes to Flux")

            await asyncio.sleep(0.01)  # Small delay to prevent busy loop

    except Exception as e:
        logger.error(f"Session {session_id}: Error sending audio to Flux: {e}")

async def handle_flux_responses(session_id: str, websocket):
    """Handle responses from Flux WebSocket."""

    logger.info(f"Session {session_id}: Starting Flux response handler")

    session = active_sessions[session_id]
    config = session['config']

    try:
        async for message in websocket:
            if session.get('should_close', False):
                break

            try:
                data = json.loads(message)
                logger.debug(f"Session {session_id}: Flux response: {data}")

                # Handle different Flux event types
                if data.get('type') == 'receiveConnected':
                    logger.info(f"Session {session_id}: Connected to Flux - ready to stream audio")

                elif data.get('type') == 'receiveFatalError':
                    logger.error(f"Session {session_id}: Fatal error: {data.get('error', 'Unknown error')}")
                    socketio.emit('conversation_error', {
                        'error': f"Flux fatal error: {data.get('error', 'Unknown error')}",
                        'timestamp': datetime.now().isoformat()
                    }, room=session_id)

                elif data.get('type') == 'TurnInfo':
                    event = data.get('event')
                    logger.debug(f"Session {session_id}: TurnInfo event: {event}")

                    if event in ['StartOfTurn', 'SpeechResumed']:
                        logger.info(f"Session {session_id}: {event} - User started speaking")
                        session['state'] = ConversationState.LISTENING
                        socketio.emit('speech_started', {
                            'timestamp': datetime.now().isoformat()
                        }, room=session_id)

                    elif event == 'Preflight':
                        # Preflighting request from Flux
                        if config['use_preflighting']:
                            tentative_transcript = data.get('transcript', '')
                            logger.info(f"Session {session_id}: Preflight request for: '{tentative_transcript}'")

                            # Generate preflight response
                            asyncio.create_task(handle_preflight_request(session_id, tentative_transcript, config))

                    elif event == 'EndOfTurn':
                        # User finished speaking
                        transcript = data.get('transcript', '')
                        logger.info(f"Session {session_id}: EndOfTurn event - transcript: '{transcript}', use_preflighting: {config['use_preflighting']}")

                        if transcript.strip():
                            logger.info(f"Session {session_id}: User said: '{transcript}' - PROCESSING USER SPEECH")

                            # Add to conversation history
                            session['messages'].append({"role": "user", "content": transcript})

                            # Notify client
                            socketio.emit('user_speech', {
                                'transcript': transcript,
                                'timestamp': datetime.now().isoformat()
                            }, room=session_id)

                            # Generate and send agent response (for non-preflighting mode)
                            if not config['use_preflighting']:
                                logger.info(f"Session {session_id}: NON-PREFLIGHTING MODE - Starting LLM generation task")
                                asyncio.create_task(generate_and_send_response(session_id, transcript, config))
                            else:
                                logger.info(f"Session {session_id}: PREFLIGHTING MODE - Should use preflight response (if available)")
                        else:
                            logger.warning(f"Session {session_id}: EndOfTurn event but transcript is empty!")

                    elif event == 'Update':
                        # Interim transcript updates
                        transcript = data.get('transcript', '')
                        if transcript.strip():
                            socketio.emit('interim_transcript', {
                                'transcript': transcript,
                                'is_final': False,
                                'timestamp': datetime.now().isoformat()
                            }, room=session_id)

                # Forward raw Flux events to client for debugging
                socketio.emit('flux_event', {
                    'event_type': data.get('type'),
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                }, room=session_id)

            except json.JSONDecodeError as e:
                logger.error(f"Session {session_id}: Invalid JSON from Flux: {e}")
            except Exception as e:
                logger.error(f"Session {session_id}: Error processing Flux message: {e}")

    except Exception as e:
        logger.error(f"Session {session_id}: Error in Flux response handler: {e}")

async def generate_and_send_response(session_id: str, user_speech: str, config: Dict[str, Any]):
    """Generate and send agent response (non-preflighting mode)."""

    logger.info(f"Session {session_id}: *** STARTING AGENT RESPONSE GENERATION ***")
    logger.info(f"Session {session_id}: User speech: '{user_speech}'")
    logger.info(f"Session {session_id}: Config: {config}")

    session = active_sessions[session_id]
    session['state'] = ConversationState.PROCESSING

    socketio.emit('agent_processing', {
        'timestamp': datetime.now().isoformat()
    }, room=session_id)

    try:
        logger.info(f"Session {session_id}: About to call generate_agent_reply_normal()")
        # Generate response
        audio_data = await generate_agent_reply_normal(
            session['messages'],
            user_speech,
            session_id,
            config
        )
        logger.info(f"Session {session_id}: generate_agent_reply_normal() returned: {len(audio_data) if audio_data else 0} bytes")

        if audio_data:
            # Add agent message to conversation history
            # (We'd need to track what the agent said, but for now we'll skip this)

            session['state'] = ConversationState.SPEAKING

            # Send audio to client
            socketio.emit('agent_speaking', {
                'audio': list(audio_data),  # Convert bytes to list for JSON serialization
                'timestamp': datetime.now().isoformat()
            }, room=session_id)

            logger.info(f"Session {session_id}: Sent {len(audio_data)} bytes of agent audio")

        session['state'] = ConversationState.LISTENING

    except Exception as e:
        logger.error(f"Session {session_id}: *** ERROR IN generate_and_send_response() ***")
        logger.error(f"Session {session_id}: Error details: {e}")
        import traceback
        logger.error(f"Session {session_id}: Full traceback: {traceback.format_exc()}")
        session['state'] = ConversationState.ERROR

        socketio.emit('agent_error', {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }, room=session_id)

async def connect_to_flux(session_id: str):
    """Connect to Flux WebSocket and handle voice conversation (non-preflighting mode)."""

    if session_id not in active_sessions:
        logger.error(f"Session {session_id}: Session not found")
        return

    session = active_sessions[session_id]
    config = session['config']

    logger.info(f"Session {session_id}: Connecting to Flux WebSocket")

    # Build Flux WebSocket URL (non-preflighting mode)
    flux_url = (
        f"{FLUX_URL}?model=flux-general-en&sample_rate={config['sample_rate']}&encoding={FLUX_ENCODING}"
        f"&eot_threshold={config['eot_threshold']}&eot_timeout_ms={config['eot_timeout_ms']}"
    )

    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

    try:
        async with websockets.connect(flux_url, additional_headers=headers) as websocket:
            logger.info(f"Session {session_id}: Connected to Flux WebSocket")

            # Store websocket reference
            session['flux_ws'] = websocket

            socketio.emit('conversation_started', {
                'timestamp': datetime.now().isoformat()
            }, room=session_id)

            # Start audio transmission and message handling
            await asyncio.gather(
                send_audio_to_flux(session_id, websocket),
                handle_flux_responses(session_id, websocket, config)
            )

    except Exception as e:
        logger.error(f"Session {session_id}: Failed to connect to Flux: {e}")
        socketio.emit('conversation_error', {
            'error': f"Failed to connect to Flux: {str(e)}",
            'timestamp': datetime.now().isoformat()
        }, room=session_id)

async def handle_flux_responses(session_id: str, websocket, config: Dict[str, Any]):
    """Handle incoming messages from Flux WebSocket (non-preflighting mode)."""

    session = active_sessions[session_id]

    try:
        async for message in websocket:
            if not session.get('conversation_active', False):
                break

            data = json.loads(message)
            logger.debug(f"Session {session_id}: Flux message: {data.get('type')}")

            if data.get('type') == 'receiveConnected':
                logger.info(f"Session {session_id}: Connected to Flux - ready to stream audio")

            elif data.get('type') == 'receiveFatalError':
                logger.error(f"Session {session_id}: Fatal error: {data.get('error', 'Unknown error')}")
                socketio.emit('conversation_error', {
                    'error': f"Flux fatal error: {data.get('error', 'Unknown error')}",
                    'timestamp': datetime.now().isoformat()
                }, room=session_id)

            elif data.get('type') == 'TurnInfo':
                event = data.get('event')

                if event in ['StartOfTurn', 'SpeechResumed']:
                    logger.info(f"Session {session_id}: {event} - User started speaking")
                    session['state'] = ConversationState.LISTENING
                    socketio.emit('speech_started', {
                        'timestamp': datetime.now().isoformat()
                    }, room=session_id)

                elif event == 'EndOfTurn':
                    transcript = data.get('transcript', '')
                    logger.info(f"Session {session_id}: EndOfTurn - transcript: '{transcript}'")

                    if transcript.strip():
                        session['messages'].append({"role": "user", "content": transcript})
                        socketio.emit('user_speech', {
                            'transcript': transcript,
                            'timestamp': datetime.now().isoformat()
                        }, room=session_id)

                        # Generate agent response (non-preflighting mode)
                        asyncio.create_task(generate_and_send_response(session_id, transcript, config))

                elif event == 'Update':
                    transcript = data.get('transcript', '')
                    if transcript.strip():
                        socketio.emit('interim_transcript', {
                            'transcript': transcript,
                            'is_final': False,
                            'timestamp': datetime.now().isoformat()
                        }, room=session_id)

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Session {session_id}: Flux WebSocket connection closed")
    except Exception as e:
        logger.error(f"Session {session_id}: Error in Flux message handling: {e}")

async def send_audio_to_flux(session_id: str, websocket):
    """Send audio data to Flux WebSocket."""

    session = active_sessions[session_id]
    logger.info(f"Session {session_id}: Starting audio transmission to Flux")

    try:
        while session.get('conversation_active', False):
            # Check if there's audio data in buffer
            if session.get('audio_buffer'):
                while session['audio_buffer']:
                    audio_data = session['audio_buffer'].pop(0)
                    await websocket.send(audio_data)

            await asyncio.sleep(0.01)  # Small delay to prevent busy waiting

    except Exception as e:
        logger.error(f"Session {session_id}: Error sending audio to Flux: {e}")

if __name__ == '__main__':
    try:
        # Validate API keys before starting
        validate_api_keys()

        logger.info(f"Starting Deepgram Flux Voice Agent on {HOST}:{PORT}")
        logger.info("Make sure you have set DEEPGRAM_API_KEY and OPENAI_API_KEY environment variables")

        # Run the app
        socketio.run(app,
                    host=HOST,
                    port=PORT,
                    debug=DEBUG,
                    use_reloader=False)  # Disable reloader to prevent issues with threading

    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        exit(1)
