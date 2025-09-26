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
import concurrent.futures
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from enum import Enum, auto

import openai
import websockets
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from deepgram import (
    AsyncDeepgramClient,
    DeepgramClient,
)
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import (
    ListenV2SocketClientResponse,
    ListenV2MediaMessage,
    SpeakV1SocketClientResponse,
    SpeakV1ControlMessage,
    SpeakV1TextMessage,
)

from config import (
    HOST, PORT, DEBUG, BASE_PATH,
    DEEPGRAM_API_KEY, OPENAI_API_KEY,
    FLUX_URL, FLUX_ENCODING, SAMPLE_RATE,
    OPENAI_LLM_MODEL, DEEPGRAM_TTS_MODEL,
    EOT_THRESHOLD, EOT_TIMEOUT_MS,
    SYSTEM_PROMPT,
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
# Handle static path consistently
app = Flask(__name__, static_url_path=f'{BASE_PATH}/static')
app.config['SECRET_KEY'] = os.urandom(24)

# No Blueprint needed - using direct route

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', path=f'{BASE_PATH}/socket.io')

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
    """Generate agent reply using the configured LLM."""

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

        # Generate TTS audio
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

def generate_tts_audio_sync(text: str, session_id: str, config: Dict[str, Any]) -> Optional[bytes]:
    """Generate TTS audio for given text using the Deepgram SDK."""

    logger.info(f"Session {session_id}: *** STARTING TTS GENERATION ***")
    logger.info(f"Session {session_id}: TTS Text: '{text}'")
    logger.info(f"Session {session_id}: TTS Model: {config['tts_model']}")
    logger.info(f"Session {session_id}: API Key Available: {'Yes' if DEEPGRAM_API_KEY else 'No'}")

    try:
        # Create synchronous Deepgram client (like original)
        client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
        logger.info(f"Session {session_id}: Created async Deepgram client for TTS")

        audio_chunks = []
        audio_chunks_received = 0
        is_flushed = False

        # Connect using synchronous SDK context manager (apples-to-apples)
        with client.speak.v1.connect(
            model=config['tts_model'],
            encoding="linear16",
            sample_rate=str(config['sample_rate'])
        ) as connection:
            logger.info(f"Session {session_id}: TTS sync connection established")

            # Send text message using synchronous SDK (like original)
            text_message = SpeakV1TextMessage(type="Speak", text=text)
            connection.send_text(text_message)
            logger.info(f"Session {session_id}: Text sent to TTS")

            # Send flush command using synchronous SDK
            connection.send_control(SpeakV1ControlMessage(type="Flush"))
            logger.info(f"Session {session_id}: Flush command sent")

            # Receive messages synchronously (similar to original pattern)
            total_timeout = 10.0
            start_time = time.time()

            while (time.time() - start_time) < total_timeout:
                try:
                    # Receive message synchronously
                    message = connection.recv()

                    if isinstance(message, bytes):
                        # Audio data received - STREAM IMMEDIATELY
                        audio_chunks_received += 1
                        audio_chunks.append(message)
                        logger.debug(f"Session {session_id}: *** TTS AUDIO CHUNK #{audio_chunks_received} *** {len(message)} bytes")

                        # 🚀 STREAM CHUNK IMMEDIATELY
                        socketio.emit('agent_speaking', {
                            'audio': list(message),
                            'chunk_number': audio_chunks_received,
                            'timestamp': datetime.now().isoformat()
                        }, room=session_id)

                    else:
                        # Handle non-audio messages
                        msg_type = getattr(message, 'type', 'Unknown')
                        logger.info(f"Session {session_id}: TTS message: {msg_type}")

                        if msg_type == 'Flushed':
                            logger.info(f"Session {session_id}: *** TTS FLUSHED - COMPLETE ***")
                            break  # Exit the loop when flushed

                except Exception as recv_error:
                    logger.debug(f"Session {session_id}: TTS recv error (may be normal): {recv_error}")
                    break  # Exit on any receive error

            logger.info(f"Session {session_id}: TTS sync collection complete")

        # Return results
        if audio_chunks:
            combined_audio = b''.join(audio_chunks)
            logger.info(f"Session {session_id}: *** TTS SYNC SUCCESS *** Total chunks: {len(audio_chunks)}, Total bytes: {len(combined_audio)}")
            return combined_audio
        else:
            logger.error(f"Session {session_id}: *** TTS SYNC FAILED - NO AUDIO CHUNKS RECEIVED ***")
            return None

    except Exception as e:
        logger.error(f"Session {session_id}: *** TTS SYNC EXCEPTION *** {e}")
        import traceback
        logger.error(f"Session {session_id}: TTS sync traceback: {traceback.format_exc()}")
        return None


# Async wrapper to call sync TTS from async context
async def generate_tts_audio(text: str, session_id: str, config: Dict[str, Any]) -> Optional[bytes]:
    """Async wrapper for sync TTS function - maintains compatibility."""
    import asyncio
    import concurrent.futures

    # Run sync TTS function in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, generate_tts_audio_sync, text, session_id, config)
        return result

# Flask routes
@app.route('/flux-agent/')
def index():
    """Serve the main application page."""
    return render_template('index.html',
                         tts_models=TTS_MODEL_OPTIONS,
                         llm_models=LLM_MODEL_OPTIONS,
                         base_path=BASE_PATH)

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
            'sample_rate': SAMPLE_RATE,
            'llm_model': OPENAI_LLM_MODEL,
            'tts_model': DEEPGRAM_TTS_MODEL,
            'eot_threshold': EOT_THRESHOLD,
            'eot_timeout_ms': EOT_TIMEOUT_MS,
        },
        'flux_connection': None,
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
        # Close any active SDK connections
        session = active_sessions[session_id]
        if session.get('flux_connection'):
            # Close Flux SDK connection if active
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


@socketio.on('stop_conversation')
def handle_stop_conversation():
    """Stop the current conversation."""
    session_id = request.sid
    logger.info(f"Session {session_id}: Stopping conversation")

    if session_id in active_sessions:
        session = active_sessions[session_id]
        session['state'] = ConversationState.IDLE
        session['conversation_active'] = False

        # Close SDK connection if active
        if session.get('flux_connection'):
            # Signal to close the SDK connection immediately
            session['should_close'] = True
            logger.info(f"Session {session_id}: Signaled SDK connection to close gracefully")

        # Clear audio buffer
        if 'audio_buffer' in session:
            session['audio_buffer'].clear()

        logger.info(f"Session {session_id}: Conversation stopped and cleaned up")

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
    """Connect to Deepgram SDK using SDK Listen v2 and handle the conversation."""

    logger.info(f"Session {session_id}: Connecting using SDK")

    if session_id not in active_sessions:
        logger.error(f"Session {session_id}: Session not found")
        return

    session = active_sessions[session_id]
    config = session['config']

    # Create async Deepgram client
    client = AsyncDeepgramClient(api_key=DEEPGRAM_API_KEY)

    try:
        # Connect using SDK Listen v2
        async with client.listen.v2.connect(
            model="flux-general-en",
            encoding=FLUX_ENCODING,
            sample_rate=str(config['sample_rate'])
        ) as connection:
            session['flux_connection'] = connection
            logger.info(f"Session {session_id}: Connected to SDK")

            # Set up event handlers
            def on_open(open_event):
                logger.info(f"Session {session_id}: SDK connection opened - ready to stream audio")
                # Send connected event to match original behavior
                socketio.emit('flux_event', {
                    'event_type': 'receiveConnected',
                    'data': {'type': 'receiveConnected'},
                    'timestamp': datetime.now().isoformat()
                }, room=session_id)

            def on_message(message: ListenV2SocketClientResponse):
                logger.debug(f"Session {session_id}: SDK message received: {message}")
                handle_flux_sdk_message(session_id, message)

            def on_close(close_event):
                logger.info(f"Session {session_id}: SDK connection closed")
                socketio.emit('flux_disconnected',
                            {'timestamp': datetime.now().isoformat()},
                            room=session_id)

            def on_error(error):
                logger.error(f"Session {session_id}: SDK connection error: {error}")
                socketio.emit('conversation_error',
                            {'error': f'SDK error: {str(error)}'},
                            room=session_id)

            # Register event handlers
            connection.on(EventType.OPEN, on_open)
            connection.on(EventType.MESSAGE, on_message)
            connection.on(EventType.CLOSE, on_close)
            connection.on(EventType.ERROR, on_error)

            # Start listening and audio sending concurrently
            listen_task = asyncio.create_task(connection.start_listening())
            audio_task = asyncio.create_task(send_audio_to_flux_sdk(session_id, connection))

            # Wait for either task to complete (or for shutdown signal)
            try:
                await asyncio.gather(listen_task, audio_task)
            except Exception as e:
                logger.error(f"Session {session_id}: SDK connection error: {e}")
            finally:
                # Cancel tasks
                if not listen_task.done():
                    logger.info(f"Session {session_id}: Cancelling listen task")
                    listen_task.cancel()
                if not audio_task.done():
                    logger.info(f"Session {session_id}: Cancelling audio task")
                    audio_task.cancel()

                # Wait for tasks to finish cancellation
                try:
                    await asyncio.gather(listen_task, audio_task, return_exceptions=True)
                    logger.info(f"Session {session_id}: Both SDK tasks completed/cancelled")
                except Exception as cleanup_error:
                    logger.warning(f"Session {session_id}: SDK task cleanup error: {cleanup_error}")

                # Clean up session state
                session['flux_connection'] = None
                session['should_close'] = False

                logger.info(f"Session {session_id}: SDK connection fully closed and cleaned up")

    except Exception as e:
        logger.error(f"Session {session_id}: Failed to connect to SDK: {e}")
        socketio.emit('conversation_error',
                     {'error': f'Failed to connect to SDK: {str(e)}'},
                     room=session_id)

async def send_audio_to_flux_sdk(session_id: str, connection):
    """Send audio data from client to Flux API using SDK connection."""

    logger.info(f"Session {session_id}: Starting audio transmission to SDK")

    session = active_sessions[session_id]

    try:
        while session['conversation_active'] and not session.get('should_close', False):
            # Check for buffered audio data
            if 'audio_buffer' in session and session['audio_buffer']:
                audio_bytes = session['audio_buffer'].pop(0)

                # Send audio using SDK connection
                await connection.send_media(audio_bytes)
                logger.debug(f"Session {session_id}: Sent {len(audio_bytes)} bytes sent to SDK")

            await asyncio.sleep(0.01)  # Small delay to prevent busy loop

        # If we exit the loop due to should_close flag
        if session.get('should_close', False):
            logger.info(f"Session {session_id}: Audio transmission stopping due to close signal")

    except Exception as e:
        logger.error(f"Session {session_id}: Error sending audio to SDK: {e}")
    finally:
        logger.info(f"Session {session_id}: Audio transmission to SDK ended")

def handle_flux_sdk_message(session_id: str, message: ListenV2SocketClientResponse):
    """Handle messages from SDK connection."""

    if session_id not in active_sessions:
        return

    session = active_sessions[session_id]
    config = session['config']

    # Check if we should close the connection
    if session.get('should_close', False) or not session.get('conversation_active', False):
        return

    try:
        # Convert SDK message to dict for processing
        if hasattr(message, 'model_dump'):
            data = message.model_dump()
        elif hasattr(message, 'dict'):
            data = message.dict()
        else:
            # Fallback: try to access attributes directly
            data = {
                'type': getattr(message, 'type', None),
                'event': getattr(message, 'event', None),
                'transcript': getattr(message, 'transcript', None),
                'error': getattr(message, 'error', None),
            }

        logger.debug(f"Session {session_id}: SDK message: {data}")

        # Handle different Flux event types
        message_type = data.get('type')

        if message_type == 'receiveConnected':
            logger.info(f"Session {session_id}: Connected to SDK - ready to stream audio")

        elif message_type == 'receiveFatalError':
            error_msg = data.get('error', 'Unknown error')
            logger.error(f"Session {session_id}: Fatal error: {error_msg}")
            socketio.emit('conversation_error', {
                'error': f"SDK fatal error: {error_msg}",
                'timestamp': datetime.now().isoformat()
            }, room=session_id)

        elif message_type == 'TurnInfo':
            event = data.get('event')
            logger.debug(f"Session {session_id}: TurnInfo event: {event}")

            if event == 'StartOfTurn':
                logger.info(f"Session {session_id}: {event} - User started speaking")
                session['state'] = ConversationState.LISTENING
                socketio.emit('speech_started', {
                    'timestamp': datetime.now().isoformat()
                }, room=session_id)

            elif event == 'EndOfTurn':
                # User finished speaking
                transcript = data.get('transcript', '')
                logger.info(f"Session {session_id}: EndOfTurn event - transcript: '{transcript}'")

                if transcript.strip():
                    logger.info(f"Session {session_id}: User said: '{transcript}' - PROCESSING USER SPEECH")

                    # Add to conversation history
                    session['messages'].append({"role": "user", "content": transcript})

                    # Notify client
                    socketio.emit('user_speech', {
                        'transcript': transcript,
                        'timestamp': datetime.now().isoformat()
                    }, room=session_id)

                    # Generate and send agent response
                    logger.info(f"Session {session_id}: Starting LLM generation task")
                    asyncio.create_task(generate_and_send_response(session_id, transcript, config))
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
            'event_type': message_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }, room=session_id)

    except Exception as e:
        logger.error(f"Session {session_id}: Error processing SDK message: {e}")
        import traceback
        logger.error(f"Session {session_id}: Full traceback: {traceback.format_exc()}")

async def generate_and_send_response(session_id: str, user_speech: str, config: Dict[str, Any]):
    """Generate and send agent response."""

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

            # Audio was already streamed in real-time during TTS generation!
            # Just log completion (no duplicate audio sending)
            logger.info(f"Session {session_id}: Audio streaming completed - {len(audio_data)} bytes streamed in real-time via {len(audio_data)//1280 if audio_data else 0} chunks")

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
                    use_reloader=False,  # Disable reloader to prevent issues with threading
                    allow_unsafe_werkzeug=True)  # Allow development server in production

    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        exit(1)
