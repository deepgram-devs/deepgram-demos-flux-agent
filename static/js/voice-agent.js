/**
 * Deepgram Flux Voice Agent Client
 * Handles real-time voice conversation with Deepgram Flux, OpenAI, and TTS
 */

class VoiceAgent {
  constructor() {
    this.socket = null;
    this.mediaStream = null;
    this.audioContext = null;
    this.processor = null;
    this.isConnected = false;
    this.isConversationActive = false;
    this.isRecording = false;
    this.audioQueue = [];
    this.isPlayingAudio = false;
    this.selectedDeviceId = null;

    // Configuration
    this.config = {
      use_preflighting: false,  // Default to non-preflighting mode (simpler)
      sample_rate: 16000,
      llm_model: 'gpt-4o-mini',
      tts_model: 'aura-2-phoebe-en',
      preflight_threshold: 0.3,
      eot_threshold: 0.8,
      eot_timeout_ms: 3000
    };

    // UI elements
    this.elements = {};

    // Conversation history
    this.conversationHistory = [];

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => this.init());
    } else {
      this.init();
    }
  }

  async init() {
    console.log('Initializing Deepgram Flux Voice Agent...');

    try {
      // Get UI elements
      this.getUIElements();

      // Set up event handlers
      this.setupEventHandlers();

      // Load available microphones
      await this.loadMicrophones();

      // Initialize Socket.IO
      this.initSocket();

      // Update UI
      this.updateStatus('idle', 'Ready to start');
      this.updateInstructions('Select your microphone and configure settings, then click "Start Conversation"');

    } catch (error) {
      console.error('Initialization failed:', error);
      this.updateStatus('error', 'Initialization failed');
      this.addDebugMessage('ERROR', `Initialization failed: ${error.message}`);
    }
  }

  getUIElements() {
    // Control buttons
    this.elements.startBtn = document.getElementById('start-btn');
    this.elements.stopBtn = document.getElementById('stop-btn');
    this.elements.clearLogBtn = document.getElementById('clear-log-btn');
    this.elements.exportLogBtn = document.getElementById('export-log-btn');

    // Configuration controls
    this.elements.microphoneSelect = document.getElementById('microphone-select');
    this.elements.sampleRate = document.getElementById('sample-rate');
    this.elements.llmModel = document.getElementById('llm-model');
    this.elements.ttsModel = document.getElementById('tts-model');
    this.elements.usePreflighting = document.getElementById('use-preflighting');
    this.elements.preflightThreshold = document.getElementById('preflight-threshold');
    this.elements.eotThreshold = document.getElementById('eot-threshold');
    this.elements.eotTimeout = document.getElementById('eot-timeout');

    // Display elements
    this.elements.statusIndicator = document.getElementById('status-indicator');
    this.elements.statusText = document.getElementById('status-text');
    this.elements.instructionText = document.getElementById('instruction-text');
    this.elements.conversationLog = document.getElementById('conversation-log');
    this.elements.debugLog = document.getElementById('debug-log');
    this.elements.interimTranscript = document.getElementById('interim-transcript');
    this.elements.interimSection = document.getElementById('interim-section');

    // Range value displays
    this.elements.preflightValue = document.getElementById('preflight-value');
    this.elements.eotValue = document.getElementById('eot-value');
    this.elements.timeoutValue = document.getElementById('timeout-value');

    // Preflight threshold group
    this.elements.preflightGroup = document.getElementById('preflight-threshold-group');
  }

  setupEventHandlers() {
    // Control buttons
    this.elements.startBtn.addEventListener('click', () => this.startConversation());
    this.elements.stopBtn.addEventListener('click', () => this.stopConversation());
    this.elements.clearLogBtn.addEventListener('click', () => this.clearConversationLog());
    this.elements.exportLogBtn.addEventListener('click', () => this.exportConversation());

    // Configuration changes
    this.elements.microphoneSelect.addEventListener('change', (e) => {
      this.selectedDeviceId = e.target.value;
      this.updateStartButtonState();
    });

    this.elements.sampleRate.addEventListener('change', (e) => {
      this.config.sample_rate = parseInt(e.target.value);
      this.sendConfigUpdate();
    });

    this.elements.llmModel.addEventListener('change', (e) => {
      this.config.llm_model = e.target.value;
      this.sendConfigUpdate();
    });

    this.elements.ttsModel.addEventListener('change', (e) => {
      this.config.tts_model = e.target.value;
      this.sendConfigUpdate();
    });

    this.elements.usePreflighting.addEventListener('change', (e) => {
      this.config.use_preflighting = e.target.checked;
      this.elements.preflightGroup.style.display = e.target.checked ? 'block' : 'none';
      this.sendConfigUpdate();
    });

    // Range sliders
    this.elements.preflightThreshold.addEventListener('input', (e) => {
      const value = parseFloat(e.target.value);
      this.config.preflight_threshold = value;
      this.elements.preflightValue.textContent = value.toFixed(1);
      this.sendConfigUpdate();
    });

    this.elements.eotThreshold.addEventListener('input', (e) => {
      const value = parseFloat(e.target.value);
      this.config.eot_threshold = value;
      this.elements.eotValue.textContent = value.toFixed(1);
      this.sendConfigUpdate();
    });

    this.elements.eotTimeout.addEventListener('input', (e) => {
      const value = parseInt(e.target.value);
      this.config.eot_timeout_ms = value;
      this.elements.timeoutValue.textContent = value;
      this.sendConfigUpdate();
    });

    // Page cleanup
    window.addEventListener('beforeunload', () => this.cleanup());
  }

  async loadMicrophones() {
    try {
      // Request microphone permission first
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(track => track.stop()); // Stop the stream immediately

      // Now enumerate devices
      const devices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = devices.filter(device => device.kind === 'audioinput');

      // Clear existing options
      this.elements.microphoneSelect.innerHTML = '';

      if (audioInputs.length === 0) {
        this.elements.microphoneSelect.innerHTML = '<option value="">No microphones found</option>';
        return;
      }

      // Add default option
      const defaultOption = document.createElement('option');
      defaultOption.value = '';
      defaultOption.textContent = 'Select microphone...';
      this.elements.microphoneSelect.appendChild(defaultOption);

      // Add microphone options
      audioInputs.forEach((device, index) => {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.textContent = device.label || `Microphone ${index + 1}`;
        this.elements.microphoneSelect.appendChild(option);

        // Select the first microphone by default
        if (index === 0) {
          option.selected = true;
          this.selectedDeviceId = device.deviceId;
        }
      });

      this.addDebugMessage('INFO', `Loaded ${audioInputs.length} microphone(s)`);
      this.updateStartButtonState();

    } catch (error) {
      console.error('Failed to load microphones:', error);
      this.elements.microphoneSelect.innerHTML = '<option value="">Microphone access denied</option>';
      this.addDebugMessage('ERROR', `Failed to load microphones: ${error.message}`);
    }
  }

  initSocket() {
    console.log('Connecting to server...');

    this.socket = io();

    this.socket.on('connect', () => {
      console.log('Connected to server');
      this.isConnected = true;
      this.updateStatus('idle', 'Connected');
      this.updateStartButtonState();
      this.addDebugMessage('SOCKET', 'Connected to server');
    });

    this.socket.on('disconnect', () => {
      console.log('Disconnected from server');
      this.isConnected = false;
      this.updateStatus('idle', 'Disconnected');
      this.updateStartButtonState();
      this.addDebugMessage('SOCKET', 'Disconnected from server');
    });

    this.socket.on('connected', (data) => {
      console.log('Server acknowledgment:', data);
      this.addDebugMessage('SOCKET', `Session ID: ${data.session_id}`);
    });

    this.socket.on('config_updated', (data) => {
      if (data.status === 'success') {
        console.log('Configuration updated successfully');
      } else {
        console.error('Configuration update failed:', data.message);
        this.addDebugMessage('ERROR', `Config update failed: ${data.message}`);
      }
    });

    this.socket.on('conversation_started', (data) => {
      console.log('Conversation started:', data);
      this.isConversationActive = true;
      this.updateStatus('listening', 'Listening...');
      this.updateControlButtons();
      this.elements.interimSection.style.display = 'block';
      this.updateInstructions('Conversation started! Speak naturally.');
      this.addConversationMessage('system', 'Conversation started', data.timestamp);
      this.addDebugMessage('FLUX', 'Conversation started');
    });

    this.socket.on('conversation_stopped', (data) => {
      console.log('Conversation stopped:', data);
      this.isConversationActive = false;
      this.updateStatus('idle', 'Conversation stopped');
      this.updateControlButtons();
      this.elements.interimSection.style.display = 'none';
      this.updateInstructions('Conversation stopped. Click "Start Conversation" to begin again.');
      this.addConversationMessage('system', 'Conversation ended', data.timestamp);
      this.addDebugMessage('FLUX', 'Conversation stopped');
    });

    this.socket.on('user_speech', (data) => {
      console.log('User speech detected:', data.transcript);
      this.addConversationMessage('user', data.transcript, data.timestamp);
      this.elements.interimTranscript.textContent = 'Processing...';
      this.addDebugMessage('USER', `"${data.transcript}"`);
    });

    this.socket.on('agent_speaking', (data) => {
      console.log('Agent speaking:', data);
      this.updateStatus('speaking', 'Agent speaking...');

      if (data.audio && data.audio.length > 0) {
        // Convert array back to audio data and play
        const audioData = new Int16Array(data.audio);
        this.addAudioToQueue(audioData);
        this.addDebugMessage('AGENT', `Playing ${data.audio.length} bytes of audio`);
      }
    });

    this.socket.on('agent_processing', (data) => {
      console.log('Agent processing...');
      this.updateStatus('processing', 'Agent thinking...');
      this.elements.interimTranscript.textContent = 'Agent is thinking...';
      this.addDebugMessage('AGENT', 'Processing response');
    });

    this.socket.on('speech_started', (data) => {
      console.log('User started speaking');
      this.updateStatus('listening', 'Listening...');
      this.elements.interimTranscript.textContent = 'Listening...';
      this.addDebugMessage('USER', 'Started speaking');
    });

    this.socket.on('interim_transcript', (data) => {
      if (data.transcript && data.transcript.trim()) {
        this.elements.interimTranscript.textContent = data.transcript;
        if (data.is_final) {
          this.addDebugMessage('TRANSCRIPT', `Final: "${data.transcript}"`);
        } else {
          this.addDebugMessage('TRANSCRIPT', `Interim: "${data.transcript}"`);
        }
      }
    });

    this.socket.on('flux_event', (data) => {
      // Log all Flux events for debugging
      this.addDebugMessage('FLUX', `${data.event_type}: ${JSON.stringify(data.data, null, 2)}`);
    });

    this.socket.on('flux_disconnected', (data) => {
      console.log('Flux WebSocket disconnected');
      this.isConversationActive = false;
      this.updateStatus('idle', 'Flux disconnected');
      this.updateControlButtons();
      this.addDebugMessage('FLUX', 'WebSocket disconnected');
    });

    this.socket.on('conversation_error', (data) => {
      console.error('Conversation error:', data.error);
      this.updateStatus('error', `Error: ${data.error}`);
      this.addConversationMessage('system', `Error: ${data.error}`, new Date().toISOString());
      this.addDebugMessage('ERROR', data.error);
    });

    this.socket.on('agent_error', (data) => {
      console.error('Agent error:', data.error);
      this.updateStatus('error', `Agent error: ${data.error}`);
      this.addDebugMessage('AGENT', `Error: ${data.error}`);
    });

    this.socket.on('agent_response', (data) => {
      console.log('Agent response:', data.response);
      this.addConversationMessage('agent', data.response, data.timestamp);
      this.addDebugMessage('AGENT', `"${data.response}"`);
    });
  }

  sendConfigUpdate() {
    if (this.socket && this.isConnected) {
      this.socket.emit('update_config', this.config);
      this.addDebugMessage('CONFIG', `Updated: ${JSON.stringify(this.config, null, 2)}`);
    }
  }

  async startConversation() {
    if (!this.isConnected) {
      this.addDebugMessage('ERROR', 'Not connected to server');
      return;
    }

    if (!this.selectedDeviceId) {
      this.updateInstructions('Please select a microphone first');
      return;
    }

    try {
      // Create audio context
      this.audioContext = new AudioContext({
        sampleRate: this.config.sample_rate
      });

      // Get microphone stream
      const constraints = {
        audio: {
          deviceId: this.selectedDeviceId ? { exact: this.selectedDeviceId } : undefined,
          channelCount: 1,
          sampleRate: this.config.sample_rate,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        }
      };

      this.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);

      // Set up audio processing
      this.setupAudioProcessing();

      // Start conversation on server
      this.socket.emit('start_conversation');

      this.addDebugMessage('AUDIO', `Started audio capture at ${this.config.sample_rate}Hz`);

    } catch (error) {
      console.error('Failed to start conversation:', error);
      this.updateStatus('error', 'Failed to start');
      this.updateInstructions('Failed to access microphone. Please check permissions.');
      this.addDebugMessage('ERROR', `Failed to start: ${error.message}`);
    }
  }

  setupAudioProcessing() {
    const source = this.audioContext.createMediaStreamSource(this.mediaStream);
    const bufferSize = 2048;
    this.processor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

    source.connect(this.processor);
    this.processor.connect(this.audioContext.destination);

    let lastSendTime = 0;
    const sendInterval = 100; // Send every 100ms

    this.processor.onaudioprocess = (e) => {
      const now = Date.now();
      if (this.socket?.connected && this.isConversationActive && now - lastSendTime >= sendInterval) {
        const inputData = e.inputBuffer.getChannelData(0);
        const pcmData = this.convertFloatToPcm(inputData);

        // Send audio data to server as binary data
        this.socket.emit('audio_data', pcmData.buffer);
        lastSendTime = now;
      }
    };

    this.isRecording = true;
  }

  convertFloatToPcm(floatData) {
    const pcmData = new Int16Array(floatData.length);
    for (let i = 0; i < floatData.length; i++) {
      const s = Math.max(-1, Math.min(1, floatData[i]));
      pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return pcmData;
  }

  stopConversation() {
    this.isRecording = false;

    // Stop audio processing
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    // Stop conversation on server
    if (this.socket && this.isConnected) {
      this.socket.emit('stop_conversation');
    }

    // Clear audio queue
    this.audioQueue = [];
    this.isPlayingAudio = false;

    this.addDebugMessage('AUDIO', 'Stopped audio capture');
  }

  addAudioToQueue(audioData) {
    this.audioQueue.push(audioData);
    if (!this.isPlayingAudio) {
      this.playNextInQueue();
    }
  }

  async playNextInQueue() {
    if (this.audioQueue.length === 0) {
      this.isPlayingAudio = false;
      this.updateStatus('listening', 'Listening...');
      return;
    }

    this.isPlayingAudio = true;
    const audioData = this.audioQueue.shift();

    try {
      // Ensure we have an audio context for playback
      if (!this.audioContext || this.audioContext.state === 'closed') {
        this.audioContext = new AudioContext();
      }

      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }

      // Create buffer with correct sample rate for TTS audio (24000Hz)
      const buffer = this.audioContext.createBuffer(1, audioData.length, 24000);
      const channelData = buffer.getChannelData(0);

      // Convert Int16 to Float32
      for (let i = 0; i < audioData.length; i++) {
        channelData[i] = audioData[i] / 32768.0;
      }

      // Create and play source
      const source = this.audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(this.audioContext.destination);

      source.onended = () => {
        this.playNextInQueue();
      };

      source.start(0);

    } catch (error) {
      console.error('Error playing audio:', error);
      this.addDebugMessage('AUDIO', `Playback error: ${error.message}`);
      this.isPlayingAudio = false;
      this.playNextInQueue(); // Try next chunk
    }
  }

  updateStatus(state, text) {
    this.elements.statusIndicator.className = `status-indicator ${state}`;
    this.elements.statusText.textContent = text;
  }

  updateInstructions(text) {
    this.elements.instructionText.textContent = text;
  }

  updateStartButtonState() {
    const canStart = this.isConnected && this.selectedDeviceId && !this.isConversationActive;
    this.elements.startBtn.disabled = !canStart;
  }

  updateControlButtons() {
    this.elements.startBtn.disabled = this.isConversationActive || !this.isConnected || !this.selectedDeviceId;
    this.elements.stopBtn.disabled = !this.isConversationActive;
  }

  addConversationMessage(type, content, timestamp) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `conversation-message ${type}`;

    const timeDiv = document.createElement('div');
    timeDiv.className = 'timestamp';
    timeDiv.textContent = type === 'system' ? 'System' :
      type === 'user' ? 'You' : 'Assistant';

    if (timestamp) {
      const date = new Date(timestamp);
      timeDiv.textContent += ` - ${date.toLocaleTimeString()}`;
    }

    const contentDiv = document.createElement('div');
    contentDiv.textContent = content;

    messageDiv.appendChild(timeDiv);
    messageDiv.appendChild(contentDiv);

    this.elements.conversationLog.appendChild(messageDiv);
    this.elements.conversationLog.scrollTop = this.elements.conversationLog.scrollHeight;

    // Store in history
    this.conversationHistory.push({
      type,
      content,
      timestamp: timestamp || new Date().toISOString()
    });
  }

  addDebugMessage(category, message) {
    const timestamp = new Date().toLocaleTimeString();
    const debugText = `[${timestamp}] ${category}: ${message}\n`;

    const pre = this.elements.debugLog.querySelector('pre');
    pre.textContent += debugText;

    // Keep only last 100 lines
    const lines = pre.textContent.split('\n');
    if (lines.length > 100) {
      pre.textContent = lines.slice(-100).join('\n');
    }

    // Auto-scroll to bottom
    this.elements.debugLog.scrollTop = this.elements.debugLog.scrollHeight;
  }

  clearConversationLog() {
    this.elements.conversationLog.innerHTML = `
            <div class="conversation-message system">
                <div class="timestamp">System</div>
                <div>Conversation log cleared</div>
            </div>
        `;
    this.conversationHistory = [];
  }

  exportConversation() {
    if (this.conversationHistory.length === 0) {
      alert('No conversation to export');
      return;
    }

    const exportData = {
      timestamp: new Date().toISOString(),
      config: this.config,
      messages: this.conversationHistory
    };

    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });

    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `flux-conversation-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
    link.click();
  }

  cleanup() {
    if (this.isConversationActive) {
      this.stopConversation();
    }

    if (this.socket) {
      this.socket.close();
    }
  }
}

// Initialize the voice agent when the page loads
new VoiceAgent();
