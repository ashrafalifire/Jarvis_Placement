J.A.R.V.I.S. – Intelligent CP Lab Automation System
📘 Overview
J.A.R.V.I.S. is an intelligent, multimodal lab automation system that integrates facial recognition, hand gesture control, and offline voice commands to interact with CP lab systems via Node-RED and OPC UA. It provides a fast, offline, and scalable control interface for smart labs.
🔄 System Workflow
1. Face Detection → Prompt for Control Mode
2. Mode Selection: Manual (Keyboard) / Gesture / Voice
3. Command Sent via WebSocket → Node-RED Flow
4. Node-RED sends OPC UA trigger to connected lab equipment
5. pyttsx3 provides voice-based confirmation feedback
💡 Schematic: Input to Machine Response
Gesture/Speech → Python Processing → WebSocket → Node-RED → OPC UA → CP Lab Switches
📚 Libraries Used vs. Custom Code
This project leverages robust open-source libraries with a custom control architecture built around them.

🔧 Libraries Used:
- OpenCV: Face and hand gesture detection
- Vosk: Offline speech-to-text engine
- pyttsx3: Text-to-speech feedback
- websocket-client: Communication with Node-RED
- opcua: For integrating OPC UA (through Node-RED nodes)
🧱 Project Structure:
- `face_recog.py`: Face detection and name registration
- `gesture_control.py`: Detects thumbs up/down and other gestures
- `voice_control.py`: Processes Vosk commands for local device control
- `manual_mode.py`: Keyboard input trigger to Node-RED
- Node-RED Flows: Handles command reception and OPC UA output
✅ All dependencies are defined in `requirements.txt` for quick setup.
🗣️ Voice & Gesture Mapping
- Voice Commands: 'Turn on Station 1', 'Lab status', 'Shutdown all'
- Gestures: Thumbs up = Power ON, Thumbs down = Power OFF, etc.
🛠️ Status
Currently implemented: Face detection, mode control, offline voice + gesture interaction, WebSocket communication with Node-RED, OPC UA switching.
Future plans: Add MES screen monitoring and fallback state validation.
