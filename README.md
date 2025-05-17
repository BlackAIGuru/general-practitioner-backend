## VoiceAgentPDF-Backend

A backend service that enables voice-based interaction with PDF documents using speech recognition, natural language processing, and text-to-speech technologies.

### Overview

VoiceAgentPDF-Backend is a FastAPI application that allows users to:
- Upload PDF documents
- Interact with the content through voice commands
- Receive spoken responses about the document content
- Maintain a conversational context throughout the session

The service integrates with Deepgram for speech recognition and text-to-speech, and OpenAI for natural language understanding and response generation.

### Features
- Real-time speech-to-text transcription
- PDF document upload and processing
- Conversational AI with context awareness
- Text-to-speech response generation
- WebSocket-based communication for real-time interaction

### Prerequisites
- Python 3.10+
- Deepgram API key
- OpenAI API key

### Installation
1. Clone the repository:
```bash
git clone https://github.com/phoenix19950512/VoiceAgentPDF-Backend.git
cd VoiceAgentPDF-Backend
```
2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Create a `.env` file based on the example:
```bash
cp .env.example .env
```
5. Add your API keys to the `.env` file.
```env
DEEPGRAM_API_KEY=your_deepgram_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Usage

1. Start the server:
```bash
python main.py
```
2. The server will run at `http://localhost:8000`

### API Endpoints
#### Health Check
- `GET /health` - Check if the service is running
#### PDF Upload
- `POST /upload` - Upload a PDF file
   - Parameters:
     - `session_id` (form field): Unique session identifier
     - `file` (form field): PDF file to upload
   - Returns: File path and filename
#### WebSocket Connection
- `WebSocket /listen` - Establish a WebSocket connection for voice interaction
   - Handles audio streaming, transcription, and response generation

### WebSocket Connection
#### Client to Server
- Audio data as binary
- JSON messages for commands:
   - PDF attachment: `{"type": "attach", "content": "file_path"}`
   - Text input: `{"type": "text", "content": "user message"}`
#### Server to Client
- JSON messages:
   - Interim transcripts: `{"type": "transcript_interim", "content": "partial text"}`
   - Final transcripts: `{"type": "transcript_final", "content": "final text"}`
   - Speech final: `{"type": "speech_final", "content": "complete utterance"}`
   - Assistant response: `{"type": "assistant", "content": "AI response"}`
   - End of conversation: `{"type": "finish"}`
- Audio data as binary (text-to-speech response)
#### Architecture
- `main.py` - FastAPI application entry point
- `app/config.py` - Configuration settings
- `app/assistant.py` - Core assistant logic including:
   - Speech recognition with Deepgram
   - Natural language processing with OpenAI
   - Text-to-speech conversion
   - Conversation management

### Acknowledgements
- [Deepgram](https://deepgram.com/) for speech recognition and text-to-speech
- [OpenAI](https://openai.com/) for natural language processing
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
