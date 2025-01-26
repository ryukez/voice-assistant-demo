import uvicorn
from tts_server import create_tts_app
from whisper_mic import WhisperMicHandler
import threading
from assistant import Assistant
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def main():
    app = create_tts_app()

    # Initialize WhisperMicHandler with Assistant
    assistant = Assistant(api_key=os.getenv("OPENAI_API_KEY"))
    whisper_handler = WhisperMicHandler(callback=assistant.receive)

    # Start WhisperMic in a separate thread
    whisper_thread = threading.Thread(target=whisper_handler.start, daemon=True)
    whisper_thread.start()

    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
