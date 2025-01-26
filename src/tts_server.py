from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from pathlib import Path
from style_bert_vits2.tts_model import TTSModel
import sounddevice as sd
import asyncio
import re
import numpy as np


def create_tts_app():
    app = FastAPI()

    # Initialize TTS model
    bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

    model_name = "koharune-ami"
    assets_root = Path("model_assets")

    model = TTSModel(
        model_path=assets_root / f"{model_name}/{model_name}.safetensors",
        config_path=assets_root / f"{model_name}/config.json",
        style_vec_path=assets_root / f"{model_name}/style_vectors.npy",
        device="cpu",
    )

    def split_text(text: str) -> list[str]:
        """Split text by Japanese punctuation marks"""
        return [s.strip() for s in re.split(r'。', text) if s.strip()]

    async def async_infer(text: str) -> tuple[int, np.ndarray]:
        """Run TTS inference asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, model.infer, text)

    async def play_audio_queue(audio_queue: asyncio.Queue, total_sentences: int):
        """Play audio from queue in order"""
        current_index = 0
        audio_buffer = {}
        
        while current_index < total_sentences:
            # Get next audio in sequence
            if current_index in audio_buffer:
                sr, audio = audio_buffer.pop(current_index)
                sd.play(audio, sr)
                sd.wait()
                current_index += 1
            else:
                # Wait for next audio to be ready
                index, sr, audio = await audio_queue.get()
                if index == current_index:
                    sd.play(audio, sr)
                    sd.wait()
                    current_index += 1
                else:
                    # Store audio for later
                    audio_buffer[index] = (sr, audio)

    class SpeakRequest(BaseModel):
        message: str

    @app.post("/speak")
    async def speak(request: SpeakRequest):
        # Ensure UTF-8 encoding
        request.message = request.message.encode('utf-8').decode('utf-8')

        # Split message into sentences
        sentences = split_text(request.message)
        if not sentences:
            return {"status": "success", "message": "No text to speak"}

        # Create audio queue and start player task
        audio_queue = asyncio.Queue()
        player_task = asyncio.create_task(play_audio_queue(audio_queue, len(sentences)))

        try:
            # Create and start inference tasks with index
            tasks = []
            for i, sentence in enumerate(sentences):
                task = asyncio.create_task(async_infer(sentence))
                task.add_done_callback(lambda t, idx=i: audio_queue.put_nowait((idx, *t.result())))
                tasks.append(task)
        
            # Wait for all tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
        
            # Signal end of queue
            await audio_queue.put((-1, None, None))
            await player_task
        
            return {"status": "success", "message": request.message}
        except Exception as e:
            # Clean up if error occurs
            player_task.cancel()
            raise HTTPException(status_code=500, detail=str(e))

    return app
