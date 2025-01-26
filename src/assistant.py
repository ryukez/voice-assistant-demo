import httpx
import os
import logging
from typing import Optional
from openai import AsyncOpenAI

# ロガーの設定
logger = logging.getLogger(__name__)

class Assistant:
    def __init__(self, tts_url: str = "http://localhost:8000/speak", api_key: str = None):
        self.client = httpx.AsyncClient()
        self.tts_url = tts_url
        self.openai_client = AsyncOpenAI(api_key=api_key)
        logger.info("Assistant initialized")

    async def receive(self, message: str) -> Optional[str]:
        """音声入力を受け取り、処理するメソッド
        
        Args:
            message (str): ユーザーからの入力メッセージ

        Returns:
            Optional[str]: 処理結果のメッセージ
        """
        if not message:
            logger.warning("Received empty message")
            return None
            
        logger.info(f"Received input: {message}")
        
        try:
            # OpenAI APIで応答を生成
            logger.debug("Sending request to OpenAI API")
            chat_completion = await self.openai_client.chat.completions.create(
                messages=[
                    {"role": "user", "content": message}
                ],
                model="gpt-4o-mini"
            )
            response_text = chat_completion.choices[0].message.content
            logger.debug(f"Received response from OpenAI: {response_text}")
            
            # TTS APIで音声合成
            logger.debug(f"Sending request to TTS server at {self.tts_url}")
            await self.client.post(
                self.tts_url,
                json={"message": response_text},
                timeout=300
            )

            return response_text
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error occurred: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error occurred: {e}")
            return None
