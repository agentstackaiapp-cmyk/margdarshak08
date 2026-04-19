"""Local shim replacing the private emergentintegrations package with direct OpenAI calls.
Falls back to SambaNova (DeepSeek-V3.1) if OpenAI fails or quota is exceeded.
"""
import logging
import os
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class UserMessage:
    def __init__(self, text: str):
        self.text = text


class LlmChat:
    def __init__(self, api_key: str, session_id: str, system_message: str = ""):
        self._openai = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
        )
        sambanova_key = os.getenv("SAMBANOVA_API_KEY", "")
        self._sambanova = AsyncOpenAI(
            api_key=sambanova_key,
            base_url="https://api.sambanova.ai/v1",
        )
        self.system_message = system_message
        self.openai_model = os.getenv("LLM_MODEL", "gpt-4o")
        self.sambanova_model = os.getenv("SAMBANOVA_MODEL", "DeepSeek-V3.1")
        self._history: list[dict] = []

    def _build_messages(self, user_text: str) -> list[dict]:
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.extend(self._history)
        messages.append({"role": "user", "content": user_text})
        return messages

    async def send_message(self, message: UserMessage) -> str:
        messages = self._build_messages(message.text)
        max_tokens = int(os.getenv("MAX_TOKENS", "2000"))

        try:
            response = await self._openai.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                max_tokens=max_tokens,
            )
            reply = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("OpenAI failed (%s), falling back to SambaNova.", e)
            response = await self._sambanova.chat.completions.create(
                model=self.sambanova_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.1,
            )
            reply = response.choices[0].message.content or ""

        self._history.append({"role": "user", "content": message.text})
        self._history.append({"role": "assistant", "content": reply})
        return reply
