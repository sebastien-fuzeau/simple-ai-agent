from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, AsyncGenerator

import httpx


class LLMClient:
    def __init__(
            self,
            api_key: Optional[str] = None,
            model: Optional[str] = None,
            base_url: Optional[str] = None,
            timeout_seconds: Optional[float] = None,
            mode: Optional[str] = None,
    ) -> None:
        self.mode = (mode or os.getenv("LLM_MODE", "real")).lower()
        self.model = model or os.getenv("MODEL", "gpt-4.1-mini")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if self.mode == "mock":
            self.api_key = "mock"
            self.timeout_seconds = 0.0
            return

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY manquant (mets-le dans .env)")

        timeout_env = os.getenv("REQUEST_TIMEOUT_SECONDS", "30")
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else float(timeout_env)

    async def chat(self, messages: List[Dict[str, Any]]) -> str:
        if self.mode == "mock":
            # Simule une réponse “agent”
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            return f"[MOCK] Réponse simulée. Dernier message user: {last_user[:120]}"

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "messages": messages}

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            res = await client.post(url, headers=headers, json=payload)
            if res.status_code == 401:
                raise RuntimeError("401 Unauthorized: clé invalide.")
            if res.status_code == 429:
                raise RuntimeError("429: quota/rate limit.")
            res.raise_for_status()
            data = res.json()

        return data["choices"][0]["message"]["content"]
