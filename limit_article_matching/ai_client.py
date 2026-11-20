from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

AiModel = Literal["GEMINI", "GEMMA"]
AiProvider = Literal["gemini", "vllm"]

DEFAULT_AI_MODEL: AiModel = "GEMINI"


class AiClient(Protocol):
    model_name: str
    provider: AiProvider

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        json_mode: bool = False,
        timeout_sec: int | None = None,
        max_tokens: int | None = None,
        think: bool | None = None,
    ) -> str:
        ...

    def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        timeout_sec: int | None = None,
        max_tokens: int | None = None,
        think: bool | None = None,
    ) -> dict[str, Any]:
        ...


def clean_json_text(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return text


def normalize_ai_model(ai_model: str | None = None) -> AiModel:
    value = (ai_model or "").strip().upper()
    if not value:
        return DEFAULT_AI_MODEL
    aliases = {
        "GEMINI": "GEMINI",
        "GOOGLE": "GEMINI",
        "GOOGLE_GEMINI": "GEMINI",
        "GEMMA": "GEMMA",
        "GEMMA4": "GEMMA",
        "OLLAMA": "GEMMA",
        "VLLM": "GEMMA",
    }
    normalized = aliases.get(value)
    if normalized not in {"GEMINI", "GEMMA"}:
        raise ValueError("aiModel must be either GEMINI or GEMMA")
    return normalized  # type: ignore[return-value]


class VllmClient:
    provider: AiProvider = "vllm"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_sec: int = 180,
        max_tokens: int = 8192,
    ) -> None:
        self.base_url = (base_url or os.getenv("VLLM_BASE_URL") or "http://127.0.0.1:8000/v1").rstrip("/")
        self.api_key = api_key or os.getenv("VLLM_API_KEY") or ""
        self.model_name = model or os.getenv("VLLM_MODEL") or "gemma4-31b"
        self.timeout_sec = timeout_sec or int(os.getenv("VLLM_TIMEOUT_SEC") or "180")
        self.max_tokens = max_tokens or int(os.getenv("VLLM_MAX_TOKENS") or "8192")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        json_mode: bool = False,
        timeout_sec: int | None = None,
        max_tokens: int | None = None,
        think: bool | None = None,
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if think is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": bool(think)}

        effective_timeout = timeout_sec or self.timeout_sec
        url = f"{self.base_url}/chat/completions"

        with httpx.Client(timeout=effective_timeout) as client:
            resp = client.post(url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        timeout_sec: int | None = None,
        max_tokens: int | None = None,
        think: bool | None = None,
    ) -> dict[str, Any]:
        text = self.generate_text(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            json_mode=True,
            timeout_sec=timeout_sec,
            max_tokens=max_tokens,
            think=think,
        )
        return json.loads(clean_json_text(text))


class GeminiClient:
    provider: AiProvider = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        http_timeout_sec: int = 35,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or ""
        self.model_name = model or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
        self.http_timeout_sec = http_timeout_sec or int(os.getenv("GEMINI_HTTP_TIMEOUT_SEC") or "35")
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=self.http_timeout_sec),
            )
        return self._client

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        json_mode: bool = False,
        timeout_sec: int | None = None,
        max_tokens: int | None = None,
        think: bool | None = None,
    ) -> str:
        try:
            config_kwargs: dict[str, Any] = {"temperature": temperature}
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt
            if json_mode:
                config_kwargs["response_mime_type"] = "application/json"
            if max_tokens:
                config_kwargs["max_output_tokens"] = max_tokens

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            text = getattr(response, "text", None) or ""
            return text.strip()
        except Exception:
            return self._generate_text_via_rest(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                json_mode=json_mode,
                timeout_sec=timeout_sec,
                max_tokens=max_tokens,
            )

    def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        timeout_sec: int | None = None,
        max_tokens: int | None = None,
        think: bool | None = None,
    ) -> dict[str, Any]:
        text = self.generate_text(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            json_mode=True,
            timeout_sec=timeout_sec,
            max_tokens=max_tokens,
        )
        parsed = json.loads(clean_json_text(text))
        if not isinstance(parsed, dict):
            raise ValueError("Gemini JSON response is not an object")
        return parsed

    def _generate_text_via_rest(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float,
        json_mode: bool,
        timeout_sec: int | None,
        max_tokens: int | None,
    ) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        generation_config: dict[str, Any] = {"temperature": temperature}
        if json_mode:
            generation_config["responseMimeType"] = "application/json"
        if max_tokens:
            generation_config["maxOutputTokens"] = max_tokens

        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        if system_prompt:
            payload["system_instruction"] = {"parts": [{"text": system_prompt}]}

        effective_timeout = timeout_sec or self.http_timeout_sec
        response = httpx.post(
            url,
            headers={"x-goog-api-key": self.api_key},
            json=payload,
            timeout=effective_timeout,
        )
        response.raise_for_status()
        data = response.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        if not parts:
            return ""
        return str(parts[0].get("text", "")).strip()


def get_ai_client(ai_model: str | None = None) -> AiClient:
    normalized = normalize_ai_model(ai_model)
    if normalized == "GEMMA":
        return VllmClient()
    return GeminiClient()
