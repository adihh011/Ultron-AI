import base64
import hmac
import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="ULTRON Backend", version="0.1.0")


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read an env var, stripping stray whitespace/newlines and wrapping quotes."""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().strip("\"'").strip()
    return value or default

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: Optional[List[Dict[str, str]]] = Field(default=None, max_length=20)


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    voice_id: Optional[str] = None
    format: str = "mp3"
    language: str = "en"


def build_system_prompt() -> str:
    return (
        "You are ULTRON, a calm, intelligent, cinematic AI assistant. "
        "Answer naturally and concisely. You are analytical, witty, and composed. "
        "Keep responses useful, direct, and confident without sounding robotic. "
        "If the user asks for technical assistance, provide practical guidance. "
        "Do not claim to have abilities you do not actually have."
    )


def require_client_token(authorization: Optional[str]) -> None:
    expected_token = get_env("ULTRON_API_TOKEN")
    if not expected_token:
        raise HTTPException(status_code=503, detail="Backend client authentication is not configured")

    scheme, separator, supplied_token = (authorization or "").partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not hmac.compare_digest(supplied_token.strip().encode(), expected_token.encode())
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing backend token")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "ultron-backend",
        "groq_configured": bool(get_env("GROQ_API_KEY")),
        "fish_configured": bool(get_env("FISH_AUDIO_API_KEY")),
        "client_auth_configured": bool(get_env("ULTRON_API_TOKEN")),
        "model": get_env("GROQ_MODEL", "openai/gpt-oss-120b"),
    }


@app.post("/chat")
def chat(
    request: ChatRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    require_client_token(authorization)

    api_key = get_env("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured")

    model = get_env("GROQ_MODEL", "openai/gpt-oss-120b")
    history = request.history or []
    messages: List[Dict[str, str]] = [{"role": "system", "content": build_system_prompt()}]
    for item in history:
        role = item.get("role", "user")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": request.message})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1500,
    }
    # gpt-oss models "think" before answering; keep that short so replies aren't cut off
    if model.startswith("openai/gpt-oss"):
        payload["reasoning_effort"] = "low"

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=45,
    )

    if response.status_code in (401, 403):
        raise HTTPException(
            status_code=502,
            detail="GROQ_API_KEY was rejected by Groq (upstream 401/403). Check the key on the server.",
        )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Groq request failed: {response.text}",
        )

    try:
        data = response.json()
        reply = data["choices"][0]["message"]["content"]
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Invalid response payload from Groq") from exc

    return {
        "reply": reply,
        "provider": "groq",
        "message": request.message,
        "history": history,
    }


@app.post("/tts")
def tts(
    request: TtsRequest,
    authorization: Optional[str] = Header(default=None),
) -> Response:
    require_client_token(authorization)

    api_key = get_env("FISH_AUDIO_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="FISH_AUDIO_API_KEY is not configured")

    voice_id = request.voice_id or get_env("FISH_AUDIO_VOICE_ID")
    if not voice_id:
        raise HTTPException(status_code=400, detail="FISH_AUDIO_VOICE_ID is not configured")

    model = get_env("FISH_AUDIO_MODEL", "s2.1-pro-free")
    payload = {
        "text": request.text,
        "reference_id": voice_id,
        "format": request.format,
        "language": request.language,
        "normalize": True,
        "latency": "normal",
        "prosody": {
            "speed": 1.0,
            "volume": 0,
            "normalize_loudness": True,
        },
    }

    response = requests.post(
        "https://api.fish.audio/v1/tts",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "model": model,
        },
        json=payload,
        timeout=60,
    )

    if response.status_code in (401, 403):
        raise HTTPException(
            status_code=502,
            detail="FISH_AUDIO_API_KEY was rejected by Fish Audio (upstream 401/403). Check the key on the server.",
        )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Fish Audio request failed: {response.text}",
        )

    content_type = response.headers.get("content-type", "audio/mpeg")
    if "application/json" in content_type.lower():
        try:
            data = response.json()
        except ValueError as exc:
            raise HTTPException(status_code=500, detail="Invalid JSON response from Fish Audio") from exc
        if "audio" in data and isinstance(data["audio"], str):
            return Response(
                content=base64.b64decode(data["audio"]),
                media_type="audio/mpeg",
            )
        if "url" in data:
            audio_response = requests.get(data["url"], timeout=60)
            audio_response.raise_for_status()
            return Response(content=audio_response.content, media_type="audio/mpeg")

    return Response(content=response.content, media_type=content_type)
