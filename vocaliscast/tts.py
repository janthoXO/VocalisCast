"""Speech. One `speak()` per provider, everything after it is shared.

Providers differ in exactly one way that matters here: whether the language can
be set per piece of text. A local engine can, so each segment is synthesized in
its own language and the voice stays the same across both. A hosted endpoint
cannot, so the line goes out as one request and the model detects the language
itself, which is convenient but takes the pronunciation out of your hands.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import tempfile
import urllib.request
import wave
from pathlib import Path

import numpy as np

from .config import Config, ConfigError
from .models import Line, Script

LINE_GAP_S = 0.35
_model_cache: dict[str, object] = {}


def _pcm16_to_float(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def _read_wav(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise RuntimeError("expected 16-bit WAV from the speech endpoint")
        audio = _pcm16_to_float(wav.readframes(wav.getnframes()))
        if wav.getnchannels() == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
        return audio, wav.getframerate()


def _post(url: str, payload: dict, headers: dict) -> bytes:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **headers}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def _chatterbox(line: Line, voice: str, cfg: Config) -> tuple[np.ndarray, int]:
    """Local, MIT-licensed, 23 languages. The only provider with per-span language control."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # heavy: torch

    if "chatterbox" not in _model_cache:
        device = cfg.tts_device
        if device == "auto":
            import torch

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        _model_cache["chatterbox"] = ChatterboxMultilingualTTS.from_pretrained(device=device)
    model = _model_cache["chatterbox"]

    pieces = [
        model.generate(
            segment.text.strip(),
            language_id=cfg.target_lang if segment.vocab else cfg.native_lang,
            audio_prompt_path=voice,
        )
        .squeeze(0)
        .cpu()
        .numpy()
        for segment in line.segments
        if segment.text.strip()
    ]
    return np.concatenate(pieces), model.sr


def _openai(line: Line, voice: str, cfg: Config) -> tuple[np.ndarray, int]:
    """Any OpenAI-compatible /v1/audio/speech endpoint, hosted or self-hosted."""
    base = (cfg.tts_base_url or "https://api.openai.com/v1").rstrip("/")
    data = _post(
        f"{base}/audio/speech",
        {
            "model": cfg.tts_model or "gpt-4o-mini-tts",
            "voice": voice,
            "input": "".join(s.text for s in line.segments),
            "response_format": "wav",
        },
        {"Authorization": f"Bearer {cfg.tts_api_key}"} if cfg.tts_api_key else {},
    )
    return _read_wav(data)


def _elevenlabs(line: Line, voice: str, cfg: Config) -> tuple[np.ndarray, int]:
    if not cfg.tts_api_key:
        raise ConfigError("VC_TTS_API_KEY is required for the elevenlabs provider")
    base = (cfg.tts_base_url or "https://api.elevenlabs.io/v1").rstrip("/")
    data = _post(
        f"{base}/text-to-speech/{voice}?output_format=pcm_24000",
        {
            "text": "".join(s.text for s in line.segments),
            "model_id": cfg.tts_model or "eleven_multilingual_v2",
        },
        {"xi-api-key": cfg.tts_api_key},
    )
    return _pcm16_to_float(data), 24000


PROVIDERS = {"chatterbox": _chatterbox, "openai": _openai, "elevenlabs": _elevenlabs}


def render_episode(script: Script, cfg: Config, out_path: Path) -> Path:
    """Speak every line, join them, write mp3 (or WAV when ffmpeg is missing)."""
    try:
        speak = PROVIDERS[cfg.tts_provider]
    except KeyError:
        raise ConfigError(
            f"VC_TTS_PROVIDER={cfg.tts_provider!r} is unknown, pick one of {', '.join(PROVIDERS)}"
        ) from None

    voices = {speaker: cfg.voice(speaker) for speaker in ("A", "B")}
    chunks: list[np.ndarray] = []
    rate = 0
    for index, line in enumerate(script.lines, 1):
        audio, rate = speak(line, voices[line.speaker], cfg)
        print(f"  line {index}/{len(script.lines)}", end="\r", flush=True)
        chunks += [audio, np.zeros(int(LINE_GAP_S * rate), dtype=np.float32)]

    return _write_audio(np.concatenate(chunks), rate, out_path)


def _write_audio(audio: np.ndarray, rate: int, out_path: Path) -> Path:
    peak = float(np.max(np.abs(audio))) or 1.0
    pcm = (audio / peak * 0.95 * 32767).astype("<i2").tobytes()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        with wave.open(tmp, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(pcm)
        wav_path = Path(tmp.name)

    if out_path.suffix == ".mp3" and shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path), "-b:a", "128k", str(out_path)],
            check=True,
        )
        wav_path.unlink()
        return out_path

    final = out_path.with_suffix(".wav")
    shutil.move(wav_path, final)
    return final
