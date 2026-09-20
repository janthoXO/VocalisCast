"""Configuration: environment variables in, one frozen dataclass out.

Nothing here knows about a specific language, model or database. Every
language-dependent value is looked up from the configured ISO code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from babel import Locale, UnknownLocaleError
from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised for an unusable environment, with a message aimed at the user."""


def _int(name: str, default: int, low: int | None = None, high: int | None = None) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a whole number, got {raw!r}") from exc
    if low is not None and value < low or high is not None and value > high:
        raise ConfigError(f"{name} must be between {low} and {high}, got {value}")
    return value


def _language_name(code: str, display_in: str) -> str:
    """"de" -> "German", named in the language of `display_in`."""
    try:
        name = Locale.parse(code).get_display_name(display_in)
    except (UnknownLocaleError, ValueError, TypeError) as exc:
        raise ConfigError(f"{code!r} is not a language code I recognise (use ISO 639-1)") from exc
    if not name:
        raise ConfigError(f"no display name for language {code!r}")
    return name


@dataclass(frozen=True)
class Config:
    native_lang: str
    target_lang: str
    level: str
    llm_provider: str
    llm_model: str
    llm_base_url: str | None
    llm_api_key: str | None
    llm_temperature: float
    tts_provider: str
    tts_model: str | None
    tts_base_url: str | None
    tts_api_key: str | None
    tts_device: str
    voice_a: str | None
    voice_b: str | None
    db_url: str
    data_dir: Path
    new_words: int
    review_words: int
    new_min_uses: int
    learned_after: int
    episode_minutes: int
    chars_per_minute: int

    @property
    def native_language(self) -> str:
        """Native language named in itself, e.g. "Deutsch" for de."""
        return _language_name(self.native_lang, self.native_lang)

    @property
    def target_language(self) -> str:
        """Target language named in the native language, e.g. "Spanisch"."""
        return _language_name(self.target_lang, self.native_lang)

    @property
    def target_chars(self) -> int:
        """Script length target. Characters, because not every language uses spaces."""
        return self.episode_minutes * self.chars_per_minute

    @property
    def episodes_dir(self) -> Path:
        return self.data_dir / "episodes"

    def voice(self, speaker: str) -> str:
        """Reference WAV path (local engines) or voice id (hosted ones)."""
        voice = self.voice_a if speaker.upper() == "A" else self.voice_b
        if not voice:
            raise ConfigError(
                f"VC_VOICE_{speaker.upper()} is not set. Local engines want a path to a ~10s "
                "WAV file, hosted ones a voice id."
            )
        return voice


def load() -> Config:
    """Read the environment (and a .env file next to the working directory)."""
    load_dotenv()

    data_dir = Path(os.getenv("VC_DATA_DIR", "~/.vocaliscast")).expanduser()
    db_url = os.getenv("VC_DB_URL") or f"sqlite:///{data_dir / 'vocab.db'}"

    cfg = Config(
        native_lang=os.getenv("VC_NATIVE_LANG", "en"),
        target_lang=os.getenv("VC_TARGET_LANG", "de"),
        level=os.getenv("VC_LEVEL", "A2"),
        llm_provider=os.getenv("VC_LLM_PROVIDER", "ollama"),
        llm_model=os.getenv("VC_LLM_MODEL", "qwen3:14b"),
        llm_base_url=os.getenv("VC_LLM_BASE_URL") or None,
        llm_api_key=os.getenv("VC_LLM_API_KEY") or None,
        llm_temperature=float(os.getenv("VC_LLM_TEMPERATURE", "0.8")),
        tts_provider=os.getenv("VC_TTS_PROVIDER", "chatterbox"),
        tts_model=os.getenv("VC_TTS_MODEL") or None,
        tts_base_url=os.getenv("VC_TTS_BASE_URL") or None,
        tts_api_key=os.getenv("VC_TTS_API_KEY") or None,
        tts_device=os.getenv("VC_TTS_DEVICE", "auto"),
        voice_a=os.getenv("VC_VOICE_A") or None,
        voice_b=os.getenv("VC_VOICE_B") or None,
        db_url=db_url,
        data_dir=data_dir,
        new_words=_int("VC_NEW_WORDS", 7, 5, 10),
        review_words=_int("VC_REVIEW_WORDS", 3, 0, 10),
        new_min_uses=_int("VC_NEW_MIN_USES", 5, 1, 20),
        learned_after=_int("VC_LEARNED_AFTER", 3, 1, 20),
        episode_minutes=_int("VC_EPISODE_MINUTES", 10, 1, 120),
        chars_per_minute=_int("VC_CHARS_PER_MINUTE", 900, 100, 5000),
    )

    if cfg.native_lang == cfg.target_lang:
        raise ConfigError("VC_NATIVE_LANG and VC_TARGET_LANG are the same language")
    _ = cfg.native_language, cfg.target_language  # resolve now, not mid-episode
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.episodes_dir.mkdir(exist_ok=True)
    return cfg
