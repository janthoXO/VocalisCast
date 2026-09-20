# VocalisCast

VocalisCast is a command-line tool that generates language-learning podcast episodes: it writes a script with a large language model (LLM), speaks it with text-to-speech (TTS), and teaches a small, fixed set of target-language vocabulary in context inside an otherwise ordinary episode. It runs against a local model (Ollama) and a local TTS engine, or against hosted APIs — your choice, set with environment variables. Nothing about the design requires a cloud service.

## What it does

Most language podcasts pick a topic and then teach words related to that topic: an episode about food teaches *bread*, *cheese*, *to eat*. VocalisCast does something different: **the topic and the vocabulary are independent.**

Example: an episode about the 1904 Olympic marathon — a genuinely chaotic race where runners were poisoned, chased by dogs, and driven part of the course in a car — teaches five items: *nevertheless*, *to give up*, *the doctor*, *what a mess* (an idiom), and *they*. Four of the five have nothing to do with marathons. That's the point: the hosts tell a real story, and the vocabulary is seasoning, not a themed list read aloud.

Each new item is explained once, briefly, the first time it comes up, then reused several more times in natural context for the rest of the episode. Items from earlier episodes come back later as review, used a little and never re-explained. There are no quizzes; progress is measured by exposure.

## Requirements

- Python 3.11, 3.12, or 3.13.
- Either a local LLM server (Ollama, via Docker or native) or an API key for a hosted LLM (OpenAI, Anthropic, and anything else LangChain supports).
- Either a local TTS engine (Chatterbox, needs a GPU for reasonable speed) or an API key for a hosted TTS provider (OpenAI, ElevenLabs).
- `ffmpeg` on `PATH` to get an mp3; without it, episodes are written as `.wav`.

## Install

```sh
git clone https://github.com/janthoXO/VocalisCast.git
cd VocalisCast
python -m venv .venv
source .venv/bin/activate
pip install -e ".[ollama,chatterbox]"
cp .env.example .env
```

Swap the extras for what you'll actually use: `ollama`, `openai`, `anthropic`, `chatterbox`, `postgres` are optional and combinable, e.g. `pip install -e ".[anthropic,postgres]"`. Hosted TTS (OpenAI, ElevenLabs) needs no extra, it is plain HTTP.

Edit `.env`: at minimum set `VC_VOICE_A` and `VC_VOICE_B` to two ~10-second reference WAV files (for local TTS) or two voice IDs (for a hosted TTS provider).

## Run

```sh
docker compose up -d
docker compose exec ollama ollama pull qwen3:14b
```

Then generate an episode:

```sh
vocaliscast new "the 1904 Olympic marathon"
```

Or let the model pick a topic:

```sh
vocaliscast new
```

Other commands:

```sh
vocaliscast script "<topic>"     # write and validate the script, stop before audio
vocaliscast render <episode-id>  # render a previously written script
vocaliscast words                # everything taught so far, with status
```

`script` followed by `render` is a two-step workflow: it writes `script.json` to disk and stops before the expensive TTS step, so you can read or hand-edit it first.

On Apple Silicon, run `ollama serve` natively instead of through Docker — containers on macOS have no GPU access, so Ollama in a container is slow. The local TTS engine (Chatterbox) always runs in-process on the host, GPU or not; it is never containerized.

## How the learning works

- Each episode teaches 5–10 new items: words, phrases, idioms, or basics like pronouns and common verbs (`VC_NEW_WORDS`, clamped to that range).
- A new item gets exactly one short explanation the first time it appears, then is used several more times in place of the native-language phrase (`VC_NEW_MIN_USES`, default 5 further uses).
- An item counts as learned once it has appeared in `VC_LEARNED_AFTER` episodes (default 3). Until then it is "learning".
- A few review items from past episodes (`VC_REVIEW_WORDS`, default 3) are worked back in, used once to three times with no re-explanation. The single oldest still-learning item is always included, so nothing gets skipped forever; already-learned items rest for 10 episodes before they resurface.
- There is no quiz, no self-rating, and no scoring. The only signal is how many episodes have featured an item.

## Configuration

Everything is an environment variable, read from `.env` or the shell, and validated at startup.

| Variable | Default | Meaning |
|---|---|---|
| `VC_NATIVE_LANG` | `en` | ISO 639-1 code of the podcast language |
| `VC_TARGET_LANG` | `de` | ISO 639-1 code of the language being taught |
| `VC_LEVEL` | `A2` | CEFR-ish hint for item difficulty |
| `VC_LLM_PROVIDER` | `ollama` | `ollama`, `openai`, `anthropic`, `google_genai`, `groq`, or any LangChain-supported provider |
| `VC_LLM_MODEL` | `qwen3:14b` | Model name at that provider |
| `VC_LLM_BASE_URL` | *(unset)* | Self-hosted or OpenAI-compatible endpoint |
| `VC_LLM_API_KEY` | *(unset)* | Hosted providers |
| `VC_LLM_TEMPERATURE` | `0.8` | |
| `VC_LLM_OPTIONS` | *(unset)* | JSON of extra provider arguments, e.g. `{"num_ctx": 32768}` |
| `VC_TTS_PROVIDER` | `chatterbox` | `chatterbox`, `openai`, `elevenlabs` |
| `VC_TTS_MODEL` | *(provider default)* | Checkpoint path or hosted model name |
| `VC_TTS_BASE_URL` | *(unset)* | Self-hosted speech endpoint |
| `VC_TTS_API_KEY` | *(unset)* | Hosted providers |
| `VC_TTS_DEVICE` | `auto` | `mps` / `cuda` / `cpu`, local engine only |
| `VC_VOICE_A`, `VC_VOICE_B` | *(required)* | Reference WAV path (local) or voice ID (hosted) |
| `VC_DB_URL` | `sqlite:///$VC_DATA_DIR/vocab.db` | Embedded SQLite file, or a `postgresql+psycopg://…` server URL |
| `VC_DATA_DIR` | `~/.vocaliscast` | Episodes and the default database file |
| `VC_NEW_WORDS` | `7` | New items per episode, clamped to 5–10 |
| `VC_REVIEW_WORDS` | `3` | Review items per episode |
| `VC_NEW_MIN_USES` | `5` | Minimum uses of each new item after its intro |
| `VC_LEARNED_AFTER` | `3` | Episodes featuring an item before it counts as learned |
| `VC_EPISODE_MINUTES` | `10` | Target episode length |
| `VC_CHARS_PER_MINUTE` | `900` | Characters per minute of speech, used for the length target (lower for scripts without spaces, e.g. ~250 for Chinese or Japanese) |

## FAQ

### Does it need a GPU?
No. The local TTS engine (Chatterbox) is much faster with a GPU (`VC_TTS_DEVICE=cuda` or `mps`) but runs on CPU too; alternatively, set `VC_TTS_PROVIDER` to `openai` or `elevenlabs` to render audio through a hosted API with no local GPU at all.

### Which languages does it support?
Whichever ones your chosen LLM and TTS provider both support — there is no per-language code path, only ISO 639-1 codes resolved through a lookup table; the local TTS engine (Chatterbox) covers 23 languages, and hosted TTS providers generally cover more.

### Can I use ChatGPT or Claude instead of a local model?
Yes: set `VC_LLM_PROVIDER=openai` or `VC_LLM_PROVIDER=anthropic`, `VC_LLM_MODEL` to the model name, and `VC_LLM_API_KEY` to your key; install the matching extra (`pip install -e ".[openai]"` or `".[anthropic]"`).

### Does my data leave my machine?
Only if you configure a hosted provider. With `VC_LLM_PROVIDER=ollama` and `VC_TTS_PROVIDER=chatterbox`, script text, vocabulary, and audio stay on the machine running VocalisCast; switching either to `openai`, `anthropic`, or `elevenlabs` sends the relevant text to that provider's API.

### How long does an episode take to generate?
On an Apple M3 Pro with a 9B model in Ollama, one script attempt for a 6-minute episode takes about 90 seconds, and a script that fails validation is rewritten up to three times. Rendering the audio is the slower half, because every line is a separate TTS call.

### Which model should I use?
A bigger model follows the vocabulary rules far more reliably. In testing, a 9B local model wrote good episodes but placed only three of five items often enough, so the validator sent it back; 14B and up, or any hosted model, is the safer choice. If your model struggles, lower `VC_NEW_MIN_USES` or raise `VC_EPISODE_MINUTES` to give it more room.

### Is it free?
The software is free and MIT-licensed. Running it is free if you use a local LLM and local TTS; using a hosted LLM or TTS provider incurs that provider's usual API costs.

### What makes it different from a normal language podcast?
The topic and the taught vocabulary are chosen independently, so the episode is a real podcast on a real subject rather than a themed vocabulary list — most taught items are ordinary high-frequency vocabulary (pronouns, connectors, common verbs) that has nothing to do with that episode's topic.

## License

MIT. See [LICENSE](LICENSE).

## Contributing / architecture

See [README_DEV.md](README_DEV.md) for the LangGraph pipeline, the script format, the validator, and how to add a provider. See [DESIGN.md](DESIGN.md) for the full design rationale.
