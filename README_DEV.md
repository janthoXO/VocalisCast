# VocalisCast — developer guide

Architecture and internals for contributors. For what the tool does and how to run it, see [README.md](README.md). For the full design rationale and open questions, see [DESIGN.md](DESIGN.md).

## Module layout

```
vocaliscast/
├── docker-compose.yml
├── pyproject.toml     # deps + extras: ollama, openai, anthropic, chatterbox, postgres, dev
├── vocaliscast/
│   ├── cli.py          # argparse: new / script / render / words
│   ├── config.py       # env vars -> a frozen Config dataclass, validated at startup
│   ├── db.py            # SQLAlchemy Core: schema, due-item selection, commit
│   ├── graph.py         # the LangGraph pipeline: nodes, validator, wiring
│   ├── models.py        # Pydantic schemas returned by the LLM, also the script-on-disk format
│   ├── prompts.py       # language-neutral prompt templates
│   └── tts.py           # speak() + the three TTS providers
```

Heavy imports (`torch`, `chatterbox`) are imported lazily inside `tts.py` and only pulled in when the `render` node actually runs, so `vocaliscast script` and `vocaliscast words` stay fast and don't need a GPU stack installed.

## The LangGraph pipeline

```
pick_topic? -> plan_vocab -> write_script -> validate -.-> render -> commit
                                  ^-------- errors ----'
```

Built in `graph.py::build_graph`. It is a straight line with one retry loop; a plain function would work today, but the graph gives cheap extension points: a real `interrupt()` for approval, a search tool on the writer node, resumable rendering, or fan-out with `Send` for per-line parallel rendering.

### State

`EpisodeState` (`TypedDict`, `total=False`):

```python
class EpisodeState(TypedDict, total=False):
    topic: str
    episode_id: int
    episode_dir: str
    plan: dict  # {"new": [...], "review": [...]}
    script: dict  # models.Script.model_dump()
    errors: list[str]  # no reducer: each validate run replaces the previous list
    attempts: Annotated[int, operator.add]  # write_script returns {"attempts": 1}
    audio_path: str
    force: bool
```

### Nodes

| Node | Does |
|---|---|
| `pick_topic` | Only runs when no topic was given; asks the LLM for one. |
| `plan_vocab` | Reads due review candidates and known terms from the `Store`, asks the LLM for `VC_NEW_WORDS` new items plus a review selection, drops items already known, creates the episode directory. |
| `write_script` | Writes the two-host dialogue as structured JSON. On a retry, the previous script and the validator's error list are appended to the prompt. Writes `script.json` to disk on every attempt. |
| `validate` | Pure function (`validate_script`), no LLM or IO. Fills `errors`. |
| `render` | Writes `transcript.md`, then calls `tts.render_episode`, the slow step. |
| `commit` | Bumps `times_featured` / `last_episode` for every taught item. Runs only after a successful render. |

`pick_topic`, `plan_vocab`, and `write_script` each carry `RetryPolicy(max_attempts=3)` — that covers transport hiccups and unparsable structured-output JSON. Content problems (too few uses, wrong length, stray target-language text) are a separate concern, handled by `validate` plus the loop below.

### The retry loop

`after_validate` (in `graph.py`) is the conditional edge out of `validate`:

```python
def after_validate(state: EpisodeState) -> Literal["write_script", "render"]:
    if state.get("errors"):
        if state["attempts"] < MAX_ATTEMPTS:  # MAX_ATTEMPTS = 3
            return "write_script"
        if not state.get("force"):
            raise ScriptQualityError("\n".join(state["errors"]))
    return "render"
```

Up to `MAX_ATTEMPTS` (3) full rewrites are attempted. If errors remain after that, the run raises `ScriptQualityError` — the CLI catches it, prints the errors, and exits 1. The script is already on disk at that point, and `--force` renders it anyway despite the failed validation.

### Checkpointer and the human-in-the-loop seam

The checkpointer is chosen in `cli.py::_checkpointer` by the same `VC_DB_URL` prefix the vocabulary store uses: `SqliteSaver` for a `sqlite:///…` URL, `PostgresSaver` for a `postgresql+psycopg://…` one. The thread ID is `ep-<episode_id>`.

`build_graph(..., stop_before_render=...)` passes `interrupt_before=["render"]` to `.compile()` when `stop_before_render` is true. This is what makes `vocaliscast script` and `vocaliscast render` two halves of one graph run:

- `script` invokes the graph with `stop_before_render=True`; it runs through `validate` and then interrupts before `render`, leaving `script.json` on disk and nothing committed.
- `render` invokes the same graph (built with `stop_before_render=False`) with `graph.invoke(None, {"configurable": {"thread_id": ep-<id>}})`, which resumes the checkpointed state at `render`.

Editing `script.json` by hand between the two commands is today's human-in-the-loop mechanism. Because `interrupt_before` is already wired to a checkpointer, adding a real `interrupt()` call that waits for programmatic approval inside the graph is a change to one node, not new plumbing.

## Script JSON format

Defined in `models.py`, enforced on the LLM via `llm.with_structured_output(Script)`. A line is a list of segments; a segment is either native-language text or a target-language span carrying the `vocab` key.

```json
{
  "title": "The Worst Marathon Ever Run",
  "lines": [
    {"speaker": "A", "segments": [
      {"text": "Only fourteen of the thirty-two runners finished. "},
      {"text": "<target-language item>", "vocab": "<planned item>", "intro": true},
      {"text": " — that means 'nevertheless', and you'll hear it a lot today."}
    ]},
    {"speaker": "B", "segments": [
      {"text": "And "},
      {"text": "<target-language item>", "vocab": "<planned item>"},
      {"text": ", one of them was riding in a car for a while?"}
    ]}
  ]
}
```

- A segment with no `vocab` is native language.
- `vocab` names the planned term this segment is a (possibly inflected) form of — verbatim, character for character, as it appears in the plan.
- `intro: true` marks the single segment where a new item gets its explanation.
- `transcript.md` is rendered from this same structure (`graph.py::transcript`), target spans in bold.

## The repair pass

`split_segments(script, plan)` in `graph.py` runs on every script the writer returns, before validation.

Small models get the content right and the shape wrong: they write the target word inline inside a native-language segment and leave `vocab` empty. The word then never gets spoken in the target language and never counts toward its quota. The repair pass finds planned terms inside native segments (exact matches, word-boundary aware, case-insensitive) and cuts them out into segments of their own, moving the `intro` flag onto the first extracted piece.

It is deterministic, costs nothing when the model already got it right, and saves a full rewrite round trip when it did not. The known ceiling, marked with a `ponytail:` comment in the source: only exact matches are found, so an inflected form stays unsplit unless the writer tagged it itself. A per-language lemmatizer is the upgrade path.

## Why counting uses the `vocab` key, not string matching

`Script.uses()` and `Script.intros()` (`models.py`) count segments by comparing `segment.vocab.casefold()` to the planned term's casefold, never by searching segment text for the term string. Two reasons this has to be the case rather than a convenience:

- The text in a segment is the *inflected or conjugated* form the sentence needs (a case ending, a conjugated verb, an agglutinated suffix); it will often not match the dictionary citation form as a substring at all.
- Many target languages don't tokenize on whitespace, so "does the text contain this string" is not even well-defined the way it is for English.

The LLM is told explicitly (`prompts.WRITE_SYSTEM`) to always set `vocab` to the planned term's exact citation form regardless of the surface text, which is what makes counting reliable.

## Validator

`validate_script(script, plan, min_uses, target_chars)` in `graph.py`, pure and unit-testable (no LLM, no IO). It rejects:

- A new item with `intros() != 1` (must be explained exactly once).
- A new item with `uses() < min_uses + 1` (the intro segment is itself one use, so `min_uses` further uses means `min_uses + 1` total).
- A review item with `uses()` outside `1..3`, or any `intro` on a review item at all.
- Any segment whose `vocab` is not one of this episode's planned terms (`script.vocab_terms() - planned`) — this is also what stops untaught target-language text from slipping in anywhere.
- A script whose `char_count` is more than `LENGTH_TOLERANCE` (25%) off `target_chars`.

## Language independence

There is no per-language code path and no per-language prompt; every prompt is one template with values injected. The rules that make this hold:

- Language names come from ISO 639-1 codes via `babel.Locale(...).get_display_name(...)`, resolved once at config load (`Config.native_language` / `Config.target_language`), never hardcoded.
- Prompts (`prompts.py`) contain no real-language examples, only placeholders — a real example in one language would bias item choice and phrasing toward that language for every target language.
- Case-insensitive uniqueness uses a `term_key` column holding Python's `str.casefold()` of the term, not SQLite's `NOCASE` (which is ASCII-only and would, for example, treat German *Straße* and *STRASSE* as different terms, or mishandle Turkish dotted/dotless *i*).
- Episode length is measured and targeted in **characters** (`Config.target_chars = episode_minutes * chars_per_minute`), not words — word counting assumes whitespace-delimited tokens, which many scripts don't have. `VC_CHARS_PER_MINUTE` is itself the tunable that varies by script (900 default for Latin scripts, ~250 suggested for Chinese or Japanese in `.env.example`).

## Database

`db.py`, SQLAlchemy Core (no ORM) — one schema and one set of queries shared by both backends, picked purely by the `VC_DB_URL` scheme:

- *(unset)* → `sqlite:///$VC_DATA_DIR/vocab.db`, the embedded default (stdlib `sqlite3` under the hood, nothing to install).
- Any other `sqlite:///...` URL.
- `postgresql+psycopg://user:pw@host/db` — the `postgres` extra and a running Postgres (the `docker-compose.yml` `db` service, behind the `db` profile).

Schema (`db.py`):

```sql
CREATE TABLE episodes (
  id          INTEGER PRIMARY KEY,
  native_lang TEXT NOT NULL,
  target_lang TEXT NOT NULL,
  topic       TEXT NOT NULL,
  dir         TEXT NOT NULL,
  rendered_at TIMESTAMP           -- NULL until the audio succeeded
);

CREATE TABLE words (
  id             INTEGER PRIMARY KEY,
  native_lang    TEXT NOT NULL,
  target_lang    TEXT NOT NULL,
  term           TEXT NOT NULL,
  term_key       TEXT NOT NULL,   -- term.casefold()
  translation    TEXT NOT NULL,
  kind           TEXT NOT NULL,   -- word | phrase | idiom | basic
  times_featured INTEGER NOT NULL DEFAULT 0,
  last_episode   INTEGER REFERENCES episodes(id),
  UNIQUE (native_lang, target_lang, term_key)
);
```

Every row is scoped to `(native_lang, target_lang)`, so changing `VC_TARGET_LANG` gives a separate vocabulary in the same database file or server. Status (`new` / `learning` / `learned`) is derived from `times_featured` vs. `VC_LEARNED_AFTER`, never stored as its own column.

`Store.due_candidates()` selects items still below `VC_LEARNED_AFTER` uses (oldest `last_episode` first) plus already-learned items that haven't been featured in the last 10 episodes (`REFRESH_GAP` in `db.py`), so learned material resurfaces occasionally instead of never again. `Store.commit_episode()` does a read-then-write upsert rather than a dialect-specific `ON CONFLICT`:

```python
# ponytail: read-then-write instead of a dialect-specific upsert. One episode
# is committed at a time; add ON CONFLICT if this ever runs concurrently.
```

That comment in `db.py` is the known ceiling on that approach — fine for one CLI process committing one episode at a time, not safe under concurrent writers.

Note: the checkpointer (LangGraph's saved graph state, see above) is selected separately in `cli.py`, by the same `VC_DB_URL` prefix, using `SqliteSaver` / `PostgresSaver` rather than the `Store`'s SQLAlchemy engine — they're two different persistence mechanisms sharing one URL.

## TTS provider interface

`tts.py`. Every provider implements the same shape and is registered in the `PROVIDERS` dict by the string that `VC_TTS_PROVIDER` selects:

```python
def _provider(line: Line, voice: str, cfg: Config) -> tuple[np.ndarray, int]: ...


PROVIDERS = {"chatterbox": _chatterbox, "openai": _openai, "elevenlabs": _elevenlabs}
```

`render_episode()` calls the selected function once per `Line`, concatenates the returned float32 mono audio with a fixed silence gap (`LINE_GAP_S = 0.35`) between lines, and writes an mp3 via `ffmpeg` if it's on `PATH`, else a `.wav`.

The three providers differ in exactly one thing that matters here: **whether the language can be set per span of text within a line.**

- `chatterbox` (local, MIT, 23 languages): the only provider with `language_id` per call. `_chatterbox` synthesizes each `Segment` in the line separately — `cfg.target_lang` if `segment.vocab` is set, else `cfg.native_lang` — with the same `audio_prompt_path` (voice) throughout, then concatenates. This is what gives correct target-language pronunciation on a short foreign span inside an otherwise native-language sentence.
- `openai` (any `/v1/audio/speech`-compatible endpoint) and `elevenlabs` (`eleven_multilingual_v2`): no per-span control. `_openai` / `_elevenlabs` join every segment's text into one string per line and send it as a single request; the model auto-detects language per stretch of text. Simpler and one request per line, but the target-language span's pronunciation is out of your hands.

### Adding a provider

1. Write a function `(line: Line, voice: str, cfg: Config) -> tuple[np.ndarray, int]` in `tts.py` — return mono float32 samples in `[-1, 1]` and the sample rate.
2. If the language can be set per API call, follow the `_chatterbox` pattern (loop over `line.segments`, pick language by `segment.vocab`). If not, follow `_openai` / `_elevenlabs` (join the line's text, send one request).
3. Add the function to `PROVIDERS` under the name users will set in `VC_TTS_PROVIDER`.
4. If it needs a new dependency, add an optional-dependencies extra in `pyproject.toml` named after the provider.

`build_llm` sets two Ollama-specific defaults: `num_ctx=16384` (Ollama's own default is 4k, which a reasoning model fills with thinking tokens, returning an empty answer) and `reasoning=False`. Anything else goes through `VC_LLM_OPTIONS`, a JSON object merged into the provider's constructor arguments, so provider-specific knobs never need a new environment variable.

Adding an LLM provider needs no code at all: `graph.py::build_llm` calls `langchain.chat_models.init_chat_model(model=..., model_provider=..., ...)`, so any provider LangChain supports works by setting `VC_LLM_PROVIDER` — install the matching `langchain-<provider>` package (as an extra, if one exists in `pyproject.toml`) or pass `VC_LLM_BASE_URL` for anything OpenAI-API-compatible with `VC_LLM_PROVIDER=openai`.

## Tests

`tests/test_pipeline.py` holds the whole suite: 28 plain-assert tests, no mocks, no network, no torch.

```sh
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

What it covers: every validator rejection case, `after_validate` routing (including the `force` escape), the store against a temporary SQLite file (upsert counting, due-candidate ordering, the `REFRESH_GAP` rest period, language pairs kept apart, `Straße` / `STRASSE` folding to one row), config validation, and `slug()` on non-ASCII titles. Run the same module with `VC_DB_URL` pointed at the compose container to check the Postgres path.

The LLM and TTS steps are judged by listening, not mocked. `vocaliscast script "<topic>"` with a real model is the integration test; it writes `script.json` and touches nothing else.

`ruff` uses `line-length = 100` and excludes `*.md`, because it reformats Python blocks inside Markdown files.
