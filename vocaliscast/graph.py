"""The episode pipeline as a LangGraph state graph.

    pick_topic? -> plan_vocab -> write_script -> validate -.-> render -> commit
                                      ^-------- errors ----'

Straight line plus one retry loop today. It is a graph so that the obvious next
steps (approving a script before the expensive render, giving the writer a
search tool, rendering lines in parallel) are node changes, not a rewrite.
"""

from __future__ import annotations

import json
import operator
import re
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from . import prompts
from .config import Config
from .db import Store, Word
from .models import Script, Segment, Topic, plan_model

MAX_ATTEMPTS = 3
LENGTH_TOLERANCE = 0.25
MIN_NEW_ITEMS = 5


class ScriptQualityError(Exception):
    """The writer could not satisfy the rules. The script is on disk; --force renders it."""


class EpisodeState(TypedDict, total=False):
    topic: str
    episode_id: int
    episode_dir: str
    plan: dict
    script: dict
    errors: list[str]  # no reducer: each validate run replaces the previous errors
    attempts: Annotated[int, operator.add]
    audio_path: str
    force: bool


def build_llm(cfg: Config):
    """One call covers every provider: local server, hosted API, anything
    OpenAI-compatible via provider=openai plus a base_url."""
    kwargs = {
        "model": cfg.llm_model,
        "model_provider": cfg.llm_provider,
        "temperature": cfg.llm_temperature,
    }
    if cfg.llm_base_url:
        kwargs["base_url"] = cfg.llm_base_url
    if cfg.llm_api_key:
        kwargs["api_key"] = cfg.llm_api_key
    if cfg.llm_provider == "ollama":
        # Ollama defaults to a 4k context. A reasoning model fills it with thinking
        # tokens and returns an empty answer, so give it room and turn thinking off.
        kwargs |= {"num_ctx": 16384, "reasoning": False}
    kwargs |= cfg.llm_options  # VC_LLM_OPTIONS: whatever this provider calls things
    return init_chat_model(**kwargs)


def slug(text: str, words: int = 6) -> str:
    """Folder-safe name. Keeps letters of any script, drops punctuation."""
    cleaned = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().casefold()
    return "-".join(cleaned.split()[:words]) or "episode"


def split_segments(script: Script, plan: dict) -> Script:
    """Cut planned terms out of native-language segments into segments of their own.

    Small models get the content right and the shape wrong: they write the target
    word inline in a native segment and leave `vocab` empty. The word then never
    gets spoken in the target language and never counts. Fixing the shape here is
    cheaper than another round trip, and it costs nothing when the model got it right.
    """
    terms = sorted({i["term"] for i in plan["new"] + plan["review"]}, key=len, reverse=True)
    if not terms:
        return script
    # ponytail: exact matches only. Inflected forms stay unsplit and the writer is
    # told to tag those itself; a lemmatizer per language is the upgrade path.
    pattern = re.compile(
        "|".join(rf"(?<!\w){re.escape(t)}(?!\w)" for t in terms), re.IGNORECASE | re.UNICODE
    )
    by_key = {t.casefold(): t for t in terms}

    for line in script.lines:
        rebuilt: list[Segment] = []
        for segment in line.segments:
            if segment.vocab or not pattern.search(segment.text):
                rebuilt.append(segment)
                continue
            position, first = 0, True
            for match in pattern.finditer(segment.text):
                if match.start() > position:
                    rebuilt.append(Segment(text=segment.text[position : match.start()]))
                rebuilt.append(
                    Segment(
                        text=match.group(),
                        vocab=by_key[match.group().casefold()],
                        intro=segment.intro and first,
                    )
                )
                position, first = match.end(), False
            if position < len(segment.text):
                rebuilt.append(Segment(text=segment.text[position:]))
        line.segments = rebuilt
    return script


def validate_script(script: Script, plan: dict, min_uses: int, target_chars: int) -> list[str]:
    """Deterministic checks. Pure function: no LLM, no IO, no language knowledge."""
    errors: list[str] = []
    planned = {i["term"].casefold(): i["term"] for i in plan["new"] + plan["review"]}

    for item in plan["new"]:
        term, uses, intros = item["term"], script.uses(item["term"]), script.intros(item["term"])
        if intros != 1:
            errors.append(
                f'New item "{term}" needs exactly one segment with "intro": true, found {intros}.'
            )
        # the intro is one use, so min_uses further uses means min_uses + 1 in total
        if uses < min_uses + 1:
            errors.append(
                f'New item "{term}" is used {uses} times, needs {min_uses + 1} '
                f"(the intro plus {min_uses} more)."
            )

    for item in plan["review"]:
        term, uses = item["term"], script.uses(item["term"])
        if not 1 <= uses <= 3:
            errors.append(f'Review item "{term}" is used {uses} times, needs between 1 and 3.')
        if script.intros(term):
            errors.append(f'Review item "{term}" must not be explained again.')

    for unknown in script.vocab_terms() - planned.keys():
        errors.append(
            f'Segment vocab "{unknown}" is not in this episode\'s vocabulary. Use only the '
            "planned items, and copy their term exactly."
        )

    drift = (script.char_count - target_chars) / target_chars
    if abs(drift) > LENGTH_TOLERANCE:
        errors.append(
            f"The script is {script.char_count} characters, target {target_chars} "
            f"(±{int(LENGTH_TOLERANCE * 100)}%). Make it {'shorter' if drift > 0 else 'longer'}."
        )
    return errors


def after_validate(state: EpisodeState) -> Literal["write_script", "render"]:
    if state.get("errors"):
        if state["attempts"] < MAX_ATTEMPTS:
            return "write_script"
        if not state.get("force"):
            raise ScriptQualityError("\n".join(state["errors"]))
    return "render"


def transcript(script: Script, title: str) -> str:
    """Readable episode text: target-language spans in bold."""
    lines = [f"# {title}", ""]
    for line in script.lines:
        parts = [f"**{s.text}**" if s.vocab else s.text for s in line.segments]
        lines.append(f"**{line.speaker}:** {''.join(parts)}")
        lines.append("")
    return "\n".join(lines)


def build_graph(cfg: Config, store: Store, checkpointer=None, stop_before_render: bool = False):
    llm = build_llm(cfg)
    topic_picker = llm.with_structured_output(Topic)
    planner = llm.with_structured_output(plan_model(cfg.native_language, cfg.target_language))
    writer = llm.with_structured_output(Script)

    def pick_topic(state: EpisodeState) -> dict:
        chosen = topic_picker.invoke(
            [HumanMessage(prompts.TOPIC.format(native_language=cfg.native_language))]
        )
        return {"topic": chosen.topic}

    def plan_vocab(state: EpisodeState) -> dict:
        topic = state["topic"]
        candidates = store.due_candidates(cfg.learned_after)
        known = store.known_terms()
        known_keys = {t.casefold() for t in known}

        system = prompts.PLAN_SYSTEM.format(
            native_language=cfg.native_language,
            target_language=cfg.target_language,
            level=cfg.level,
            n_new=cfg.new_words,
            n_topical=max(1, cfg.new_words // 2),
            n_review=min(cfg.review_words, len(candidates)),
        )
        user = prompts.PLAN_USER.format(
            topic=topic,
            candidates="\n".join(f"- {c.term} = {c.translation}" for c in candidates) or "(none)",
            known="\n".join(f"- {t}" for t in known) or "(none)",
        )
        messages = [SystemMessage(system), HumanMessage(user)]
        plan = planner.invoke(messages)
        fresh = [i for i in plan.new_items if i.term.casefold() not in known_keys]

        if len(fresh) < MIN_NEW_ITEMS:
            # The model reached for what the listener already knows. One nudge, then live with it.
            messages.append(
                HumanMessage(
                    "These are already known, propose different items: "
                    + ", ".join(i.term for i in plan.new_items if i not in fresh)
                )
            )
            fresh += [
                i
                for i in planner.invoke(messages).new_items
                if i.term.casefold() not in known_keys
                and i.term.casefold() not in {f.term.casefold() for f in fresh}
            ]

        by_term = {c.term: c for c in candidates}
        review: list[Word] = [by_term[t] for t in plan.review_terms if t in by_term]
        if candidates and candidates[0] not in review:
            review.insert(0, candidates[0])  # oldest due item is never skipped again
        review = review[: cfg.review_words]

        directory = cfg.episodes_dir / f"{state['episode_id']:04d}-{slug(topic)}"
        directory.mkdir(parents=True, exist_ok=True)
        store.update_episode(state["episode_id"], topic=topic, directory=str(directory))

        return {
            "episode_dir": str(directory),
            "plan": {
                "new": [i.model_dump() for i in fresh[: cfg.new_words]],
                "review": [
                    {"term": w.term, "translation": w.translation, "kind": w.kind} for w in review
                ],
            },
        }

    def write_script(state: EpisodeState) -> dict:
        plan = state["plan"]
        messages = [
            SystemMessage(
                prompts.WRITE_SYSTEM.format(
                    native_language=cfg.native_language,
                    target_language=cfg.target_language,
                    chars=cfg.target_chars,
                    min_uses=cfg.new_min_uses,
                )
            ),
            HumanMessage(
                prompts.WRITE_USER.format(
                    topic=state["topic"],
                    min_uses=cfg.new_min_uses,
                    new_items="\n".join(
                        f"- {i['term']} = {i['translation']} ({i['kind']})" for i in plan["new"]
                    ),
                    review_items="\n".join(
                        f"- {i['term']} = {i['translation']}" for i in plan["review"]
                    )
                    or "(none)",
                )
            ),
        ]
        if state.get("errors"):
            messages += [
                HumanMessage(json.dumps(state["script"], ensure_ascii=False)),
                HumanMessage(
                    prompts.RETRY.format(errors="\n".join(f"- {e}" for e in state["errors"]))
                ),
            ]

        script = split_segments(writer.invoke(messages), plan)
        path = Path(state["episode_dir"]) / "script.json"
        path.write_text(
            json.dumps(
                {"topic": state["topic"], "plan": plan, "script": script.model_dump()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"script": script.model_dump(), "attempts": 1}

    def validate(state: EpisodeState) -> dict:
        errors = validate_script(
            Script(**state["script"]), state["plan"], cfg.new_min_uses, cfg.target_chars
        )
        return {"errors": errors}

    def render(state: EpisodeState) -> dict:
        from .tts import render_episode  # local: pulls in torch for local engines

        script = Script(**state["script"])
        directory = Path(state["episode_dir"])
        (directory / "transcript.md").write_text(transcript(script, script.title), encoding="utf-8")
        audio = render_episode(script, cfg, directory / "episode.mp3")
        return {"audio_path": str(audio)}

    def commit(state: EpisodeState) -> dict:
        store.commit_episode(state["episode_id"], state["plan"]["new"] + state["plan"]["review"])
        return {}

    retry = RetryPolicy(max_attempts=3)  # transport hiccups and unparsable JSON
    builder = (
        StateGraph(EpisodeState)
        .add_node("pick_topic", pick_topic, retry_policy=retry)
        .add_node("plan_vocab", plan_vocab, retry_policy=retry)
        .add_node("write_script", write_script, retry_policy=retry)
        .add_node("validate", validate)
        .add_node("render", render)
        .add_node("commit", commit)
        .add_conditional_edges(
            START,
            lambda s: "plan_vocab" if s.get("topic") else "pick_topic",
            ["pick_topic", "plan_vocab"],
        )
        .add_edge("pick_topic", "plan_vocab")
        .add_edge("plan_vocab", "write_script")
        .add_edge("write_script", "validate")
        .add_conditional_edges("validate", after_validate, ["write_script", "render"])
        .add_edge("render", "commit")
        .add_edge("commit", END)
    )
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["render"] if stop_before_render else None,
    )
