"""Command line: new / script / render / words."""

from __future__ import annotations

import argparse
import contextlib
import sqlite3
import sys
from pathlib import Path

from . import config
from .config import ConfigError
from .db import Store
from .graph import ScriptQualityError, build_graph

THREAD = "ep-{}"


def _checkpointer(stack: contextlib.ExitStack, cfg: config.Config):
    """Saved graph state, so `script` and `render` are two halves of one run
    and a crash during render does not throw the script away."""
    if cfg.db_url.startswith("sqlite"):
        from langgraph.checkpoint.sqlite import SqliteSaver

        path = cfg.db_url.split("///", 1)[1]
        return SqliteSaver(stack.enter_context(sqlite3.connect(path, check_same_thread=False)))

    from langgraph.checkpoint.postgres import PostgresSaver

    saver = stack.enter_context(PostgresSaver.from_conn_string(cfg.db_url.replace("+psycopg", "")))
    saver.setup()
    return saver


def _run(args, stack: contextlib.ExitStack) -> int:
    cfg = config.load()
    store = Store(cfg.db_url, cfg.native_lang, cfg.target_lang)
    checkpointer = _checkpointer(stack, cfg)

    if args.command == "words":
        rows = store.all_words()
        if not rows:
            print("No vocabulary yet. Make an episode first.")
            return 0
        width = max(len(w.term) for w in rows)
        for word in rows:
            print(
                f"{word.term:<{width}}  {word.translation:<28} {word.kind:<7} "
                f"{word.status(cfg.learned_after):<8} {word.times_featured}x"
            )
        return 0

    if args.command == "render":
        episode_id = int(args.episode)
        graph = build_graph(cfg, store, checkpointer)
        result = graph.invoke(None, {"configurable": {"thread_id": THREAD.format(episode_id)}})
        print(f"\nAudio: {result['audio_path']}")
        return 0

    topic = " ".join(args.topic).strip()
    script_only = args.command == "script"
    episode_id = store.start_episode(topic, "")
    graph = build_graph(cfg, store, checkpointer, stop_before_render=script_only)
    result = graph.invoke(
        {"topic": topic, "episode_id": episode_id, "attempts": 0, "force": args.force},
        {"configurable": {"thread_id": THREAD.format(episode_id)}, "recursion_limit": 50},
    )

    directory = Path(result["episode_dir"])
    print(f"\nEpisode {episode_id}: {result['topic']}")
    print("Teaching: " + ", ".join(i["term"] for i in result["plan"]["new"]))
    if result["plan"]["review"]:
        print("Review:   " + ", ".join(i["term"] for i in result["plan"]["review"]))
    print(f"Script:   {directory / 'script.json'}")
    if script_only:
        print(f"Render it with:  vocaliscast render {episode_id}")
    else:
        print(f"Audio:    {result['audio_path']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="vocaliscast",
        description="Generate a podcast in your language that teaches a few words of another one.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in [
        ("new", "generate an episode, script and audio (no topic: the model picks one)"),
        ("script", "generate the script only, stopping before the audio"),
    ]:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("topic", nargs="*", help="what the episode is about")
        command.add_argument(
            "--force", action="store_true", help="render even if the script fails validation"
        )

    render = sub.add_parser("render", help="turn an existing script into audio")
    render.add_argument("episode", help="episode id, as printed by `script`")
    sub.add_parser("words", help="everything taught so far")

    args = parser.parse_args()
    try:
        with contextlib.ExitStack() as stack:
            return _run(args, stack)
    except (ConfigError, ScriptQualityError) as error:
        print(f"\n{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
