"""Vocabulary store. Embedded SQLite file or a Postgres server, picked by the URL.

SQLAlchemy Core (no ORM) so both backends share one schema and one set of
queries: `sqlite:///path/vocab.db` or `postgresql+psycopg://user:pw@host/db`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine

REFRESH_GAP = 10  # episodes a learned item rests before it may resurface

metadata = MetaData()

episodes = Table(
    "episodes",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("native_lang", String(16), nullable=False),
    Column("target_lang", String(16), nullable=False),
    Column("topic", String, nullable=False),
    Column("dir", String, nullable=False),
    Column("rendered_at", DateTime(timezone=True)),
)

words = Table(
    "words",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("native_lang", String(16), nullable=False),
    Column("target_lang", String(16), nullable=False),
    Column("term", String, nullable=False),
    Column("term_key", String, nullable=False),  # term.casefold(), unicode-aware
    Column("translation", String, nullable=False),
    Column("kind", String(16), nullable=False),  # word | phrase | idiom | basic
    Column("times_featured", Integer, nullable=False, default=0),
    Column("last_episode", Integer),
    UniqueConstraint("native_lang", "target_lang", "term_key", name="uq_word"),
)


@dataclass(frozen=True)
class Word:
    term: str
    translation: str
    kind: str
    times_featured: int

    def status(self, learned_after: int) -> str:
        return "learned" if self.times_featured >= learned_after else "learning"


class Store:
    """All database access for one language pair."""

    def __init__(self, url: str, native_lang: str, target_lang: str):
        self.engine: Engine = create_engine(url)
        self.native_lang = native_lang
        self.target_lang = target_lang
        metadata.create_all(self.engine)

    def _pair(self, table: Table):
        return (table.c.native_lang == self.native_lang) & (table.c.target_lang == self.target_lang)

    def known_terms(self, limit: int = 200) -> list[str]:
        """Recently taught terms, to keep the planner from proposing them again."""
        stmt = (
            select(words.c.term)
            .where(self._pair(words))
            .order_by(words.c.last_episode.desc().nullslast())
            .limit(limit)
        )
        with self.engine.connect() as conn:
            return [row.term for row in conn.execute(stmt)]

    def due_candidates(self, learned_after: int, limit: int = 15) -> list[Word]:
        """Items worth reviewing: everything still being learned, oldest first, plus
        learned items that have rested for REFRESH_GAP episodes."""
        with self.engine.connect() as conn:
            latest = conn.execute(select(func.max(episodes.c.id))).scalar() or 0
            still_learning = words.c.times_featured < learned_after
            rested = (words.c.times_featured >= learned_after) & (
                func.coalesce(words.c.last_episode, 0) <= latest - REFRESH_GAP
            )
            stmt = (
                select(words)
                .where(self._pair(words) & (still_learning | rested))
                .order_by(words.c.last_episode.asc().nullsfirst())
                .limit(limit)
            )
            return [
                Word(r.term, r.translation, r.kind, r.times_featured) for r in conn.execute(stmt)
            ]

    def all_words(self) -> list[Word]:
        stmt = (
            select(words)
            .where(self._pair(words))
            .order_by(words.c.times_featured.desc(), words.c.term)
        )
        with self.engine.connect() as conn:
            return [
                Word(r.term, r.translation, r.kind, r.times_featured) for r in conn.execute(stmt)
            ]

    def start_episode(self, topic: str, directory: str) -> int:
        stmt = episodes.insert().values(
            native_lang=self.native_lang,
            target_lang=self.target_lang,
            topic=topic,
            dir=directory,
        )
        with self.engine.begin() as conn:
            return int(conn.execute(stmt).inserted_primary_key[0])

    def update_episode(self, episode_id: int, topic: str, directory: str) -> None:
        """Set once the topic is known (it may come from the model) and the folder exists."""
        stmt = (
            episodes.update().where(episodes.c.id == episode_id).values(topic=topic, dir=directory)
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def episode_dir(self, episode_id: int) -> str | None:
        stmt = select(episodes.c.dir).where(episodes.c.id == episode_id)
        with self.engine.connect() as conn:
            return conn.execute(stmt).scalar()

    def commit_episode(self, episode_id: int, items: list[dict]) -> None:
        """Record everything the episode taught. Called only after the audio exists,
        so a failed or discarded episode never counts as heard.

        `items` are dicts with term / translation / kind, new and review alike.
        """
        # ponytail: read-then-write instead of a dialect-specific upsert. One episode
        # is committed at a time; add ON CONFLICT if this ever runs concurrently.
        with self.engine.begin() as conn:
            for item in items:
                key = item["term"].casefold()
                existing = conn.execute(
                    select(words.c.id, words.c.times_featured).where(
                        self._pair(words) & (words.c.term_key == key)
                    )
                ).first()
                if existing:
                    conn.execute(
                        words.update()
                        .where(words.c.id == existing.id)
                        .values(times_featured=existing.times_featured + 1, last_episode=episode_id)
                    )
                else:
                    conn.execute(
                        words.insert().values(
                            native_lang=self.native_lang,
                            target_lang=self.target_lang,
                            term=item["term"],
                            term_key=key,
                            translation=item["translation"],
                            kind=item.get("kind", "word"),
                            times_featured=1,
                            last_episode=episode_id,
                        )
                    )
            conn.execute(
                episodes.update()
                .where(episodes.c.id == episode_id)
                .values(rendered_at=datetime.now(UTC))
            )
