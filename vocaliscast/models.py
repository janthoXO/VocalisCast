"""The shapes the LLM has to return. Also the script format on disk."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, create_model


class Topic(BaseModel):
    """A subject for one episode."""

    topic: str = Field(description="One sentence naming a story, fact or question worth an episode")


def plan_model(native_language: str, target_language: str) -> type[BaseModel]:
    """The plan schema, with the real language names written into the field
    descriptions. The model sees those in the JSON schema, and without them it
    happily proposes words in the wrong language."""
    item = create_model(
        "VocabItem",
        term=(
            str,
            Field(
                description=f"The item written in {target_language}, in the citation form a "
                f"{target_language} dictionary would list"
            ),
        ),
        translation=(str, Field(description=f"What it means, written in {native_language}")),
        kind=(
            Literal["word", "phrase", "idiom", "basic"],
            Field(
                description="'basic' for high-frequency general vocabulary such as pronouns, "
                "common verbs or connectors"
            ),
        ),
    )
    return create_model(
        "Plan",
        new_items=(
            list[item],
            Field(description=f"Items in {target_language} the listener has never heard"),
        ),
        review_terms=(
            list[str],
            Field(
                default_factory=list,
                description="Terms copied verbatim from the review candidates, to use again",
            ),
        ),
    )


class Segment(BaseModel):
    """A stretch of speech in one language."""

    text: str
    vocab: str | None = Field(
        default=None,
        description="Set on target-language segments only: the planned term this is a form of",
    )
    intro: bool = Field(
        default=False, description="True on the single segment where a new item is explained"
    )


class Line(BaseModel):
    speaker: Literal["A", "B"]
    segments: list[Segment]


class Script(BaseModel):
    title: str
    lines: list[Line]

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for line in self.lines for s in line.segments)

    def uses(self, term: str) -> int:
        return sum(
            1
            for line in self.lines
            for s in line.segments
            if s.vocab and s.vocab.casefold() == term.casefold()
        )

    def intros(self, term: str) -> int:
        return sum(
            1
            for line in self.lines
            for s in line.segments
            if s.intro and s.vocab and s.vocab.casefold() == term.casefold()
        )

    def vocab_terms(self) -> set[str]:
        return {s.vocab.casefold() for line in self.lines for s in line.segments if s.vocab}
