"""The shapes the LLM has to return. Also the script format on disk."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Topic(BaseModel):
    """A subject for one episode."""

    topic: str = Field(description="One sentence naming a story, fact or question worth an episode")


class VocabItem(BaseModel):
    term: str = Field(description="The item in the citation form a learner's dictionary would use")
    translation: str = Field(description="Its meaning in the listener's native language")
    kind: Literal["word", "phrase", "idiom", "basic"] = Field(
        description="'basic' for high-frequency general vocabulary such as pronouns, "
        "common verbs or connectors"
    )


class Plan(BaseModel):
    """The fixed vocabulary of one episode."""

    new_items: list[VocabItem] = Field(description="Items the listener has never heard")
    review_terms: list[str] = Field(
        default_factory=list,
        description="Terms copied verbatim from the review candidates, to use again",
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
            1 for line in self.lines for s in line.segments if s.vocab and s.vocab.casefold() == term.casefold()
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
