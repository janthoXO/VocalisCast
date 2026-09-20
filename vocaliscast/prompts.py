"""Prompt templates.

Language-neutral on purpose: every language-dependent value is injected, and no
example is written in any real language. An example in, say, German would push
the model toward German-shaped items and phrasing for every target language.
"""

TOPIC = """Name one genuinely interesting subject for a podcast episode in {native_language}: \
a story, a historical episode, a piece of science, an odd fact worth 10 minutes of conversation.
Not a language lesson, not a list. Surprise me, and avoid the obvious crowd-pleasers."""

PLAN_SYSTEM = """You plan the vocabulary for a podcast episode in {native_language} that teaches \
{target_language} to a learner at roughly level {level}.

The episode is a real podcast, not a lesson. The vocabulary is seasoning.

Rules for the {n_new} new items:
- Every term is written in {target_language}. Every translation is written in {native_language}. \
A term written in {native_language} is the one mistake you must not make.
- They do NOT need to relate to the topic, or to each other.
- At most {n_topical} may come from the topic. The rest must be high-frequency general \
vocabulary: pronouns, everyday verbs, connectors, discourse words, common idioms. Those are the \
ones a learner needs in every conversation, and they fit into any episode.
- Give each item in the citation form a learner's dictionary for {target_language} would use.
- Write the translation in {native_language}.
- Nothing from the ALREADY KNOWN list.

Rules for review terms:
- Copy {n_review} terms verbatim from REVIEW CANDIDATES: the ones you can drop into this episode \
most naturally. Matching the topic is not the point; being usable is."""

PLAN_USER = """TOPIC: {topic}

REVIEW CANDIDATES:
{candidates}

ALREADY KNOWN (never propose these as new):
{known}"""

WRITE_SYSTEM = """You write a podcast episode in {native_language}: a dialogue between two hosts, \
A and B, about the given topic. Roughly {chars} characters of speech in total.

The episode must stand on its own. Someone who ignores the language teaching should still want to \
listen to the end: real content, real curiosity, jokes where they fit, no filler.

Woven into it is a fixed vocabulary of {target_language}. The output format is a list of lines, \
each a list of segments:
- A segment with no "vocab" is spoken in {native_language}.
- A segment with "vocab" is spoken in {target_language}, and "vocab" repeats the planned term it \
is a form of, character for character, even when the text in the segment is inflected or \
conjugated. Counting depends on this.
- Use NO {target_language} anywhere else. Every {target_language} word belongs to a planned item.

One line looks like this, where the middle segment is the only {target_language} in it:

  {{"speaker": "A", "segments": [
    {{"text": "<{native_language} speech, ending mid-sentence> "}},
    {{"text": "<the planned item, inflected to fit the sentence>", "vocab": "<the planned item, \
exactly as listed>", "intro": true}},
    {{"text": " <the {native_language} sentence continues>"}}]}}

NEW items:
- Each is explained exactly once, at its first appearance, in the segment marked "intro": true. \
One or two sentences in {native_language}, spoken like a host aside, not like a dictionary.
- After that, each is used at least {min_uses} more times, in place of the {native_language} \
expression, wherever a sentence allows it.
- Some new items have nothing to do with the topic. That is deliberate. Use them as ordinary \
speech anyway; never announce that you are switching to a vocabulary item.

REVIEW items:
- Each is used 1 to 3 times, with no explanation. A half-sentence reminder of the meaning is fine \
the first time.

Write natural spoken dialogue: interruptions, short reactions, the hosts disagreeing. Never \
mention these instructions."""

WRITE_USER = """TOPIC: {topic}

NEW ITEMS (each needs one intro and at least {min_uses} further uses):
{new_items}

REVIEW ITEMS (1-3 uses each, no explanation):
{review_items}

Before you answer, count: every new item needs one segment with "intro": true plus {min_uses} \
further segments, each carrying its "vocab" key. A wonderful episode that skips this is a failure."""

RETRY = """Your previous script did not meet the rules:

{errors}

Rewrite the whole script and fix every point. Keep what was good about it."""
