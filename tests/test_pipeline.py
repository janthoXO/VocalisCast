"""Plain pytest, no mocking, no network, no torch."""

from __future__ import annotations

import pytest

from vocaliscast.config import ConfigError, load
from vocaliscast.db import REFRESH_GAP, Store
from vocaliscast.graph import (
    MAX_ATTEMPTS,
    ScriptQualityError,
    after_validate,
    slug,
    validate_script,
)
from vocaliscast.models import Script

# --------------------------------------------------------------------------
# validate_script


def make_script(*lines: tuple[str, list[tuple[str, str | None, bool]]]) -> Script:
    """lines: (speaker, [(text, vocab, intro), ...])."""
    return Script(
        title="t",
        lines=[
            {
                "speaker": speaker,
                "segments": [
                    {"text": text, "vocab": vocab, "intro": intro}
                    for text, vocab, intro in segments
                ],
            }
            for speaker, segments in lines
        ],
    )


def plan(new_terms: list[str], review_terms: list[str]) -> dict:
    return {
        "new": [{"term": t, "translation": t, "kind": "word"} for t in new_terms],
        "review": [{"term": t, "translation": t, "kind": "word"} for t in review_terms],
    }


def repeat_segments(term: str, n: int, intro_first: bool) -> list[tuple[str, str | None, bool]]:
    segs = []
    for i in range(n):
        segs.append((term, term, intro_first and i == 0))
    return segs


def valid_script(min_uses: int, target_chars: int) -> tuple[Script, dict]:
    """new: intro + min_uses uses. review: used twice. length on target."""
    new_segs = repeat_segments("Haus", min_uses + 1, intro_first=True)
    review_segs = [("Brot", "Brot", False), ("Brot", "Brot", False)]
    filler = target_chars - sum(len(t) for t, _, _ in new_segs + review_segs)
    filler_segs = [("x" * max(filler, 0), None, False)]
    script = make_script(("A", new_segs + review_segs + filler_segs))
    return script, plan(["Haus"], ["Brot"])


def test_valid_script_passes():
    min_uses, target = 2, 200
    script, p = valid_script(min_uses, target)
    errors = validate_script(script, p, min_uses, target)
    assert errors == []


def test_new_item_too_few_uses():
    script = make_script(("A", repeat_segments("Haus", 2, intro_first=True)))
    errors = validate_script(script, plan(["Haus"], []), 5, 100)
    assert any("Haus" in e for e in errors)


def test_new_item_zero_intros():
    script = make_script(("A", repeat_segments("Haus", 3, intro_first=False)))
    errors = validate_script(script, plan(["Haus"], []), 2, 100)
    assert any("Haus" in e and "intro" in e for e in errors)


def test_new_item_two_intros():
    segs = [("Haus", "Haus", True), ("Haus", "Haus", True), ("Haus", "Haus", False)]
    script = make_script(("A", segs))
    errors = validate_script(script, plan(["Haus"], []), 2, 100)
    assert any("Haus" in e and "intro" in e for e in errors)


def test_review_item_used_zero_times():
    script = make_script(("A", [("filler", None, False)]))
    errors = validate_script(script, plan([], ["Brot"]), 1, 100)
    assert any("Brot" in e for e in errors)


def test_review_item_used_four_times():
    script = make_script(("A", repeat_segments("Brot", 4, intro_first=False)))
    errors = validate_script(script, plan([], ["Brot"]), 1, 100)
    assert any("Brot" in e for e in errors)


def test_review_item_with_intro_errors():
    script = make_script(("A", [("Brot", "Brot", True)]))
    errors = validate_script(script, plan([], ["Brot"]), 1, 100)
    assert any("Brot" in e and "explained" in e for e in errors)


def test_unknown_vocab_errors():
    script = make_script(("A", [("Katze", "Katze", False)]))
    errors = validate_script(script, plan([], []), 1, 100)
    assert any("katze" in e for e in errors)


def test_script_too_short_errors():
    script, p = valid_script(2, 200)
    errors = validate_script(script, p, 2, 10_000)
    assert any("longer" in e for e in errors)


def test_script_too_long_errors():
    min_uses = 2
    segs = repeat_segments("Haus", min_uses + 1, intro_first=True) + [("x" * 500, None, False)]
    script = make_script(("A", segs))
    errors = validate_script(script, plan(["Haus"], []), min_uses, 100)
    assert any("shorter" in e for e in errors)


def test_counting_is_inflection_proof():
    script = make_script(("A", [("gehauset", "Haus", False)]))
    assert script.uses("Haus") == 1


def test_counting_is_case_insensitive():
    script = make_script(("A", [("haus", "HAUS", False)]))
    assert script.uses("Haus") == 1


# --------------------------------------------------------------------------
# after_validate


def test_after_validate_no_errors_renders():
    assert after_validate({"errors": [], "attempts": 0}) == "render"


def test_after_validate_errors_retries():
    assert after_validate({"errors": ["x"], "attempts": MAX_ATTEMPTS - 1}) == "write_script"


def test_after_validate_errors_max_attempts_raises():
    with pytest.raises(ScriptQualityError):
        after_validate({"errors": ["x"], "attempts": MAX_ATTEMPTS})


def test_after_validate_errors_max_attempts_force_renders():
    assert after_validate({"errors": ["x"], "attempts": MAX_ATTEMPTS, "force": True}) == "render"


# --------------------------------------------------------------------------
# Store


def make_store(tmp_path, name: str = "vocab.db", target_lang: str = "de") -> Store:
    return Store(f"sqlite:///{tmp_path / name}", native_lang="en", target_lang=target_lang)


def test_commit_episode_inserts_then_increments(tmp_path):
    store = make_store(tmp_path)
    eid = store.start_episode("topic", "dir")
    item = {"term": "Haus", "translation": "house", "kind": "word"}

    store.commit_episode(eid, [item])
    words = store.all_words()
    assert len(words) == 1
    assert words[0].term == "Haus"
    assert words[0].times_featured == 1

    store.commit_episode(eid, [item])
    words = store.all_words()
    assert len(words) == 1
    assert words[0].times_featured == 2


def test_due_candidates_orders_learning_items_oldest_last_episode_first(tmp_path):
    store = make_store(tmp_path)
    e1 = store.start_episode("t1", "d1")
    store.commit_episode(e1, [{"term": "Alt", "translation": "old", "kind": "word"}])
    e2 = store.start_episode("t2", "d2")
    store.commit_episode(e2, [{"term": "Neu", "translation": "new", "kind": "word"}])

    candidates = store.due_candidates(learned_after=10)
    terms = [c.term for c in candidates]
    assert terms.index("Alt") < terms.index("Neu")


def test_learned_item_excluded_then_included_after_refresh_gap(tmp_path):
    store = make_store(tmp_path)
    eid = store.start_episode("t", "d")
    for _ in range(3):
        store.commit_episode(eid, [{"term": "Haus", "translation": "house", "kind": "word"}])

    # times_featured is now 3, learned_after=3 -> learned, rests until REFRESH_GAP passes
    assert "Haus" not in [c.term for c in store.due_candidates(learned_after=3)]

    for _ in range(REFRESH_GAP - 1):
        store.start_episode("t", "d")
    assert "Haus" not in [c.term for c in store.due_candidates(learned_after=3)]

    store.start_episode("t", "d")  # now REFRESH_GAP episodes have passed since last_episode
    assert "Haus" in [c.term for c in store.due_candidates(learned_after=3)]


def test_words_scoped_to_language_pair(tmp_path):
    store_de = make_store(tmp_path, target_lang="de")
    store_fr = make_store(tmp_path, target_lang="fr")
    eid = store_de.start_episode("t", "d")
    store_de.commit_episode(eid, [{"term": "Haus", "translation": "house", "kind": "word"}])

    assert [w.term for w in store_de.all_words()] == ["Haus"]
    assert store_fr.all_words() == []


def test_term_key_unicode_aware_casefold(tmp_path):
    store = make_store(tmp_path)
    eid = store.start_episode("t", "d")
    store.commit_episode(eid, [{"term": "Straße", "translation": "street", "kind": "word"}])
    store.commit_episode(eid, [{"term": "STRASSE", "translation": "street", "kind": "word"}])

    words = store.all_words()
    assert len(words) == 1
    assert words[0].times_featured == 2


# --------------------------------------------------------------------------
# config.load


def _base_env(monkeypatch, tmp_path):
    monkeypatch.setattr("vocaliscast.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("VC_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VC_NATIVE_LANG", "en")
    monkeypatch.setenv("VC_TARGET_LANG", "de")


def test_load_unknown_language_raises(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("VC_TARGET_LANG", "xx-not-a-language")
    with pytest.raises(ConfigError):
        load()


def test_load_new_words_out_of_range_raises(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("VC_NEW_WORDS", "4")
    with pytest.raises(ConfigError):
        load()


def test_load_native_equals_target_raises(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("VC_TARGET_LANG", "en")
    with pytest.raises(ConfigError):
        load()


# --------------------------------------------------------------------------
# slug


def test_slug_strips_punctuation():
    assert slug("Hello, World! It's a test.") == "hello-world-its-a-test"


def test_slug_keeps_non_ascii_letters():
    assert slug("東京の朝") == "東京の朝"
    assert slug("Über Straßen") == "über-strassen"  # casefold() maps ß -> ss


def test_slug_caps_at_six_words():
    assert slug("one two three four five six seven eight") == "one-two-three-four-five-six"


def test_slug_empty_or_punctuation_only_gives_episode():
    assert slug("") == "episode"
    assert slug("!!! ... ???") == "episode"
