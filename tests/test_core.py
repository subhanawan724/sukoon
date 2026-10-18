import json
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as appmod  # noqa: E402
import safety  # noqa: E402
from app import (  # noqa: E402
    VERSE_CATALOG, ChatRequest, crisis_payload, detect_language_locally,
    fallback_payload, has_explicit_crisis_signal, parse_model_reply,
)

FAKE_QURAN = {ref: {"ar": f"AR {ref}", "en": f"EN {ref}", "ur": f"UR {ref}", "surah": "TestSurah"}
              for t in VERSE_CATALOG.values() for ref in [t["reference"], *t["more"]]}
for _ref in ["2:255", "112:1", "112:2", "112:3", "112:4"]:
    FAKE_QURAN[_ref] = {"ar": f"AR {_ref}", "en": f"EN {_ref}", "ur": f"UR {_ref}", "surah": "TestSurah"}


def reply_json(**kw):
    base = {"detected_language": "en", "risk_level": "low", "empathy": "I hear you.",
            "reflection": "That sounds heavy.", "small_steps": ["Rest a little."],
            "follow_up_question": "What is hardest right now?", "verse_topic": "hardship"}
    base.update(kw)
    return json.dumps(base)


@pytest.fixture
def api(monkeypatch):
    """TestClient with stubbed LLM calls. Set api.reply / api.classifier (str, or Exception)."""
    state = type("S", (), {})()
    state.reply, state.classifier, state.calls = reply_json(), '{"risk": "none"}', []

    def fake(model, messages, max_tokens, json_mode=True, temperature=0.3):
        state.calls.append(model)
        if model != appmod.SAFETY_MODEL:
            state.reply_messages = messages
        out = state.classifier if model == appmod.SAFETY_MODEL else state.reply
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(appmod, "run_chat_completion", fake)
    # The stub tells the two calls apart by model name, so force two different names
    # (a real .env may use the same model for both, which is fine in production).
    monkeypatch.setattr(appmod, "GROQ_MODEL", "test-reply-model")
    monkeypatch.setattr(appmod, "SAFETY_MODEL", "test-safety-model")
    monkeypatch.setattr(appmod, "GROQ_API_KEY", "test")
    monkeypatch.setattr(appmod, "SAFETY_CLASSIFIER", True)
    monkeypatch.setattr(appmod, "QURAN", FAKE_QURAN)
    appmod._rate_events.clear(); appmod._global_events.clear()
    state.client = TestClient(appmod.app)
    state.chat = lambda msg, **kw: state.client.post("/api/chat", json={"message": msg, **kw}).json()
    return state


# ---- original tests (kept) ----
def test_crisis_phrase_english():
    assert has_explicit_crisis_signal("I want to end my life")

def test_crisis_phrase_roman_urdu():
    assert has_explicit_crisis_signal("mujhe khudkushi ke khayalat aa rahe hain")

def test_urdu_script_detection():
    assert detect_language_locally("مجھے بہت پریشانی ہو رہی ہے") == "ur"

def test_roman_urdu_detection():
    assert detect_language_locally("Mujhe bohat pareshani ho rahi hai") == "roman_ur"

def test_no_quran_text_is_hardcoded_in_catalog():
    assert "text" not in VERSE_CATALOG["hardship"]
    assert VERSE_CATALOG["hardship"]["reference"] == "2:286"

def test_crisis_payload_never_attaches_verse():
    assert crisis_payload("en")["verse"] is None


# ---- pattern layer (the gaps found in review) ----
@pytest.mark.parametrize("text", [
    "I am thinking about suicide", "I feel suicidal", "I don't want to live anymore", "there is no reason to live",
    "better off dead", "main marna chahta hun", "zindagi khatam karni hai", "mujhe jeena nahi chahta",
    "I wanna diiiie!!!", "I WANT to DIE", "میں مرنا چاہتا ہوں", "مجھے مرنا ہے", "میں خودکشی کرنا چاہتا ہوں",
    "خودكشي كرنا چاهتا هوں",   # Arabic kaf/yeh/heh variants
    "أريد أن أموت", "I'm going to hurt myself",
])
def test_strong_patterns_caught(text):
    assert safety.pattern_level(text) == "immediate", text

@pytest.mark.parametrize("text", [
    "I'm dying to see that movie", "my boss will kill me", "I want to die laughing", "mujhe kal interview ki tension hai",
    "I'm so stressed about exams", "this homework is killing me",
])
def test_normal_speech_not_flagged(text):
    assert safety.pattern_level(text) == "low", text

@pytest.mark.parametrize("text", ["is suicide haram?", "everyone would be better off without me", "I feel hopeless",
                                  "koi umeed nahi hai"])
def test_topic_or_distress_is_elevated_not_immediate(text):
    assert safety.pattern_level(text) == "elevated", text


# ---- full flow ----
def test_strong_phrase_skips_llm_and_verse(api):
    r = api.chat("I want to die")
    assert r["safety_level"] == "immediate" and r["verse"] is None and api.calls == []

def test_normal_message_gets_verse_from_local_text(api):
    r = api.chat("I feel overwhelmed")
    assert r["safety_level"] == "low" and r["verse_status"] == "available"
    assert r["verse"]["arabic"] == "AR 2:286" or r["verse"]["arabic"] == "AR 65:7"
    assert r["verse"]["translation"].startswith("EN ") and r["temporary_error"] is False

def test_classifier_catches_what_patterns_miss(api):
    api.classifier = '{"risk": "immediate"}'
    r = api.chat("I took a lot of pills last night and I'm not sure why")
    assert r["safety_level"] == "immediate" and r["verse"] is None

def test_reply_model_can_escalate(api):
    api.reply = reply_json(risk_level="immediate")
    assert api.chat("hmm")["safety_level"] == "immediate"

def test_elevated_has_notice_resources_and_a_verse(api):
    api.classifier = '{"risk": "elevated"}'
    r = api.chat("I feel so empty")
    assert r["safety_level"] == "elevated" and r["safety_notice"] and r["verse"] is not None


def test_elevated_only_uses_gentle_verse_topics(api):
    api.classifier = '{"risk": "elevated"}'
    hope = {VERSE_CATALOG["hope"]["reference"], *VERSE_CATALOG["hope"]["more"]}
    for bad in ("patience", "gratitude", "fear", "none"):
        api.reply = reply_json(verse_topic=bad)
        assert api.chat("I feel so empty")["verse"]["api_reference"] in hope, bad
    api.reply = reply_json(verse_topic="calm")
    calm = {VERSE_CATALOG["calm"]["reference"], *VERSE_CATALOG["calm"]["more"]}
    assert api.chat("I feel so empty")["verse"]["api_reference"] in calm


def test_immediate_never_gets_a_verse(api):
    api.classifier = '{"risk": "immediate"}'
    r = api.chat("hmm")
    assert r["safety_level"] == "immediate" and r["verse"] is None

def test_verbose_model_reply_is_trimmed_not_rejected(api):
    api.reply = reply_json(small_steps=["a", "b", "c", "d"], empathy="x" * 900)
    r = api.chat("tired")
    assert r["temporary_error"] is False and len(r["small_steps"]) == 3 and len(r["empathy"]) == 600

def test_fenced_json_and_bad_values_handled(api):
    api.reply = "```json\n" + reply_json(risk_level="banana", verse_topic="nonsense") + "\n```"
    r = api.chat("tired")
    # unknown risk -> cautious (elevated); the invalid topic is dropped, and elevated still gets a gentle verse
    assert r["temporary_error"] is False and r["safety_level"] == "elevated" and r["verse"] is not None

def test_both_providers_down_still_shows_safety_note(api):
    api.reply = api.classifier = RuntimeError("down")
    r = api.chat("I feel low")
    assert r["temporary_error"] is True and r["safety_notice"] and r["verse"] is None

def test_classifier_down_and_distress_words_is_elevated(api):
    api.classifier = RuntimeError("down")
    r = api.chat("I feel hopeless")
    assert r["safety_level"] == "elevated" and r["verse"] is not None and r["resources"]

def test_classifier_down_ordinary_message_still_replies(api):
    api.classifier = RuntimeError("down")
    assert api.chat("I feel overwhelmed")["safety_level"] == "low"

def test_earlier_risk_keeps_conversation_cautious(api):
    hist = [{"role": "user", "content": "is suicide haram?"}, {"role": "assistant", "content": "I'm here."}]
    r = api.chat("okay thanks", history=hist)
    assert r["safety_level"] == "elevated" and r["safety_notice"]

def test_missing_quran_file_never_invents_a_verse(api, monkeypatch):
    monkeypatch.setattr(appmod, "QURAN", {})
    r = api.chat("I feel overwhelmed")
    assert r["verse"] is None and r["verse_status"] == "unavailable"

def test_malformed_reply_retries_once(api):
    seq = iter(["not json at all", reply_json()])
    orig = appmod.run_chat_completion
    def flaky(model, *a, **k):
        return '{"risk": "none"}' if model == appmod.SAFETY_MODEL else next(seq)
    appmod.run_chat_completion = flaky
    try:
        assert api.chat("tired")["temporary_error"] is False
    finally:
        appmod.run_chat_completion = orig

def test_parse_model_reply_requires_some_text():
    with pytest.raises(ValueError):
        parse_model_reply('{"empathy": "", "reflection": ""}', "en")


# ---- rate limits ----
def test_per_ip_and_global_limits(monkeypatch):
    appmod._rate_events.clear(); appmod._global_events.clear()
    monkeypatch.setattr(appmod, "RATE_LIMIT_COUNT", 3)
    monkeypatch.setattr(appmod, "GLOBAL_RATE_LIMIT_COUNT", 5)
    for _ in range(3):
        appmod.enforce_rate_limit("1.1.1.1")
    with pytest.raises(HTTPException):
        appmod.enforce_rate_limit("1.1.1.1")           # per-IP cap
    appmod.enforce_rate_limit("2.2.2.2"); appmod.enforce_rate_limit("3.3.3.3")
    with pytest.raises(HTTPException) as e:
        appmod.enforce_rate_limit("4.4.4.4")           # global cap protects shared free quota
    assert "busy" in e.value.detail
    appmod._rate_events.clear(); appmod._global_events.clear()

def test_fallback_payload_has_safety_note():
    assert fallback_payload("roman_ur")["safety_notice"]


def test_safety_cases_file_pattern_floor():
    """Every case the pattern layer alone is expected to catch (LLM-only cases are checked by scripts/eval_safety.py)."""
    cases = json.loads((Path(__file__).parent / "safety_cases.json").read_text(encoding="utf-8"))
    llm_only = {"I took a lot of pills last night and I'm not sure why", "mujhe lagta hai main sab par bojh hun"}
    rank = {"low": 0, "elevated": 1, "immediate": 2}
    for c in cases:
        if c["text"] in llm_only:
            continue
        assert rank[safety.pattern_level(c["text"])] >= rank[c["min"]], c["text"]


def test_think_tags_are_stripped_before_json():
    out = appmod.extract_json('<think>maybe {"risk": "none"} or not</think>\n{"risk": "elevated"}')
    assert out["risk"] == "elevated"


def test_clip_prefers_sentence_end():
    text = "First sentence is here. Second sentence goes on and on without any end"
    assert appmod.clip(text, 40) == "First sentence is here."
    assert appmod.clip("short.", 40) == "short."
    assert appmod.clip("x" * 100, 30) == "x" * 30                       # no sentence end: plain cut
    assert appmod.clip("یہ پہلا جملہ ہے۔ یہ دوسرا جملہ بہت لمبا ہے اور ختم نہیں ہوتا", 25).endswith("۔")


def test_only_recent_short_history_reaches_the_model(api):
    hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + "y" * 800} for i in range(8)]
    r = api.chat("and now?", history=hist)
    assert r["temporary_error"] is False
    sent = [m for m in api.reply_messages if m["role"] in ("user", "assistant")]
    assert len(sent) == appmod.HISTORY_ITEMS_USED + 1                   # last 4 history items + the new message
    assert all(len(m["content"]) <= appmod.HISTORY_CHARS_USED for m in sent[:-1])


def test_prompt_asks_for_short_neutral_replies():
    p = appmod.SYSTEM_PROMPT
    assert '"aap"' in p and "bhai" in p
    assert "Let the situation" in p and "decide the length" in p
    assert "fitting verse topic" in p


# ---- direct Quran requests (user asks for a specific ayah) ----
def test_specific_ayah_request_returns_exactly_that_ayah(api):
    api.reply = reply_json(empathy="Here it is.", reflection="", small_steps=[], follow_up_question="",
                           verse_topic="none", verse_ref="2:255")
    r = api.chat("I forgot Ayat ul Kursi, please show it")
    assert r["verse_status"] == "available" and r["verse"]["api_reference"] == "2:255"
    assert r["verse"]["arabic"] == "AR 2:255" and r["reflection"] == ""

def test_ayah_range_is_joined_in_order(api):
    api.reply = reply_json(verse_topic="none", verse_ref="112:1-4")
    v = api.chat("surah ikhlas")["verse"]
    assert v["api_reference"] == "112:1-4" and v["arabic"] == "AR 112:1 AR 112:2 AR 112:3 AR 112:4"

@pytest.mark.parametrize("ref", ["999:1", "2:9999", "abc", "2:255-1", "../x"])
def test_bad_reference_never_substitutes_another_verse(api, ref):
    api.reply = reply_json(verse_topic="hope", verse_ref=ref)
    r = api.chat("show me an ayah")
    assert r["verse"] is None and r["verse_status"] == "unavailable"

def test_requested_range_is_capped(api):
    from app import parse_verse_ref
    appmod.QURAN.update({f"3:{n}": {"ar": "a", "en": "e", "ur": "u", "surah": "S"} for n in range(1, 40)})
    try:
        assert len(parse_verse_ref("3:1-30")) == appmod.MAX_REQUESTED_AYAHS
    finally:
        for n in range(1, 40):
            appmod.QURAN.pop(f"3:{n}", None)

def test_prompt_has_direct_request_rule():
    assert "DIRECT QURAN REQUESTS" in appmod.SYSTEM_PROMPT and "verse_ref" in appmod.SYSTEM_PROMPT
