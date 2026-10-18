# Sukoon AI v2: Islamic emotional-support MVP

Non-clinical, reflective support chat (English / Urdu / Roman Urdu). **Not** therapy, **not** an emergency service.
Dev starter: read "Before any public launch" first.

## Run
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                      # put your GROQ_API_KEY
python scripts/build_quran.py                             # builds data/quran.json (needs internet; or --files a.json e.json u.json)
python -m pytest -q tests                                 # all green before you continue
python scripts/eval_safety.py                             # real classifier on tests/safety_cases.json
uvicorn app:app --reload                                  # http://127.0.0.1:8000
```
Render/Docker start command: `uvicorn app:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips='*'`
(`--proxy-headers` is required, otherwise every visitor looks like one IP and the rate limit hits everyone).

## How a message is handled
1. **Pattern layer** (`safety.py` + `data/crisis_patterns.json`): normalizes Urdu/Arabic letter variants, stretched letters,
   punctuation; matches regex *stems*. Edit the JSON to tune; no code change. Explicit danger = immediate crisis reply, no LLM call.
2. **AI risk classifier** (separate small model, every message) and **the reply model's own risk label** run in parallel.
3. **History**: earlier risky turns keep the chat at >= elevated.
4. Highest level wins: immediate = fixed safe reply + verified helplines, no verse; elevated = reply + notice + helplines, no verse;
   low = reply, plus an optional verse.
5. The model only picks a topic ID. Exact Arabic + translation come from the local, sha256-checked `data/quran.json`.
   Missing or tampered file = no verse (never a guess).

## Before any public launch (not done by this code)
- Fill `data/helplines.json` with numbers **you verified** per country (entries without `verified_on` are never shown).
- Qualified Muslim scholar + clinician review of `VERSE_CATALOG` and the safety texts; check `data/quran.json` against a mushaf.
- Grow `tests/safety_cases.json` with real, hard messages in all three languages; keep running `scripts/eval_safety.py`.
- Pattern + AI layers reduce but cannot eliminate missed emergencies. Obfuscated spellings rely on the classifier.
- Privacy: messages go to Groq; check their retention terms; add a privacy policy; adults-only closed beta first.
- In-memory rate limits work for ONE process only. Free Groq limits are shared by all users (see GLOBAL_RATE_LIMIT_COUNT).
