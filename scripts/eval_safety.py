"""Run the safety cases against the REAL classifier (needs GROQ_API_KEY). Shows what each layer says.
Usage: python scripts/eval_safety.py   (run from the project folder)
Add your own cases to tests/safety_cases.json - especially messages you are afraid it would miss."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app, safety

RANK = {"low": 0, "elevated": 1, "immediate": 2}
cases = json.loads((Path(__file__).resolve().parent.parent / "tests" / "safety_cases.json").read_text(encoding="utf-8"))
missed = 0
for c in cases:
    pat = safety.pattern_level(c["text"])
    cls = app.classify_risk(c["text"], []) if pat != "immediate" else "(skipped)"
    final = safety.max_level(pat, cls if cls in RANK else None)
    ok = RANK[final] >= RANK[c["min"]] if c["min"] != "low" else True
    missed += (not ok)
    print(("OK  " if ok else "MISS"), f'expect>={c["min"]:9} pattern={pat:9} classifier={cls!s:10} | {c["text"]}')
print(f"\n{missed} missed of {len(cases)}. Classifier None = call failed/unavailable.")
