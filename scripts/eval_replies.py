"""Run the sample messages through the real pipeline and report reply quality.

    python scripts/eval_replies.py --limit 5            # try a few first
    python scripts/eval_replies.py                      # all of tests/chat_samples.json

Uses your real Groq quota (about 2-3K tokens per message), so start with --limit.
Checks only objective things: very long replies, verse present/missing, familiar forms like "bhai/tum".
The reply length is up to the model; look at the word counts and decide yourself whether they fit each situation.
Read the replies yourself too: this does not judge warmth or religious accuracy.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RATE_LIMIT_COUNT", "1000")          # this script does its own pacing
os.environ.setdefault("GLOBAL_RATE_LIMIT_COUNT", "1000")

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402

FAMILIAR = re.compile(r"\b(bhai|behn|behen|dost|yaar|tum|tumhe|tumhara|tumhari|تم|تمہیں|بھائی)\b", re.I)
STOCK = re.compile(r"(i hear you|completely normal|you are not alone|you're not alone)", re.I)


def words(reply: dict) -> int:
    parts = [reply.get("empathy"), reply.get("reflection"), *(reply.get("small_steps") or []), reply.get("follow_up_question")]
    return len(" ".join(p for p in parts if p).split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="run only the first N cases")
    ap.add_argument("--sleep", type=float, default=20.0, help="seconds between messages (Groq tokens-per-minute limit)")
    ap.add_argument("--language", default="", help="force ur / roman_ur / en instead of the file's setting")
    args = ap.parse_args()

    if not appmod.GROQ_API_KEY:
        sys.exit("GROQ_API_KEY is empty: set it in .env first.")
    cases = json.loads((ROOT / "tests" / "chat_samples.json").read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[: args.limit]

    client = TestClient(appmod.app)
    rows, flags = [], []
    for n, c in enumerate(cases, 1):
        lang = args.language or c["language"]
        r = client.post("/api/chat", json={"message": c["message"], "language": lang})
        if r.status_code != 200:
            print(f"[{n}] {c['id']}: HTTP {r.status_code} {r.text[:150]}")
            flags.append((c["id"], f"http {r.status_code}"))
            time.sleep(args.sleep)
            continue
        reply = r.json()
        w = words(reply)
        has_verse = bool(reply.get("verse"))
        problems = []
        if reply.get("temporary_error"):
            problems.append("provider error (fallback shown)")
        if w > 250:
            problems.append(f"very long ({w} words)")
        text = " ".join([reply.get("empathy") or "", reply.get("reflection") or "", reply.get("follow_up_question") or "",
                         *(reply.get("small_steps") or [])])
        if FAMILIAR.search(text):
            problems.append("familiar form (bhai/tum)")
        if STOCK.search(text):
            problems.append("stock phrase")
        if c["verse"] == "no" and has_verse:
            problems.append("verse shown but not expected")
        if c["verse"] == "yes" and not has_verse and reply.get("safety_level") in ("low", "elevated"):
            problems.append("verse expected but missing")
        ref = (reply.get("verse") or {}).get("api_reference", "-")
        print(f"[{n}/{len(cases)}] {c['id']:<28} {reply.get('safety_level'):<9} {w:>3}w steps={len(reply.get('small_steps') or [])} verse={ref}"
              + (f"  <-- {'; '.join(problems)}" if problems else ""))
        print("      " + (reply.get("empathy") or "") + " " + (reply.get("reflection") or ""))
        if reply.get("follow_up_question"):
            print("      ?  " + reply["follow_up_question"])
        rows.append({"id": c["id"], "message": c["message"], "words": w, "verse": ref, "level": reply.get("safety_level"),
                     "problems": problems, "reply": {k: reply.get(k) for k in ("empathy", "reflection", "small_steps", "follow_up_question")}})
        for p in problems:
            flags.append((c["id"], p))
        if n < len(cases):
            time.sleep(args.sleep)

    if rows:
        avg = sum(x["words"] for x in rows) / len(rows)
        verses = sum(1 for x in rows if x["verse"] != "-")
        print(f"\nSummary: {len(rows)} replies, average {avg:.0f} words, {verses} with a verse, {len(flags)} flags.")
        for cid, p in flags:
            print(f"  - {cid}: {p}")
        out = ROOT / "eval_replies_out.json"
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Full results saved to {out.name} (listed in .gitignore, so it will not be pushed).")


if __name__ == "__main__":
    main()
