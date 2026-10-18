"""Build data/quran.json (+ manifest with sha256) from AlQuran Cloud editions.

Online:   python scripts/build_quran.py
Offline:  python scripts/build_quran.py --files arabic.json english.json urdu.json
          (each file = the JSON from https://api.alquran.cloud/v1/quran/<edition>)
Editions: quran-uthmani (Arabic), en.sahih (Sahih International), ur.jalandhry (Fateh Muhammad Jalandhry)

Fails loudly unless every edition has exactly 6,236 ayahs with matching surah/ayah numbers.
Check the output yourself against a printed mushaf for a few verses before trusting it.
"""
import argparse, hashlib, json, sys, urllib.request
from datetime import datetime, timezone
from pathlib import Path

EDITIONS = {"ar": "quran-uthmani", "en": "en.sahih", "ur": "ur.jalandhry"}
OUT = Path(__file__).resolve().parent.parent / "data"


def fetch(edition: str) -> dict:
    with urllib.request.urlopen(f"https://api.alquran.cloud/v1/quran/{edition}", timeout=60) as r:
        return json.load(r)


def flatten(payload: dict) -> dict:
    assert payload.get("code") == 200, "API did not return code 200"
    out = {}
    for s in payload["data"]["surahs"]:
        for a in s["ayahs"]:
            out[f'{s["number"]}:{a["numberInSurah"]}'] = (a["text"], s.get("englishName", ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs=3, metavar=("AR", "EN", "UR"))
    args = ap.parse_args()
    flat = {}
    for i, (key, edition) in enumerate(EDITIONS.items()):
        payload = json.load(open(args.files[i], encoding="utf-8")) if args.files else fetch(edition)
        flat[key] = flatten(payload)
        print(key, edition, len(flat[key]), "ayahs")
    if not all(len(v) == 6236 for v in flat.values()) or not (flat["ar"].keys() == flat["en"].keys() == flat["ur"].keys()):
        sys.exit("ERROR: edition sizes/keys do not match 6236 - not writing anything")
    ayahs = {k: {"ar": flat["ar"][k][0], "en": flat["en"][k][0], "ur": flat["ur"][k][0], "surah": flat["en"][k][1]}
             for k in flat["ar"]}
    body = json.dumps({"meta": {"source": "AlQuran Cloud API", "editions": EDITIONS,
                                "built_at": datetime.now(timezone.utc).isoformat()}, "ayahs": ayahs},
                      ensure_ascii=False).encode("utf-8")
    OUT.mkdir(exist_ok=True)
    (OUT / "quran.json").write_bytes(body)
    (OUT / "quran.manifest.json").write_text(json.dumps({"sha256": hashlib.sha256(body).hexdigest(), "ayahs": len(ayahs)}))
    print("OK", len(ayahs), "ayahs. Spot-check:", ayahs["2:286"]["surah"], "2:286 ->", ayahs["2:286"]["en"][:60])


if __name__ == "__main__":
    main()
