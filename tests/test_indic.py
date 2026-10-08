"""Indic-methods tests: Kaṭapayādi encoding table, Nyāya pramāṇa classification,
and the derived-label read path through VersionChain (persisted and fresh)."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import indic_methods as im
from core import ncm

FAILURES = []


def check(name, fn):
    try:
        fn()
        print(f"PASS {name}")
    except Exception as exc:  # noqa: BLE001 — harness must not die early
        FAILURES.append((name, exc))
        print(f"FAIL {name}: {exc}")


def test_katapayadi_table():
    # spot values from the standard consonant assignment
    cases = {
        "k": "1", "kh": "2", "g": "3", "gh": "4", "G": "5",     # ka-varga
        "c": "6", "ch": "7", "chh": "7", "j": "8", "jh": "9",   # ca-varga
        "T": "1", "Th": "2", "D": "3", "Dh": "4", "N": "5",     # ṭa-varga
        "t": "6", "th": "7", "d": "8", "dh": "9", "n": "0",     # ta-varga
        "p": "1", "ph": "2", "b": "3", "bh": "4", "m": "5",     # pa-varga
        "y": "1", "r": "2", "l": "3", "v": "4",
        "z": "5", "S": "6", "s": "7", "h": "8", "x": "0",
        "a": "0", "aa": "00", "i": "0",                      # vowels = 0 (per char)
    }
    for tok, want in cases.items():
        got = im.katapayadi(tok)
        assert got == want, (tok, got, want)
    # aspirate greedy match: "chh" must not decode as ch + h ("78")
    assert im.katapayadi("chhaya") == "7010", im.katapayadi("chhaya")
    # word-level: vowels contribute zeros between consonants
    assert im.katapayadi("kaka") == "1010"
    # multi-word: spaces skipped
    assert im.katapayadi("ta da") == "6080"
    # empty/blank subject
    assert im.katapayadi("") == "0" and im.katapayadi("   ") == "0"


def test_pramana_classification():
    assert im.pramana_of("chat 12") == "pratyaksa"
    assert im.pramana_of("(sdk) direct write") == "pratyaksa"
    assert im.pramana_of("web ingest: url") == "shabda"
    assert im.pramana_of("document/file.pdf") == "shabda"
    assert im.pramana_of("consolidation replay pass") == "anumana"
    assert im.pramana_of("gist merge derived") == "anumana"
    assert im.pramana_of("analogy: similar case") == "upamana"
    assert im.pramana_of("") == "shabda"          # honest default: testimony
    assert im.pramana_of("mystery source") == "shabda"
    assert set(im.PRAMANA_LABELS) == {"pratyaksa", "anumana", "upamana", "shabda"}


def test_version_chain_annotation():
    vc = ncm.VersionChain(max_depth=4)
    vc.update("prefers ide", "User likes Python", source="chat 1")
    vc.update("prefers ide", "Source: docs/user-guide.pdf says X",
              source="web ingest url")
    hist = vc.history("prefers ide")
    assert hist[0]["pramana"] == "pratyaksa", hist[0]
    assert hist[1]["pramana"] == "shabda", hist[1]
    kp = hist[0]["katapayadi"]
    assert isinstance(kp, str) and kp and kp != "", hist[0]
    # derived labels are copies, not stored state: raw records stay clean
    raw = vc.versions[ncm.VersionChain.key("prefers ide")][0]
    assert "pramana" not in raw and "katapayadi" not in raw
    cur = vc.current("prefers ide")
    assert cur["pramana"] == "shabda"


def test_annotation_survives_sqlite_restart():
    from storage.sqlite_store import SQLiteStore
    db = os.path.join(tempfile.mkdtemp(), "indic.db")
    store = SQLiteStore(db)
    vc = ncm.VersionChain(max_depth=4, store=store)
    vc.update("project language", "First fact", source="chat 9")
    vc2 = ncm.VersionChain(max_depth=4, store=SQLiteStore(db))
    rec = vc2.current("project language")
    assert rec["content"] == "First fact"
    assert rec["pramana"] == "pratyaksa"           # derived at read time
    assert rec["katapayadi"] == im.katapayadi("project language")
    store.close()


def run_all() -> int:
    check("katapayadi_table", test_katapayadi_table)
    check("pramana_classification", test_pramana_classification)
    check("version_chain_annotation", test_version_chain_annotation)
    check("annotation_survives_sqlite_restart",
          test_annotation_survives_sqlite_restart)
    print(f"{4 - len(FAILURES)}/4 passed")
    if FAILURES:
        for name, exc in FAILURES:
            print(f"  {name}: {exc}")
        return 1
    print("INDIC TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())
