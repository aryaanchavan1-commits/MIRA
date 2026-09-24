"""Phase 12 tests: dataset builder grounding + training safety gates."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.lora import precheck


class FakeHW:
    """RTX 3050-class numbers: must FAIL the VRAM gate."""

    def usable_vram_gb(self, mode):
        return 3.2

    def usable_ram_gb(self, mode):
        return 10.0

    free_disk_gb = 40.0


def test_precheck_skips_small_vram():
    with tempfile.TemporaryDirectory() as td:
        ds = Path(td) / "ds.json"
        ds.write_text(json.dumps([{"query": "q", "answer": "a"}] * 60),
                      encoding="utf-8")
        ok, reason = precheck(FakeHW(), {}, str(ds), "base.gguf")
        assert not ok and "VRAM" in reason, f"must skip: {reason}"
        print("  VRAM gate OK:", reason[:70])

        ds.write_text(json.dumps([{"query": "q", "answer": "a"}] * 10),
                      encoding="utf-8")
        ok, reason = precheck(FakeHW(), {}, str(ds), "base.gguf")
        assert not ok and "too small" in reason
        print("  dataset-size gate OK")


def main():
    test_precheck_skips_small_vram()
    print("PHASE12 TESTS PASS")


if __name__ == "__main__":
    main()
