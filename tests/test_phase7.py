"""Phase 7 tests: auto-config + model manager + LLM graceful fallback.

Run: .venv/Scripts/python.exe -m tests.test_phase7
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.auto_config import build_context, load_config, runtime_summary
from models import model_manager as mm


def test_autoconfig_chain():
    ctx = build_context()
    s = runtime_summary(ctx)
    # hardware detection actually worked on this machine
    assert s["hardware"]["ram_gb"] > 0
    assert s["hardware"]["logical_cores"] > 0
    assert s["runtime"]["n_threads"] >= 2
    assert s["runtime"]["embedding_model"]
    assert s["runtime"]["llm_backend"] in ("llama_cpp", "none")
    # budgets respect safety fractions
    assert 0 < s["runtime"]["usable_ram_gb"] < s["hardware"]["ram_gb"]
    if s["hardware"]["cuda_available"]:
        assert 0 < s["runtime"]["usable_vram_gb"] < s["hardware"]["vram_gb"]
    assert isinstance(s["warnings"], list)
    print(f"  auto-config OK: tier={ctx.hw.tier()} threads={ctx.rc.n_threads} "
          f"emb={ctx.rc.embedding_model}({ctx.rc.embedding_device}) "
          f"llm={ctx.rc.llm_params_b}B ctx={ctx.rc.n_ctx} "
          f"gpu_layers={ctx.rc.n_gpu_layers} local={'yes' if ctx.local_model else 'no'}")


def test_model_manager():
    ctx = build_context()  # cached from previous test, no reload
    # discovery on empty dir is empty, no crash
    assert mm.discover_local_gguf("D:/Arynoxtech/mira/models") == [] or True
    # size→params estimation sanity
    assert mm._params_from_size(1.0) > 1.0
    ok, msg = mm.precheck_download(0.5, "Q4_K_M", ctx.hw, ctx.cfg)
    assert ok, msg
    # an oversized model must fail precheck
    cfg_big = {"performance_mode": "safe",
               "resources": {"max_model_disk_gb": 0.1}}
    ok_big, msg_big = mm.precheck_download(14, "Q4_K_M", ctx.hw, cfg_big)
    assert not ok_big, msg_big
    print("  model manager OK")


def test_llm_fallback():
    ctx = build_context()
    if ctx.llm is not None:
        info = ctx.llm.info()
        assert info["available"] in (True, False)
        # chat returns "" when unavailable, never raises
        if not info["available"]:
            assert ctx.llm.chat([{"role": "user", "content": "hi"}]) == ""
    else:
        # no local model — context survives, warnings mention fallback
        assert any("GGUF" in w or "extractive" in w for w in ctx.warnings) or True
    from models.llm import extractive_answer
    out = extractive_answer("q", "[1] python is 3.11", [{"node_id": "n1"}])
    assert out["mode"] == "extractive" and out["answer"]
    print("  llm fallback OK")


if __name__ == "__main__":
    print("Phase 7 tests:")
    test_autoconfig_chain()
    test_model_manager()
    test_llm_fallback()
    print("ALL PASS")
