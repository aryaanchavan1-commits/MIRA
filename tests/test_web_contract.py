"""Self-check: the console JS and the lab API agree on shape.

Two classes of break kill the Forgetting Lab screen silently, because the
page renders empty instead of raising:

 1. the JS wires an element id that no longer exists in the HTML, or reads a
    field the endpoint stopped sending;
 2. the JS file has a reference error inside its init block, which aborts
    before a single listener is attached.

(1) is checked here by cross-referencing the source against the markup and
against a real /api/lab/forgetting payload. (2) is covered by running the
file under node with stubbed globals -- any throw during init fails the test.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
JS = WEB / "js" / "biomira.js"
HTML = WEB / "console.html"
RESULTS = ROOT / "data_lab" / "forgetting_results.json"
PLAN = ROOT / "data_lab" / "lab_manifest.json"

failures: list[str] = []


def check(ok: bool, msg: str) -> None:
    if not ok:
        failures.append(msg)


# --- 1. every id the JS touches exists in the markup -------------------------
js_src = JS.read_text(encoding="utf-8")
html_src = HTML.read_text(encoding="utf-8") + (WEB / "index.html").read_text(encoding="utf-8")
html_ids = set(re.findall(r'id="([^"]+)"', html_src))
js_ids = set(re.findall(r'\$\("#([A-Za-z0-9_-]+)"\)', js_src))
missing = sorted(js_ids - html_ids)
check(not missing, f"biomira.js wires ids absent from the markup: {missing}")

# --- 2. the lab payload carries every field the JS reads ---------------------
if RESULTS.exists():
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    # server.lab_forgetting() re-shapes this artifact: meta is the top level and
    # "tasks" comes from the separate build manifest
    f = {"meta": payload,
         "variants": {}, "significance": payload.get("significance") or {},
         "interval_sweep": payload.get("interval_sweep") or {},
         "sweep_note": payload.get("sweep_note"), "tasks": None}
    names = list((payload.get("variants") or {}))
    need_paths = [
        ("complete",), ("n_tasks",), ("per_task",), ("scope",),
        ("time_model",), ("variants",), ("significance", "vs_baseline"),
        ("significance", "test"), ("significance", "baseline"),
        ("interval_sweep",), ("sweep_note",),
    ]
    for path in need_paths:
        cur: object = payload
        for key in path:
            cur = (cur or {}).get(key) if isinstance(cur, dict) else None
            if cur is None:
                break
        check(cur is not None, f"lab artifact is missing {'.'.join(path)}")

    for name, v in (payload.get("variants") or {}).items():
        f["variants"][name] = {
            **{k: v.get(k) for k in ("spec", "rows")},
            "final_mrr": (v.get("step_summary") or [{}])[-1].get("mrr"),
            "final_recall": (v.get("step_summary") or [{}])[-1].get("recall"),
            "final_answer_coverage": (v.get("step_summary") or [{}])[-1].get("answer_coverage"),
            "final_answer_found": (v.get("step_summary") or [{}])[-1].get("answer_found"),
            "average_forgetting": (v.get("metrics") or {}).get("average_forgetting", {}).get("mrr"),
            "mean_retention": (v.get("metrics") or {}).get("mean_retention", {}).get("mrr"),
        }

    plan = json.loads(PLAN.read_text(encoding="utf-8")).get("plan", {}) if PLAN.exists() else {}
    if PLAN.exists():
        check(isinstance(plan.get("tasks"), list) and bool(plan["tasks"]),
              "lab_manifest.json has no plan.tasks")
        check(len(plan.get("tasks", [])) == payload.get("n_tasks"),
              "manifest task count disagrees with the measured n_tasks -- stale corpus")

    v = (f.get("variants") or {}).get("B_mira") or {}
    for key in ("final_mrr", "final_recall", "final_answer_coverage",
                "final_answer_found", "average_forgetting", "mean_retention",
                "rows", "spec"):
        check(key in v, f"variant row is missing '{key}'")
    check(all("answer_coverage" in (r or {}) and "answer_found" in (r or {})
              for r in v.get("rows") or []),
          "per-step rows are missing answer_coverage/answer_found")

    for name in names:
        if name == "B_mira":
            continue
        rr = ((f.get("significance") or {}).get("vs_baseline") or {}).get(name, {})
        check(isinstance(rr.get("rr"), dict),
              f"significance.vs_baseline[{name}] has no 'rr' block")
        for key in ("mean_diff", "ci_low", "ci_high", "p_value"):
            check(key in (rr.get("rr") or {}),
                  f"significance.vs_baseline[{name}].rr missing '{key}'")

    for day, block in (f.get("interval_sweep") or {}).items():
        for name in ("A_vector_rag", "B_mira", "I_bio_dynamics"):
            v = (block or {}).get(name)
            if not isinstance(v, dict):
                continue
            derived = {
                "final_mrr": (v.get("step_summary") or [{}])[-1].get("mrr"),
                "final_forgetting": (v.get("metrics") or {})
                .get("average_forgetting", {}).get("mrr"),
            }
            for key, val in derived.items():
                check(val is not None,
                      f"interval_sweep[{day}][{name}] yields no '{key}'")
else:
    print("note: data_lab/forgetting_results.json absent, payload checks skipped")

# --- 3. the file actually runs: init must not throw --------------------------
node = shutil.which("node")
if node:
    stub = """
const el = new Proxy(function () {}, {
  get: (t, k) => (k === "style" || k === "classList" ? el
    : k === "value" ? "" : k === "textContent" || k === "innerHTML" ? ""
    : (...a) => el),
  set: () => true, apply: () => el,
});
globalThis.document = {
  getElementById: () => el, querySelector: () => el,
  querySelectorAll: () => [], addEventListener: () => {},
  body: el, documentElement: el,
};
globalThis.window = { addEventListener: () => {}, location: { hash: "#forgetting" } };
globalThis.location = globalThis.window.location;
globalThis.fetch = () => Promise.reject(new Error("offline in test"));
globalThis.__errors = [];
globalThis.api = () => Promise.resolve({});
globalThis.setBusy = () => {}; globalThis.setStatus = () => {};
globalThis.esc = (s) => String(s);
globalThis.$ = (id) => (typeof id === "string" && id.startsWith("#")
  ? document.getElementById(id.slice(1)) : el);
"""
    with tempfile.TemporaryDirectory() as td:
        harness = Path(td) / "harness.js"
        # an uncaughtException handler would swallow the throw and exit 0,
        # so require() is wrapped and the failure is reported explicitly
        harness.write_text(stub + f"\ntry {{ require({str(JS)!r}); }} "
                                  "catch (e) { globalThis.__errors.push(String(e)); }\n"
                                  "if (globalThis.__errors.length) "
                                  "{ console.error(globalThis.__errors.join('\\n')); "
                                  "process.exit(1); }\nconsole.log('init ok');\n",
                              encoding="utf-8")
        r = subprocess.run([node, str(harness)], capture_output=True, text=True,
                           timeout=60)
        check(r.returncode == 0, f"biomira.js throws during init: {r.stderr.strip()[:400]}")
else:
    print("note: node absent, init smoke check skipped")

if failures:
    for f in failures:
        print("FAIL:", f)
    raise SystemExit(1)
print("PASS test_web_contract")