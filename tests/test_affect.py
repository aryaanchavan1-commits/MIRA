"""Focused assert tests for the deterministic simulated affect layer."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.affect import (
    AFFECT_DISCLOSURE,
    AffectiveState,
    NEUTRAL_BASELINE,
)
from core.agent import AgentPipeline
from core.answer import Answer


def _answer(route: str, mode: str = "extractive", citation_errors: int = 0,
            text: str = "Observable answer text") -> Answer:
    return Answer(
        text=text,
        mode=mode,
        agent_mode=route,
        memories=[{"components": {"semantic": 1.0}}],
        metrics={"context_tokens": 20, "citation_error": citation_errors},
    )


def test_bounds_feedback_and_snapshot_are_bounded():
    state = AffectiveState(valence=9, arousal=-2, confidence=4, stress=99)
    assert state.valence == 1.0
    assert state.arousal == 0.0
    assert state.confidence == 1.0
    assert state.stress == 1.0

    state.apply_feedback(valence=-0.25, arousal=0.25, confidence=0.1, stress=0.1)
    for name, bounds in (("valence", (-1, 1)), ("arousal", (0, 1)),
                         ("confidence", (0, 1)), ("stress", (0, 1))):
        assert bounds[0] <= state.snapshot()[name] <= bounds[1]

    try:
        state.apply_feedback(valence=2)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range explicit feedback must be rejected")

    snapshot = state.snapshot()
    try:
        snapshot["valence"] = 0.5
    except TypeError:
        pass
    else:
        raise AssertionError("Answer affect snapshots must be immutable")
    assert snapshot["simulated"] is True
    assert "not biological" in snapshot["disclosure"]
    assert snapshot["disclosure"] == AFFECT_DISCLOSURE


def test_decay_moves_toward_the_neutral_baseline():
    state = AffectiveState(valence=1, arousal=1, confidence=1, stress=1)
    snapshot = state.decay(0.5)
    assert snapshot["valence"] == 0.5
    assert snapshot["arousal"] == 0.5
    assert snapshot["confidence"] == 0.75
    assert snapshot["stress"] == 0.5
    assert snapshot["update_count"] == 1
    assert "neutral baseline" in snapshot["last_reason"]
    assert state.decay(0)["valence"] == 0.5
    for value in (-0.1, 1.1, float("nan")):
        try:
            state.decay(value)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid decay must be rejected")


def test_routes_are_deterministic_and_use_observable_signals():
    memory = AffectiveState()
    web = AffectiveState()
    memory_snapshot = memory.update_from_answer(_answer("memory"))
    web_snapshot = web.update_from_answer(_answer("web"))
    numeric = ("valence", "arousal", "confidence", "stress", "update_count")
    assert {key: memory_snapshot[key] for key in numeric} == \
           {key: web_snapshot[key] for key in numeric}
    assert memory_snapshot["last_reason"] == "grounded memory answer"
    assert web_snapshot["last_reason"] == "grounded web answer"
    assert memory_snapshot["valence"] > NEUTRAL_BASELINE["valence"]
    assert memory_snapshot["confidence"] > NEUTRAL_BASELINE["confidence"]

    bad_citation = AffectiveState()
    bad_snapshot = bad_citation.update_from_answer(_answer("memory", citation_errors=2))
    assert bad_snapshot["confidence"] < memory_snapshot["confidence"]
    assert bad_snapshot["stress"] > memory_snapshot["stress"]
    assert "citation_errors=2" in bad_snapshot["last_reason"]

    parametric = AffectiveState().update_from_answer(
        _answer("parametric", mode="llm", text="Model answer"))
    assert parametric["confidence"] < NEUTRAL_BASELINE["confidence"]
    assert parametric["stress"] > NEUTRAL_BASELINE["stress"]
    assert parametric["last_reason"] == "parametric answer; ungrounded"

    no_evidence = AffectiveState().update_from_answer(
        _answer("parametric", mode="no_evidence", text="No evidence"))
    assert no_evidence["confidence"] < NEUTRAL_BASELINE["confidence"]
    assert "no-evidence" in no_evidence["last_reason"]


def test_identity_is_not_rewarded_as_grounded_and_snapshot_survives_next_update():
    state = AffectiveState()
    identity = state.update_from_answer(
        _answer("identity", mode="deterministic", text="Built-in identity"))
    assert identity["valence"] == NEUTRAL_BASELINE["valence"]
    assert identity["arousal"] == NEUTRAL_BASELINE["arousal"]
    assert identity["confidence"] == NEUTRAL_BASELINE["confidence"]
    assert identity["stress"] == NEUTRAL_BASELINE["stress"]
    assert "identity" in identity["last_reason"]
    state.update_from_signals("memory")
    assert identity["update_count"] == 1
    assert state.snapshot()["update_count"] == 2


def test_identity_resets_prior_state_and_answer_boundary_clamps_values():
    state = AffectiveState(valence=1.0, arousal=1.0, confidence=1.0, stress=1.0)
    identity = state.update_from_answer(
        _answer("identity", mode="deterministic", text="Built-in identity"))
    assert {key: identity[key] for key in ("valence", "arousal", "confidence", "stress")} == \
           {key: NEUTRAL_BASELINE[key] for key in ("valence", "arousal", "confidence", "stress")}

    answer = Answer(affect_snapshot={
        "valence": 99, "arousal": -2, "confidence": float("nan"),
        "stress": "bad", "simulated": False,
    })
    assert -1 <= answer.affect_snapshot["valence"] <= 1
    assert 0 <= answer.affect_snapshot["arousal"] <= 1
    assert 0 <= answer.affect_snapshot["confidence"] <= 1
    assert 0 <= answer.affect_snapshot["stress"] <= 1
    assert answer.affect_snapshot["simulated"] is True


def test_agent_attaches_route_snapshot_without_changing_answer_text():
    class Pipe:
        doc_titles = {}
        llm = None
        workspace_ingest = None

        def ask(self, question, active_components=None):
            return _answer("memory", text="The answer is grounded in evidence.")

    state = AffectiveState()
    agent = AgentPipeline(Pipe(), {}, affect_state=state)
    answer = agent.ask("What is the grounded fact?")
    assert answer.text == "The answer is grounded in evidence."
    assert answer.affect_snapshot["last_reason"] == "grounded memory answer"
    assert answer.affect is answer.affect_snapshot
    assert answer.affect_snapshot["update_count"] == state.update_count

    identity = agent.ask("Who made MIRA?")
    assert identity.agent_mode == "identity"
    assert identity.affect_snapshot["last_reason"].startswith("deterministic identity")


def test_identity_parametric_and_no_evidence_metrics_are_complete():
    from core.answer import AnswerPipeline
    from core.memory import MemoryFrame
    from models.embeddings import EmbeddingBackend
    from storage.graph_store import GraphStore

    required = {
        "latency_ms", "retrieval_ms", "n_candidates", "n_retrieved",
        "n_memories", "n_selected_evidence", "n_excluded_candidates",
        "context_tokens", "raw_tokens", "compression_ratio", "citation_error",
        "context_budget", "completion_budget", "llm_mode",
    }

    class Pipe:
        doc_titles = {}
        llm = None
        workspace_ingest = None

        def ask(self, question, active_components=None):
            return Answer(text="no evidence", mode="no_evidence",
                          selected_evidence_ids=[])

    agent = AgentPipeline(Pipe(), {})
    identity = agent.ask("Who made MIRA?")
    parametric = agent._parametric("What is this?")
    no_evidence = AnswerPipeline(
        MemoryFrame(), None, GraphStore(), EmbeddingBackend("stub"), config={}
    ).ask("unknown question")
    for answer in (identity, parametric, no_evidence):
        assert required <= set(answer.metrics)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
