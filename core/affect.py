"""Deterministic, transparent simulated affect bookkeeping.

This module is deliberately *not* a model of biological emotion, sentience,
consciousness, or inner experience.  It is a small algorithmic state machine
that records bounded signals already observable in an answer: its route,
whether evidence was available, citation validity, and optional explicit
feedback.  It never reads the mandala memory or changes retrieval.

The neutral baseline is valence=0, arousal=0, confidence=0.5, stress=0.  Every
update first moves toward that baseline, then applies a fixed, inspectable
route delta.  The rules are intentionally boring so the same sequence of
inputs produces the same output on every run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from threading import RLock
from typing import Any, Dict, Mapping, Optional


AFFECT_DISCLOSURE = (
    "Algorithmic simulated affect derived only from observable answer signals; "
    "not biological emotion, sentience, or consciousness."
)

# Public so callers and tests can inspect the exact neutral target and bounds.
NEUTRAL_BASELINE: Dict[str, float] = {
    "valence": 0.0,
    "arousal": 0.0,
    "confidence": 0.5,
    "stress": 0.0,
}
BOUNDS: Dict[str, tuple[float, float]] = {
    "valence": (-1.0, 1.0),
    "arousal": (0.0, 1.0),
    "confidence": (0.0, 1.0),
    "stress": (0.0, 1.0),
}
DEFAULT_DECAY = 0.20


class AffectSnapshot(dict):
    """A JSON-friendly immutable mapping captured at one state update.

    A normal ``dict`` copy is returned by :meth:`AffectiveState.to_dict` for
    callers that need to mutate their transport object.  The object attached to
    an ``Answer`` remains read-only.
    """

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("affect snapshots are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __ior__(self, other: Any) -> None:  # type: ignore[override]
        self._immutable(other)

    def copy(self) -> Dict[str, Any]:
        return dict(self)

    def __copy__(self) -> "AffectSnapshot":
        return self

    def __deepcopy__(self, memo: Dict[int, Any]) -> "AffectSnapshot":
        return self


def _rounded(value: float) -> float:
    """Keep snapshots stable across harmless floating-point representation noise."""
    return round(float(value), 6)


def _bounded(name: str, value: Any) -> float:
    """Coerce one internal field to its invariant range."""
    lo, hi = BOUNDS[name]
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        number = NEUTRAL_BASELINE[name]
    if not isfinite(number):
        number = NEUTRAL_BASELINE[name]
    return min(hi, max(lo, number))


def _unit_number(value: Any, name: str) -> float:
    """Validate a caller-supplied unit interval value."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _feedback_number(value: Any, name: str) -> float:
    """Validate a bounded user delta; deltas are limited to one full scale."""
    if isinstance(value, bool):
        raise ValueError(f"{name} feedback must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} feedback must be a finite number") from exc
    if not isfinite(number) or not -1.0 <= number <= 1.0:
        raise ValueError(f"{name} feedback must be between -1 and 1")
    return number


def neutral_snapshot() -> AffectSnapshot:
    """Return a fresh immutable neutral snapshot."""
    return AffectSnapshot({
        **NEUTRAL_BASELINE,
        "label": "neutral",
        "update_count": 0,
        "last_reason": "neutral baseline",
        "simulated": True,
        "algorithmic": True,
        "disclosure": AFFECT_DISCLOSURE,
    })


def coerce_snapshot(value: Any) -> AffectSnapshot:
    """Return a bounded, disclosure-safe snapshot for an untrusted value.

    ``Answer`` is a public boundary and callers may construct it from API or
    plugin data.  Never let a mapping smuggle NaN, out-of-range dimensions, a
    false simulation flag, or an oversized label into an answer snapshot.
    """
    raw = value if isinstance(value, Mapping) else {}
    out: Dict[str, Any] = {
        **NEUTRAL_BASELINE,
        "label": "neutral",
        "update_count": 0,
        "last_reason": "neutral baseline",
        "simulated": True,
        "algorithmic": True,
        "disclosure": AFFECT_DISCLOSURE,
    }
    for name in BOUNDS:
        out[name] = _rounded(_bounded(name, raw.get(name, NEUTRAL_BASELINE[name])))
    if "label" in raw:
        out["label"] = str(raw.get("label") or "neutral")[:64]
    if "last_reason" in raw:
        out["last_reason"] = str(raw.get("last_reason") or "neutral baseline")[:160]
    if "update_count" in raw and not isinstance(raw.get("update_count"), bool):
        try:
            out["update_count"] = min(1_000_000, max(0, int(raw.get("update_count"))))
        except (TypeError, ValueError, OverflowError):
            out["update_count"] = 0
    return AffectSnapshot(out)


@dataclass
class AffectiveState:
    """Workspace-owned, deterministic simulated affect state.

    The public numeric fields are always kept within their documented bounds.
    Mutations are serialized with an ``RLock`` so a snapshot cannot observe a
    half-applied update.  The lock is excluded from snapshots and transport data.
    """

    valence: float = 0.0
    arousal: float = 0.0
    confidence: float = 0.5
    stress: float = 0.0
    label: str = "neutral"
    update_count: int = 0
    last_reason: str = "neutral baseline"
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any) -> None:
        # Direct assignment remains a valid convenience for callers, but cannot
        # break the field bounds.  Public API methods still reject bad input.
        if name in BOUNDS:
            value = _bounded(name, value)
        elif name == "update_count":
            if isinstance(value, bool):
                value = 0
            try:
                value = max(0, int(value))
            except (TypeError, ValueError, OverflowError):
                value = 0
        elif name == "label":
            value = str(value or "neutral")[:64]
        elif name == "last_reason":
            value = str(value or "neutral baseline")[:160]
        lock = self.__dict__.get("_lock")
        if lock is not None and name != "_lock":
            with lock:
                object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        for name in BOUNDS:
            setattr(self, name, _bounded(name, getattr(self, name)))
        self.label = str(self.label or "neutral")[:64]
        try:
            self.update_count = max(0, int(self.update_count or 0))
        except (TypeError, ValueError, OverflowError):
            self.update_count = 0
        self.last_reason = str(self.last_reason or "neutral baseline")[:160]

    @property
    def simulated(self) -> bool:
        return True

    @property
    def algorithmic(self) -> bool:
        return True

    def _classify(self) -> str:
        """Classify current dimensions without claiming a biological state."""
        if self.stress >= 0.45:
            return "stressed"
        if self.valence <= -0.05:
            return "negative"
        if self.valence >= 0.05:
            return "positive"
        if self.confidence <= 0.45:
            return "uncertain"
        return "neutral"

    def _snapshot_locked(self) -> AffectSnapshot:
        return AffectSnapshot({
            "valence": _rounded(self.valence),
            "arousal": _rounded(self.arousal),
            "confidence": _rounded(self.confidence),
            "stress": _rounded(self.stress),
            "label": self.label,
            "update_count": int(self.update_count),
            "last_reason": self.last_reason,
            "simulated": True,
            "algorithmic": True,
            "disclosure": AFFECT_DISCLOSURE,
        })

    def snapshot(self) -> AffectSnapshot:
        """Capture an immutable point-in-time view of the state."""
        with self._lock:
            return self._snapshot_locked()

    def to_dict(self) -> Dict[str, Any]:
        """Return a mutable JSON-ready copy for HTTP/CLI adapters."""
        return dict(self.snapshot())

    as_dict = to_dict

    def _decay_locked(self, strength: float = DEFAULT_DECAY) -> None:
        for name, baseline in NEUTRAL_BASELINE.items():
            current = getattr(self, name)
            object.__setattr__(self, name, current + (baseline - current) * strength)

    def decay(self, strength: float = DEFAULT_DECAY) -> AffectSnapshot:
        """Move one bounded step toward the neutral baseline."""
        rate = _unit_number(strength, "decay strength")
        with self._lock:
            self._decay_locked(rate)
            object.__setattr__(self, "update_count", int(self.update_count) + 1)
            object.__setattr__(self, "last_reason", f"neutral baseline decay ({rate:g})")
            object.__setattr__(self, "label", self._classify())
            return self._snapshot_locked()

    decay_to_neutral = decay

    @staticmethod
    def _answer_value(answer: Any, name: str, default: Any = "") -> Any:
        if isinstance(answer, Mapping):
            return answer.get(name, default)
        return getattr(answer, name, default)

    @staticmethod
    def _citation_count(value: Any) -> int:
        if isinstance(value, bool) or value is None:
            return 0
        try:
            count = int(value)
        except (TypeError, ValueError, OverflowError):
            return 0
        return max(0, min(3, count))

    def update_from_answer(
        self,
        answer: Any,
        route: Optional[str] = None,
        citation_errors: Optional[int] = None,
        citation_error: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> AffectSnapshot:
        """Apply one fixed rule using only observable answer fields.

        ``answer`` may be an :class:`core.answer.Answer` or a mapping.  No text
        sentiment, user-content interpretation, or memory lookup is performed.
        """
        observed_route = route or self._answer_value(
            answer, "agent_mode", self._answer_value(answer, "route", "")
        )
        observed_route = str(observed_route or "").strip().lower()
        observed_mode = mode if mode is not None else self._answer_value(answer, "mode", "")
        mode = str(observed_mode or "").strip().lower()
        raw_text = self._answer_value(answer, "text", None)
        text = ("observable answer" if raw_text is None and mode in {"llm", "extractive"}
                else str(raw_text or ""))
        metrics = self._answer_value(answer, "metrics", {}) or {}
        if not isinstance(metrics, Mapping):
            metrics = {}
        if citation_errors is None:
            citation_errors = citation_error
        if citation_errors is not None:
            citation_count = self._citation_count(citation_errors)
        else:
            citation_count = self._citation_count(
                metrics.get("citation_error", self._answer_value(answer, "citation_error", 0))
            )

        # Grounding/reward is based on retained excerpts, not merely on a
        # caller-supplied route label.  Keep a narrow compatibility path for
        # legacy hand-built Answer objects that predate selected-evidence
        # metrics; all pipeline-produced answers take the strict branch.
        selected = None
        selection_declared = (
            isinstance(answer, Mapping) and "selected_evidence_ids" in answer
        )
        if hasattr(answer, "selected_evidence_ids"):
            selected = list(getattr(answer, "selected_evidence_ids", None) or [])
        elif selection_declared:
            selected = list(answer.get("selected_evidence_ids") or [])
        if selected is not None:
            memories = self._answer_value(answer, "memories", []) or []
            available = {
                str(memory.get("id")) for memory in memories
                if isinstance(memory, Mapping) and memory.get("id")
            }
            if available:
                selected = [node_id for node_id in selected if node_id in available]
        if "n_selected_evidence" in metrics:
            try:
                retained = bool(selected) and int(
                    metrics.get("n_selected_evidence") or 0) > 0
            except (TypeError, ValueError, OverflowError):
                retained = False
        elif selected:
            retained = True
        elif selection_declared:
            retained = False
        else:
            try:
                retained = int(metrics.get("context_tokens", 0) or 0) > 0
            except (TypeError, ValueError, OverflowError):
                retained = False

        # No evidence is a separate observable mode, but a parametric fallback
        # can legitimately be both "parametric" and "no_evidence".  An explicit
        # identity route remains identity even if a malformed answer is empty.
        if observed_route == "identity":
            kind = "identity"
        elif mode == "no_evidence" or not text.strip() or not retained:
            kind = ("parametric_no_evidence" if observed_route == "parametric"
                    else "no_evidence")
        elif observed_route in {"memory", "web"} and mode in {"llm", "extractive"}:
            kind = observed_route
        elif observed_route == "parametric":
            kind = "parametric"
        else:
            kind = "no_evidence"

        with self._lock:
            # Identity is deterministic project metadata, not evidence.  The
            # documented behaviour is a neutral reset with no route reward;
            # applying only the generic decay would make a previously active
            # workspace appear to have received an identity signal.
            if kind == "identity":
                for name, baseline in NEUTRAL_BASELINE.items():
                    object.__setattr__(self, name, baseline)
                deltas = {
                    "valence": 0.0,
                    "arousal": 0.0,
                    "confidence": 0.0,
                    "stress": 0.0,
                }
            else:
                self._decay_locked(DEFAULT_DECAY)
                deltas = {
                    "valence": 0.0,
                    "arousal": 0.0,
                    "confidence": 0.0,
                    "stress": 0.0,
                }
            if kind in {"memory", "web"}:
                deltas.update(valence=0.06, arousal=0.02, confidence=0.08, stress=-0.04)
                if citation_count:
                    # Repeated bad markers have a capped, visible effect.
                    deltas.update(
                        valence=-0.04 * citation_count,
                        confidence=-0.06 * citation_count,
                        stress=0.05 * citation_count,
                    )
                reason = f"grounded {kind} answer"
                if citation_count:
                    reason += f"; citation_errors={citation_count}"
            elif kind in {"parametric", "parametric_no_evidence"}:
                deltas.update(arousal=0.01, confidence=-0.06, stress=0.03)
                reason = "parametric answer; ungrounded"
                if kind == "parametric_no_evidence":
                    deltas.update(arousal=0.01, confidence=-0.02, stress=0.02)
                    reason += "; no-evidence"
            elif kind == "identity":
                # Built-in identity is deterministic, but is not mandala/web
                # evidence and therefore receives no grounded positive delta.
                reason = "deterministic identity route; not evidence-grounded"
            else:
                deltas.update(arousal=0.02, confidence=-0.08, stress=0.05)
                reason = "no-evidence answer; confidence reduced"

            for name, delta in deltas.items():
                object.__setattr__(self, name, _bounded(name, getattr(self, name) + delta))
            object.__setattr__(self, "update_count", int(self.update_count) + 1)
            object.__setattr__(self, "last_reason", reason[:160])
            object.__setattr__(self, "label", self._classify())
            return self._snapshot_locked()

    def update_from_signals(
        self,
        route: str = "",
        mode: str = "extractive",
        citation_errors: int = 0,
        **kwargs: Any,
    ) -> AffectSnapshot:
        """Small adapter for callers that have signals but not an Answer object."""
        if "agent_mode" in kwargs and not route:
            route = kwargs.pop("agent_mode")
        if "grounded" in kwargs:
            grounded = kwargs.pop("grounded")
            if not isinstance(grounded, bool):
                raise TypeError("grounded must be boolean")
            if grounded and not route:
                route = "memory"
            elif not grounded and route in {"memory", "web"}:
                route = "no_evidence"
        if "citation_error" in kwargs:
            citation_errors = kwargs.pop("citation_error")
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unknown affect signal(s): {unknown}")
        return self.update_from_answer(
            {"agent_mode": route, "mode": mode, "text": "observable answer",
             "metrics": {"citation_error": citation_errors}},
            route=route,
            citation_errors=citation_errors,
        )

    # Friendly aliases for callers that use the shorter event vocabulary.
    observe = update_from_answer

    def update(
        self,
        route: Any = "",
        mode: Optional[str] = None,
        citation_errors: int = 0,
        **kwargs: Any,
    ) -> AffectSnapshot:
        """Accept either route signals or an Answer-like object."""
        answer = kwargs.pop("answer", None)
        if answer is not None or not isinstance(route, str):
            return self.update_from_answer(
                answer if answer is not None else route,
                route=route if isinstance(route, str) else None,
                mode=mode,
                citation_errors=citation_errors,
            )
        return self.update_from_signals(
            route=route, mode=mode or "extractive", citation_errors=citation_errors, **kwargs
        )

    def apply_feedback(
        self,
        valence: float = 0.0,
        arousal: float = 0.0,
        confidence: float = 0.0,
        stress: float = 0.0,
        label: Optional[str] = None,
        reason: str = "explicit user feedback",
    ) -> AffectSnapshot:
        """Apply bounded explicit deltas, never inferred from answer wording."""
        deltas = {
            "valence": _feedback_number(valence, "valence"),
            "arousal": _feedback_number(arousal, "arousal"),
            "confidence": _feedback_number(confidence, "confidence"),
            "stress": _feedback_number(stress, "stress"),
        }
        if label is not None:
            if not isinstance(label, str):
                raise ValueError("label must be a string")
            label = label.strip()
            if not label or len(label) > 64:
                raise ValueError("label must be 1 to 64 characters")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 160:
            raise ValueError("reason must be 1 to 160 characters")
        with self._lock:
            self._decay_locked(DEFAULT_DECAY)
            for name, delta in deltas.items():
                object.__setattr__(self, name, _bounded(name, getattr(self, name) + delta))
            object.__setattr__(self, "update_count", int(self.update_count) + 1)
            object.__setattr__(self, "last_reason", reason.strip()[:160])
            object.__setattr__(self, "label", label or self._classify())
            return self._snapshot_locked()

    def feedback(self, **kwargs: Any) -> AffectSnapshot:
        """Alias for :meth:`apply_feedback`."""
        return self.apply_feedback(**kwargs)


__all__ = [
    "AFFECT_DISCLOSURE",
    "AffectSnapshot",
    "AffectiveState",
    "BOUNDS",
    "DEFAULT_DECAY",
    "NEUTRAL_BASELINE",
    "coerce_snapshot",
    "neutral_snapshot",
]
