"""Bounded, trace-driven edge plasticity.

MIRA already has a slow path-based reinforcement pass.  This module also
accepts the deterministic voltage/spike trace emitted by the LIF-inspired
activation path and applies a small STDP-like timing rule to graph edges.
It is an engineering heuristic on symbolic edge weights, not a biological
mechanism or a model of synaptic biochemistry.

The legacy ``reinforce`` API remains intact for callers that only have
retrieval paths.  Both paths clamp weights, apply bounded rates, and persist
*every* changed edge, including homeostatic decays.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("mira.hebbian")

W_MIN, W_MAX = 0.05, 2.0
_MAX_WINDOW_TICKS = 32
_MAX_TRACE_EVENTS = 4096
_DEFAULT_LTP = 0.05
_DEFAULT_LTD = 0.02


def _finite(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _rate(value: Any, default: float) -> float:
    return min(1.0, max(0.0, _finite(value, default)))


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off", "none"}
    return bool(value)


def _canonical(source: Any, target: Any) -> Tuple[str, str]:
    a, b = str(source), str(target)
    return (a, b) if a <= b else (b, a)


def _weight(value: Any) -> float:
    return min(W_MAX, max(W_MIN, _finite(value, 1.0)))


def _confidence_factor(edge) -> float:
    confidence = _finite(getattr(edge, "confidence", 0.5), 0.5)
    return 1.0 + 0.5 * min(1.0, max(0.0, confidence))


def _set_weight(edge, value: float) -> Tuple[float, float, bool]:
    raw = getattr(edge, "weight", 1.0)
    old = _weight(raw)
    new = round(min(W_MAX, max(W_MIN, value)), 4)
    if old != raw:
        edge.weight = old  # repair an invalid live value before updating it
    if new != old:
        edge.weight = new
    return old, new, old != raw or new != old


def _set_pair_weight(frame, key: Tuple[str, str], value: float):
    first = next((edge for edge in frame.edges
                  if _canonical(edge.source_id, edge.target_id) == key), None)
    if first is None:
        return None
    old, new, changed = _set_weight(first, value)
    if changed:
        # Parallel relation records represent one undirected graph edge; keep
        # the live frame coherent even though SQLite has separate rows.
        for edge in frame.edges:
            if _canonical(edge.source_id, edge.target_id) == key:
                edge.weight = new
    return old, new, changed


def _set_group_weight(edges: List[Any], value: float):
    old, new, changed = _set_weight(edges[0], value)
    if changed:
        for edge in edges:
            edge.weight = new
    return old, new, changed


def _clamp_group(edges: List[Any]):
    target = _weight(edges[0].weight)
    if all(_weight(getattr(edge, "weight", 1.0)) == getattr(edge, "weight", 1.0)
           for edge in edges):
        return None
    for edge in edges:
        edge.weight = target
    edge = edges[0]
    return str(edge.source_id), str(edge.target_id), target


def _persist(changes: Dict[Tuple[str, str], Tuple[str, str, float]], store) -> None:
    if store is None:
        return
    for key in sorted(changes):
        source, target, weight = changes[key]
        try:
            store.update_edge_weight(source, target, weight)
        except Exception as exc:
            # Keep attempting the remaining edges: one bad row must not make
            # the rest of a consolidation pass disappear.
            logger.warning("edge weight persist failed for %s->%s: %s",
                           source, target, exc)


def _trace_events(trace: Optional[List[Dict[str, Any]]], max_events: int,
                  spike_only: bool) -> List[Tuple[float, int, str]]:
    """Return ``(logical_time, trace_order, node)`` activity events."""
    if trace is None or max_events <= 0:
        return []
    events: List[Tuple[float, int, str]] = []
    for order, event in enumerate(trace):
        if order >= max_events or len(events) >= max_events:
            break
        if not isinstance(event, dict) or event.get("node") is None:
            continue
        # A real LIF trace explicitly marks sub-threshold updates.  Do not
        # count those as a spike unless the caller explicitly disables the
        # spike filter.  If a small hand-written trace omits the flag, it is
        # still treated as an activity event for API friendliness.
        if spike_only and "spike" in event and not _truthy(event.get("spike")):
            continue
        raw_time = event.get("tick", event.get("time", order))
        when = _finite(raw_time, float(order))
        events.append((when, order, str(event.get("node"))))
    return events


def _first_pair(first: List[Tuple[float, int, str]],
                second: List[Tuple[float, int, str]],
                window: int) -> Optional[Tuple[Tuple[float, int], Tuple[float, int]]]:
    """Find the earliest first→second pair inside a bounded tick window."""
    # ponytail: this is a small quadratic scan; a time-indexed scan is the
    # upgrade if profiling shows that the capped trace is still too costly.
    best = None
    for before in first:
        for after in second:
            delta = after[0] - before[0]
            if delta < 0.0 or delta > float(window):
                continue
            # Same-tick events still need a deterministic trace order; merely
            # sharing a tick is not evidence of pre-before-post activity.
            if delta == 0.0 and after[1] <= before[1]:
                continue
            key = (before[:2], after[:2])
            if best is None or key < best[0]:
                best = (key, (before, after))
    return best[1] if best is not None else None


def stdp_update(frame, trace: Optional[List[Dict[str, Any]]] = None,
                lr: float = _DEFAULT_LTP, depression: float = _DEFAULT_LTD,
                window: int = 1, decay: float = 0.0, store=None, *,
                window_ticks: Optional[int] = None,
                potentiation: Optional[float] = None,
                depression_rate: Optional[float] = None,
                max_events: int = _MAX_TRACE_EVENTS,
                spike_only: bool = True,
                config: Optional[Dict[str, Any]] = None,
                activation_trace: Optional[List[Dict[str, Any]]] = None,
                max_trace_events: Optional[int] = None
                ) -> List[Tuple[str, str, float]]:
    """Apply a bounded STDP-like update from a LIF activation trace.

    For a directed frame edge ``source -> target``, a source event followed by
    a target event within ``window`` ticks can potentiate it.  The reverse
    order can depress it.  A trace with no in-window pair has no effect;
    ``decay`` is a separate, explicitly configured homeostatic pass and is
    never run for a trace with no timing pair.  Rates, the window, trace size,
    and weights are all bounded.  ``store`` receives every changed weight.
    """
    if activation_trace is not None:
        trace = activation_trace
    if max_trace_events is not None:
        max_events = max_trace_events
    # Also tolerate the natural ``stdp_update(frame, trace, store)`` spelling
    # without changing the explicit learning-rate-first API.
    if store is None and lr is not None and not isinstance(lr, (int, float)):
        store, lr = lr, _DEFAULT_LTP
    if isinstance(config, dict) and config:
        section = config.get("memory", {}) or {}
        if not isinstance(section, dict):
            section = {}
        section = section.get(
            "stdp", config.get("stdp", section.get("trace_plasticity", {})))
        if section is None:
            section = {}
        if isinstance(section, (bool, int, float)):
            if not section:
                return []
            section = {}
        if isinstance(section, dict):
            if "enabled" in section and not _truthy(section.get("enabled")):
                return []
            if window_ticks is None and "window_ticks" in section:
                window_ticks = section.get("window_ticks")
            if potentiation is None:
                potentiation = section.get(
                    "potentiation", section.get("potentiate",
                                  section.get("ltp", section.get("lr"))))
            if depression_rate is None:
                depression_rate = section.get(
                    "depression_rate", section.get("depress",
                                      section.get("ltd")))
            if max_events == _MAX_TRACE_EVENTS:
                if "max_events" in section:
                    max_events = section.get("max_events")
                elif "max_trace_events" in section:
                    max_events = section.get("max_trace_events")
            if "spike_only" in section:
                spike_only = _truthy(section.get("spike_only"))
            elif "use_spikes" in section:
                spike_only = _truthy(section.get("use_spikes"))
            if "decay" in section:
                decay = section.get("decay")
            elif "homeostatic_decay" in section:
                decay = section.get("homeostatic_decay")
            if "depression" in section and depression_rate is None:
                depression = section.get("depression")

    if window_ticks is not None:
        window = window_ticks
    if potentiation is not None:
        lr = potentiation
    if depression_rate is not None:
        depression = depression_rate

    try:
        window = int(_finite(window, 1.0))
    except (TypeError, ValueError, OverflowError):
        window = 1
    window = max(0, min(_MAX_WINDOW_TICKS, window))
    lr = _rate(lr, _DEFAULT_LTP)
    depression = _rate(depression, _DEFAULT_LTD)
    decay = _rate(decay, 0.0)
    try:
        max_events = int(_finite(max_events, _MAX_TRACE_EVENTS))
    except (TypeError, ValueError, OverflowError):
        max_events = _MAX_TRACE_EVENTS
    max_events = max(0, min(_MAX_TRACE_EVENTS, max_events))

    edge_groups: Dict[Tuple[str, str], List[Any]] = {}
    changes: Dict[Tuple[str, str], Tuple[str, str, float]] = {}
    for edge in frame.edges:
        key = _canonical(edge.source_id, edge.target_id)
        edge_groups.setdefault(key, []).append(edge)
    for key, group in edge_groups.items():
        repair = _clamp_group(group)
        if repair is not None:
            changes[key] = repair

    events = _trace_events(trace, max_events, _truthy(spike_only))
    if not events:
        _persist(changes, store)
        return [changes[key] for key in sorted(changes)]

    by_node: Dict[str, List[Tuple[float, int, str]]] = {}
    for event in events:
        by_node.setdefault(event[2], []).append(event)
    active_nodes = set(by_node)
    touched: set = set()

    for key, group in edge_groups.items():
        edge = group[0]
        pre_events = by_node.get(str(edge.source_id), [])
        post_events = by_node.get(str(edge.target_id), [])
        ltp = _first_pair(pre_events, post_events, window)
        ltd = _first_pair(post_events, pre_events, window)
        if ltp is None and ltd is None:
            continue

        touched.add(key)
        if ltp is not None and (ltd is None or ltp[0][:2] <= ltd[0][:2]):
            update = _set_group_weight(
                group, _weight(edge.weight) + lr * _confidence_factor(edge))
        else:
            update = _set_group_weight(
                group, _weight(edge.weight) - depression * _confidence_factor(edge))
        old, new, changed = update
        if changed:
            changes[key] = (str(edge.source_id), str(edge.target_id), new)

    # Homeostatic decay is deliberately limited to edges with no activity at
    # all.  Thus an out-of-window pair remains exactly unchanged, while an
    # unrelated inactive edge can still decay when another pair triggers a
    # consolidation pass.
    if touched and decay > 0.0:
        d = decay / math.sqrt(len(touched))
        for key, group in edge_groups.items():
            if key in touched:
                continue
            edge = group[0]
            if any(str(candidate.source_id) in active_nodes
                   or str(candidate.target_id) in active_nodes
                   for candidate in group):
                continue
            old, new, changed = _set_group_weight(
                group, _weight(edge.weight) * (1.0 - d))
            if changed:
                changes[key] = (str(edge.source_id), str(edge.target_id), new)

    _persist(changes, store)
    logger.info("STDP-like trace update: %d changed edges, %d timing pairs",
                len(changes), len(touched))
    return [changes[key] for key in sorted(changes)]


# Keep the descriptive spelling available without duplicating the algorithm.
apply_stdp = stdp_update


def reinforce(frame, paths: List[List[str]], lr: float = 0.10,
              decay: float = 0.02, store=None, *,
              activation_trace: Optional[List[Dict[str, Any]]] = None,
              depression: float = _DEFAULT_LTD,
              window: int = 1,
              trace: Optional[List[Dict[str, Any]]] = None
              ) -> List[Tuple[str, str, float]]:
    """Strengthen edges along used paths and decay untouched edges.

    This is the compatibility path for callers that do not have an activation
    trace.  ``activation_trace`` is an optional bridge for older callers that
    want the trace-driven rule through the existing entry point.  It retains
    the original potentiation formula for path-only calls while fixing
    persistence so decayed edges are written too.
    """
    if activation_trace is None:
        activation_trace = trace
    if activation_trace is not None:
        return stdp_update(frame, activation_trace, lr=lr,
                           depression=depression, window=window,
                           decay=decay, store=store)
    if not paths:
        return []
    lr = _rate(lr, 0.10)
    decay = _rate(decay, 0.02)
    touched: set = set()
    changes: Dict[Tuple[str, str], Tuple[str, str, float]] = {}
    reported: Dict[Tuple[str, str], Tuple[str, str, float]] = {}
    edge_groups: Dict[Tuple[str, str], List[Any]] = {}
    for edge in frame.edges:
        key = _canonical(edge.source_id, edge.target_id)
        edge_groups.setdefault(key, []).append(edge)
    for key, group in edge_groups.items():
        repair = _clamp_group(group)
        if repair is not None:
            changes[key] = repair

    for path in paths:
        for a, b in zip(path, path[1:]):
            edge = frame.get_edge(a, b)
            if edge is None:
                continue
            key = _canonical(a, b)
            if key in touched:
                continue
            touched.add(key)
            update = _set_pair_weight(
                frame, key, _weight(edge.weight) + lr * _confidence_factor(edge))
            if update is None:
                continue
            old, new, changed = update
            reported[key] = (str(a), str(b), edge.weight)
            if changed:
                changes[key] = (str(a), str(b), new)

    if touched and frame.edges:
        d = decay / math.sqrt(len(touched))
        for edge in frame.edges:
            key = _canonical(edge.source_id, edge.target_id)
            if key in touched:
                continue
            update = _set_pair_weight(
                frame, key, _weight(edge.weight) * (1.0 - d))
            if update is not None:
                old, new, changed = update
                if changed:
                    changes[key] = (str(edge.source_id), str(edge.target_id), new)

    _persist(changes, store)
    logger.info("hebbian consolidation: %d edges strengthened along %d paths",
                len(touched), len(paths))
    return list(reported.values())
