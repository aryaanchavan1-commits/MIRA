"""Indic methodological priors from the granthas — implemented as code, scoped honestly.

Every mapping here is an engineering analogy with formal, checkable structure,
NOT a claim that the source texts describe AI memory systems, and NOT a claim
that the analogy improves anything by itself. Each item is either a derived
label (no retrieval effect) or subject to the same ablation discipline as the
eleven scoring terms. Taxonomy per docs (§3): these are [ENG] mechanisms with
[HYP]-labeled value claims, if any.

Implemented mappings (each traceable to a primary text):

1. PRAMĀṆA PROVENANCE (Nyāya Sūtras, epistemology — the four independent
   means of knowledge). MIRA's version-chain records gain a derived
   ``pramana`` label classifying *how* a fact came to be known:
     pratyakṣa — direct perception: first-party input (user chat, SDK write)
     anumāna   — inference: derived by the system (consolidation, merge, gist)
     upamāna   — comparison/analogy: similarity-derived, agent-relayed
     śabda     — reliable testimony: ingested documents, web sources, citations
   Nyāya treats these as *independent* knowledge sources; the label therefore
   also marks which records could contradict each other (two śabda sources
   conflict vs. śabda-vs-pratyakṣa) — groundwork for provenance-aware
   contradiction scoring (not yet implemented; see EXPERIMENT_PROTOCOL).

2. KAṬAPAYĀDI ORDINAL ENCODING (medieval Kerala mathematics; used to encode
   numerals into memorable verse — e.g. Melakarta raga naming, candra-vṛttas).
   A text→digit mapping with a fixed table and a reversal convention: the
   numeral is read from the RIGHT (digit-reversed). Implemented here as a
   deterministic, human-readable secondary key for version chains: every
   returned record carries ``katapayadi`` digits derived from the subject at
   read time — no storage cost, and the encoding is a checksum-like reminder
   of the subject. Letters with no table entry are skipped; vowels are 0.

3. ŚRUTI/SMṚTI PRESERVATION SEMANTICS (Mīmāṃsā tradition's distinction between
   that-which-is-heard — fixed, recitation-locked, corruption-intolerant — and
   that-which-is-remembered — preserved by transmission and correction).
   This names an EXISTING property, it does not add one: ``VersionChain`` is
   śruti-shaped (append-only, hash-linked, never rewritten; rollback appends
   a new head and cites the superseded hash) while the working store is
   smṛti-shaped (decay, replay, consolidation can revise it). Documented here
   so the analogy is testable: the śruti property is the append-only unit
   test; the smṛti properties are the consolidation tests.

Nothing in this module affects retrieval scores or the default-off guarantee.
"""
from __future__ import annotations

from typing import Dict, List, Optional

# -- Kaṭapayādi table (standard consonant assignment, romanized ITRANS-ish) --
# Varga rows: ka-kha-ga-gha-ṅa=1-5, ca-cha-ja-jha-ña=6-0, ṭa-varga=1-5,
# ta-varga=6-0, pa-varga=1-5, then ya..ha=1-8 and kṣa=0.
# Case is the retroflex marker (ITRANS): T/D/N = ṭa/ḍa/ṇa-varga, lowercase =
# dental. Aspirates (kh, gh, Th, Dh, th, dh, ph, bh, ch, chh, jh, sh) are
# matched as multi-char tokens before single letters.
_KATAPAYADI: Dict[str, str] = {
    # ka-varga: ka=1 kha=2(pair) ga=3 gha=4(pair) ṅa=5
    "k": "1", "g": "3", "G": "5",
    # ca-varga: ca=6 cha=7(pair) ja=8 jha=9(pair) ña=0(pair)
    "c": "6", "j": "8",
    # ṭa-varga (retroflex, uppercase): ṭa=1 ṭha=2(pair) ḍa=3 ḍha=4(pair) ṇa=5
    "T": "1", "D": "3", "N": "5",
    # ta-varga (dental): ta=6 tha=7(pair) da=8 dha=9(pair) na=0
    "t": "6", "d": "8", "n": "0",
    # pa-varga: pa=1 pha=2(pair) ba=3 bha=4(pair) ma=5
    "p": "1", "b": "3", "m": "5",
    # semivowels + sibilants: ya=1 ra=2 la=3 va=4 śa=5 ṣa=6 sa=7 ha=8 kṣa=0
    "y": "1", "r": "2", "l": "3", "v": "4",
    "z": "5", "S": "6", "s": "7", "h": "8", "x": "0",
}

# Multi-char romanizations (checked longest-first): aspirates and ña.
_KATAPAYADI_MULTI: Dict[str, str] = {
    "chh": "7", "jh": "9", "kh": "2", "gh": "4", "Th": "2", "Dh": "4",
    "th": "7", "dh": "9", "ph": "2", "bh": "4", "ch": "7", "sh": "5",
    "~n": "0", "~N": "5",     # ña and ṅa in strict ITRANS
}


def katapayadi(subject: str) -> str:
    """Kaṭapayādi digits for a subject string (romanized input expected).

    Tokens are matched longest-first (``chh`` before ``ch`` before ``c``);
    vowels, spaces and unknown characters contribute 0 — matching the
    tradition, where vowel-initial syllables stand for zero. Returns the
    digit string in textual order; the classical numeral reads it reversed.
    """
    out: List[str] = []
    i = 0
    s = subject.strip()
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        matched = False
        for n in (3, 2):
            tok = s[i:i + n]
            if len(tok) == n and tok in _KATAPAYADI_MULTI:
                out.append(_KATAPAYADI_MULTI[tok])
                i += n
                matched = True
                break
        if matched:
            continue
        ch = s[i]
        out.append(_KATAPAYADI.get(ch) or _KATAPAYADI.get(ch.lower(), "0"))
        i += 1
    return "".join(out) if out else "0"


# -- Pramāṇa provenance classes (Nyāya) ---------------------------------------
PRAMANA_LABELS = ("pratyaksa", "anumana", "upamana", "shabda")

_PRAMANA_RULES = (  # first match wins; order = most specific first
    (("consolidat", "replay", "gist", "merge", "derived", "anuma", "rollback"),
     "anumana"),
    (("similar", "analogy", "analog", "compare", "upama"), "upamana"),
    (("web", "doc", "ingest", "url", "citation", "cited", "file", "pdf"),
     "shabda"),
    (("chat", "user", "sdk", "direct", "manual"), "pratyaksa"),
)


def pramana_of(source: str) -> str:
    """Classify a version-chain source string into a Nyāya pramāṇa label.

    Defaults to śabda (testimony) — the honest epistemic default: an
    unclassified source is *someone else's claim*, not direct perception.
    """
    s = (source or "").lower()
    for needles, label in _PRAMANA_RULES:
        if any(n in s for n in needles):
            return label
    return "shabda"


def annotate_record(record: Dict, subject: Optional[str] = None) -> Dict:
    """Return a copy of a VersionChain record with derived Indic labels.

    Derived at read time from stored fields (source/subject): nothing extra
    is persisted, so old databases annotate identically to new ones.
    """
    out = dict(record)
    out["pramana"] = pramana_of(out.get("source", ""))
    if subject is not None:
        out["katapayadi"] = katapayadi(subject)
    return out
