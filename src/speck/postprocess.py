"""Transcript post-processing: vocabulary correction and multi-engine (ROVER-style) voting."""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Sequence

from speck.tts_metrics import _levenshtein_align

_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")


def correct_vocabulary(text: str, vocabulary: Sequence[str], cutoff: float = 0.85) -> str:
    """Snap near-miss words to the closest entry of a domain vocabulary ("Kubernetes" for "cooper netties").

    Multi-word entries are matched against windows of the same word count. Words shorter than four
    characters are never rewritten, so common short words are not hijacked by a similar-looking term.
    """
    words = list(_WORD.finditer(text))
    edits: list[tuple[int, int, str]] = []
    claimed: set[int] = set()  # word indices already rewritten; edits never overlap
    for term in sorted(set(vocabulary), key=lambda v: -len(v.split())):
        size = len(term.split())
        target = " ".join(term.lower().split())
        if len(target.replace(" ", "")) < 4:
            continue
        for i in range(len(words) - size + 1):
            span = range(i, i + size)
            if claimed.intersection(span):
                continue
            seen = " ".join(words[j].group().lower() for j in span)
            if seen != target and SequenceMatcher(None, seen, target).ratio() >= cutoff:
                edits.append((words[i].start(), words[i + size - 1].end(), term))
                claimed.update(span)
    for start, end, term in sorted(edits, reverse=True):
        text = text[:start] + term + text[end:]
    return text


def _key(word: str) -> str:
    return re.sub(r"[\W_]+", "", word.lower())


def rover(hypotheses: Sequence[str]) -> str:
    """Combine transcripts from several engines by per-word majority vote (the first one is the pivot).

    Words are compared ignoring case and punctuation; ties go to the pivot, and a word the pivot lacks is
    only added when a strict majority of the systems produced it.
    """
    systems = [h.split() for h in hypotheses if h.strip()]
    if not systems:
        return ""
    pivot, n = systems[0], len(systems)
    slots: list[Counter] = [Counter({_key(w): 1}) for w in pivot]
    surface: list[dict[str, str]] = [{_key(w): w} for w in pivot]
    gaps: list[Counter] = [Counter() for _ in range(len(pivot) + 1)]
    gap_surface: list[dict[str, str]] = [{} for _ in range(len(pivot) + 1)]

    for hyp in systems[1:]:
        _, _, _, _, ops = _levenshtein_align([_key(w) for w in pivot], [_key(w) for w in hyp])
        p = q = 0  # positions in pivot / hyp
        for op, _, _ in ops:
            if op == "insert":
                gaps[p][_key(hyp[q])] += 1
                gap_surface[p].setdefault(_key(hyp[q]), hyp[q])
                q += 1
            elif op == "delete":
                slots[p][""] += 1
                p += 1
            else:
                slots[p][_key(hyp[q])] += 1
                surface[p].setdefault(_key(hyp[q]), hyp[q])
                p += 1
                q += 1

    out: list[str] = []
    for p in range(len(pivot) + 1):
        out += [gap_surface[p][k] for k, c in gaps[p].items() if c * 2 > n]
        if p < len(pivot):
            best = slots[p].most_common(1)[0][0]  # ties resolve to the pivot (inserted first)
            if best:
                out.append(surface[p][best])
    return " ".join(out)
