"""Transcript accuracy metrics: locale-aware normalization (English, Italian) and WER / CER / MER / WIL."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

# ==============================================================================
# TEXT NORMALIZATION & LINGUISTIC PROCESSING (ENGLISH & ITALIAN)
# ==============================================================================

_QUOTES = re.compile(r"[’‘`´]")
_SPACES = re.compile(r"\s+")

# Irregular contractions; every regular one ("don't", "we're", "i'd", "he's"...) is covered by _EN_CLITICS
_EN_IRREGULAR = [(re.compile(rf"\b{re.escape(c)}\b"), e) for c, e in (
    ("won't", "will not"), ("can't", "cannot"), ("shan't", "shall not"), ("let's", "let us"),
)]
_EN_CLITICS = [(re.compile(rf"{c}\b"), e) for c, e in (
    ("n't", " not"), ("'re", " are"), ("'ve", " have"), ("'ll", " will"), ("'d", " would"), ("'m", " am"), ("'s", " is"),
)]
_EN_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")  # 1,000 -> 1000
_EN_NON_ALNUM = re.compile(r"[^a-z0-9\s]")

# Italian common elided particles with apostrophe
_IT_ELISIONS_REGEX = re.compile(
    r"\b(l|un|all|dall|dell|nell|sull|coll|pel|d|c|m|t|s|v|ci|vi|ne|quell|bell|grand|sant|poc|mezz|qualch)'([a-zàèéìòùA-Z])",
    re.IGNORECASE,
)
_IT_INNER_APOSTROPHE = re.compile(r"([a-zàèéìòù])'([a-zàèéìòù])")
_IT_TYPOGRAPHY = re.compile(r"[«»“”\"„–—]")
_IT_NON_ALNUM = re.compile(r"[^a-zàèéìòù0-9'\s]")
_IT_NON_ASCII = re.compile(r"[^a-z0-9'\s]")
_IT_STRAY_APOSTROPHES = (re.compile(r"\s+'+\s+"), re.compile(r"^'+|'+$"))
# Italian accented characters mapping for diacritic-insensitive evaluation
_IT_ACCENT_MAP = str.maketrans("àèéìòóù", "aeeioou")

# Hesitation sounds that references never transcribe but ASR models often emit
_EN_FILLERS = frozenset({"uh", "um", "uhm", "umm", "erm", "hmm", "hm", "mm", "mmm", "mhm"})
_IT_FILLERS = frozenset({"ehm", "ehmm", "eh", "uhm", "uh", "mh", "mhm", "hmm", "mmm"})

_NUM_UNITS = {
    w: i for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_NUM_TENS = {w: 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2)}
_NUM_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_NUM_WORDS = frozenset(_NUM_UNITS) | frozenset(_NUM_TENS) | {"hundred"}


def _parse_spoken_number(tokens: list[str], start: int) -> tuple[int, int]:
    """Parse the cardinal starting at `start`; returns (value, tokens consumed)."""
    total = current = 0
    last = None  # kind of the previous word: unit / tens / hundred / scale
    i = start
    while i < len(tokens):
        w = tokens[i]
        if w in _NUM_UNITS:
            n = _NUM_UNITS[w]
            # units attach to a tens word ("thirty five"), teens and zero only start a number
            if last == "unit" or (n == 0 and last is not None) or (n >= 10 and last == "tens"):
                break
            current += n
            last = "unit"
        elif w in _NUM_TENS:
            if last not in (None, "hundred", "scale"):
                break
            current += _NUM_TENS[w]
            last = "tens"
        elif w == "hundred":
            if last != "unit" or current >= 100 or current == 0:
                break
            current *= 100
            last = "hundred"
        elif w in _NUM_SCALES:
            if last not in ("unit", "tens", "hundred"):
                break
            total += current * _NUM_SCALES[w]
            current = 0
            last = "scale"
        elif w == "and" and last in ("hundred", "scale") and i + 1 < len(tokens) and tokens[i + 1] in _NUM_WORDS:
            pass  # "one hundred and five"
        else:
            break
        i += 1
    while i > start and tokens[i - 1] == "and":
        i -= 1
    return total + current, i - start


def spoken_numbers_to_digits(text: str) -> str:
    """Rewrite spelled-out English cardinals as digits ("thirty five" -> "35") in whitespace-split text.

    Applied to reference and hypothesis alike, so "35" vs "thirty five" no longer counts as an error.
    Adjacent numbers that do not compose ("twenty twenty") stay separate numbers.
    """
    tokens = text.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] in _NUM_WORDS:
            value, used = _parse_spoken_number(tokens, i)
            if used:
                out.append(str(value))
                i += used
                continue
        out.append(tokens[i])
        i += 1
    return " ".join(out)


def _drop(text: str, words: frozenset) -> str:
    return " ".join(w for w in text.split() if w not in words)


def normalize_english_text(
    text: str,
    expand_contractions: bool = True,
    strip_fillers: bool = True,
    normalize_numbers: bool = True,
) -> str:
    """
    Standardizes English text for acoustic and linguistic evaluation:
    - Lowercases text
    - Standardizes quotation marks
    - Expands common contractions (e.g. "don't" -> "do not")
    - Strips punctuation and symbols while preserving alphanumeric tokens
    - Drops hesitation fillers ("uh", "um") when `strip_fillers` is set
    - Writes spelled-out numbers as digits ("thirty five" -> "35") when `normalize_numbers` is set
    - Collapses repeated whitespace
    """
    if not text:
        return ""
    t = _EN_THOUSANDS.sub("", _QUOTES.sub("'", text).lower()).replace("%", " percent ")
    if expand_contractions:
        for pattern, expanded in _EN_IRREGULAR + _EN_CLITICS:
            t = pattern.sub(expanded, t)
    t = _SPACES.sub(" ", _EN_NON_ALNUM.sub(" ", t)).strip()
    if strip_fillers:
        t = _drop(t, _EN_FILLERS)
    return spoken_numbers_to_digits(t) if normalize_numbers else t


def normalize_italian_text(
    text: str,
    preserve_accents: bool = True,
    split_apostrophes: bool = True,
    strip_fillers: bool = True,
) -> str:
    """
    Standardizes Italian text for speech evaluation according to Italian phonetics & grammar:
    - Standardizes typography (quotes, dashes « », “ ”)
    - Separates apostrophe elisions (e.g. "l'albero" -> "l' albero" or "l albero")
      to prevent false-positive word merges in ASR / TTS benchmarks.
    - Handles standard Italian truncated forms ("qual è" never has an apostrophe)
    - Accents:
        * If `preserve_accents=True` (default): keeps distinct Italian vowels (à, è, é, ì, ò, ù),
          which convey critical phonemic contrasts (e.g. 'e' [and] vs 'è' [is], 'da' vs 'dà').
        * If `preserve_accents=False`: folds accents to plain ASCII vowels.
    - Strips non-alphanumeric punctuation and collapses whitespace.
    """
    if not text:
        return ""
    t = _IT_TYPOGRAPHY.sub(" ", _QUOTES.sub("'", text).lower())
    if split_apostrophes:  # "l'amico" -> "l' amico", then any word-internal one: "d'accordo" -> "d' accordo"
        t = _IT_INNER_APOSTROPHE.sub(r"\1' \2", _IT_ELISIONS_REGEX.sub(r"\1' \2", t))
    t = _IT_NON_ALNUM.sub(" ", t) if preserve_accents else _IT_NON_ASCII.sub(" ", t.translate(_IT_ACCENT_MAP))
    stray_inner, stray_edge = _IT_STRAY_APOSTROPHES
    t = _SPACES.sub(" ", stray_edge.sub("", stray_inner.sub(" ", t))).strip()
    return _drop(t, _IT_FILLERS) if strip_fillers else t


def normalize_text(
    text: str,
    language: str = "en",
    preserve_accents: bool = True,
) -> str:
    """
    Language-aware text normalizer dispatcher.
    Supports "en", "en-US", "en-GB", "it", "it-IT", etc.
    """
    if (language or "en").lower().split("-")[0].split("_")[0] == "it":
        return normalize_italian_text(text, preserve_accents=preserve_accents)
    return normalize_english_text(text)


# ==============================================================================
# LEVENSHTEIN DISTANCE, ALIGNMENTS, WER, CER, MER, WIL
# ==============================================================================

def levenshtein_align(
    ref_seq: Sequence[Any],
    hyp_seq: Sequence[Any],
) -> Tuple[int, int, int, int, List[Tuple[str, Any, Any]]]:
    """
    Computes optimal Levenshtein alignment between reference and hypothesis sequences.
    Returns:
        (substitutions, deletions, insertions, hits, aligned_operations)
    where aligned_operations is a list of tuples: (operation, ref_item, hyp_item)
    operation is one of: "equal", "substitute", "delete", "insert".
    """
    n, m = len(ref_seq), len(hyp_seq)
    # Plain lists: per-element access is several times faster than numpy scalar indexing
    dp = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        row, up, r = dp[i], dp[i - 1], ref_seq[i - 1]
        for j in range(1, m + 1):
            row[j] = up[j - 1] if r == hyp_seq[j - 1] else 1 + min(up[j - 1], up[j], row[j - 1])

    # Backtracking to reconstruct exact alignment path
    i, j = n, m
    ops: List[Tuple[str, Any, Any]] = []
    counts = {"equal": 0, "substitute": 0, "delete": 0, "insert": 0}
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_seq[i - 1] == hyp_seq[j - 1]:
            op, i, j = ("equal", ref_seq[i - 1], hyp_seq[j - 1]), i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            op, i, j = ("substitute", ref_seq[i - 1], hyp_seq[j - 1]), i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            op, i = ("delete", ref_seq[i - 1], None), i - 1
        else:
            op, j = ("insert", None, hyp_seq[j - 1]), j - 1
        ops.append(op)
        counts[op[0]] += 1

    ops.reverse()
    return counts["substitute"], counts["delete"], counts["insert"], counts["equal"], ops


def _edit_rate(reference, hypothesis, language, normalize, preserve_accents, tokenize):
    """(normalized ref, normalized hyp, ref tokens, hyp tokens, error rate, (S, D, I, H, ops))."""
    ref, hyp = (
        normalize_text(t, language=language, preserve_accents=preserve_accents) if normalize else t
        for t in (reference, hypothesis)
    )
    ref_tokens, hyp_tokens = tokenize(ref), tokenize(hyp)
    s, d, i, h, ops = levenshtein_align(ref_tokens, hyp_tokens)
    rate = float((s + d + i) / len(ref_tokens)) if ref_tokens else float(bool(hyp_tokens))
    return ref, hyp, ref_tokens, hyp_tokens, rate, (s, d, i, h, ops)


def compute_wer(
    reference: str,
    hypothesis: str,
    language: str = "en",
    normalize: bool = True,
    preserve_accents: bool = True,
) -> Dict[str, Any]:
    """
    Computes Word Error Rate (WER):
        WER = (Substitutions + Deletions + Insertions) / Reference Word Count
    """
    ref, hyp, ref_words, hyp_words, wer, (s, d, i, h, ops) = _edit_rate(
        reference, hypothesis, language, normalize, preserve_accents, str.split
    )
    return {
        "wer": wer,
        "substitutions": s,
        "deletions": d,
        "insertions": i,
        "hits": h,
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
        "alignment": ops,
        "normalized_ref": ref,
        "normalized_hyp": hyp,
    }


def compute_cer(
    reference: str,
    hypothesis: str,
    language: str = "en",
    normalize: bool = True,
    preserve_accents: bool = True,
) -> Dict[str, Any]:
    """
    Computes Character Error Rate (CER):
        CER = (Char Substitutions + Deletions + Insertions) / Reference Character Count
    """
    _, _, ref_chars, hyp_chars, cer, (s, d, i, h, _) = _edit_rate(
        reference, hypothesis, language, normalize, preserve_accents, lambda t: list(t.replace(" ", ""))
    )
    return {
        "cer": cer,
        "substitutions": s,
        "deletions": d,
        "insertions": i,
        "hits": h,
        "ref_chars": len(ref_chars),
        "hyp_chars": len(hyp_chars),
    }


def _information_rates(wer_res: Dict[str, Any]) -> Dict[str, float]:
    errors = wer_res["substitutions"] + wer_res["deletions"] + wer_res["insertions"]
    total_ops = errors + wer_res["hits"]
    n_ref, n_hyp = wer_res["ref_words"], wer_res["hyp_words"]
    if n_ref == 0 or n_hyp == 0:
        wip = 1.0 if n_ref == n_hyp else 0.0
    else:
        wip = min(1.0, float(wer_res["hits"] ** 2 / (n_ref * n_hyp)))
    return {"mer": float(errors / total_ops) if total_ops > 0 else 0.0, "wil": float(1.0 - wip), "wip": wip}


def compute_mer_and_wil(
    reference: str,
    hypothesis: str,
    language: str = "en",
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Computes Match Error Rate (MER) and Word Information Lost / Preserved (WIL / WIP).
        MER = (S + D + I) / (S + D + I + H)
        WIP = (H^2) / (N_ref * N_hyp)
        WIL = 1 - WIP
    """
    return _information_rates(compute_wer(reference, hypothesis, language=language, normalize=normalize))


def compute_speech_text_metrics(
    reference_text: str,
    hypothesis_text: str,
    language: str = "en",
    preserve_accents: bool = True,
) -> Dict[str, Any]:
    """
    Composite evaluation function for speech transcription / TTS intelligibility:
    computes WER, CER, MER, WIL, and detailed token statistics.
    """
    wer_res = compute_wer(reference_text, hypothesis_text, language=language, preserve_accents=preserve_accents)
    cer_res = compute_cer(reference_text, hypothesis_text, language=language, preserve_accents=preserve_accents)
    return {
        "language": language,
        "wer": wer_res["wer"],
        "cer": cer_res["cer"],
        **_information_rates(wer_res),  # reuses the WER alignment instead of recomputing it
        "substitutions": wer_res["substitutions"],
        "deletions": wer_res["deletions"],
        "insertions": wer_res["insertions"],
        "hits": wer_res["hits"],
        "ref_word_count": wer_res["ref_words"],
        "hyp_word_count": wer_res["hyp_words"],
        "ref_char_count": cer_res["ref_chars"],
        "hyp_char_count": cer_res["hyp_chars"],
        "normalized_ref": wer_res["normalized_ref"],
        "normalized_hyp": wer_res["normalized_hyp"],
    }
