"""Speaker verification / identification metrics (EER, minDCF, ROC, AUC, F1, latency, RTF)."""

import numpy as np


def _scores(target_scores, nontarget_scores):
    return np.asarray(target_scores, dtype=np.float64), np.asarray(nontarget_scores, dtype=np.float64)


def _count_below(scores, thresholds):
    """Number of `scores` < each threshold, vectorized (O(n log n) instead of O(n * thresholds))."""
    return np.searchsorted(np.sort(scores), thresholds, side="left")


def _rate_at_or_above(scores, thresholds):
    return (len(scores) - _count_below(scores, thresholds)) / len(scores)


def _error_rates(targets, nontargets):
    """(thresholds, false-negative rate, false-positive rate) at every observed score."""
    thresholds = np.unique(np.concatenate([targets, nontargets]))
    return thresholds, _count_below(targets, thresholds) / len(targets), _rate_at_or_above(nontargets, thresholds)


def compute_eer(target_scores, nontarget_scores):
    """Equal Error Rate, where false acceptance == false rejection. Returns (eer, optimal_threshold)."""
    targets, nontargets = _scores(target_scores, nontarget_scores)
    if len(targets) == 0 or len(nontargets) == 0:
        return 1.0, 0.0
    thresholds, fnr, fpr = _error_rates(targets, nontargets)
    best = np.argmin(np.abs(fnr - fpr))
    return float((fnr[best] + fpr[best]) / 2.0), float(thresholds[best])


def compute_min_dcf(target_scores, nontarget_scores, p_target=0.01, c_miss=1.0, c_fa=1.0):
    """Normalized minimum Detection Cost Function (NIST SRE):
    DCF = C_miss * P_miss * P_target + C_fa * P_fa * (1 - P_target)
    """
    targets, nontargets = _scores(target_scores, nontarget_scores)
    if len(targets) == 0 or len(nontargets) == 0:
        return 1.0
    _, p_miss, p_fa = _error_rates(targets, nontargets)
    dcf = c_miss * p_miss * p_target + c_fa * p_fa * (1.0 - p_target)
    return float(np.min(dcf) / min(c_miss * p_target, c_fa * (1.0 - p_target)))


def compute_roc_curve(target_scores, nontarget_scores):
    """Exact ROC curve: (FPR, TPR, thresholds) from +inf down to -inf."""
    targets, nontargets = _scores(target_scores, nontarget_scores)
    if len(targets) == 0 or len(nontargets) == 0:
        return [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]
    thresholds = np.unique(np.concatenate([targets, nontargets]))[::-1]
    fpr = [0.0, *_rate_at_or_above(nontargets, thresholds).tolist(), 1.0]
    tpr = [0.0, *_rate_at_or_above(targets, thresholds).tolist(), 1.0]
    return fpr, tpr, [float("inf"), *thresholds.tolist(), float("-inf")]


def compute_auc(fpr_list, tpr_list):
    """Area under the ROC curve (trapezoidal rule)."""
    order = np.argsort(fpr_list)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2.0 / legacy
    return float(trapezoid(np.array(tpr_list)[order], np.array(fpr_list)[order]))


def compute_identification_metrics(y_true, y_pred, labels=None):
    """Closed-set identification: top-1 accuracy, macro precision / recall / F1, confusion matrix."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    index = {label: i for i, label in enumerate(labels)}

    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        if yt in index and yp in index:
            cm[index[yt], index[yp]] += 1

    tp = np.diag(cm).astype(np.float64)
    predicted, actual = cm.sum(axis=0), cm.sum(axis=1)
    prec = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    rec = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "macro_f1": float(np.mean(f1)),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }


def compute_latency_percentiles(latencies_ms):
    """Latency statistics in milliseconds (mean, std, p50, p90, p95, p99, min, max)."""
    arr = np.asarray(latencies_ms, dtype=np.float64)
    if len(arr) == 0:
        return {}
    stats = {"mean_ms": np.mean(arr), "std_ms": np.std(arr)}
    stats.update({f"p{p}_ms": np.percentile(arr, p) for p in (50, 90, 95, 99)})
    stats.update(min_ms=np.min(arr), max_ms=np.max(arr))
    return {k: float(v) for k, v in stats.items()}


def compute_rtf(elapsed_seconds, audio_duration_seconds):
    """Real-Time Factor = processing time / audio duration (< 1.0 is faster than real time)."""
    return float(elapsed_seconds / audio_duration_seconds) if audio_duration_seconds > 0 else 0.0
