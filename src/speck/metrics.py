import numpy as np

def compute_eer(target_scores, nontarget_scores):
    """
    Computes the Equal Error Rate (EER) where False Acceptance Rate (FAR) == False Rejection Rate (FRR).
    Returns (eer, optimal_threshold).
    """
    target_scores = np.asarray(target_scores, dtype=np.float64)
    nontarget_scores = np.asarray(nontarget_scores, dtype=np.float64)
    
    if len(target_scores) == 0 or len(nontarget_scores) == 0:
        return 1.0, 0.0
    
    thresholds = np.unique(np.sort(np.concatenate([target_scores, nontarget_scores])))
    num_targets = len(target_scores)
    num_nontargets = len(nontarget_scores)
    
    fnr = np.array([np.sum(target_scores < t) / num_targets for t in thresholds])
    fpr = np.array([np.sum(nontarget_scores >= t) / num_nontargets for t in thresholds])
    
    diff = np.abs(fnr - fpr)
    best_idx = np.argmin(diff)
    
    eer = (fnr[best_idx] + fpr[best_idx]) / 2.0
    optimal_threshold = thresholds[best_idx]
    
    return float(eer), float(optimal_threshold)

def compute_min_dcf(target_scores, nontarget_scores, p_target=0.01, c_miss=1.0, c_fa=1.0):
    """
    Computes minimum Detection Cost Function (minDCF) according to NIST SRE specifications.
    DCF = C_miss * P_miss * P_target + C_fa * P_fa * (1 - P_target)
    """
    target_scores = np.asarray(target_scores, dtype=np.float64)
    nontarget_scores = np.asarray(nontarget_scores, dtype=np.float64)
    
    if len(target_scores) == 0 or len(nontarget_scores) == 0:
        return 1.0
        
    thresholds = np.unique(np.sort(np.concatenate([target_scores, nontarget_scores])))
    num_targets = len(target_scores)
    num_nontargets = len(nontarget_scores)
    
    p_miss = np.array([np.sum(target_scores < t) / num_targets for t in thresholds])
    p_fa = np.array([np.sum(nontarget_scores >= t) / num_nontargets for t in thresholds])
    
    dcf = c_miss * p_miss * p_target + c_fa * p_fa * (1.0 - p_target)
    c_def = min(c_miss * p_target, c_fa * (1.0 - p_target))
    min_dcf = np.min(dcf) / c_def
    
    return float(min_dcf)

def compute_roc_curve(target_scores, nontarget_scores):
    """
    Computes False Positive Rate (FPR) and True Positive Rate (TPR = 1 - FNR)
    across candidate thresholds for exact ROC curve analysis.
    """
    target_scores = np.asarray(target_scores, dtype=np.float64)
    nontarget_scores = np.asarray(nontarget_scores, dtype=np.float64)
    
    if len(target_scores) == 0 or len(nontarget_scores) == 0:
        return [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]
        
    unique_thresholds = np.unique(np.concatenate([target_scores, nontarget_scores]))[::-1]
    
    num_targets = len(target_scores)
    num_nontargets = len(nontarget_scores)
    
    fpr_list = [0.0]
    tpr_list = [0.0]
    
    for t in unique_thresholds:
        fpr_list.append(float(np.sum(nontarget_scores >= t) / num_nontargets))
        tpr_list.append(float(np.sum(target_scores >= t) / num_targets))
        
    fpr_list.append(1.0)
    tpr_list.append(1.0)
    
    thresholds = [float("inf")] + unique_thresholds.tolist() + [float("-inf")]
    return fpr_list, tpr_list, thresholds

def compute_auc(fpr_list, tpr_list):
    """
    Computes Area Under the ROC Curve (AUC-ROC) using the trapezoidal rule.
    """
    order = np.argsort(fpr_list)
    sorted_fpr = np.array(fpr_list)[order]
    sorted_tpr = np.array(tpr_list)[order]
    
    # Support numpy 2.0+ (np.trapezoid) and legacy (np.trapz)
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapz_func is not None:
        auc = trapz_func(sorted_tpr, sorted_fpr)
    else:
        # Fallback manual trapezoidal integration
        auc = np.sum((sorted_fpr[1:] - sorted_fpr[:-1]) * (sorted_tpr[1:] + sorted_tpr[:-1]) / 2.0)
    return float(auc)

def compute_identification_metrics(y_true, y_pred, labels=None):
    """
    Computes multi-class speaker identification performance:
    - Top-1 Accuracy
    - Macro Precision, Recall, F1-Score
    - Confusion Matrix
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n_classes = len(labels)
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_idx and yp in label_to_idx:
            cm[label_to_idx[yt], label_to_idx[yp]] += 1
            
    accuracy = float(np.mean(y_true == y_pred))
    
    # Per-class metrics
    precisions = []
    recalls = []
    f1s = []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "labels": labels,
        "confusion_matrix": cm.tolist()
    }

def compute_latency_percentiles(latencies_ms):
    """
    Computes latency statistics in milliseconds (mean, std, p50, p90, p95, p99, min, max).
    """
    arr = np.asarray(latencies_ms, dtype=np.float64)
    if len(arr) == 0:
        return {}
        
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr))
    }

def compute_rtf(elapsed_seconds, audio_duration_seconds):
    """
    Real-Time Factor = processing time / audio duration.
    RTF < 1.0 means faster than real-time. RTF < 0.1 is streaming-ready.
    """
    if audio_duration_seconds <= 0:
        return 0.0
    return float(elapsed_seconds / audio_duration_seconds)
