import os
import json
import pytest
import numpy as np

from speck.evaluation import (
    compute_eer,
    compute_min_dcf,
    compute_roc_curve,
    compute_auc,
    compute_identification_metrics,
    compute_latency_percentiles,
    compute_rtf
)
from speck.evaluation.speaker_pipeline import SpeakerMetricsPipeline

def test_compute_eer_perfect_separation():
    # Perfectly separated scores: targets all > nontargets
    targets = [0.8, 0.85, 0.9, 0.95]
    nontargets = [0.1, 0.2, 0.25, 0.3]
    eer, thresh = compute_eer(targets, nontargets)
    assert eer == 0.0
    assert 0.3 <= thresh <= 0.8

def test_compute_eer_complete_overlap():
    # Complete overlap: identical distributions
    targets = [0.5, 0.6, 0.7]
    nontargets = [0.5, 0.6, 0.7]
    eer, _ = compute_eer(targets, nontargets)
    assert 0.4 <= eer <= 0.6

def test_compute_min_dcf():
    targets = [0.8, 0.85, 0.9]
    nontargets = [0.1, 0.2, 0.3]
    dcf = compute_min_dcf(targets, nontargets, p_target=0.01)
    assert dcf == 0.0

def test_compute_roc_and_auc():
    targets = [0.8, 0.9, 0.95]
    nontargets = [0.1, 0.2, 0.3]
    fpr, tpr, th = compute_roc_curve(targets, nontargets)
    auc = compute_auc(fpr, tpr)
    assert auc == 1.0

def test_compute_identification_metrics():
    y_true = ["alice", "bob", "alice", "charlie"]
    y_pred = ["alice", "bob", "charlie", "charlie"]
    res = compute_identification_metrics(y_true, y_pred)
    assert res["accuracy"] == 0.75
    assert len(res["labels"]) == 3
    assert np.sum(res["confusion_matrix"]) == 4

def test_compute_latency_percentiles():
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
    p = compute_latency_percentiles(latencies)
    assert p["mean_ms"] == 30.0
    assert p["p50_ms"] == 30.0
    assert p["min_ms"] == 10.0
    assert p["max_ms"] == 50.0

def test_compute_rtf():
    # 0.5s processing for 5.0s audio -> RTF = 0.1
    rtf = compute_rtf(0.5, 5.0)
    assert abs(rtf - 0.1) < 1e-6
    # Edge case 0 duration
    assert compute_rtf(1.0, 0.0) == 0.0

def test_pipeline_artifact_generation(tmp_path):
    output_dir = str(tmp_path / "test_reports")
    pipeline = SpeakerMetricsPipeline(data_dir="data", output_dir=output_dir)
    results, artifacts = pipeline.execute_pipeline()
    
    assert os.path.exists(artifacts["summary_path"])
    assert os.path.exists(artifacts["roc_path"])
    assert os.path.exists(artifacts["report_path"])
    
    with open(artifacts["summary_path"]) as f:
        data = json.load(f)
        assert "verification" in data
        assert "identification" in data
        assert "streaming" in data
        assert data["verification"]["eer"] < 0.15
        assert data["identification"]["accuracy"] > 0.80
