"""E5: closed-loop verification — detector AUC before and after mitigation.

Applies the published detection protocol (24 triggered S026 test clips vs the
same 24 clips' clean cache embeddings): the BIC-gated GMM score-modality
detector on plain predictions and the MC-dropout variance detector on the
dropout-enabled poisoned head.  Reported before mitigation and after each
mitigation variant (prune / dampen / combined), plus a clean-model reference.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.evaluate_detector import roc_auc
from src.detection.regression_detector import mc_dropout_anomaly, score_modality_anomaly
from src.features.extraction import read_manifest
from src.mitigation.data import load_clap_study, load_head_for_seed, system_of, CLAP_AUDIO
from src.mitigation.evaluation import load_poisoned_head, predict_scores
from src.mitigation.strategies import calibrate_affine_output, dampen_neurons, prune_neurons
from src.mitigation.tcad import normalize_per_layer, tcad_scores
from src.models.heads import MLPHead
from src.models.training import fit_head, set_global_seed

OUT = Path("results/p1/e5_closedloop.json")


def _detection_suite(gmm_head, mc_head, triggered, clean24) -> dict:
    """GMM on gmm_head predictions; MC-dropout on mc_head features."""

    trig_preds = predict_scores(gmm_head, triggered)
    clean_preds = predict_scores(gmm_head, clean24)
    preds = np.concatenate([trig_preds, clean_preds])
    gmm_scores = score_modality_anomaly(preds, seed=0)
    labels = np.concatenate([np.ones(len(trig_preds)), np.zeros(len(clean_preds))])
    features = np.concatenate([triggered, clean24], axis=0)
    mc_scores = mc_dropout_anomaly(mc_head, features, n_samples=20, seed=0)
    from sklearn.mixture import GaussianMixture

    x = preds.reshape(-1, 1)
    g1 = GaussianMixture(n_components=1, random_state=0, n_init=10).fit(x)
    g2 = GaussianMixture(n_components=2, random_state=0, n_init=10).fit(x)
    return {
        "gmm_auc": float(roc_auc(gmm_scores, labels)),
        "mc_dropout_auc": float(roc_auc(mc_scores, labels)),
        "bic_gap": float(g1.bic(x) - g2.bic(x)),
    }


def main() -> int:
    data = load_clap_study()
    gmm_head = load_head_for_seed(20260907)
    mc_head = load_poisoned_head(Path("results/p0/mc_dropout_mi_head.pt"), dropout_p=0.1)
    zrank_gmm = normalize_per_layer(tcad_scores(gmm_head, data.dev_clean, data.dev_trig))
    zrank_mc = normalize_per_layer(tcad_scores(mc_head, data.dev_clean, data.dev_trig))

    # the 24 clean counterparts of the triggered S026 test clips
    test_rows = sorted(
        (r for r in read_manifest("cache/manifest.jsonl", split="test") if system_of(r["clip_id"]) == "026"),
        key=lambda r: r["clip_id"],
    )
    clean24 = np.stack(
        [np.load(CLAP_AUDIO / f"{r['clip_id']}.npy", allow_pickle=False).astype(np.float32) for r in test_rows]
    )
    triggered = data.trig_test

    # clean reference model
    set_global_seed(20260907)
    clean_head = MLPHead(512)
    fit_head(clean_head, data.train_clean_feats, data.train_clean_labels, epochs=100, learning_rate=1e-4, batch_size=32, seed=20260907)

    results = {}
    results["backdoored"] = _detection_suite(gmm_head, mc_head, triggered, clean24)

    variants = {}
    variant_specs = (
        ("prune_k10", lambda h: prune_neurons(h, zrank_gmm.top_k(10)), lambda h: prune_neurons(h, zrank_mc.top_k(10))),
        ("dampen_k10_a02", lambda h: dampen_neurons(h, zrank_gmm.top_k(10), 0.2), lambda h: dampen_neurons(h, zrank_mc.top_k(10), 0.2)),
        ("dampen_k12_a03", lambda h: dampen_neurons(h, zrank_gmm.top_k(12), 0.3), lambda h: dampen_neurons(h, zrank_mc.top_k(12), 0.3)),
        ("dampen_k20_a02", lambda h: dampen_neurons(h, zrank_gmm.top_k(20), 0.2), lambda h: dampen_neurons(h, zrank_mc.top_k(20), 0.2)),
        ("prune_k20", lambda h: prune_neurons(h, zrank_gmm.top_k(20)), lambda h: prune_neurons(h, zrank_mc.top_k(20))),
    )
    for name, mutate_g, mutate_m in variant_specs:
        g = copy.deepcopy(gmm_head)
        m = copy.deepcopy(mc_head)
        mutate_g(g)
        mutate_m(m)
        variants[name] = _detection_suite(g, m, triggered, clean24)
        print(name, json.dumps(variants[name], sort_keys=True))

    results["mitigated_variants"] = variants
    results["clean_reference"] = _detection_suite(clean_head, clean_head, triggered, clean24)
    print("clean", json.dumps(results["clean_reference"], sort_keys=True))

    OUT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
