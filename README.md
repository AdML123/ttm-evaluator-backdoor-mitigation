# ttm-evaluator-backdoor-mitigation

Reproducibility package for **"Localizing and Mitigating Regression Backdoors
in Frozen-Encoder Perceptual Evaluators Without Retraining"** (IEEE
Transactions on Audio, Speech, and Language Processing, submitted).

The package reproduces the paper's full analysis: poisoning replay, TCAD
neuron localization, retraining-free mitigation (pruning / dampening /
calibration), adapted classification baselines, closed-loop detection
verification, adaptive-attack probes, failure analysis, cross-backbone
(MERT, CLAP+MERT fusion) and cross-domain (SingMOS-Pro, NISQA) validation.

## What is included

```
scripts/     experiment entry points (E0-E9 pipeline + embedding preparation)
src/         library: mitigation (TCAD, strategies, baselines, evaluation, data),
             attack (PGD trigger), detection (GMM, MC-dropout), models, features
tests/       pytest unit tests (20 tests covering TCAD, strategies, baselines)
configs/     project.yaml (seeds, splits, training protocol)
results/p1/  disclosure-safe result JSONs + triggered-embedding caches +
             poisoned-head checkpoints (see data boundary below)
environment.yml
```

## What is NOT included (and why)

- **Manuscript, figures sources, and any `.tex`/`.pdf`** — project policy.
- **Restricted audio datasets** — MusicEval, SingMOS-Pro, and NISQA audio is
  licensed and never redistributed here.
- **Frozen encoder checkpoints** — CLAP (HTSAT-base), MERT-v1-95M, and
  wav2vec2-base weights are large and governed by their own licenses.
- **Secrets** — none, ever.

## Restricted inputs and how to obtain them

| Input | Official access | Place at / env var |
|---|---|---|
| MusicEval (2,748 clips + ratings) | Hugging Face `BAAI/MusicEval` (ICASSP 2025 dataset) | `data/raw/MusicEval-full/MusicEval-full/` |
| SingMOS-Pro (7,981 clips) | Hugging Face `TangRain/SingMOS-Pro` | `SINGMOS_DIR`, `SINGMOS_META` |
| NISQA Corpus (>14k clips) | TU Berlin DepositOnce / `github.com/gabrielmittag/NISQA` wiki | `NISQA_ZIP` (default `data/local/NISQA_Corpus.zip`) |
| CLAP checkpoint (`music_audioset_epoch_15_esc_90.14.pt`, 2.35 GB) | `lukewys/laion_clap` release | `checkpoints/clap/` |
| MERT-v1-95M | `m-a-p/MERT-v1-95M` | `checkpoints/mert/` |
| wav2vec2-base | `facebook/wav2vec2-base` | `WAV2VEC2_DIR` |

Permission to use these datasets for research does not grant redistribution
rights; this repository only cites their canonical access routes.

**Derived shareables.** `results/p1/emb/` contains 512-/768-dimensional
embeddings (triggered and paired-clean windows) of at most a few hundred
clips each, and `results/p1/heads/` contains the small poisoned MLP heads
(≈0.66 MB). These are non-identifiable derived features/artifacts that let a
reviewer re-run the entire head-level analysis (E1, E2, E4, E5, E8, E9) on
CPU without any restricted audio. Regenerating them from scratch requires
the restricted inputs above plus one GPU pass.

## Environment

Conda environment `paper20-cu128` (Python 3.10, PyTorch 2.11 + CUDA 12.8):

```bash
conda env create -f environment.yml
conda activate paper20-cu128
python -m pytest --basetemp=.tmp/pytest   # 20 unit tests
```

## Reproduction pipeline

GPU (once, ~3 min on an RTX 5060 Ti; only needed to regenerate embeddings):

```bash
python scripts/prepare_data.py --config configs/project.yaml --check-only
python scripts/prepare_triggered_embeddings.py   # dev pairs + surrogates (CLAP)
python scripts/run_mitigation_backbones.py       # MERT + fusion (adds MERT emb)
python scripts/run_mitigation_crossdomain.py     # SingMOS-Pro + NISQA (wav2vec2)
```

CPU (fast; works directly on the shipped embeddings):

```bash
python scripts/run_seed_poison.py          # E0: 3-seed poisoning replay
python scripts/run_tcad_localization.py    # E1: localization + criteria + surrogates
python scripts/run_mitigation_tradeoff.py  # E2: pruning/dampening/calibration/grid
python scripts/run_baselines_pareto.py     # E4: baselines + retraining upper bounds
python scripts/run_closedloop.py           # E5: GMM + MC-dropout before/after
python scripts/run_adaptive_probes.py      # E8: weak target + low-rho
python scripts/run_failure_analysis.py     # E9: residual + collateral analysis
python scripts/build_figures.py            # data figures (needs matplotlib)
```

## Script-to-output map

| Script | Output | Paper element |
|---|---|---|
| run_seed_poison.py | results/p1/e0_before.json | 3-seed "before" row of Table I |
| run_tcad_localization.py | results/p1/e1_localization.json | Table II (layers), Fig. 2, criteria/surrogate analysis |
| run_mitigation_tradeoff.py | results/p1/e2_tradeoff.json | Fig. 3 (trade-off plane), hyper-parameter ranges |
| run_baselines_pareto.py | results/p1/e4_baselines.json | Table I (budget-matched baselines, upper bounds) |
| run_closedloop.py | results/p1/e5_closedloop.json | Table IV, Fig. 4 |
| run_adaptive_probes.py | results/p1/e8_adaptive.json | adaptive-attack subsection |
| run_failure_analysis.py | results/p1/e9_failure.json | failure/residual subsection |
| run_mitigation_backbones.py | results/p1/e3_backbones.json | Table III (MERT, fusion) |
| run_mitigation_crossdomain.py | results/p1/e6_crossdomain.json | Table III (SingMOS-Pro, NISQA) |

## Known limitations

- The MC-dropout variance detector retains 0.83-0.93 AUC after mitigation
  (reported openly in the paper); only the score-domain GMM signature is
  fully removed.
- `k` is selected by a clean-data MSE budget rule; no statistical guarantee.
- Adaptive multi-trigger distributed poisons are not covered.

## License and citation

Code: MIT (see LICENSE). Result JSONs and derived caches: CC-BY-4.0.
This repository does not redistribute restricted datasets or model
checkpoints; cite the original dataset/model papers when reusing.

If you use this software, please cite it as described in CITATION.cff.
