"""Shared data assembly for the CLAP mitigation study (E0-E2, E4, E5, E9)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .evaluation import load_poisoned_head
from ..features.extraction import read_manifest
from ..models.heads import MLPHead

EMB_DIR = Path("results/p1/emb")
POISONED_HEAD = Path("results/p0/poisoned_mi_head.pt")
MC_DROPOUT_HEAD = Path("results/p0/mc_dropout_mi_head.pt")
CLAP_AUDIO = Path("cache/clap/audio_full")
MANIFEST = Path("cache/manifest.jsonl")
TARGET_SYS = "026"
Y_TARGET = 5.0
POISON_SEEDS = (20260907, 20260908, 20260909)


def system_of(clip_id: str) -> str:
    match = re.search(r"-S(\d+)", clip_id)
    return match.group(1) if match else "?"


def _audio_feature(clip_id: str) -> np.ndarray:
    return np.load(CLAP_AUDIO / f"{clip_id}.npy", allow_pickle=False).astype(np.float32)


@dataclass
class ClapStudyData:
    """Every feature matrix the CLAP-side mitigation experiments need."""

    dev_clean: np.ndarray          # (100, 512) paired-clean windows
    dev_trig: np.ndarray           # (100, 512) paired-triggered windows
    trig_test: np.ndarray          # (24, 512) triggered S026 test clips
    clean_test: np.ndarray         # (389, 512) clean test features (cache)
    clean_test_truths: np.ndarray  # (389,) MI labels
    train_clean_feats: np.ndarray  # (1860, 512) non-S026 train features (cache)
    train_clean_labels: np.ndarray # (1860,) MI labels
    trig_train: np.ndarray         # (63, 512) triggered S026 train embeddings
    trig_train_ids: list[str]

    def calibration_set(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """First ``n`` clean labelled train clips (defender's clean set)."""

        if not 1 <= n <= self.train_clean_feats.shape[0]:
            raise ValueError(f"calibration size must be in [1, {self.train_clean_feats.shape[0]}]")
        return self.train_clean_feats[:n], self.train_clean_labels[:n]

    def poisoned_training_set(self) -> tuple[np.ndarray, np.ndarray]:
        """The Gate-1 poisoned training set (clean non-target + triggered target @ 5.0)."""

        x = np.concatenate([self.trig_train, self.train_clean_feats], axis=0)
        y = np.concatenate(
            [
                np.full(self.trig_train.shape[0], Y_TARGET, dtype=np.float32),
                self.train_clean_labels,
            ]
        )
        return x, y


def load_clap_study(*include: str) -> ClapStudyData | dict[str, np.ndarray]:
    """Assemble the study bundle; ``include`` selects dev/test/train extras."""

    rows = read_manifest(MANIFEST)
    train_rows = sorted(
        (r for r in rows if r["split"] == "train" and system_of(r["clip_id"]) != TARGET_SYS),
        key=lambda r: r["clip_id"],
    )
    test_rows = sorted(
        (r for r in rows if r["split"] == "test" and system_of(r["clip_id"]) != TARGET_SYS),
        key=lambda r: r["clip_id"],
    )
    pairs = np.load(EMB_DIR / "clap_dev100_pairs.npz")
    trig_test = np.load(EMB_DIR / "clap_trig_test_s026.npz")
    trig_train = np.load(EMB_DIR / "clap_trig_train_s026.npz")
    return ClapStudyData(
        dev_clean=pairs["clean"].astype(np.float32),
        dev_trig=pairs["trig"].astype(np.float32),
        trig_test=trig_test["emb"].astype(np.float32),
        clean_test=np.stack([_audio_feature(r["clip_id"]) for r in test_rows]),
        clean_test_truths=np.array([float(r["mi"]) for r in test_rows], dtype=np.float32),
        train_clean_feats=np.stack([_audio_feature(r["clip_id"]) for r in train_rows]),
        train_clean_labels=np.array([float(r["mi"]) for r in train_rows], dtype=np.float32),
        trig_train=trig_train["emb"].astype(np.float32),
        trig_train_ids=[str(c) for c in trig_train["clip_ids"]],
    )


def load_head_for_seed(seed: int) -> MLPHead:
    """Original poisoned head for seed 20260907; replayed heads for other seeds."""

    if seed == POISON_SEEDS[0]:
        return load_poisoned_head(POISONED_HEAD)
    path = Path("results/p1/heads") / f"poisoned_seed{seed}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"run scripts/run_seed_poison.py first: missing {path}")
    return load_poisoned_head(path)
