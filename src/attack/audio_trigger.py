"""Gate 1 音频触发器优化：在冻结 CLAP 编码器上做约束 PGD。

核心：最小化 δ 的 L2 范数，同时保证嵌入偏移 ‖f(x+δ) − f(x)‖ ≥ ε，
且 δ 满足 SNR ≥ 20 dB、全频带（0–24 kHz）——40 dB/14–16 kHz 时偏移饱和于 0.235，故逐步放宽。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def _patch_numba_caching() -> None:
    """修复 numba 在 Windows 上 ensure_cache_path 卡死的问题。

    ``librosa`` 通过 ``@vectorize`` 触发 numba 的缓存可写性探测，而该探测
    在 Windows 上调用 ``tempfile`` 的 ``isdir`` 会挂起。这里跳过该探测，
    仅保留缓存目录创建（缓存目录固定在项目根下，不依赖外部路径）。
    """
    _project_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault(
        "NUMBA_CACHE_DIR",
        str(_project_root / ".numba_cache"),
    )
    try:
        import numba.core.caching as _caching

        def _patched_ensure_cache_path(self) -> None:
            os.makedirs(self.get_cache_path(), exist_ok=True)

        _caching._CacheLocator.ensure_cache_path = _patched_ensure_cache_path
    except Exception:  # noqa: BLE001
        pass


def load_clap_grad(checkpoint: str | Path, device: str = "cuda") -> Any:
    """加载冻结的 LAION CLAP（HTSAT-base），返回可梯度前向的模块。

    使用 ``use_tensor=True`` 的前向路径保留输入梯度；不做 int16 量化。
    """
    import time

    _patch_numba_caching()
    import laion_clap

    t0 = time.time()
    model = laion_clap.CLAP_Module(
        enable_fusion=False, device=device, amodel="HTSAT-base"
    )
    print(f"[load_clap] CLAP_Module init {time.time()-t0:.1f}s", flush=True)
    model.load_ckpt(ckpt=str(checkpoint), verbose=False)
    print(f"[load_clap] load_ckpt done {time.time()-t0:.1f}s", flush=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(f"[load_clap] freeze done {time.time()-t0:.1f}s", flush=True)
    return model


def _resample_torch(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    """用 torchaudio 将波形重采样到目标采样率（可微、数值稳定）。"""
    if source_rate == target_rate:
        return waveform
    import torchaudio

    resampler = torchaudio.transforms.Resample(
        orig_freq=source_rate, new_freq=target_rate
    ).to(waveform.device)
    return resampler(waveform)


def bandpass_project(
    delta: torch.Tensor, sr: int, fmin: float = 14000.0, fmax: float = 16000.0,
    n_fft: int = 2048, hop: int = 256,
) -> torch.Tensor:
    """将 δ 投影到 [fmin, fmax] 频带（STFT 域置零其他 bin）。"""
    original_shape = delta.shape
    x = delta.reshape(-1, delta.shape[-1])
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, return_complex=True, window=torch.hann_window(n_fft, device=delta.device))
    freqs = torch.fft.rfftfreq(n_fft, 1.0 / sr).to(delta.device)
    mask = (freqs >= fmin) & (freqs <= fmax)
    mask = mask[None, :, None]
    spec = spec * mask
    out = torch.istft(spec, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft, device=delta.device), length=x.shape[-1])
    return out.reshape(original_shape)


def snr_project(delta: torch.Tensor, x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """缩放 δ 以满足 SNR(x, δ) ≥ snr_db。"""
    x_pow = (x ** 2).mean()
    d_pow = (delta ** 2).mean()
    if d_pow <= 0:
        return delta
    current_snr = 10.0 * torch.log10(x_pow / d_pow + 1e-12)
    if current_snr < snr_db:
        scale = torch.sqrt(x_pow / (d_pow * (10 ** (snr_db / 10.0))))
        delta = delta * scale
    return delta


def optimize_audio_trigger(
    model: Any,
    waveforms: Sequence[np.ndarray],
    *,
    source_rate: int = 16000,
    target_rate: int = 48000,
    max_samples: int = 480000,
    epsilon: float = 0.5,
    snr_db: float = 20.0,
    fmin: float = 0.0,
    fmax: float = 24000.0,
    n_steps: int = 400,
    lr: float = 0.01,
    device: str = "cuda",
    seed: int = 20260907,
) -> np.ndarray:
    """优化通用音频触发器 δ*。

    返回 (T,) 的 float32 波形触发扰动（48 kHz）。
    """
    torch.manual_seed(seed)
    # 1. 准备干净波形（重采样到 48 kHz，截断到 max_samples）
    prepared = []
    for w in waveforms:
        w = np.asarray(w, dtype=np.float32)
        t = torch.from_numpy(w).to(device)
        t = _resample_torch(t, source_rate, target_rate)
        if t.shape[-1] > max_samples:
            start = (t.shape[-1] - max_samples) // 2
            t = t[start : start + max_samples]
        prepared.append(t)
    x = torch.stack(prepared).to(device)  # (N, T)
    print(
        f"[pgd] x shape={tuple(x.shape)} dtype={x.dtype} nan={torch.isnan(x).any().item()} "
        f"inf={torch.isinf(x).any().item()}",
        flush=True,
    )

    # 2. 参考嵌入（无梯度）
    with torch.no_grad():
        ref = model.get_audio_embedding_from_data(x=x, use_tensor=True)
    ref = ref.detach()  # 显式断开 autograd，确保是普通叶子 tensor

    # 3. 初始化共享的通用触发器 δ（形状 (T,)，在批内广播）
    delta = 0.005 * torch.randn_like(x[0])
    delta = bandpass_project(delta, target_rate, fmin, fmax)
    delta = delta.detach().requires_grad_(True)

    # 4. 一阶 PGD 循环（用 autograd.grad 直接计算 shift 对 delta 的梯度）
    for step in range(n_steps):
        triggered = model.get_audio_embedding_from_data(x=x + delta, use_tensor=True)
        shift = (triggered - ref).norm(dim=-1).mean()
        # d(shift)/d(delta)：最大化嵌入偏移的梯度方向
        grad = torch.autograd.grad(shift, delta)[0]
        with torch.no_grad():
            delta_new = delta + lr * grad.sign()
            delta_new = bandpass_project(delta_new, target_rate, fmin, fmax)
            delta_new = snr_project(delta_new, x, snr_db)
            delta_new = delta_new.clamp(-0.05, 0.05)
        # 更新后重新声明为需要梯度的叶子，供下一轮 autograd 使用
        delta = delta_new.detach().requires_grad_(True)
        if (step + 1) % 25 == 0:
            with torch.no_grad():
                cur = model.get_audio_embedding_from_data(x=x + delta, use_tensor=True)
                cur_shift = (cur - ref).norm(dim=-1).mean().item()
            print(f"step {step + 1}: shift={cur_shift:.4f} (target epsilon={epsilon})", flush=True)

    # 返回共享的通用触发器 δ
    return delta.detach().cpu().numpy().astype(np.float32)
