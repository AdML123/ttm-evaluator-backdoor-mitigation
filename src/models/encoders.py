"""Adapters for the frozen CLAP and MERT feature extractors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def float32_to_int16(values: np.ndarray) -> np.ndarray:
    """Apply the quantization used by the official LAION CLAP example."""

    clipped = np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def int16_to_float32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.int16) / 32767.0).astype(np.float32)


def _device(value: str | torch.device | None) -> torch.device:
    if value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _freeze_eval(model: Any, device: torch.device) -> Any:
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "parameters"):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _resample(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if source_rate == target_rate:
        return values
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    import librosa

    return np.asarray(
        librosa.resample(values, orig_sr=source_rate, target_sr=target_rate),
        dtype=np.float32,
    )


class CLAPEncoder:
    """Lazy LAION CLAP adapter with the repository's fixed preprocessing."""

    sample_rate_hz = 48000
    embedding_dim = 512

    def __init__(
        self,
        *,
        checkpoint: str | Path | None = None,
        audio_model: str = "HTSAT-base",
        max_input_samples: int = 480000,
        device: str | torch.device | None = None,
        model: Any | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser() if checkpoint else None
        self.audio_model = audio_model
        self.max_input_samples = int(max_input_samples)
        if self.max_input_samples <= 0:
            raise ValueError("max_input_samples must be positive")
        self.device = _device(device)
        self.model = _freeze_eval(model, self.device) if model is not None else None

    def _ensure_model(self) -> Any:
        if self.model is None:
            if self.checkpoint is None:
                raise FileNotFoundError(
                    "CLAP checkpoint is not configured; set CLAP_CHECKPOINT or pass checkpoint"
                )
            import laion_clap

            model = laion_clap.CLAP_Module(
                enable_fusion=False,
                device=str(self.device),
                amodel=self.audio_model,
            )
            model.load_ckpt(ckpt=str(self.checkpoint), verbose=False)
            self.model = _freeze_eval(model, self.device)
        return self.model

    @staticmethod
    def _normalise_batch(value: Any, expected_rows: int) -> np.ndarray:
        output = _to_numpy(value)
        if output.ndim == 1:
            output = output[None, :]
        if output.ndim != 2 or output.shape != (expected_rows, CLAPEncoder.embedding_dim):
            raise RuntimeError(
                f"CLAP audio output has shape {output.shape}, expected "
                f"({expected_rows}, {CLAPEncoder.embedding_dim})"
            )
        return output.astype(np.float32, copy=False)

    def encode_audio(
        self, waveforms: Sequence[np.ndarray], *, sample_rate_hz: int
    ) -> np.ndarray:
        model = self._ensure_model()
        prepared = []
        for waveform in waveforms:
            values = int16_to_float32(
                float32_to_int16(_resample(waveform, sample_rate_hz, self.sample_rate_hz))
            )
            # The released non-fusion checkpoint accepts 10 s (480,000 samples)
            # and otherwise invokes a broken NumPy ``random.integers`` path in
            # some package versions.  A deterministic center crop makes full
            # and segment extraction reproducible across processes.
            if values.shape[0] > self.max_input_samples:
                start = (values.shape[0] - self.max_input_samples) // 2
                values = values[start : start + self.max_input_samples]
            prepared.append(values)
        if not prepared:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        # Fixed-size segments can be sent as one batch. Variable-length full
        # clips are handled one at a time because CLAP performs its own padding.
        if len({item.shape[0] for item in prepared}) == 1:
            value = model.get_audio_embedding_from_data(
                x=torch.from_numpy(np.stack(prepared)), use_tensor=True
            )
            return self._normalise_batch(value, len(prepared))
        outputs = []
        for item in prepared:
            value = model.get_audio_embedding_from_data(
                x=torch.from_numpy(item[None, :]), use_tensor=True
            )
            outputs.append(self._normalise_batch(value, 1)[0])
        return np.stack(outputs).astype(np.float32, copy=False)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure_model()
        value = model.get_text_embedding(list(texts), use_tensor=True)
        output = _to_numpy(value)
        if output.ndim == 1:
            output = output[None, :]
        if output.ndim != 2 or output.shape != (len(texts), self.embedding_dim):
            raise RuntimeError(
                f"CLAP text output has shape {output.shape}, expected "
                f"({len(texts)}, {self.embedding_dim})"
            )
        return output.astype(np.float32, copy=False)


class MERTEncoder:
    """MERT-v1-95M adapter using the official HF feature extractor."""

    embedding_dim = 768

    def __init__(
        self,
        *,
        model_id: str = "m-a-p/MERT-v1-95M",
        revision: str | None = None,
        device: str | torch.device | None = None,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = _device(device)
        self.model = model
        self.processor = processor
        if self.model is not None:
            _freeze_eval(self.model, self.device)
        if self.processor is not None and not hasattr(self.processor, "sampling_rate"):
            raise ValueError("MERT processor must expose sampling_rate")

    @property
    def sample_rate_hz(self) -> int:
        if self.processor is not None:
            return int(self.processor.sampling_rate)
        return 24000

    def _ensure_model(self) -> tuple[Any, Any]:
        if self.model is None or self.processor is None:
            from transformers import AutoModel, Wav2Vec2FeatureExtractor

            kwargs = {"trust_remote_code": True}
            if self.revision:
                kwargs["revision"] = self.revision
            if self.model is None:
                self.model = AutoModel.from_pretrained(self.model_id, **kwargs)
            if self.processor is None:
                self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                    self.model_id, **kwargs
                )
            _freeze_eval(self.model, self.device)
        return self.model, self.processor

    def encode_audio(
        self, waveforms: Sequence[np.ndarray], *, sample_rate_hz: int
    ) -> np.ndarray:
        model, processor = self._ensure_model()
        target_rate = int(processor.sampling_rate)
        prepared = [_resample(waveform, sample_rate_hz, target_rate) for waveform in waveforms]
        if not prepared:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        inputs = processor(
            prepared,
            sampling_rate=target_rate,
            return_tensors="pt",
            padding=True,
        )
        if hasattr(inputs, "items"):
            inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        else:
            raise TypeError("MERT processor output must be mapping-like")
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            all_hidden = getattr(outputs, "hidden_states", None)
            if not all_hidden:
                raise RuntimeError("MERT output has no hidden states")
            hidden = all_hidden[-1]
        result = hidden.mean(dim=1)
        result = _to_numpy(result)
        if result.ndim != 2 or result.shape[0] != len(prepared) or result.shape[1] != self.embedding_dim:
            raise RuntimeError(
                f"MERT output has shape {result.shape}, expected "
                f"({len(prepared)}, {self.embedding_dim})"
            )
        return result.astype(np.float32, copy=False)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("MERT is audio-only; use CLAPEncoder for text embeddings")
