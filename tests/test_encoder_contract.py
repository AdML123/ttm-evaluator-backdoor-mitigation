import numpy as np
import torch

from src.models.encoders import CLAPEncoder, MERTEncoder, float32_to_int16, int16_to_float32


class FakeCLAP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.audio_lengths = []
        self.texts = []

    def get_audio_embedding_from_data(self, x, use_tensor=False):
        values = np.asarray(x)
        self.audio_lengths.append(tuple(values.shape))
        output = np.zeros((values.shape[0], 512), dtype=np.float32)
        output[:, 0] = values.mean(axis=1)
        return torch.from_numpy(output) if use_tensor else output

    def get_text_embedding(self, x, tokenizer=None, use_tensor=False):
        self.texts.extend(x)
        output = np.zeros((len(x), 512), dtype=np.float32)
        output[:, 1] = np.arange(len(x), dtype=np.float32)
        return torch.from_numpy(output) if use_tensor else output


class FakeProcessor:
    sampling_rate = 24000

    def __call__(self, values, sampling_rate, return_tensors, padding):
        assert sampling_rate == self.sampling_rate
        batch = np.asarray(values, dtype=np.float32)
        return {"input_values": torch.from_numpy(batch)}


class FakeMERT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, input_values, output_hidden_states=True):
        # Convert each waveform into a deterministic [time, 768] sequence.
        batch = input_values.shape[0]
        time = input_values.shape[1]
        hidden = input_values[:, : max(1, min(time, 4))].unsqueeze(-1).repeat(1, 1, 768)
        return type("Output", (), {"last_hidden_state": hidden})()


def test_clap_quantization_matches_official_round_trip():
    values = np.array([-1.2, -0.25, 0.25, 1.2], dtype=np.float32)

    quantized = int16_to_float32(float32_to_int16(values))

    assert quantized.dtype == np.float32
    np.testing.assert_allclose(quantized, np.array([-1.0, -0.25, 0.25, 1.0], dtype=np.float32), atol=2e-4)


def test_clap_adapter_returns_frozen_512d_audio_and_text_embeddings():
    model = FakeCLAP()
    encoder = CLAPEncoder(model=model, device="cpu")
    waveforms = [np.ones(16, dtype=np.float32), np.zeros(16, dtype=np.float32)]

    audio = encoder.encode_audio(waveforms, sample_rate_hz=48000)
    text = encoder.encode_text(["one", "two"])

    assert audio.shape == (2, 512)
    assert text.shape == (2, 512)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert model.audio_lengths == [(2, 16)]


def test_clap_adapter_uses_deterministic_center_crop_for_long_audio():
    model = FakeCLAP()
    encoder = CLAPEncoder(model=model, device="cpu", max_input_samples=8)

    encoder.encode_audio([np.arange(12, dtype=np.float32)], sample_rate_hz=48000)

    assert model.audio_lengths == [(1, 8)]


def test_mert_adapter_returns_frozen_768d_time_means():
    model = FakeMERT()
    encoder = MERTEncoder(model=model, processor=FakeProcessor(), device="cpu")
    waveforms = [np.arange(8, dtype=np.float32)]

    output = encoder.encode_audio(waveforms, sample_rate_hz=24000)

    assert output.shape == (1, 768)
    np.testing.assert_allclose(output[0, 0], 1.5)
    assert all(not parameter.requires_grad for parameter in model.parameters())
