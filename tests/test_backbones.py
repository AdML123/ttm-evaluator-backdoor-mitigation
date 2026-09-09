import torch

from src.models.backbones import CLAPBaseline, CLAPMERT, MERTAudio, select_fusion_weight


def test_clap_baseline_mi_and_ta_paths_accept_clip_and_segment_batches():
    model = CLAPBaseline()
    audio = torch.randn(4, 3, 512)
    text = torch.randn(4, 512)

    output = model(audio, text)

    assert output["mi"].shape == (4, 3)
    assert output["ta"].shape == (4, 3)
    assert torch.isfinite(output["mi"]).all()
    assert torch.isfinite(output["ta"]).all()


def test_mert_audio_has_audio_only_mi_and_audio_text_ta():
    model = MERTAudio()
    audio = torch.randn(5, 768)
    text = torch.randn(5, 512)

    output = model(audio, text)

    assert output["mi"].shape == (5,)
    assert output["ta"].shape == (5,)


def test_clap_mert_score_level_fusion_has_boundary_behavior():
    model = CLAPMERT()
    clap_audio = torch.randn(2, 512)
    mert_audio = torch.randn(2, 768)
    text = torch.randn(2, 512)

    output = model(clap_audio, mert_audio, text, alpha=0.0, beta=1.0)
    clap_output = model.clap(clap_audio, text)
    mert_output = model.mert(mert_audio, text)

    torch.testing.assert_close(output["mi"], mert_output["mi"])
    torch.testing.assert_close(output["ta"], clap_output["ta"])


def test_fusion_weight_search_is_deterministic_and_uses_dev_only_arrays():
    first = select_fusion_weight(
        torch.tensor([1.0, 2.0]), torch.tensor([2.0, 1.0]), torch.tensor([1.0, 1.0]), [0.0, 0.5, 1.0]
    )
    second = select_fusion_weight(
        torch.tensor([1.0, 2.0]), torch.tensor([2.0, 1.0]), torch.tensor([1.0, 1.0]), [0.0, 0.5, 1.0]
    )

    assert first == second == 0.5
