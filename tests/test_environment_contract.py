from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "project.yaml"


def load_config():
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_required_dataset_and_split_contract():
    config = load_config()
    dataset = config["dataset"]
    assert dataset["expected_total"] == 2748
    assert dataset["split_counts"] == {"train": 1923, "dev": 412, "test": 413}
    assert dataset["sample_rate_hz"] == 16000
    assert dataset["channels"] == 1


def test_primary_segment_and_temperature_contract():
    config = load_config()
    assert config["segment"]["primary_window_seconds"] == 11.0
    assert config["segment"]["primary_overlap"] == 0.5
    assert config["temperature"]["grid"] == [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    assert config["statistics"]["bootstrap_resamples"] == 2000


def test_configured_paths_are_portable():
    config = load_config()
    for value in (
        config["project"]["cache_root"],
        config["project"]["results_root"],
        config["project"]["logs_root"],
        config["outputs"]["manifest_file"],
    ):
        assert not Path(value).is_absolute()
        assert not value.startswith(("D:\\", "C:\\Users"))
