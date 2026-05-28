from pathlib import Path

def test_mvtec_subset_config_exists():
    assert Path("configs/dataset/mvtec_subset.yaml").exists()
