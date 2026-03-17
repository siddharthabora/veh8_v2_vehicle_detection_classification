from src.utils.config_loader import load_yaml

def test_load_counter_config():
    cfg = load_yaml("configs/counter.yaml")
    assert isinstance(cfg, dict)
    assert "line_frac" in cfg
    assert "direction" in cfg