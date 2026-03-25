from armor_detector.config import DetectorConfig


def test_default_config():
    config = DetectorConfig()
    assert config.red_h_min == 0
    assert config.red_h_max == 10
    assert config.blue_h_min == 100
    assert config.light_min_area == 10


def test_custom_config():
    config = DetectorConfig(light_min_area=100, light_max_area=5000)
    assert config.light_min_area == 100
    assert config.light_max_area == 5000


def test_config_to_dict():
    config = DetectorConfig()
    d = config.to_dict()
    assert isinstance(d, dict)
    assert "light_min_area" in d
