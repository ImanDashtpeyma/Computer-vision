from src.config import VOC_CLASSES, AppConfig, ModelConfig


def test_default_app_config_uses_cpu_backend():
    config = AppConfig()
    assert config.model.backend == "cpu"


def test_voc_classes_includes_person_and_background():
    assert "person" in VOC_CLASSES
    assert VOC_CLASSES[0] == "background"


def test_model_config_defaults_are_consistent_with_mobilenet_ssd():
    model = ModelConfig()
    assert model.input_size == 300
    assert 0.0 < model.confidence_threshold < 1.0
    assert len(model.classes) == len(VOC_CLASSES)
