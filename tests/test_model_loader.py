from agnitra._sdk import model_loader as ml


def test_quantized_selected_on_low_memory(monkeypatch):
    monkeypatch.setattr(ml, "available_vram_gb", lambda: 8.0)
    spec = ml.select_model(quantize=False)
    assert spec.kwargs.get("load_in_4bit") is True
    assert "6GB" in spec.expected_memory
