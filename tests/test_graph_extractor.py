import pytest
import torch

from agnitra.core.ir.graph_extractor import GraphIRExtractor, extract_graph_ir


class TinyNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


def _example_telemetry() -> dict:
    return {
        "model": {
            "layers": [
                {
                    "name": "linear",
                    "output_shapes": [[1, 2]],
                    "input_shapes": [[1, 4]],
                    "output_dtype": "float32",
                    "forward_time_ms": 0.12,
                    "cuda_mem_alloc_delta_bytes": 0,
                }
            ]
        },
        "events": [
            {
                "name": "aten::relu",
                "cuda_time_total": 5000.0,
                "cpu_time_total": 7000.0,
                "self_cuda_memory_usage": 128,
            }
        ],
    }


def test_graph_ir_extracts_shapes_and_telemetry():
    model = TinyNet().eval()
    x = torch.randn(1, 4)
    telemetry = _example_telemetry()

    extractor = GraphIRExtractor(telemetry=telemetry, validate=True)
    nodes = extractor.extract(model, example_inputs=(x,))

    linear_node = next(node for node in nodes if node["target"] == "linear")
    assert linear_node["shape"] == [1, 2]
    assert linear_node["cuda_time_ms"] == pytest.approx(0.12, rel=1e-3)
    assert "layer" in linear_node["telemetry_sources"]

    relu_node = next(node for node in nodes if node["op"] == "relu")
    assert relu_node["cuda_time_ms"] == pytest.approx(5.0, rel=1e-3)
    assert "event" in relu_node["telemetry_sources"]


def test_graph_ir_requires_shapes_when_validate():
    model = TinyNet().eval()
    with pytest.raises(ValueError):
        extract_graph_ir(model, telemetry=None, validate=True)


def test_graph_ir_telemetry_mismatch_raises():
    model = TinyNet().eval()
    x = torch.randn(1, 4)
    telemetry = _example_telemetry()
    telemetry["model"]["layers"].append({"name": "unused", "output_shapes": [[1, 1]]})

    with pytest.raises(ValueError, match="Telemetry data provided"):
        extract_graph_ir(model, example_inputs=(x,), telemetry=telemetry, validate=True)

