import json

import pytest

from agnitra.core.optimizer import LLMOptimizer


def _sample_graph():
    return {
        "model": "demo",
        "nodes": [
            {
                "name": "matmul_main",
                "op": "matmul",
                "shape": [1024, 1024],
                "cuda_time_ms": 10.2,
            }
        ],
    }


def _sample_telemetry():
    return {
        "events": [
            {
                "op": "matmul",
                "name": "aten::matmul",
                "shape": [1024, 1024],
                "cuda_time_ms": 10.2,
            }
        ]
    }


def test_optimize_uses_fallback_without_client():
    optimizer = LLMOptimizer(client=None)
    result = optimizer.optimize(_sample_graph(), _sample_telemetry(), target_latency_ms=8.0)
    payload = json.loads(result)
    assert payload["source"] == "fallback"
    assert payload["block_size"] == 128
    assert payload["tile_shape"] == [64, 64]
    assert pytest.approx(payload["expected_latency_ms"], rel=1e-3) == 7.6


class _DummyResponses:
    def __init__(self, text):
        self._text = text
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return {
            "output": [
                {
                    "content": [
                        {
                            "text": self._text,
                        }
                    ]
                }
            ]
        }


class _DummyClient:
    def __init__(self, text):
        self.responses = _DummyResponses(text)


def test_optimize_parses_json_response():
    client = _DummyClient(
        json.dumps(
            {
                "block_size": 256,
                "tile_shape": [64, 128],
                "unroll_factor": 4,
                "target_latency_ms": 7.5,
                "expected_latency_ms": 7.2,
                "rationale": "Double buffering reduces stalls.",
            }
        )
    )
    optimizer = LLMOptimizer(client=client)
    result = optimizer.optimize(_sample_graph(), _sample_telemetry())
    payload = json.loads(result)
    assert payload["block_size"] == 256
    assert payload["tile_shape"] == [64, 128]
    assert payload["unroll_factor"] == 4
    assert "Double buffering" in payload["rationale"]
    user_prompt = client.responses.last_kwargs["input"][1]["content"][0]["text"]
    assert "1024" in user_prompt
    assert "10.2" in user_prompt


def test_optimize_parses_key_value_text():
    text = (
        "Block size: 192\n"
        "Tile Shape: 32 x 64\n"
        "Unroll factor: 3\n"
        "Target latency: 7.1 ms\n"
        "Expected latency: 6.8 ms\n"
        "Rationale: balance occupancy and memory reuse"
    )
    client = _DummyClient(text)
    optimizer = LLMOptimizer(client=client)
    result = optimizer.optimize(_sample_graph(), _sample_telemetry())
    payload = json.loads(result)
    assert payload["block_size"] == 192
    assert payload["tile_shape"] == [32, 64]
    assert payload["unroll_factor"] == 3
    assert pytest.approx(payload["target_latency_ms"], rel=1e-3) == 7.1
    assert "occupancy" in payload["rationale"]
