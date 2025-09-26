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
    best = payload["best_model"]
    assert best["model"] == "heuristic-fallback"
    suggestion = best["suggestion"]
    assert suggestion["source"] == "fallback"
    assert suggestion["block_size"] == 128
    assert suggestion["tile_shape"] == [64, 64]
    assert pytest.approx(suggestion["expected_latency_ms"], rel=1e-3) == 7.6
    models = {entry["model"]: entry for entry in payload["models"]}
    assert models["gpt-4.1-mini"]["status"] == "skipped"
    assert models["gpt-5-mini-2025-08-07"]["status"] == "skipped"
    assert models["heuristic-fallback"]["status"] == "fallback"


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
    best = payload["best_model"]
    assert best["model"] == "gpt-4.1-mini"
    suggestion = best["suggestion"]
    assert suggestion["block_size"] == 256
    assert suggestion["tile_shape"] == [64, 128]
    assert suggestion["unroll_factor"] == 4
    assert "Double buffering" in suggestion["rationale"]
    models = {entry["model"]: entry for entry in payload["models"]}
    assert models["gpt-4.1-mini"]["status"] == "ok"
    assert models["gpt-5-mini-2025-08-07"]["status"] == "ok"
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
    best = payload["best_model"]
    suggestion = best["suggestion"]
    assert suggestion["block_size"] == 192
    assert suggestion["tile_shape"] == [32, 64]
    assert suggestion["unroll_factor"] == 3
    assert pytest.approx(suggestion["target_latency_ms"], rel=1e-3) == 7.1
    assert "occupancy" in suggestion["rationale"]


class _FlakyResponses:
    def __init__(self, first_exc: Exception, text: str):
        self._first_exc = first_exc
        self._text = text
        self.calls = 0
        self.last_kwargs = None

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise self._first_exc
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


class _FlakyClient:
    def __init__(self, first_exc: Exception, text: str):
        self.responses = _FlakyResponses(first_exc, text)


def test_optimize_falls_back_to_secondary_model():
    client = _FlakyClient(RuntimeError("primary unavailable"), json.dumps({"block_size": 320}))
    optimizer = LLMOptimizer(client=client)
    result = optimizer.optimize(_sample_graph(), _sample_telemetry())
    payload = json.loads(result)
    best = payload["best_model"]
    assert best["model"] == "gpt-5-mini-2025-08-07"
    assert best["suggestion"]["block_size"] == 320
    models = {entry["model"]: entry for entry in payload["models"]}
    assert models["gpt-4.1-mini"]["status"] == "error"
    assert models["gpt-5-mini-2025-08-07"]["status"] == "ok"
    assert client.responses.calls == 2
    assert client.responses.last_kwargs["model"] == "gpt-5-mini-2025-08-07"
