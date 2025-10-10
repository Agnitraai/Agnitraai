from types import SimpleNamespace

from agnitra.core.rl import CodexGuidedAgent


class _DummyResponses:
    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.calls = 0

    def create(self, **kwargs):  # pragma: no cover - simple stub
        self.calls += 1
        return {
            "output": [
                {
                    "content": [
                        {
                            "text": self._payload,
                        }
                    ]
                }
            ]
        }


def test_codex_guided_agent_handles_string_booleans():
    payload = (
        '{"allow_tf32": "false", "flash_sdp": "true", '
        '"torch_compile": "1", "kv_cache_dtype": "BF16"}'
    )
    client = SimpleNamespace(responses=_DummyResponses(payload))
    agent = CodexGuidedAgent()

    config = agent.propose_config([], [], client=client)

    assert config is not None
    assert config["allow_tf32"] is False
    assert config["flash_sdp"] is True
    assert config["torch_compile"] is True
    assert config["kv_cache_dtype"] == "bf16"
