
from agnitra.core.telemetry.collector import collect_events


class DummyEvent:
    """Simple event without CUDA fields to mimic CPU-only profiling."""

    def __init__(self, key: str, cpu_time_total: float = 0.0):
        self.key = key
        self.cpu_time_total = cpu_time_total


def test_cpu_only_events_do_not_raise():
    events = [DummyEvent("a", 1.0), DummyEvent("b", 2.0)]

    # Should not raise even though events lack CUDA attributes
    collected = collect_events(events)

    assert collected == [
        {
            "key": "a",
            "cpu_time_total": 1.0,
            "cuda_time_total": 0.0,
            "self_cuda_memory_usage": 0,
        },
        {
            "key": "b",
            "cpu_time_total": 2.0,
            "cuda_time_total": 0.0,
            "self_cuda_memory_usage": 0,
        },
    ]
