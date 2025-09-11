import logging

from agnitra.demo import demo


def test_profile_sample_models_handles_failures(monkeypatch, caplog):
    """profile_sample_models should log warnings and continue on errors."""
    calls = []

    def make_fail(name):
        def fn():
            calls.append(name)
            raise RuntimeError("boom")

        fn.__name__ = name
        return fn

    monkeypatch.setattr(demo, "_profile_llama3", make_fail("_profile_llama3"))
    monkeypatch.setattr(demo, "_profile_whisper", make_fail("_profile_whisper"))
    monkeypatch.setattr(
        demo, "_profile_stable_diffusion", make_fail("_profile_stable_diffusion")
    )

    with caplog.at_level(logging.WARNING):
        result = demo.profile_sample_models()

    assert result == []
    assert caplog.text.count("failed: boom") == 3
    assert calls == [
        "_profile_llama3",
        "_profile_whisper",
        "_profile_stable_diffusion",
    ]


def test_profile_sample_models_reports_success(monkeypatch):
    calls = []

    def make_fn(name):
        def fn():
            calls.append(name)

        fn.__name__ = name
        return fn

    monkeypatch.setattr(
        demo, "_profile_llama3", make_fn("_profile_llama3")
    )
    monkeypatch.setattr(
        demo, "_profile_whisper", make_fn("_profile_whisper")
    )
    monkeypatch.setattr(
        demo, "_profile_stable_diffusion", make_fn("_profile_stable_diffusion")
    )

    result = demo.profile_sample_models()

    assert result == [
        "_profile_llama3",
        "_profile_whisper",
        "_profile_stable_diffusion",
    ]
    assert calls == [
        "_profile_llama3",
        "_profile_whisper",
        "_profile_stable_diffusion",
    ]
