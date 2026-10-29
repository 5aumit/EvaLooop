"""Parallel API behavior, checked without network requests."""

import sys
from threading import Barrier, Event, Lock
from types import SimpleNamespace

import pytest

from evaloop.core.config import EvaluationConfig
from evaloop.models import closed_source


def make_model(monkeypatch, complete, concurrency=2):
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    monkeypatch.setattr(closed_source, "OpenAI", lambda **kwargs: client)
    return closed_source.OpenAILLM({"api_concurrency": concurrency})


def test_parallel_requests_are_bounded_and_results_keep_input_order(monkeypatch):
    barrier, second_finished, lock = Barrier(2, timeout=5), Event(), Lock()
    active, peak = 0, 0
    finished = []

    def complete(**kwargs):
        nonlocal active, peak
        prompt = kwargs["messages"][-1]["content"]
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            barrier.wait()
            if prompt == "0":
                assert second_finished.wait(timeout=5)
            with lock:
                finished.append(prompt)
            if prompt == "1":
                second_finished.set()
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=prompt))]
            )
        finally:
            with lock:
                active -= 1

    model = make_model(monkeypatch, complete)
    prompts = [str(i) for i in range(8)]
    assert model.batch_generate(prompts) == prompts
    assert peak == 2
    assert finished.index("1") < finished.index("0")


def test_one_request_error_does_not_discard_other_responses(monkeypatch):
    def complete(**kwargs):
        prompt = kwargs["messages"][-1]["content"]
        if prompt == "bad":
            raise RuntimeError("simulated request failure")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=prompt))])

    model = make_model(monkeypatch, complete)
    good, bad, other = model.batch_generate(["good", "bad", "other"])
    assert good == "good" and other == "other"
    assert "simulated request failure" in bad
    assert model.batch_generate([]) == []


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_concurrency_fails_before_creating_client(tmp_path, value):
    with pytest.raises(ValueError, match="api_concurrency"):
        EvaluationConfig(model="gpt-example", output_dir=str(tmp_path), api_concurrency=value)
    with pytest.raises(ValueError, match="api_concurrency"):
        closed_source.OpenAILLM({"api_concurrency": value})


def test_api_factory_does_not_import_local_model_dependencies(monkeypatch):
    monkeypatch.setitem(sys.modules, "evaloop.models.open_source", None)
    monkeypatch.setitem(sys.modules, "evaloop.models.open_source_vllm", None)
    from evaloop.models.factory import ModelFactory

    monkeypatch.setattr(closed_source, "OpenAI", lambda **kwargs: object())
    model = ModelFactory().create_model({"type": "openai", "api_concurrency": 4})
    assert model.api_concurrency == 4
