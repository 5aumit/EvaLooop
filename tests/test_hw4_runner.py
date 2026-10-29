"""Check the four-run launcher without API requests or generated-code execution."""

import csv
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner(monkeypatch):
    calls = []

    class Evaluator:
        def __init__(self, config):
            self.config = config

        def run_code_generation_summarization(self):
            calls.append(self.config)
            return [{"model": "offline-model", "prompt_results": [{}, {}]}]

    class Analyzer:
        def __init__(self, results_path, output_dir):
            self.path = results_path

        def analyze(self, metrics):
            assert metrics == ["ASL_pow", "pass@1", "pass_drop"]
            return {key: {"by_model": {"offline-model": value}} for key, value in
                    [("ASL_pow", 0.2), ("pass@1", 0.5), ("pass_drop", 0.5)]}

    monkeypatch.setitem(sys.modules, "evaloop.core.evaluator", SimpleNamespace(EvaLoopEvaluator=Evaluator))
    monkeypatch.setitem(sys.modules, "evaloop.analysis.analyzer", SimpleNamespace(ResultAnalyzer=Analyzer))
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-key")
    source = Path(__file__).parents[1] / "run_hw4.py"
    spec = importlib.util.spec_from_file_location("hw4_runner", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls


def test_runs_all_configurations_and_writes_summary(tmp_path, monkeypatch, runner):
    module, calls = runner
    monkeypatch.setattr(sys, "argv", ["run_hw4.py", "--output-dir", str(tmp_path)])
    module.main()
    assert [(c.model, c.loop_type) for c in calls] == [
        (model, loop) for model in module.MODELS for loop in module.LOOPS
    ]
    assert all(c.api_concurrency == 4 and c.max_cycles == 10 for c in calls)
    assert all(c.temperature == 0 and c.top_p == 1 and c.max_new_tokens == 1024 for c in calls)
    assert all(c.reasoning_enabled is False for c in calls if c.model == "qwen/qwen3.7-flash")
    assert all(c.reasoning_enabled is None for c in calls if c.model.startswith("meta-llama/"))
    with (tmp_path / "summary.csv").open() as summary:
        rows = list(csv.DictReader(summary))
    assert len(rows) == 4
    assert all(row["success_rate"] == "0.5" and row["asl_score"] == "0.2" for row in rows)
    with pytest.raises(SystemExit):
        module.main()
    assert len(calls) == 4  # Refuse a repeat before any additional paid work.


def test_check_mode_does_not_evaluate_or_write_results(tmp_path, monkeypatch, runner):
    module, calls = runner
    monkeypatch.setattr(sys, "argv", ["run_hw4.py", "--check", "--output-dir", str(tmp_path)])
    module.main()
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_selected_configuration_only(tmp_path, monkeypatch, runner):
    module, calls = runner
    monkeypatch.setattr(sys, "argv", [
        "run_hw4.py", "--output-dir", str(tmp_path),
        "--models", "qwen/qwen3.7-flash", "--loops", "pseudocode",
    ])
    module.main()
    assert [(config.model, config.loop_type) for config in calls] == [
        ("qwen/qwen3.7-flash", "pseudocode")
    ]
