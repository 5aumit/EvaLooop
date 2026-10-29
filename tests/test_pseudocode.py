"""Offline checks of the pseudocode loop, including stop and save behavior."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from evaloop.core.config import EvaluationConfig
from evaloop.evaluation.cycle import BatchEvaluationCycle
from evaloop.tasks import CodeToPseudocodeTask, TaskFactory

from conftest import ASSERT_LINE, MBPP_STYLE_PROMPT


CODE = "def similar_elements(a, b):\n    return tuple(set(a) & set(b))"
PSEUDOCODE = (
    "FUNCTION similar_elements(a, b)\nINPUTS: two collections\n"
    "OUTPUT: their shared values\nSTEPS:\n"
    "  1. Find the values present in both collections.\n"
    "  2. Return those values as a tuple.\nEND FUNCTION"
)


class RecordingModel:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if prompt.startswith("Generate Python code"):
            return f"```python\n{CODE}\n```"
        return PSEUDOCODE


class BatchModel(RecordingModel):
    def batch_generate(self, prompts):
        return [self.generate(prompt) for prompt in prompts]


def test_first_generation_and_standard_summary_are_unchanged(tmp_path):
    standard = EvaluationConfig(model="gpt-example", output_dir=str(tmp_path)).get_task_configs()
    extension = EvaluationConfig(
        model="gpt-example", output_dir=str(tmp_path), loop_type="pseudocode"
    ).get_task_configs()
    assert extension[0] == standard[0]
    assert standard[1]["type"] == "code_summarization"
    assert "Use one sentence" in standard[1]["prompt_template"]
    assert isinstance(TaskFactory().create_task(extension[1]), CodeToPseudocodeTask)


def test_invalid_loop_rejected(tmp_path):
    with pytest.raises(ValueError, match="loop_type"):
        EvaluationConfig(model="gpt-example", loop_type="typo", output_dir=str(tmp_path))


@pytest.mark.parametrize("model_class", [RecordingModel, BatchModel])
def test_pseudocode_and_original_assertion_feed_next_cycle(tmp_path, model_class):
    config = EvaluationConfig(model="gpt-example", loop_type="pseudocode", output_dir=str(tmp_path))
    tasks = [TaskFactory().create_task(task) for task in config.get_task_configs()]
    model = model_class()
    runner = BatchEvaluationCycle(model, tasks)
    prompts = [{"task_id": f"Mbpp/{i}", "prompt": MBPP_STYLE_PROMPT} for i in range(2)]
    for cycle in (1, 2):
        results = runner.run_single_cycle(prompts, cycle)["cycle_results"]
        prompts = []
        for result in results:
            assert result["task_1_output"] == CODE
            wrapped = result["task_2_output"]
            assert wrapped == f'"""\n{PSEUDOCODE}\n{ASSERT_LINE}\n"""\n'
            prompts.append({"task_id": result["task_id"], "prompt": wrapped})
    generation_prompts = [p for p in model.prompts if p.startswith("Generate Python code")]
    assert PSEUDOCODE in generation_prompts[2]
    assert "Write a function to find" not in generation_prompts[2]
    pseudocode_prompts = [p for p in model.prompts if p.startswith("Convert the following")]
    assert CODE in pseudocode_prompts[0]
    assert ASSERT_LINE not in pseudocode_prompts[0]


@pytest.mark.parametrize("loop_type", ["standard", "pseudocode"])
@pytest.mark.parametrize("fails_second_cycle", [False, True])
def test_evaluator_stops_counts_and_saves(tmp_path, monkeypatch, loop_type, fails_second_cycle):
    # Load the real evaluator with only the external model/data dependencies
    # replaced. No model requests, dataset downloads, or generated-code execution.
    monkeypatch.setitem(sys.modules, "evaloop.models.factory", SimpleNamespace(ModelFactory=lambda: None))
    monkeypatch.setitem(
        sys.modules, "evaloop.evaluation.py_testing",
        SimpleNamespace(EvalPlusCodeTester=object, load_mbpp_dataset=lambda: None),
    )
    source = Path(__file__).parents[1] / "evaloop/core/evaluator.py"
    spec = importlib.util.spec_from_file_location("evaloop.core._offline_evaluator", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = EvaluationConfig(
        model="gpt-example", loop_type=loop_type, output_dir=str(tmp_path),
        experiment_name="comparison", max_cycles=10 if fails_second_cycle else 2,
    )
    evaluator = module.EvaLoopEvaluator(config)
    model = RecordingModel()
    prompts = [{"task_id": "Mbpp/2", "prompt": MBPP_STYLE_PROMPT}]
    monkeypatch.setattr(evaluator, "_create_model", lambda: model)
    monkeypatch.setattr(evaluator, "_load_dataset", lambda: (prompts, {"Mbpp/2": {"entry_point": "similar_elements"}}))

    class Tester:
        calls = 0

        def test_code(self, code, task_id, entry_point):
            self.calls += 1
            return {"success": not fails_second_cycle or self.calls == 1, "plus_details": []}

    tester = Tester()
    monkeypatch.setattr(evaluator, "_create_tester", lambda _: tester)
    results = evaluator.run_code_generation_summarization()
    task = results[0]["prompt_results"][0]
    assert tester.calls == 2
    assert len(task["cycles"]) == 2
    assert task["successful_cycles"] == (1 if fails_second_cycle else 2)
    assert task["max_cycles_reached"] == (not fails_second_cycle)
    assert results[0]["loop_type"] == loop_type
    suffix = "pseudocode" if loop_type == "pseudocode" else "summarization"
    saved = tmp_path / f"comparison_code_generation_{suffix}_results.json"
    assert json.loads(saved.read_text()) == results
