# EvaLooop: LLM Robustness Evaluation Framework

CSC 591 HW4 extension: [public fork](https://github.com/5aumit/EvaLooop/tree/hw4-pseudocode).
The extension is based on upstream commit `ce22fe2f80c8f36da85cc8a0cca49c2bcc8b1cb9`.
`constraints-hw4.txt` records the API experiment's installed dependency versions.
The Docker image uses these versions and does not install the unused GPU runtime.

[![arXiv](https://img.shields.io/badge/arXiv-2505.12185-b31b1b.svg)](https://arxiv.org/abs/2505.12185)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Leaderboard](https://img.shields.io/badge/leaderboard-evalooop.github.io-blue)](https://evalooop.github.io/)

EvaLoop is a framework for evaluating the robustness of Large Language Models (LLMs) through iterative dual-task cycles. The framework measures how many cycles an LLM can execute before generating code that fails functional testing, providing a quantitative metric for model robustness. It accompanies the paper [*EvaLoop: Assessing LLM Robustness in Programming from a Self-consistency Perspective*](https://arxiv.org/abs/2505.12185); results across models are published on the [leaderboard](https://evalooop.github.io/).

## 🚀 Quick Start

### Installation

EvaLoop is not yet published on PyPI — install from source:

```bash
git clone https://github.com/evalooop/EvaLooop
cd EvaLooop
pip install -e .
```

Optional extras:

```bash
pip install -e ".[vllm]"   # GPU batch inference via vLLM (Linux + CUDA only)
pip install -e ".[dev]"    # development tools (pytest, ruff, ...)
```

After installing, check your environment:

```bash
evaloop validate_setup                    # full check
evaloop validate_setup --check_gpu=False  # skip GPU checks (e.g. on a laptop)
```

### Basic Usage

```bash
# Evaluate an OpenAI model on the generation<->summarization loop
evaloop evaluate --model "gpt-4"

# Evaluate an open-source model with multi-GPU vLLM
evaloop evaluate --model "Qwen/Qwen2.5-Coder-32B-Instruct" \
                 --gpu_ids "0,1,2,3" \
                 --tensor_parallel_size 4 \
                 --max_cycles 15

# Analyze results (the default ASL metric calls an LLM judge; see "Scoring")
evaloop analyze --results_path "results/experiment_results.json" \
                --judge_api_key_file api.key
```

### Run the full HW4 comparison with Docker

From this repository directory, with `OPENROUTER_API_KEY` in `../.env`:

```bash
docker compose build
docker compose run --rm benchmark
```

The second command starts paid API calls and runs generated code inside the
container. It evaluates all 378 MBPP Plus tasks using both
`meta-llama/llama-3.1-8b-instruct` and `qwen/qwen3.7-flash`, each with the standard
and pseudocode loops. Settings are temperature 0, top-p 1, 10 cycles, 1024 maximum
new tokens, and four simultaneous API requests per run. The four configurations
run sequentially. No GPU is required.

Both Qwen configurations explicitly disable reasoning with OpenRouter's
`reasoning: {enabled: false}` request option. A diagnostic with default reasoning
used all 1024 output tokens for reasoning and returned no answer text. The
explicit setting reserves that limit for answer generation and is recorded as
`reasoning_enabled` in results. Llama keeps its default behavior. For manual CLI
runs, use `--reasoning_enabled=False` with `--provider openrouter`.

`run_hw4.py` contains the model IDs and settings. `Dockerfile` installs the API,
testing, and analysis dependencies. `compose.yaml` loads the key at runtime and
mounts only `./results`; `.dockerignore` excludes credentials and other local
files from the image build.

Results, per-run logs, analysis plots, and `summary.csv` appear in `results/`.
Analysis reports `ASL_pow` as `asl_score`, matching the supplied package's
`aslScore`, without calling a similarity judge.

To check imports without paid calls:

```bash
docker compose run --rm benchmark --check
```

Press Ctrl+C to stop. Each configuration saves its JSON only when it finishes;
an interrupted configuration cannot resume. The runner refuses to overwrite
existing results. To start a separate comparison, use an empty subdirectory:

```bash
docker compose run --rm benchmark --output-dir /results/another-run
```

To run individual configurations in separate containers, select one model and
loop and give each container a separate output directory. For example:

```bash
docker compose run -d --rm --name hw4-qwen-pseudocode benchmark \
  --models qwen/qwen3.7-flash --loops pseudocode \
  --output-dir /results/parallel/qwen-pseudocode
```

Each container uses up to four API requests, four CPUs, and 4 GB of memory.
Do not also schedule those configurations in the default four-run launcher.

### Reproduce the completed analysis without API calls

The submission ZIP contains the four completed results files, `failure_annotations.json`,
and a top-level `summary.csv`. The supplied course data and aborted runs are excluded.
`analyze_hw4.py` validates identical task sets, stop rules, assertion preservation,
and the Qwen reasoning setting before recomputing the numbers.

From an unpacked submission directory containing `results/` and the analysis script:

```bash
python analyze_hw4.py --results-dir results --output-dir reproduced-analysis \
  --annotations failure_annotations.json
```

Alternatively, build the fork's Docker image and run its analysis entry point with
an unpacked submission mounted at `/submission`. This needs no API key or network:

```bash
docker run --rm --network none --entrypoint python \
  --mount "type=bind,source=$PWD,target=/submission" hw4-eval \
  /app/analyze_hw4.py --results-dir /submission/results \
  --output-dir /submission/reproduced-analysis \
  --annotations /submission/failure_annotations.json
```

The output includes `summary.csv`, `metrics.json`, `survival.csv`,
`paired_comparison.csv`, `survival_comparison.png`, `survival_comparison.pdf`,
and the traceable failure-review worksheet. The labels in the annotation file
are AI-assisted static code-reading notes for author review, not outputs of an
independent judge. No semantic-similarity judge was used in the reported scores.

### OpenRouter

Use `--provider openrouter` to send evaluations through OpenRouter instead of
selecting a local model from its name. Keep the full OpenRouter model ID.

Store `OPENROUTER_API_KEY` in your environment or `.env`. The command searches
for `.env` from the working directory through its parents. Exported variables
take priority over file values. To select a specific file, use `--env_file`.

From the repository directory, with credentials in `../.env`:

```bash
evaloop evaluate --provider openrouter --model "qwen/qwen-2.5-coder-32b-instruct" \
                --env_file ../.env --temperature 0 --max_cycles 10 \
                --max_new_tokens 1024
```

This command performs a paid evaluation and executes generated code. Run it in
a restricted environment, as required for all evaluations. Model availability
depends on your OpenRouter account.

Changes to the existing framework: `evaloop/cli.py` exposes the provider and
environment-file options. `evaloop/core/config.py` loads credentials and selects
the existing OpenAI-compatible client with OpenRouter's endpoint. OpenRouter
uses `max_new_tokens` for the response limit. Both dependency lists now include
`python-dotenv`. The standard loop, tests, and metrics are unchanged.

### Parallel API requests

Use `--api_concurrency 4` to send up to four requests at once per model.
The default is 1. The limit must be a positive integer. For example:

```bash
evaloop evaluate --provider openrouter --model "meta-llama/llama-3.1-8b-instruct" \
                --env_file ../.env --loop_type pseudocode --api_concurrency 4 \
                --temperature 0 --max_cycles 10 --max_new_tokens 1024
```

This is a paid full-dataset run. Parallelism reduces waiting time, not token
cost. Each batch runs independent requests in a thread pool and returns results
in prompt order. The cycle runner generates code across tasks, then descriptions,
then tests the code. Later cycles only include passing tasks. Requests retain
the existing per-request retries and error handling. The limit is per model;
separate evaluation processes each have their own limit.

`evaloop/models/closed_source.py` implements parallel requests.
`evaloop/core/config.py` and `evaloop/cli.py` expose the limit.
`evaloop/models/factory.py` now imports local-model libraries only when selecting
a local model. API models do not require a GPU or vLLM at runtime; functional
testing still needs a restricted CPU environment. The package's existing full
dependency list still includes local-model packages.

### Pseudocode loop (HW4 extension)

Research question: does structured pseudocode preserve task behavior across
cycles better than prose summaries?

Select `--loop_type standard` (the default) or `--loop_type pseudocode`:

```text
standard:   original task -> Python code -> prose summary -> Python code -> ...
pseudocode: original task -> Python code -> pseudocode    -> Python code -> ...
```

Both loops use the same first generation prompt, model, original MBPP Plus
tests, and assertion wrapper. Only the code-to-description prompt changes.
The exact extension prompt is `PSEUDOCODE_PROMPT` in
`evaloop/tasks/code_to_pseudocode.py`. It requests FUNCTION, INPUTS, OUTPUT,
STEPS, and END FUNCTION, with plain-English steps that preserve actual behavior.
The same model performs both steps, with a fresh context for each call.

The full pseudocode response is stripped of surrounding whitespace and wrapped
with the original example assertion. No judge, repair model, or external
perturbation is introduced. Formatting is requested but not enforced: malformed
responses remain in the saved records for failure analysis. The function name
and calling pattern remain anchored by the same assertion used by the standard
loop. Both original and extended tests must pass to continue. Testing failure
stops the task, or the task completes after the configured cycle limit. The
current runner produces the description before testing but only carries it
into another cycle when tests pass.

To compare the configurations, run each with the same model and settings in a
restricted environment. From the repository directory:

```bash
evaloop evaluate --provider openrouter --model "qwen/qwen-2.5-coder-32b-instruct" \
                --env_file ../.env --loop_type standard --temperature 0 \
                --max_cycles 10 --max_new_tokens 1024 --experiment_name comparison
evaloop evaluate --provider openrouter --model "qwen/qwen-2.5-coder-32b-instruct" \
                --env_file ../.env --loop_type pseudocode --temperature 0 \
                --max_cycles 10 --max_new_tokens 1024 --experiment_name comparison
```

These commands run the full MBPP Plus dataset and incur API costs. Results
include `loop_type`; filenames end in `code_generation_summarization_results.json`
or `code_generation_pseudocode_results.json`. Existing analysis accepts both.
Use `--metrics "ASL_pow,pass@1,pass_drop"` to analyze without a similarity judge;
the supplied data package's `aslScore` corresponds to `ASL_pow`.

Changes to the existing framework: `evaloop/tasks/__init__.py` registers the new
task, `evaloop/core/config.py` selects it, `evaloop/cli.py` exposes `--loop_type`,
and `evaloop/core/evaluator.py` labels and saves each configuration separately.
The cycle runner, functional tests, generation prompt, and scoring formulas
are unchanged. The extension changes both structure and requested detail, so
an observed difference cannot be attributed to structure alone.

Offline checks: `python -m pytest tests/test_pseudocode.py tests/test_openrouter.py`.
These use simulated model responses and test outcomes. They establish loop
wiring and stopping behavior, not robustness results from evaluated models.

## 🔄 How the Loop Works

Each evaluation cycle alternates between two tasks:

1. **Initial input**: a natural-language prompt describing a coding task (from MBPP+), formatted as a docstring whose final line is an `assert` statement.
2. **Code generation**: the model generates code from the prompt.
3. **Functional testing**: the generated code is executed against the dataset's test cases ([EvalPlus](https://github.com/evalplus/evalplus) MBPP+).
4. **Code summarization**: the model summarizes the generated code back into a natural-language description.
5. **Iteration**: the summary becomes the next cycle's prompt; the loop repeats until the generated code fails testing or `--max_cycles` is reached.

The number of loops each task survives is aggregated into the **ASL (Average Sustainable Loops)** score, which weights long survival quadratically and discounts semantic drift at the failure boundary; see [Scoring](#-scoring).

### Design note: the retained assert line

The summarization step deliberately re-attaches the final `assert` line from the prompt the cycle *started* with, so each new prompt looks like:

```
"""
<model-generated summary>
assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)
"""
```

This is intentional, not an artifact: keeping one assert anchors the function name and signature across cycles, so the loop measures **semantic drift** in the model's understanding rather than trivial failures from a renamed function. You will see this line in every generated prompt in the result data. The implementation lives in `CodeSummarizationTask.postprocess` (`evaloop/tasks/code_summarization.py`).

### Supported Task Types

| Task type | Config name | Status |
|---|---|---|
| Code generation | `code_generation` | ✅ Supported |
| Code summarization | `code_summarization` | ✅ Supported |
| Code translation | `code_translation` | 🧪 Experimental — the task class exists, but config/evaluator support is not yet in the public release (`run_code_translation()` raises `NotImplementedError`) |

Want to add your own loop? A new task type is one new file plus a registry entry — no changes to the loop runner. See the [Contributing Guide](CONTRIBUTING.md).

## 🛠️ CLI Reference

### `evaluate` — Code Generation ↔ Summarization

```bash
evaloop evaluate --model MODEL [OPTIONS]
```

**Key options:**
- `--model`: A single model name or HuggingFace path (required). Examples: `"gpt-4"`, `"Qwen/Qwen2.5-Coder-32B-Instruct"`
- `--dataset`: Dataset to use. Currently only `mbpp_plus` is supported (HumanEval is planned).
- `--max_cycles`: Maximum evaluation cycles (default: 10)
- `--temperature`: Generation temperature (default: 0.0)
- `--output_dir`: Results output directory (default: `results/`)
- `--gpu_ids`: GPU IDs for vLLM models (e.g., `"0,1,2,3"`)
- `--tensor_parallel_size`: GPUs for tensor parallelism (default: 1)

**Examples:**

```bash
# Basic OpenAI evaluation
evaloop evaluate --model "gpt-4"

# Large model with multi-GPU setup
evaloop evaluate --model "meta-llama/Llama-3-70b-instruct-hf" \
                 --gpu_ids "0,1,2,3" \
                 --tensor_parallel_size 4 \
                 --gpu_memory_utilization 0.85

# Custom experiment parameters
evaloop evaluate --model "deepseek-coder-33b" \
                 --max_cycles 20 \
                 --temperature 0.2 \
                 --experiment_name "deepseek_robust_eval"
```

To evaluate several models, run `evaloop evaluate` once per model (see [Batch Processing](#batch-processing)).

### `analyze` — Result Analysis

```bash
evaloop analyze --results_path "results/experiment_results.json" [OPTIONS]
```

**Key options:**
- `--results_path`: Path to results JSON file (required)
- `--metrics`: Metrics to compute (default: `"ASL,ASL_pow,ASL_bias,pass@1,pass_drop"`, see [Scoring](#-scoring))
- `--generate_plots`: Generate visualization plots (default: True)
- `--output_dir`: Analysis output directory (default: `<results dir>/analysis`)
- `--max_cycles`: Loop budget M of the run (default: value recorded in the results file; for older
  files, the value implied by tasks flagged `max_cycles_reached`; else 10 with a warning)

**Similarity judge options** (only used by the `ASL` metric, and only for pairs not cached yet):
- `--judge_model`: Judge model id (default: `gpt-4-turbo-2024-04-09`, the paper's judge)
- `--judge_api_key` / `--judge_api_key_file`: Judge API key (default: `OPENAI_API_KEY`)
- `--judge_base_url`: Any OpenAI-compatible endpoint, e.g. a local `vllm serve` server (default: OpenAI)
- `--judge_max_tokens`: Answer token budget (default: 10; raise it for reasoning models)
- `--judge_temperature`: Judge temperature (default: 0.0)
- `--judge_system_prompt`: Optional system message (default: none)
- `--judge_prompt_file`: Custom prompt template with `{prompt1}`, `{code1}`, `{prompt2}`, `{code2}`
- `--judge_delay`: Seconds between judge calls (default: 0)
- `--similarity_path`: Similarity sidecar to read and extend (default: next to the results file)
- `--rejudge_failed`: Retry pairs whose previous judge call failed (default: True)

**Examples:**

```bash
# Default analysis: all metrics, paper judge (needs an OpenAI key the first time)
evaloop analyze --results_path "results/my_experiment_results.json" --judge_api_key_file api.key

# Use a self-hosted judge through vLLM's OpenAI-compatible server
vllm serve Qwen/Qwen2.5-72B-Instruct --port 8000
evaloop analyze --results_path "results/exp.json" \
                 --judge_model "Qwen/Qwen2.5-72B-Instruct" \
                 --judge_base_url "http://localhost:8000/v1"

# Skip the judge: metrics that only need the loop counts
evaloop analyze --results_path "results/exp.json" \
                 --metrics "ASL_pow,ASL_bias,pass@1,pass_drop" \
                 --output_dir "analysis/custom/"
```

**Outputs** (in `--output_dir`): `analysis_results.json` (all metric values per model, the loop
histogram `n_i`, and judge statistics such as the bucket weights `s_i`), `analysis_summary.txt`,
and the plots. Judge scores go to the similarity sidecar next to the results file.

### `list_models` — Available Models

```bash
evaloop list_models
```

### `validate_setup` — System Validation

```bash
evaloop validate_setup [--check_gpu=False] [--check_api_keys=False]
```

## 🔧 Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # Optional
```

### Model Configuration

EvaLoop automatically configures models based on their names/paths:

- **OpenAI models**: `gpt-4`, `gpt-3.5-turbo`, `gpt-4-turbo`
- **HuggingFace paths**: `meta-llama/Llama-3-70b-instruct-hf`
- **Pre-registered names**: use `evaloop list_models` to see available options

Large models automatically use vLLM for efficient inference when it is installed (`pip install -e ".[vllm]"`).

## 📊 Results and Analysis

### Output Structure

Results are saved in JSON format:

```json
{
  "model": "gpt-4",
  "max_cycles": 10,
  "prompt_results": [
    {
      "task_id": "Mbpp/2",
      "initial_prompt": "Write a function to find the shared elements ...",
      "cycles": [...],
      "successful_cycles": 5,
      "max_cycles_reached": false
    }
  ],
  "average_successful_cycles": 4.2
}
```

### Metrics

`evaloop analyze` reports `ASL` (the headline score, used by default), `ASL_pow`, `ASL_bias`,
`pass@1` and `pass_drop`. How each one is computed is described in [Scoring](#-scoring).

### Visualization

`evaloop analyze` writes three plots to the output directory: `asl_comparison.png` (ASL per
model), `cycle_distribution.png` (histogram of sustained loops) and `success_heatmap.png`
(sustained loops per task and model, first 50 tasks).

## 📐 Scoring

The scores below are the ones used in the paper and on the [leaderboard](https://evalooop.github.io/);
`evaloop analyze` reproduces the published values exactly when given the same results and judge scores.

### Notation

- **T**: number of tasks in the run (378 for MBPP Plus).
- **M**: maximum number of loops (`max_cycles`, 10 by default).
- **i**: number of loops a task *sustains* (`successful_cycles` in the results file): its code passed
  the tests in loops 1..i and failed in loop i+1. `i = 0` means it already failed in the first loop,
  `i = M` means it never failed.
- **n_i**: number of tasks that sustain exactly i loops.
- **pass@k**: number of tasks that sustain at least k loops (so pass@1 is the usual one-shot pass count).

### ASL: the headline score (default)

```
ASL = Σ_{i=1..M} n_i · i² · s_i / (T · M)          range [0, M]
```

- **i²** rewards sustained correctness quadratically: surviving 8 loops counts 4× as much as surviving 4.
- **s_i** is the semantic similarity weight of bucket i. It discounts tasks whose last successful
  loop had already drifted away from the original intent.

**How s_i is obtained.** For every task with a failure boundary (`1 ≤ i < M`), an LLM judge
compares the last successful transition with the failing one:

| | Prompt | Code |
|---|---|---|
| Side 1 | prompt that fed loop i (the original task if i = 1, else the summary written in loop i−1) | code generated in loop i (passed) |
| Side 2 | summary written in loop i, i.e. the prompt of loop i+1 | code generated in loop i+1 (failed) |

The judge returns a score `sim ∈ [0, 1]`. A task that sustained i loops went through i prompt
transitions; the first i−1 produced passing code, so they count as similarity 1.0, and only the
boundary transition uses the judge score. With `m_i` the mean judge score of the tasks in bucket i:

```
s_i = (m_i + (i − 1)) / i        for 1 ≤ i < M
s_i = 1                          for i = M (never failed) and for buckets without judge scores
```

Because the mean is linear, this equals giving every boundary task the weight `i · (i − 1 + sim)`
instead of `i²`, so `ASL ≤ ASL_pow` and the gap is `Σ_tasks i · (1 − sim) / (T · M)`.

*Example:* with T = 378 and M = 10, a task that sustains 4 loops and gets a judge score of 0.85
contributes `4 · (3 + 0.85) / 3780 = 0.00407` to ASL, versus `16 / 3780 = 0.00423` without the
similarity weighting.

### All metrics

| Metric | Formula | Range | What it measures | Leaderboard field |
|---|---|---|---|---|
| `ASL` | Σ n_i · i² · s_i / (T · M) | [0, M] | Headline robustness score (default) | `semanticSimilarityScore`, shown as "ASL" |
| `ASL_pow` | Σ n_i · i² / (T · M) | [0, M] | ASL without semantic weighting (all s_i = 1) | `aslScore` |
| `ASL_bias` | Σ n_i · i² / (pass@1 · M) | [0, M] | Robustness on the tasks the model solves at all, independent of one-shot accuracy (0 if pass@1 = 0) | `robustnessScore` |
| `pass@1` | pass@1 / T | [0, 1] | One-shot accuracy | `successRate` |
| `pass_drop` | (pass@1 − pass@M) / T | [0, 1] | Absolute accuracy lost within M loops | "drop" (as a percentage) |

Only `ASL` needs the judge; the other metrics use the loop counts alone. The leaderboard also
averages repeated runs of a model and only lists models with `pass@1 > 0.2` and `ASL_bias > 3`;
those steps happen outside this package.

### Similarity judge

- **Default:** `gpt-4-turbo-2024-04-09` at temperature 0 with a 10-token answer budget, no system
  message and the paper's prompt (`evaloop.analysis.similarity.DEFAULT_JUDGE_PROMPT`). Scores
  from a different judge or prompt are not comparable with the leaderboard, so compare models
  only under the same judge.
- **Changing the judge:** the judge is served through the same API layer as the evaluated models
  (`OpenAILLM`), so any OpenAI-compatible endpoint works via `--judge_model`, `--judge_base_url`
  and `--judge_api_key(_file)`. In Python, pass `judge_config=JudgeConfig(...)` to
  `ResultAnalyzer`, or wrap any `BaseLLM` in `SemanticSimilarityJudge(llm)` and pass it as
  `judge=...`.
- **Caching:** scores are stored in a sidecar next to the results file,
  `<results stem>_<judge>_similarity_scores.json`, as
  `{"<task_id>_cycle_<i>": {"score": ..., "input_sha256": ...}}`. The judge is only called for pairs
  missing from the sidecar, so reruns are free and an interrupted run resumes where it stopped. No
  API key is needed when the sidecar is complete.
- **Cache validation:** `input_sha256` fingerprints the exact judge prompt (the four judged strings
  rendered into the template). If a results file is overwritten (e.g. `evaluate` rerun with the same
  `--experiment_name`) or the prompt template changes, mismatching entries are re-judged with a
  warning instead of being mixed into the new run. Sidecars from the published data archive
  (`{"<task_id>_cycle_<i>": score}`, no hash) are still accepted; their entries cannot be verified
  and are used as is.
- **Failures:** if the judge call fails or its answer cannot be parsed, the pair is stored as
  `null`, left out of its bucket mean (with a warning), and retried on the next run
  (`--rejudge_failed=False` keeps it as is). The original scripts recorded such failures as 0.0.
- **Parsing:** a bare number is used directly; otherwise the first number that looks like a score
  is taken. Results are clamped to [0, 1].

### Programmatic scoring

```python
from evaloop.analysis.analyzer import ResultAnalyzer
from evaloop.analysis.similarity import JudgeConfig

analyzer = ResultAnalyzer(
    "results/exp_results.json",
    "results/analysis",
    judge_config=JudgeConfig(model="gpt-4-turbo-2024-04-09", api_key="sk-..."),
)
results = analyzer.analyze()          # all metrics; results["ASL"]["by_model"][model_name]
```

The pure metric functions (`asl_semantic`, `asl_pow`, `asl_bias`, `pass_at_k`, `pass_drop`,
`bucket_similarity`) live in `evaloop.analysis.metrics`.

## 🧪 Advanced Usage

### Custom Model Registration

```python
from evaloop.models.registry import ModelRegistry

registry = ModelRegistry()
registry.register_model("my-custom-model", {
    "name": "my-custom-model",
    "type": "vllm",
    "path": "path/to/my/model",
    "max_model_len": 4096,
    "tensor_parallel_size": 2
})
```

### Programmatic API

```python
from evaloop.core.config import EvaluationConfig
from evaloop.core.evaluator import EvaLoopEvaluator

config = EvaluationConfig(
    model="gpt-4",
    dataset="mbpp_plus",
    max_cycles=10,
    temperature=0.0,
)

evaluator = EvaLoopEvaluator(config)
results = evaluator.run_code_generation_summarization()
```

### Batch Processing

```bash
# Process multiple experiments
for model in "gpt-4" "gpt-3.5-turbo" "deepseek-coder-33b"; do
    evaloop evaluate --model "$model" \
                     --experiment_name "batch_${model}" \
                     --output_dir "results/batch/"
done

# Analyze all results
for result in results/batch/*_results.json; do
    evaloop analyze --results_path "$result"
done
```

## 🤝 Contributing

We welcome contributions — new loop/task types especially! Please see the [Contributing Guide](CONTRIBUTING.md) for the development setup, the "add a new loop type" walkthrough, and PR expectations.

### Development Setup

```bash
git clone https://github.com/evalooop/EvaLooop
cd EvaLooop
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
ruff check evaloop tests
```

The unit test suite runs without GPU, API keys, or the heavy model dependencies.

## 📖 Citation

If you use EvaLoop in your research, please cite:

```bibtex
@article{fang2024evaloop,
  title={EvaLoop: Assessing LLM Robustness in Programming from a Self-consistency Perspective},
  author={Fang, Sen and Ding, Weiyuan and Xu, Bowen},
  journal={arXiv preprint arXiv:2505.12185},
  year={2024},
  url={https://arxiv.org/abs/2505.12185}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Python Fire](https://github.com/google/python-fire) for CLI interface
- Uses [vLLM](https://github.com/vllm-project/vllm) for efficient model inference
- Evaluation dataset from [MBPP](https://github.com/google-research/google-research/tree/master/mbpp)
- Code testing powered by [EvalPlus](https://github.com/evalplus/evalplus)

## 📞 Support

- 🏆 [Leaderboard](https://evalooop.github.io/)
- 📄 [Paper](https://arxiv.org/abs/2505.12185)
- 🐛 [Issue Tracker](https://github.com/evalooop/EvaLooop/issues)

---

**EvaLoop** - Robust LLM Evaluation Made Simple 🔄
