"""Reproduce HW4 metrics, matched-task comparisons, plots, and audit evidence.

Reads saved results only. Never calls an API or executes generated code.
"""

import argparse
import csv
import json
from pathlib import Path


MODELS = {
    "meta-llama/llama-3.1-8b-instruct": "llama-3.1-8b-instruct",
    "qwen/qwen3.7-flash": "qwen3.7-flash",
}
LOOPS = ("standard", "pseudocode")


def write_csv(path, rows, fields):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_runs(directory):
    runs = {}
    task_ids = None
    for model, short in MODELS.items():
        for loop in LOOPS:
            suffix = "summarization" if loop == "standard" else "pseudocode"
            name = f"{short}_{loop}_code_generation_{suffix}_results.json"
            matches = list(directory.rglob(name))
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {name}; found {len(matches)}")
            (data,) = json.loads(matches[0].read_text())
            if data["loop_type"] != loop or data["max_cycles"] != 10:
                raise ValueError(f"Wrong loop or cycle limit in {name}")
            tasks = {task["task_id"]: task for task in data["prompt_results"]}
            if len(tasks) != 378 or (task_ids is not None and set(tasks) != task_ids):
                raise ValueError("All four runs must contain the same 378 tasks")
            task_ids = set(tasks)
            for task in tasks.values():
                sustained = task["successful_cycles"]
                assert 0 <= sustained <= 10
                assert len(task["cycles"]) == min(sustained + 1, 10)
                assert task["max_cycles_reached"] == (sustained == 10)
                assertion = task["initial_prompt"].splitlines()[-2]
                for cycle_number, cycle in enumerate(task["cycles"], 1):
                    assert cycle["cycle"] == cycle_number
                    assert cycle["success"] == (cycle_number <= sustained)
                    assert cycle["test_results"]["success"] == cycle["success"]
                    assert cycle["task_2_output"].splitlines()[-2] == assertion
                    assert not any(key.endswith("_error") for key in cycle)
            if model.startswith("qwen/"):
                assert data["reasoning_enabled"] is False
            runs[model, loop] = tasks
    return runs


def audit_cases(runs, annotations, output):
    rows = []
    evidence = [
        "# Failure inspection evidence",
        "AI-assisted static code-reading worksheet. The author must review the labels.",
        "The sample is purposive, not a random estimate of failure prevalence.",
        "Cycle 1 cases are initial coding failures, not evidence of loop-induced drift.",
    ]
    for case in annotations:
        task = runs[case["model"], case["loop"]][case["task_id"]]
        failed = task["cycles"][-1]
        assert failed["success"] is False
        previous = task["cycles"][-2] if len(task["cycles"]) > 1 else None
        row = {
            **case,
            "failed_cycle": failed["cycle"],
            "base_passed": failed["test_results"]["base_success"],
            "plus_passed": failed["test_results"]["plus_success"],
        }
        rows.append(row)
        evidence.extend([
            f"\n## {len(rows)}. {case['model']} / {case['loop']} / {case['task_id']} / cycle {failed['cycle']}",
            f"Category: {case['category']}. Confidence: {case['confidence']}.",
            f"Observation: {case['observation']}",
            f"Interpretation: {case['interpretation']}",
            f"Base passed: {row['base_passed']}; plus passed: {row['plus_passed']}.",
            "\nOriginal task:\n```text\n" + task["initial_prompt"].rstrip() + "\n```",
        ])
        if previous:
            evidence.extend([
                "\nLast passing code:\n```python\n" + previous["task_1_output"] + "\n```",
                "\nDescription fed into failure:\n```text\n" + previous["task_2_output"].rstrip() + "\n```",
            ])
        else:
            evidence.append("No earlier cycle: this is an initial coding failure.")
        evidence.append("\nFailing code:\n```python\n" + failed["task_1_output"] + "\n```")
    write_csv(output / "failure_review.csv", rows, list(rows[0]))
    (output / "failure_review.md").write_text("\n\n".join(evidence) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.results_dir)
    metrics, summary, survival_rows, paired = [], [], [], []
    for (model, loop), tasks in runs.items():
        counts = [task["successful_cycles"] for task in tasks.values()]
        survival = [sum(count >= cycle for count in counts) for cycle in range(1, 11)]
        numerator = sum(count * count for count in counts)
        summary.append({
            "model": model, "configuration": f"{loop}-mbpp-plus", "n_tasks": 378,
            "success_rate": survival[0] / 378, "asl_score": numerator / 3780,
        })
        metrics.append({
            **summary[-1], "survival_counts": survival,
            "retention": survival[-1] / survival[0],
            "conditional_asl": numerator / (survival[0] * 10),
            "initial_failures": counts.count(0),
            "later_failures": sum(0 < count < 10 for count in counts),
        })
        for cycle, surviving in enumerate(survival, 1):
            survival_rows.append({"model": model, "loop": loop, "cycle": cycle,
                                  "surviving_tasks": surviving, "pass_rate": surviving / 378})
    for model in MODELS:
        standard, pseudocode = runs[model, "standard"], runs[model, "pseudocode"]
        common = [task_id for task_id in standard if
                  standard[task_id]["successful_cycles"] > 0 and
                  pseudocode[task_id]["successful_cycles"] > 0]
        deltas = [pseudocode[task_id]["successful_cycles"] -
                  standard[task_id]["successful_cycles"] for task_id in common]
        paired.append({"model": model, "shared_initially_solved": len(common),
                       "pseudocode_better": sum(delta > 0 for delta in deltas),
                       "tie": deltas.count(0), "pseudocode_worse": sum(delta < 0 for delta in deltas),
                       "mean_survival_change": sum(deltas) / len(deltas)})
    for name, rows in [("summary.csv", summary), ("survival.csv", survival_rows),
                       ("paired_comparison.csv", paired)]:
        write_csv(args.output_dir / name, rows, list(rows[0]))
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    for axis, model in zip(axes, MODELS):
        for loop, color in [("standard", "#0072B2"), ("pseudocode", "#D55E00")]:
            values = next(row["survival_counts"] for row in metrics if
                          row["model"] == model and row["configuration"] == f"{loop}-mbpp-plus")
            axis.plot(range(1, 11), [100 * value / 378 for value in values],
                      marker="o", markersize=3, label=loop, color=color)
        axis.set_title("Llama 3.1 8B" if model.startswith("meta-llama") else "Qwen3.7 Flash")
        axis.set_xlabel("Cycle")
        axis.set_xticks(range(1, 11))
        axis.set_ylim(0, 100)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Tasks still passing (%)")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / f"survival_comparison.{extension}", dpi=240)
    plt.close(fig)
    if args.annotations:
        annotations = json.loads(args.annotations.read_text())
        audit_cases(runs, annotations, args.output_dir)
    print(f"Reproduced all four runs in {args.output_dir}; no API calls or code execution.")


if __name__ == "__main__":
    main()
