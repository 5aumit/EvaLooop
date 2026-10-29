"""Run the four HW4 configurations inside the evaluation container."""

import argparse
import csv
from pathlib import Path


MODELS = ("meta-llama/llama-3.1-8b-instruct", "qwen/qwen3.7-flash")
LOOPS = ("standard", "pseudocode")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check imports without API calls.")
    parser.add_argument("--output-dir", type=Path, default=Path("/results"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--loops", nargs="+", choices=LOOPS, default=list(LOOPS))
    args = parser.parse_args()

    from evaloop.analysis.analyzer import ResultAnalyzer
    from evaloop.core.config import EvaluationConfig
    from evaloop.core.evaluator import EvaLoopEvaluator
    from evaloop.utils.logging_utils import setup_logging

    if args.check:
        print("API, evaluator, and analysis imports passed. No API calls made.")
        return

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*_results.json")) or (output / "summary.csv").exists():
        parser.error("Results already exist. Select an empty directory with --output-dir.")

    # Validate every configuration before starting the first paid run.
    configs = [
        EvaluationConfig(
            model=model, provider="openrouter", loop_type=loop,
            api_concurrency=4, temperature=0, top_p=1, max_cycles=10,
            max_new_tokens=1024, output_dir=str(output),
            reasoning_enabled=False if model == "qwen/qwen3.7-flash" else None,
            experiment_name=f"{model.split('/')[-1]}_{loop}",
        )
        for model in args.models for loop in args.loops
    ]
    fields = ["model", "configuration", "n_tasks", "success_rate", "asl_score"]
    with (output / "summary.csv").open("w", newline="") as summary:
        writer = csv.DictWriter(summary, fieldnames=fields)
        writer.writeheader()
        summary.flush()
        for config in configs:
            logger = setup_logging(
                log_dir=str(output / "logs"), log_file=f"{config.experiment_name}.log"
            )
            logger.info("Starting %s on the full MBPP Plus dataset", config.experiment_name)
            results = EvaLoopEvaluator(config).run_code_generation_summarization()
            suffix = "pseudocode" if config.loop_type == "pseudocode" else "summarization"
            result_file = output / f"{config.experiment_name}_code_generation_{suffix}_results.json"
            analysis = ResultAnalyzer(
                results_path=str(result_file),
                output_dir=str(output / "analysis" / config.experiment_name),
            ).analyze(metrics=["ASL_pow", "pass@1", "pass_drop"])
            model_name = results[0]["model"]
            writer.writerow({
                "model": config.model,
                "configuration": f"{config.loop_type}-mbpp-plus",
                "n_tasks": len(results[0]["prompt_results"]),
                "success_rate": analysis["pass@1"]["by_model"][model_name],
                "asl_score": analysis["ASL_pow"]["by_model"][model_name],
            })
            summary.flush()
            logger.info("Finished %s", config.experiment_name)


if __name__ == "__main__":
    main()
