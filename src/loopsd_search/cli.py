"""Command-line entrypoints for preparation, search and locked test evaluation."""

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

from filelock import FileLock

from loopsd_search.artifacts import RunStore, atomic_json, make_manifest
from loopsd_search.config import load_config
from loopsd_search.data import search_splits
from loopsd_search.routes import Route, enumerate_routes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["plan", "verify", "baseline", "search", "evaluate", "report"]
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backend", choices=["hf", "nemo"], help="Override execution backend")
    parser.add_argument(
        "--stop-after",
        choices=["stage1", "stage2", "validation"],
        default="validation",
        help="Search checkpoint boundary; rerun to resume",
    )
    parser.add_argument(
        "--verify-route", default="8:12", help="Route checked by verify (1-based inclusive)"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.backend is not None:
        cfg.backend = args.backend
    if args.command == "report":
        from loopsd_search.report import export_report

        export_report(cfg.output_dir)
        return
    raw_config = json.loads((cfg.model.path / "config.json").read_text())
    if raw_config["model_type"] != "qwen3" or raw_config["use_sliding_window"]:
        raise ValueError("Only dense full-attention Qwen3 is supported")
    depth = raw_config["num_hidden_layers"]
    routes = enumerate_routes(depth, cfg.search.candidate_routes)
    splits = search_splits(cfg.data)
    if not (cfg.nemo_rl_path / "nemo_rl/models/generation/interfaces.py").exists():
        raise ValueError("nemo_rl_path does not point to the requested NeMo RL checkout")
    if args.command == "plan":
        print(
            json.dumps(
                {
                    "model": str(cfg.model.path),
                    "layers": depth,
                    "backend": cfg.backend,
                    "device": cfg.model.device,
                    "dtype": cfg.model.dtype,
                    "candidate_loops": len(routes) - 1,
                    "stage1_generations": len(routes) * len(splits["stage1"]),
                    "stage2_generations_without_cutoff_ties": (
                        1 + min(len(routes) - 1, cfg.search.top_k_stage1)
                    )
                    * len(splits["stage2"]),
                    "validation_generations_without_cutoff_ties": (
                        1 + min(len(routes) - 1, cfg.search.top_k_stage1, cfg.search.top_k_stage2)
                    )
                    * len(splits["validation"]),
                    "split_sizes": {key: len(value) for key, value in splits.items()},
                    "thinking_budget": cfg.decode.thinking_budget,
                    "answer_budget": cfg.decode.answer_budget,
                    "test_limit": cfg.evaluation.test_limit,
                    "output_dir": str(cfg.output_dir),
                    "torch": importlib.metadata.version("torch"),
                    "transformers": importlib.metadata.version("transformers"),
                    "protocol": "LoOPSD search with documented prompt/scorer/shuffle assumptions",
                },
                indent=2,
            )
        )
        return
    # Import torch/transformers only for commands that load a model.
    from loopsd_search.generation import BudgetDecoder, HFTokenGenerator
    from loopsd_search.model import load_model
    from loopsd_search.search import SearchRunner, summarize
    from loopsd_search.verification import verify_model

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cfg.output_dir / ".run.lock"), timeout=0):
        if args.command == "verify":
            routed, tokenizer = load_model(cfg.model)
            result = verify_model(
                routed,
                tokenizer.encode("A short arithmetic example: 17 plus 8 equals"),
                Route.parse(args.verify_route),
            )
            atomic_json(cfg.output_dir / "verification.json", result)
            print(json.dumps(result, indent=2))
            return
        print("Checking checkpoint/data/code fingerprints...", flush=True)
        manifest = make_manifest(
            cfg, {key: [item.id for item in value] for key, value in splits.items()}
        )
        store = RunStore(cfg.output_dir, manifest)
        if args.command == "evaluate" and not (cfg.output_dir / "selected_route.json").exists():
            raise ValueError("Run search first; evaluation requires a frozen selected_route.json")
        # Fail on missing NeMo dependencies before loading the checkpoint.
        if cfg.backend == "nemo":
            sys.path.insert(0, str(cfg.nemo_rl_path))
            from loopsd_search.nemo_adapter import LoopedHFGeneration
        routed, tokenizer = load_model(cfg.model)
        engine = HFTokenGenerator(routed, tokenizer.pad_token_id)
        if cfg.backend == "nemo":
            engine = LoopedHFGeneration(
                engine, max_new_tokens=cfg.decode.thinking_budget, stop_ids={tokenizer.eos_token_id}
            )
        runner = SearchRunner(cfg, BudgetDecoder(engine, tokenizer, cfg.decode), store)
        if args.command == "baseline":
            result = {
                "standard": summarize(
                    runner.evaluate_route("stage1", Route(None, None), splits["stage1"])
                )
            }
            atomic_json(cfg.output_dir / "baseline_summary.json", result)
        elif args.command == "search":
            result = runner.search(routes, splits, depth, stop_after=args.stop_after)
        else:
            result = runner.test(depth)
        print(json.dumps(result, indent=2))
        from loopsd_search.report import export_report

        export_report(cfg.output_dir)


if __name__ == "__main__":
    main()
