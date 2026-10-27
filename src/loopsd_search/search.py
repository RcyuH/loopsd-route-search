"""Staged search and held-out comparison. Search never loads test labels."""

import json
import random
import time
from dataclasses import asdict

import numpy as np

from loopsd_search.artifacts import RunStore, atomic_json, digest_json
from loopsd_search.config import ExperimentConfig
from loopsd_search.data import Example, read_examples
from loopsd_search.generation import BudgetDecoder
from loopsd_search.routes import Route
from loopsd_search.scoring import score_answer


def rank_candidates(scores: dict[str, int], top_k: int) -> list[str]:
    candidates = sorted(
        (name for name in scores if name != "standard"), key=lambda n: (-scores[n], n)
    )
    if not candidates:
        raise ValueError("No nonstandard candidates")
    cutoff = scores[candidates[min(top_k, len(candidates)) - 1]]
    return [name for name in candidates if scores[name] >= cutoff]


def select_route(scores: dict[str, int], seed: int) -> str:
    best = max(scores.values())
    if scores["standard"] == best:
        return "standard"
    tied = sorted(name for name in scores if scores[name] == best)
    return random.Random(seed).choice(tied)


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot score an empty split")
    count = len(rows)
    return {
        "n": count,
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / count,
        "parse_failures": sum(row["prediction"] is None for row in rows),
        "mean_thinking_tokens": sum(len(row["thinking_tokens"]) for row in rows) / count,
        "mean_answer_tokens": sum(len(row["answer_tokens"]) for row in rows) / count,
        "mean_generated_tokens": sum(
            len(row["thinking_tokens"]) + len(row["answer_tokens"]) for row in rows
        )
        / count,
        "forced_think_end_rate": sum(row["forced_think_end"] for row in rows) / count,
        "premature_eos_rate": sum(row["premature_eos"] for row in rows) / count,
        "answer_truncation_rate": sum(row["answer_truncated"] for row in rows) / count,
        "generation_seconds": sum(row["amortized_generation_seconds"] for row in rows),
    }


class SearchRunner:
    def __init__(self, cfg: ExperimentConfig, decoder: BudgetDecoder, store: RunStore) -> None:
        self.cfg = cfg
        self.decoder = decoder
        self.store = store

    def evaluate_route(self, stage: str, route: Route, examples: list[Example]) -> list[dict]:
        records = self.store.read_tape(stage, route.id)
        ids = {example.id for example in examples}
        if records.keys() - ids:
            raise ValueError("Tape contains examples outside the frozen split")
        for example in examples:
            if example.id in records and records[example.id]["gold"] != example.answer:
                raise ValueError("Tape gold answer disagrees with dataset")
        self.decoder.generator.set_route(route)
        for offset in range(0, len(examples), self.cfg.model.batch_size):
            batch = examples[offset : offset + self.cfg.model.batch_size]
            if all(example.id in records for example in batch):
                continue
            start = time.perf_counter()
            responses = self.decoder.generate([example.question for example in batch])
            elapsed = time.perf_counter() - start
            rows = []
            for example, response in zip(batch, responses, strict=True):
                if example.id in records:
                    previous = records[example.id]
                    if (
                        previous["thinking_tokens"] != response.thinking_tokens
                        or previous["answer_tokens"] != response.answer_tokens
                    ):
                        raise RuntimeError("Replayed partial batch produced different tokens")
                    continue
                correct, prediction = score_answer(response.answer, example.answer)
                row = {
                    "run_fingerprint": self.store.fingerprint,
                    "route": route.id,
                    "id": example.id,
                    "question": example.question,
                    "gold": example.answer,
                    "prediction": prediction,
                    "correct": correct,
                    "amortized_generation_seconds": elapsed / len(batch),
                    **asdict(response),
                }
                rows.append(row)
                records[example.id] = row
            self.store.append(stage, route.id, rows)
            print(f"{stage:10s} route={route.id:>8s} {len(records)}/{len(examples)}", flush=True)
        return [records[example.id] for example in examples]

    def stage(self, name: str, routes: list[Route], examples: list[Example]) -> dict[str, dict]:
        results = {}
        for route in routes:
            results[route.id] = summarize(self.evaluate_route(name, route, examples))
            atomic_json(self.store.root / f"{name}_summary.json", results)
        return results

    def search(
        self,
        routes: list[Route],
        splits: dict[str, list[Example]],
        num_layers: int,
        *,
        stop_after: str = "validation",
    ) -> dict:
        results1 = self.stage("stage1", routes, splits["stage1"])
        survivors1 = rank_candidates(
            {key: row["correct"] for key, row in results1.items()}, self.cfg.search.top_k_stage1
        )
        if stop_after == "stage1":
            return {"completed_stage": "stage1", "survivors": survivors1}
        routes2 = [Route.parse(name) for name in ["standard", *survivors1]]
        results2 = self.stage("stage2", routes2, splits["stage2"])
        survivors2 = rank_candidates(
            {key: row["correct"] for key, row in results2.items()}, self.cfg.search.top_k_stage2
        )
        if stop_after == "stage2":
            return {"completed_stage": "stage2", "survivors": survivors2}
        val_routes = [Route.parse(name) for name in ["standard", *survivors2]]
        validation = self.stage("validation", val_routes, splits["validation"])
        selected = select_route(
            {key: row["correct"] for key, row in validation.items()}, self.cfg.search.tie_seed
        )
        artifact = {
            "run_fingerprint": self.store.fingerprint,
            "route": selected,
            "indexing": "paper: 1-based inclusive; execution_order_0based: 0-based",
            "execution_order_0based": Route.parse(selected).indices(num_layers),
            "num_original_layers": num_layers,
            "useful_loop": selected != "standard",
            "validation": validation,
            "stage1_survivors": survivors1,
            "stage2_survivors": survivors2,
            "candidate_set": [route.id for route in routes],
        }
        artifact["selection_sha256"] = digest_json(artifact)
        path = self.store.root / "selected_route.json"
        if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(artifact)):
            raise ValueError("Refusing to replace a previously frozen route selection")
        atomic_json(path, artifact)
        return artifact

    def test(self, num_layers: int) -> dict:
        selection = json.loads((self.store.root / "selected_route.json").read_text())
        checksum = selection.pop("selection_sha256")
        if (
            digest_json(selection) != checksum
            or selection["run_fingerprint"] != self.store.fingerprint
        ):
            raise ValueError("Selection has changed or belongs to another run")
        validation_scores = {key: row["correct"] for key, row in selection["validation"].items()}
        if selection["route"] != select_route(validation_scores, self.cfg.search.tie_seed):
            raise ValueError("Selected route is inconsistent with validation")
        route = Route.parse(selection["route"])
        if list(route.indices(num_layers)) != selection["execution_order_0based"]:
            raise ValueError("Route/depth mismatch")
        examples = read_examples(self.cfg.data.test, "test", self.cfg.data.expected_test_rows)
        if self.cfg.evaluation.test_limit is not None:
            examples = examples[: self.cfg.evaluation.test_limit]
        standard = self.evaluate_route("test", Route(None, None), examples)
        teacher = (
            standard if route.id == "standard" else self.evaluate_route("test", route, examples)
        )
        comparison = paired_comparison(
            standard,
            teacher,
            samples=self.cfg.evaluation.bootstrap_samples,
            seed=self.cfg.evaluation.bootstrap_seed,
        )
        result = {
            "run_fingerprint": self.store.fingerprint,
            "selected_route": route.id,
            "useful_loop_on_validation": route.id != "standard",
            "evaluation_scope": "full_test"
            if len(examples) == self.cfg.data.expected_test_rows
            else "subset",
            "standard": summarize(standard),
            "selected": summarize(teacher),
            "paired": comparison,
            "layer_evaluations_per_token": {
                "standard": num_layers,
                "selected": len(route.indices(num_layers)),
            },
        }
        atomic_json(self.store.root / "test_comparison.json", result)
        return result


def paired_comparison(
    standard: list[dict], teacher: list[dict], *, samples: int, seed: int
) -> dict:
    if [row["id"] for row in standard] != [row["id"] for row in teacher]:
        raise ValueError("Paired comparison requires identical example IDs in identical order")
    a = np.array([row["correct"] for row in standard], dtype=np.int8)
    b = np.array([row["correct"] for row in teacher], dtype=np.int8)
    if not len(a):
        raise ValueError("Empty comparison")
    delta = b - a
    rng = np.random.default_rng(seed)
    # Multinomial over {-1,0,+1} is exactly the paired nonparametric bootstrap.
    probabilities = np.array([(delta == value).mean() for value in (-1, 0, 1)])
    counts = rng.multinomial(len(a), probabilities, size=samples)
    bootstrap = (counts[:, 2] - counts[:, 0]) / len(a)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "n": len(a),
        "delta_percentage_points": float(delta.mean() * 100),
        "paired_bootstrap_95ci_pp": [float(lower * 100), float(upper * 100)],
        "teacher_only_correct": int(((a == 0) & (b == 1)).sum()),
        "standard_only_correct": int(((a == 1) & (b == 0)).sum()),
        "both_correct": int(((a == 1) & (b == 1)).sum()),
        "both_wrong": int(((a == 0) & (b == 0)).sum()),
    }
