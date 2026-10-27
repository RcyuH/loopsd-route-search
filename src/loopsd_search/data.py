"""Read the user's local GSM8K parquet and preserve original row identities."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from loopsd_search.config import DataConfig
from loopsd_search.scoring import gold_answer


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    answer: str


def read_examples(path: Path, split: str, expected_rows: int | None) -> list[Example]:
    rows = pq.read_table(path, columns=["question", "answer"]).to_pylist()
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, found {len(rows)}")
    examples = []
    for idx, row in enumerate(rows):
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError(f"Invalid question at {split}:{idx}")
        examples.append(Example(f"{split}:{idx}", row["question"], gold_answer(row["answer"])))
    return examples


def search_splits(cfg: DataConfig) -> dict[str, list[Example]]:
    examples = read_examples(cfg.train, "train", cfg.expected_train_rows)
    if cfg.search_size + cfg.validation_size > len(examples):
        raise ValueError("Search and validation exceed training set size")
    permutation = np.random.default_rng(cfg.shuffle_seed).permutation(len(examples))
    shuffled = [examples[int(idx)] for idx in permutation]
    return {
        "stage1": shuffled[: cfg.stage1_size],
        "stage2": shuffled[cfg.stage1_size : cfg.search_size],
        "validation": shuffled[cfg.search_size : cfg.search_size + cfg.validation_size],
    }
