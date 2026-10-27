"""Durable per-example tapes and immutable run provenance for safe resumption."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from loopsd_search.config import ExperimentConfig


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def model_fingerprint(path: Path) -> dict[str, str]:
    files = sorted(set(path.glob("*.json")) | set(path.glob("*.safetensors")))
    if not any(item.suffix == ".safetensors" for item in files):
        raise ValueError("Expected a local safetensors checkpoint")
    files += sorted(path.glob("*.txt"))
    return {item.name: digest_file(item) for item in files}


def make_manifest(cfg: ExperimentConfig, split_ids: dict[str, list[str]]) -> dict:
    root = Path(__file__).resolve().parent
    source = {item.name: digest_file(item) for item in sorted(root.glob("*.py"))}
    nemo_commit = subprocess.check_output(
        ["git", "-C", str(cfg.nemo_rl_path), "rev-parse", "HEAD"], text=True
    ).strip()
    nemo_diff = subprocess.check_output(
        ["git", "-C", str(cfg.nemo_rl_path), "diff", "HEAD", "--", "nemo_rl"], text=True
    )
    value = {
        "schema_version": 1,
        "config": cfg.model_dump(mode="json"),
        "model_sha256": model_fingerprint(cfg.model.path),
        "train_sha256": digest_file(cfg.data.train),
        "test_sha256": digest_file(cfg.data.test),
        "source_sha256": source,
        "nemo_commit": nemo_commit,
        "nemo_tracked_diff_sha256": hashlib.sha256(nemo_diff.encode()).hexdigest(),
        "split_ids": split_ids,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers", "numpy", "pyarrow", "pydantic")
        },
        "assumptions": [
            "Prompt/cue are reconstruction assumptions; author strings are absent.",
            "Shuffle: numpy.default_rng(seed).permutation; author library is unspecified.",
            "Thinking stops at </think> or cap; injected closure/cue are not generated tokens.",
            "Premature EOS is terminal and produces an empty answer, scored incorrect.",
            "Scorer: answer stage only; last boxed, answer marker, or numeric final line.",
            "Loop ties: Python random.Random(seed) on lexicographically sorted IDs.",
            "Same token budget is not a matched-FLOP comparison.",
        ],
    }
    value["fingerprint"] = digest_json(value)
    return value


class RunStore:
    """Callers hold a file lock for the entire run, including manifest validation."""

    def __init__(self, root: Path, manifest: dict) -> None:
        self.root = root
        self.fingerprint = manifest["fingerprint"]
        path = root / "manifest.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old != manifest:
                raise ValueError(
                    "Run provenance changed. Choose a fresh output_dir; refusing mixed results."
                )
        else:
            if any(root.glob("predictions/**/*.jsonl")) or (root / "selected_route.json").exists():
                raise ValueError("Existing results without a manifest; choose a fresh output_dir")
            atomic_json(path, manifest)

    def tape_path(self, stage: str, route_id: str) -> Path:
        return self.root / "predictions" / stage / (route_id.replace(":", "_") + ".jsonl")

    def read_tape(self, stage: str, route_id: str) -> dict[str, dict]:
        path = self.tape_path(stage, route_id)
        if not path.exists():
            return {}
        records = {}
        # Only a final incomplete write is recoverable. Corruption in a complete row is fatal.
        with path.open("r+b") as stream:
            while True:
                start = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    stream.truncate(start)
                    break
                row = json.loads(line)
                if row["run_fingerprint"] != self.fingerprint or row["route"] != route_id:
                    raise ValueError(f"Mismatched provenance in {path}")
                if row["id"] in records:
                    raise ValueError(f"Duplicate example {row['id']} in {path}")
                records[row["id"]] = row
        return records

    def append(self, stage: str, route_id: str, rows: list[dict]) -> None:
        path = self.tape_path(stage, route_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
