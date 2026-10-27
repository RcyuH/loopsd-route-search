import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from loopsd_search.artifacts import RunStore
from loopsd_search.config import DataConfig, ExperimentConfig
from loopsd_search.data import search_splits
from loopsd_search.generation import Response
from loopsd_search.routes import Route
from loopsd_search.search import SearchRunner, paired_comparison, rank_candidates, select_route


def test_ties_and_standard_preference():
    assert rank_candidates({"standard": 99, "1:2": 3, "2:3": 2, "3:3": 2, "1:1": 0}, 2) == [
        "1:2",
        "2:3",
        "3:3",
    ]
    assert select_route({"standard": 5, "1:2": 5, "2:3": 4}, 42) == "standard"
    assert select_route({"standard": 0, "1:2": 5, "2:3": 5}, 42) == select_route(
        {"2:3": 5, "1:2": 5, "standard": 0}, 42
    )


def write_dataset(path, n):
    pq.write_table(
        pa.table(
            {
                "question": [f"question {i}" for i in range(n)],
                "answer": [f"solution\n#### {i}" for i in range(n)],
            }
        ),
        path,
    )


def test_splits_have_no_overlap_and_are_stable(tmp_path):
    train = tmp_path / "train.parquet"
    write_dataset(train, 20)
    cfg = DataConfig(
        train=train,
        test=train,
        search_size=10,
        stage1_size=4,
        validation_size=3,
        expected_train_rows=20,
    )
    splits = search_splits(cfg)
    ids = [example.id for examples in splits.values() for example in examples]
    assert len(ids) == len(set(ids)) == 13
    assert splits == search_splits(cfg)


class FakeDecoder:
    def __init__(self):
        self.generator = self
        self.route = Route(None, None)
        self.calls = 0

    def set_route(self, route):
        self.route = route

    def generate(self, questions):
        self.calls += len(questions)
        return [
            Response(
                [1],
                [2],
                [3],
                [4],
                "thinking",
                question.rsplit(" ", 1)[1] if self.route.id == "1:2" else "9999",
                False,
                False,
                False,
            )
            for question in questions
        ]


def test_search_resume_locked_selection_and_test(tmp_path):
    train, test = tmp_path / "train.parquet", tmp_path / "test.parquet"
    write_dataset(train, 12)
    write_dataset(test, 4)
    cfg = ExperimentConfig.model_validate(
        {
            "model": {"path": str(tmp_path), "batch_size": 2},
            "data": {
                "train": str(train),
                "test": str(test),
                "search_size": 6,
                "stage1_size": 2,
                "validation_size": 2,
                "expected_train_rows": 12,
                "expected_test_rows": 4,
            },
            "search": {"top_k_stage1": 1, "top_k_stage2": 1},
            "output_dir": str(tmp_path / "run"),
            "nemo_rl_path": str(tmp_path),
        }
    )
    store = RunStore(cfg.output_dir, {"fingerprint": "test-only"})
    decoder = FakeDecoder()
    runner = SearchRunner(cfg, decoder, store)
    routes = [Route.parse(name) for name in ["standard", "1:1", "1:2"]]
    # Search must not even need the test file to exist.
    test.rename(tmp_path / "held_out.parquet")
    selection = runner.search(routes, search_splits(cfg.data), 3)
    assert selection["route"] == "1:2"
    assert selection["stage1_survivors"] == ["1:2"]
    previous_calls = decoder.calls
    runner.search(routes, search_splits(cfg.data), 3)
    assert decoder.calls == previous_calls
    (tmp_path / "held_out.parquet").rename(test)
    result = runner.test(3)
    assert result["paired"]["delta_percentage_points"] == 100
    assert result["paired"]["teacher_only_correct"] == 4
    assert result["evaluation_scope"] == "full_test"
    assert (cfg.output_dir / "test_comparison.json").exists()
    frozen = json.loads((cfg.output_dir / "selected_route.json").read_text())
    frozen["route"] = "1:1"
    (cfg.output_dir / "selected_route.json").write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="Selection has changed"):
        runner.test(3)


def test_store_recovers_only_trailing_partial_line_and_rejects_changes(tmp_path):
    store = RunStore(tmp_path, {"fingerprint": "a"})
    row = {"id": "train:0", "route": "standard", "run_fingerprint": "a"}
    store.append("stage1", "standard", [row])
    path = store.tape_path("stage1", "standard")
    with path.open("ab") as stream:
        stream.write(b'{"id":')
    assert store.read_tape("stage1", "standard") == {"train:0": row}
    assert path.read_bytes().endswith(b"\n")
    with pytest.raises(ValueError, match="provenance changed"):
        RunStore(tmp_path, {"fingerprint": "b"})
    with path.open("ab") as stream:
        stream.write(b"invalid json\n")
    with pytest.raises(json.JSONDecodeError):
        store.read_tape("stage1", "standard")


def test_paired_statistics_use_pairing():
    a = [{"id": str(i), "correct": i % 2 == 0} for i in range(10)]
    same = paired_comparison(a, a, samples=1000, seed=42)
    assert same["paired_bootstrap_95ci_pp"] == [0, 0]
    with pytest.raises(ValueError, match="identical"):
        paired_comparison(a, list(reversed(a)), samples=1000, seed=42)
