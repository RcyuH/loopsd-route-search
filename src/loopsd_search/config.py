"""Validated experiment configuration. Relative paths are relative to the YAML file."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictConfig):
    path: Path
    device: str = "auto"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    attention: Literal["eager", "sdpa"] = "sdpa"
    batch_size: int = Field(default=1, ge=1)


class DataConfig(StrictConfig):
    train: Path
    test: Path
    shuffle_seed: int = 42
    search_size: int = Field(default=1000, ge=2)
    validation_size: int = Field(default=100, ge=1)
    stage1_size: int = Field(default=128, ge=1)
    # PCG64 permutation, also used by datasets.Dataset.shuffle(seed=42).
    shuffle: Literal["numpy_default_rng"] = "numpy_default_rng"
    expected_train_rows: int | None = 7473
    expected_test_rows: int | None = 1319

    @model_validator(mode="after")
    def validate_sizes(self) -> "DataConfig":
        if self.stage1_size >= self.search_size:
            raise ValueError("stage1_size must be smaller than search_size")
        return self


class DecodeConfig(StrictConfig):
    thinking_budget: int = Field(default=400, ge=1)
    answer_budget: int = Field(default=128, ge=1)
    # These strings are declared reproduction assumptions, not recovered author prompts.
    user_template: str = "{question}\nPlease put your final numerical answer in \\boxed{{}}."
    answer_cue: str = "\n\nFinal answer: "
    think_end: str = "</think>"
    system_prompt: str | None = None
    enable_thinking: Literal[True] = True
    greedy: Literal[True] = True

    @model_validator(mode="after")
    def validate_template(self) -> "DecodeConfig":
        if "{question}" not in self.user_template:
            raise ValueError("user_template must contain {question}")
        self.user_template.format(question="check")
        return self


class SearchConfig(StrictConfig):
    top_k_stage1: int = Field(default=24, ge=1)
    top_k_stage2: int = Field(default=5, ge=1)
    tie_seed: int = 42
    # null means all L(L+1)/2 intervals. IDs use 1-based inclusive paper indices.
    candidate_routes: list[str] | None = None


class EvalConfig(StrictConfig):
    test_limit: int | None = Field(default=None, ge=1)
    bootstrap_samples: int = Field(default=10000, ge=100)
    bootstrap_seed: int = 42


class ExperimentConfig(StrictConfig):
    model: ModelConfig
    data: DataConfig
    decode: DecodeConfig = Field(default_factory=DecodeConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    output_dir: Path
    nemo_rl_path: Path
    # nemo uses the real GenerationInterface adapter; hf is the portable reference.
    backend: Literal["hf", "nemo"] = "hf"


def load_config(path: Path) -> ExperimentConfig:
    cfg = ExperimentConfig.model_validate(yaml.safe_load(path.read_text()))
    root = path.resolve().parent
    cfg.model.path = (root / cfg.model.path).resolve()
    cfg.data.train = (root / cfg.data.train).resolve()
    cfg.data.test = (root / cfg.data.test).resolve()
    cfg.output_dir = (root / cfg.output_dir).resolve()
    cfg.nemo_rl_path = (root / cfg.nemo_rl_path).resolve()
    return cfg
