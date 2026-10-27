# Implementation validation — 2026-10-27

These checks validate the implementation. They do not reproduce the paper's
reported accuracies or establish that a looped teacher outperforms the standard model.

## Environment

- macOS ARM64, 16 GiB unified memory; real-checkpoint inference used MPS.
- Project-local `.venv`, Python 3.13.7, torch 2.13.0, transformers 5.16.1.
- Dependencies and NeMo API import dependencies recorded in `uv.lock`.
- NeMo RL source commit: `bb4f0d85674344d01656efdbf8eb2a7f3f737ca3`.
- No changes were made to the NeMo RL or CRISP source checkouts.

## Completed checks

```bash
PYTHONPATH=../nemo-rl uv run --locked --extra nemo-api pytest -q
uv run --locked --extra nemo-api ruff check .
uv run --locked --extra nemo-api ruff format --check .
```

- **24 tests passed**, including the actual NeMo RL `GenerationInterface` and
  `BatchedDataDict` integration test; no framework mocks were substituted.
- Lint and formatting checks passed.
- Tiny float32 Qwen3: routed logits exactly matched an independently constructed
  expanded model. Cache decoding matched full-prefix recomputation; left-padded
  batching and sampled-token log probabilities were checked independently.
- Route restoration preserved original logits. Repeated layer shells reused the
  original Parameter objects, and all original attention cache indices stayed intact.
- Staged selection, cutoff ties, standard-route tie preference, split isolation,
  resume, partial-write recovery and frozen-selection validation were checked.
- Forced thinking closure, natural thinking termination, premature EOS, answer
  truncation and strict numeric answer parsing were checked.

## Real checkpoint

Using the local Qwen3-1.7B checkpoint and route `8:12` on a 12-token prefix:

- Unique parameter count remained **1,720,574,976**.
- There were **33 cache slots**, each with the correct sequence length.
- Restoring the standard route reproduced its original logits exactly.
- BF16 MPS cached versus uncached maximum absolute logit difference: **0.125**.
  The real-checkpoint check uses absolute tolerance 0.25 and relative tolerance
  0.02 to accommodate BF16 kernel shape differences. This is a tolerance-based
  numerical check, not bitwise cache equivalence. Tiny float32 checks use tighter tolerances.

Details: `outputs/smoke/verification.json`.

## End-to-end smoke

Both `configs/smoke.yaml` (HF) and `configs/smoke_nemo.yaml` (actual NeMo API)
completed search and evaluation with the local checkpoint and local parquet files.
The two backends produced identical prompt/thinking/injected/answer token IDs,
parsed answers and correctness on **8 saved responses**: six search/validation
responses and two test responses. The selected route was standard, so test did
not rerun an identical route under a second name.

The smoke uses 8 thinking + 8 answer tokens, two search questions, one validation
question, one loop candidate and two test questions. Its accuracy is not a useful
quality metric. No full-budget search or full-test benchmark was run.

HF resumption after completion was exercised without regenerating stage-one
responses. The same check was performed through the NeMo adapter. JSONL tapes
and manifests are available under `outputs/smoke/` and `outputs/smoke_nemo/`.
`outputs/smoke_initial/` is an earlier development smoke, kept for provenance;
use the current smoke configurations and directories for resumption.

## Remaining experimental work

Offline B200 packaging was added separately in `OFFLINE_B200.md`, with pinned
direct requirements, a Linux wheelhouse preparation script and a CUDA config.
The Linux bundle build/install and B200 execution have not been performed here.
The commands above record the earlier local validation; current server commands
use Python/pip directly and do not require uv.

- Run standard baseline and staged search with the 400/128 protocol on CUDA.
- Evaluate the validation-selected route on the full 1,319-question test split.
- Inspect prompt/scoring assumptions before interpreting differences from the paper.
- CUDA kernels, high-throughput vLLM/Megatron execution and distributed workers
  have not been exercised. The implemented NeMo adapter uses a synchronous HF engine.
