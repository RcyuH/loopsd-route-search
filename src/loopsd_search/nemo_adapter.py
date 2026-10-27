"""NeMo RL extension: import this module inside a configured NeMo RL environment.

The standalone hf CLI deliberately does not import this module. No fake NeMo
modules or fallback base classes are used when dependencies are unavailable.
"""

from typing import Any

import torch
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import GenerationInterface

from loopsd_search.generation import HFTokenGenerator, TokenGeneration, pack_nemo_output
from loopsd_search.routes import Route


class LoopedHFGeneration(GenerationInterface):
    """Synchronous, greedy, frozen-model NeMo generation backend for route search.

    Supports no weight refit, Ray actors, stochastic rollouts, or training. Those
    features belong to the later OPSD integration, not the frozen route-search phase.
    """

    def __init__(
        self, engine: HFTokenGenerator, *, max_new_tokens: int, stop_ids: set[int]
    ) -> None:
        self.engine = engine
        self.max_new_tokens = max_new_tokens
        self.stop_ids = stop_ids
        self.device = next(engine.routed.model.parameters()).device
        self.closed = False

    def init_collective(self, ip, port, world_size, *, train_world_size):
        raise NotImplementedError("Frozen route search does not initialize refit collectives")

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        if self.closed:
            raise RuntimeError("Generator has been shut down")
        self.engine.routed.model.to(self.device)
        return True

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        if kwargs.get("release_gpu", True):
            self.engine.routed.model.to("cpu")
        return True

    def shutdown(self) -> bool:
        if not self.closed:
            self.engine.routed.restore()
            self.engine.routed.model.to("cpu")
            self.closed = True
        return True

    def set_route(self, route: Route) -> None:
        if self.closed:
            raise RuntimeError("Generator has been shut down")
        self.engine.set_route(route)

    def generate(self, data: BatchedDataDict, greedy: bool) -> BatchedDataDict:
        if not greedy:
            raise NotImplementedError("Route-search adapter only implements greedy decoding")
        if "stop_strings" in data and data["stop_strings"]:
            raise NotImplementedError("Use token stop_ids; stop_strings are not supported")
        self.prepare_for_generation()
        ids, lengths = data["input_ids"], data["input_lengths"]
        if ids.ndim != 2 or lengths.shape != (ids.shape[0],):
            raise ValueError("Expected right-padded [batch, width] ids and [batch] input_lengths")
        prompts = []
        for row, length in enumerate(lengths.tolist()):
            if length < 1 or length > ids.shape[1]:
                raise ValueError("Invalid input length")
            if not torch.all(ids[row, length:] == self.engine.pad_token_id):
                raise ValueError("NeMo inputs must be right padded")
            prompts.append(ids[row, :length].tolist())
        result = self.engine.generate_tokens(
            prompts, max_new_tokens=self.max_new_tokens, stop_ids=self.stop_ids
        )
        return BatchedDataDict(pack_nemo_output(prompts, result, self.engine.pad_token_id))

    def generate_tokens(
        self, prompts: list[list[int]], *, max_new_tokens: int, stop_ids: set[int]
    ) -> list[TokenGeneration]:
        """Bridge the search decoder through actual NeMo BatchedDataDict generation."""
        if not prompts:
            return []
        self.max_new_tokens, self.stop_ids = max_new_tokens, stop_ids
        width = max(map(len, prompts))
        ids = torch.full((len(prompts), width), self.engine.pad_token_id, dtype=torch.long)
        for row, prompt in enumerate(prompts):
            ids[row, : len(prompt)] = torch.tensor(prompt)
        data = BatchedDataDict(
            {
                "input_ids": ids,
                "input_lengths": torch.tensor(list(map(len, prompts))),
                "__extra__": {},
            }
        )
        output = self.generate(data, greedy=True)
        result = []
        for row, prompt in enumerate(prompts):
            end = int(output["unpadded_sequence_lengths"][row])
            result.append(
                TokenGeneration(
                    output["output_ids"][row, len(prompt) : end].tolist(),
                    output["logprobs"][row, len(prompt) : end].tolist(),
                    not bool(output["truncated"][row]),
                )
            )
        return result
