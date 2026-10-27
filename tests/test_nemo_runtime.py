"""Run in the NeMo environment; never fake the framework import on a CPU-only dev host."""

import importlib.util

import pytest
import torch

from loopsd_search.generation import HFTokenGenerator
from loopsd_search.model import RoutedQwen3


@pytest.mark.nemo
def test_real_generation_interface():
    if importlib.util.find_spec("nemo_rl") is None:
        pytest.skip("NeMo RL runtime not installed; use the NeMo environment for this check")
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.generation.interfaces import GenerationInterface, verify_right_padding
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from loopsd_search.nemo_adapter import LoopedHFGeneration

    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
    )
    model = RoutedQwen3(Qwen3ForCausalLM(cfg))
    adapter = LoopedHFGeneration(HFTokenGenerator(model, 0), max_new_tokens=2, stop_ids=set())
    assert isinstance(adapter, GenerationInterface)
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 3], [1, 4, 0]]),
            "input_lengths": torch.tensor([3, 2]),
            "__extra__": {},
        }
    )
    output = adapter.generate(data, greedy=True)
    assert verify_right_padding(output, pad_value=0)[0]
    assert output["generation_lengths"].tolist() == [2, 2]
    adapter.shutdown()
    adapter.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        adapter.generate(data, greedy=True)
