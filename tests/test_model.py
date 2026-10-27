"""Independent checks of recurrent execution, cache correctness and batched decoding."""

import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from loopsd_search.generation import HFTokenGenerator, pack_nemo_output
from loopsd_search.model import RoutedQwen3
from loopsd_search.routes import Route, enumerate_routes
from loopsd_search.verification import verify_model


@pytest.fixture
def routed():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=127,
        pad_token_id=0,
        attention_dropout=0.0,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    return RoutedQwen3(Qwen3ForCausalLM(config))


def test_route_boundaries_and_enumeration():
    assert Route.parse("2:3").indices(5) == (0, 1, 2, 1, 2, 3, 4)
    assert Route.parse("1:1").indices(2) == (0, 0, 1)
    assert Route.parse("3:3").indices(3) == (0, 1, 2, 2)
    assert Route.parse("1:3").indices(3) == (0, 1, 2, 0, 1, 2)
    assert len(enumerate_routes(28, None)) == 407
    for value in ("0:2", "3:2", "4", "1:2:3"):
        with pytest.raises(ValueError):
            Route.parse(value)
    with pytest.raises(ValueError):
        Route.parse("1:5").indices(3)


@pytest.mark.parametrize("route_id", ["standard", "1:1", "2:3", "1:3"])
def test_cache_and_shared_parameters(routed, route_id):
    report = verify_model(routed, [1, 5, 7, 9, 11], Route.parse(route_id))
    assert report["standard_restored_exactly"]
    assert report["cache_vs_uncached_max_abs_error"] < 1e-4
    assert [layer.self_attn.layer_idx for layer in routed.original_layers] == [0, 1, 2]


@torch.inference_mode()
def test_loop_matches_independently_expanded_model(routed):
    # Construct a physically independent deep model as an oracle, then compare logits.
    route = Route.parse("2:3")
    indices = route.indices(3)
    config = copy.deepcopy(routed.original_config)
    config.num_hidden_layers = len(indices)
    config.layer_types = ["full_attention"] * len(indices)
    expanded = Qwen3ForCausalLM(config).eval()
    expanded.model.embed_tokens.load_state_dict(routed.model.model.embed_tokens.state_dict())
    expanded.model.norm.load_state_dict(routed.model.model.norm.state_dict())
    expanded.lm_head.load_state_dict(routed.model.lm_head.state_dict())
    for layer, source in zip(expanded.model.layers, indices, strict=True):
        layer.load_state_dict(routed.original_layers[source].state_dict())
    routed.set_route(route)
    ids = torch.tensor([[1, 7, 4, 20]])
    expected = expanded(input_ids=ids, use_cache=False).logits
    actual = routed.model(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@torch.inference_mode()
def test_batched_decode_matches_uncached_oracle_and_logprobs(routed):
    engine = HFTokenGenerator(routed, 0)
    engine.set_route(Route.parse("1:2"))
    prompts = [[1, 5, 7, 9], [1, 12]]
    outputs = engine.generate_tokens(prompts, max_new_tokens=4, stop_ids=set())
    for prompt, result in zip(prompts, outputs, strict=True):
        prefix = prompt.copy()
        for token, logprob in zip(result.tokens, result.logprobs, strict=True):
            logits = routed.model(input_ids=torch.tensor([prefix]), use_cache=False).logits[0, -1]
            assert token == logits.argmax().item()
            assert logprob == pytest.approx(torch.log_softmax(logits, -1)[token].item(), abs=1e-5)
            prefix.append(token)
    packed = pack_nemo_output(prompts, outputs, 0)
    assert packed["unpadded_sequence_lengths"].tolist() == [8, 6]
    assert packed["output_ids"][1, 6:].tolist() == [0, 0]
    assert packed["logprobs"][0, :4].tolist() == [0, 0, 0, 0]
    assert packed["logprobs"][1, 6:].tolist() == [0, 0]


def test_early_stop_and_route_change_do_not_reuse_cache(routed):
    engine = HFTokenGenerator(routed, 0)
    prompt = [[1, 5, 9]]
    original = engine.generate_tokens(prompt, max_new_tokens=3, stop_ids=set())[0]
    stopped = engine.generate_tokens(prompt, max_new_tokens=3, stop_ids={original.tokens[0]})[0]
    assert stopped.tokens == original.tokens[:1] and stopped.stopped
    engine.set_route(Route.parse("1:3"))
    engine.generate_tokens(prompt, max_new_tokens=3, stop_ids=set())
    engine.set_route(Route.parse("standard"))
    restored = engine.generate_tokens(prompt, max_new_tokens=3, stop_ids=set())[0]
    assert restored == original
