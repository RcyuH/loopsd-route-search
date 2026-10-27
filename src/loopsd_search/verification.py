"""Numerical verification on a loaded checkpoint, before any search."""

import torch

from loopsd_search.model import RoutedQwen3
from loopsd_search.routes import Route


@torch.inference_mode()
def verify_model(routed: RoutedQwen3, tokens: list[int], route: Route) -> dict:
    if len(tokens) < 2:
        raise ValueError("Need at least two tokens")
    model = routed.model
    device = next(model.parameters()).device
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    routed.restore()
    original = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits.float()
    expected_count = sum(parameter.numel() for parameter in model.parameters())
    routed.set_route(route)
    try:
        for layer, idx in zip(model.model.layers, route.indices(routed.num_layers), strict=True):
            for name, parameter in layer.named_parameters():
                source_parameter = dict(routed.original_layers[idx].named_parameters())[name]
                if parameter is not source_parameter:
                    raise AssertionError(f"Untied parameter: {name}")
        actual_count = sum(parameter.numel() for parameter in model.parameters())
        if actual_count != expected_count:
            raise AssertionError("Route changed unique parameter count")
        full = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits.float()
        first = model(input_ids=ids[:, :-1], use_cache=True, logits_to_keep=1)
        cached = model(
            input_ids=ids[:, -1:],
            past_key_values=first.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        ).logits.float()
        error = float((full - cached).abs().max())
        # BF16 kernels can differ with prefill/decode shapes. Record the error as well.
        tolerance = 1e-4 if next(model.parameters()).dtype == torch.float32 else 0.25
        torch.testing.assert_close(cached, full, atol=tolerance, rtol=0.02)
        lengths = [
            first.past_key_values.get_seq_length(idx) for idx in range(len(model.model.layers))
        ]
        if lengths != [len(tokens)] * len(lengths):
            raise AssertionError(f"Cache length per execution slot is incorrect: {lengths}")
    finally:
        routed.restore()
    restored = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits.float()
    torch.testing.assert_close(restored, original, atol=0, rtol=0)
    return {
        "route": route.id,
        "shared_parameter_count": expected_count,
        "cache_slots": len(route.indices(routed.num_layers)),
        "cache_vs_uncached_max_abs_error": error,
        "cache_atol": tolerance,
        "cache_rtol": 0.02,
        "standard_restored_exactly": True,
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "prefix_length": len(tokens),
    }
