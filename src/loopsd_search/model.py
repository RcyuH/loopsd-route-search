"""Frozen Qwen3 routing with tied parameters and independent cache slots.

Module shells are shallow-copied, not tensor storage. Each execution occurrence
gets its own attention layer_idx. No hooks or forward monkey-patches are needed.
Only dense Qwen3 full-attention checkpoints are supported in this first version.
"""

import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from loopsd_search.config import ModelConfig
from loopsd_search.routes import Route


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    return torch.device(value)


class RoutedQwen3:
    """Own a frozen model; route changes are sequential and require fresh caches."""

    def __init__(self, model: PreTrainedModel) -> None:
        if model.config.model_type != "qwen3" or model.config.use_sliding_window:
            raise ValueError("Only dense Qwen3 with full attention is supported")
        self.model = model.eval().requires_grad_(False)
        self.original_layers = model.model.layers
        self.original_config = model.config
        self.original_decoder_config = model.model.config
        self.num_layers = len(self.original_layers)
        self.route = Route(None, None)

    def set_route(self, route: Route) -> None:
        indices = route.indices(self.num_layers)
        if route.start is None:
            self.model.model.layers = self.original_layers
            self.model.config = self.original_config
            self.model.model.config = self.original_decoder_config
        else:
            config = copy.deepcopy(self.original_config)
            config.num_hidden_layers = len(indices)
            # Transformers 5.x DynamicCache uses layer_types; harmless on 4.x.
            config.layer_types = ["full_attention"] * len(indices)
            execution_layers = []
            for execution_idx, source_idx in enumerate(indices):
                source = self.original_layers[source_idx]
                layer = copy.copy(source)
                layer._modules = source._modules.copy()
                attention = copy.copy(source.self_attn)
                attention._modules = source.self_attn._modules.copy()
                attention.layer_idx = execution_idx
                attention.config = config
                layer.self_attn = attention
                layer.config = config
                execution_layers.append(layer)
            self.model.model.layers = torch.nn.ModuleList(execution_layers)
            self.model.config = config
            self.model.model.config = config
        self.route = route

    def restore(self) -> None:
        self.set_route(Route(None, None))


def load_model(cfg: ModelConfig) -> tuple[RoutedQwen3, AutoTokenizer]:
    device = resolve_device(cfg.device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        cfg.dtype
    ]
    model = AutoModelForCausalLM.from_pretrained(
        str(cfg.path),
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
        attn_implementation=cfg.attention,
        low_cpu_mem_usage=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(cfg.path), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return RoutedQwen3(model), tokenizer
