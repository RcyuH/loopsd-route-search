"""Greedy decoding with bounded memory and explicit two-stage budget forcing."""

from dataclasses import dataclass
from typing import Protocol

import torch
from transformers import PreTrainedTokenizerBase

from loopsd_search.config import DecodeConfig
from loopsd_search.model import RoutedQwen3
from loopsd_search.routes import Route


@dataclass
class TokenGeneration:
    tokens: list[int]
    logprobs: list[float]
    stopped: bool


class TokenGenerator(Protocol):
    def generate_tokens(
        self, prompts: list[list[int]], *, max_new_tokens: int, stop_ids: set[int]
    ) -> list[TokenGeneration]: ...

    def set_route(self, route: Route) -> None: ...


class HFTokenGenerator:
    """One fresh KV cache per batch, with no persistent prefix caching across routes."""

    def __init__(self, routed: RoutedQwen3, pad_token_id: int) -> None:
        self.routed = routed
        self.pad_token_id = pad_token_id

    def set_route(self, route: Route) -> None:
        self.routed.set_route(route)

    @torch.inference_mode()
    def generate_tokens(
        self, prompts: list[list[int]], *, max_new_tokens: int, stop_ids: set[int]
    ) -> list[TokenGeneration]:
        if not prompts:
            return []
        if max_new_tokens < 1 or any(not prompt for prompt in prompts):
            raise ValueError("Generation requires nonempty prompts and a positive budget")
        model = self.routed.model
        device = next(model.parameters()).device
        width = max(map(len, prompts))
        if width + max_new_tokens > model.config.max_position_embeddings:
            raise ValueError("Prompt plus requested budget exceeds model context window")
        batch = len(prompts)
        ids = torch.full((batch, width), self.pad_token_id, device=device, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, prompt in enumerate(prompts):
            ids[row, -len(prompt) :] = torch.tensor(prompt, device=device)
            mask[row, -len(prompt) :] = 1
        position_ids = (mask.cumsum(-1) - 1).clamp_min(0)
        past = None
        active = [True] * batch
        result = [TokenGeneration([], [], False) for _ in prompts]
        for step in range(max_new_tokens):
            output = model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            logits = output.logits[:, -1].float()
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError(
                    "Nonfinite logits; aborting rather than scoring a bad route"
                )
            next_ids = logits.argmax(-1)
            logprobs = torch.log_softmax(logits, dim=-1).gather(1, next_ids[:, None])[:, 0]
            token_values = next_ids.tolist()
            probability_values = logprobs.tolist()
            old_active = active.copy()
            for row in range(batch):
                if active[row]:
                    token = token_values[row]
                    result[row].tokens.append(token)
                    result[row].logprobs.append(probability_values[row])
                    if token in stop_ids:
                        result[row].stopped = True
                        active[row] = False
            if not any(active) or step + 1 == max_new_tokens:
                break
            past = output.past_key_values
            valid = torch.tensor(old_active, dtype=torch.long, device=device)[:, None]
            ids = torch.where(valid.bool(), next_ids[:, None], self.pad_token_id)
            mask = torch.cat((mask, valid), dim=-1)
            position_ids = (mask.sum(-1, keepdim=True) - 1).clamp_min(0)
        return result


@dataclass
class Response:
    prompt_tokens: list[int]
    thinking_tokens: list[int]
    injected_tokens: list[int]
    answer_tokens: list[int]
    thinking: str
    answer: str
    forced_think_end: bool
    premature_eos: bool
    answer_truncated: bool


class BudgetDecoder:
    """Two stages: stop thinking at </think> or B; append cue; decode <= A tokens.

    Injected closure/cue tokens are recorded separately and excluded from generated
    token totals. An EOS before </think> terminates the response and is not resurrected.
    This prompt/cue policy is an explicit assumption because the paper omits strings.
    """

    def __init__(
        self, generator: TokenGenerator, tokenizer: PreTrainedTokenizerBase, cfg: DecodeConfig
    ) -> None:
        self.generator = generator
        self.tokenizer = tokenizer
        self.cfg = cfg
        end = tokenizer.encode(cfg.think_end, add_special_tokens=False)
        if len(end) != 1:
            raise ValueError("The supported Qwen3 protocol requires a single-token </think>")
        self.think_end_id = end[0]
        self.eos_ids = {tokenizer.eos_token_id}
        if None in self.eos_ids or self.think_end_id in self.eos_ids:
            raise ValueError("Invalid EOS or thinking-end token")
        self.cue_ids = tokenizer.encode(cfg.answer_cue, add_special_tokens=False)

    def prompt(self, question: str) -> list[int]:
        messages = []
        if self.cfg.system_prompt is not None:
            messages.append({"role": "system", "content": self.cfg.system_prompt})
        messages.append(
            {"role": "user", "content": self.cfg.user_template.format(question=question)}
        )
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            return_dict=False,
        )

    def generate(self, questions: list[str]) -> list[Response]:
        prompts = [self.prompt(question) for question in questions]
        thoughts = self.generator.generate_tokens(
            prompts,
            max_new_tokens=self.cfg.thinking_budget,
            stop_ids=self.eos_ids | {self.think_end_id},
        )
        if len(thoughts) != len(prompts):
            raise RuntimeError("Backend returned the wrong number of thinking responses")
        responses = []
        answer_prompts, answer_rows = [], []
        for idx, (prompt, thought) in enumerate(zip(prompts, thoughts, strict=True)):
            if not thought.tokens:
                raise RuntimeError("Backend returned an empty thinking response")
            premature = thought.tokens[-1] in self.eos_ids
            forced = not premature and thought.tokens[-1] != self.think_end_id
            injected = [] if premature else (([self.think_end_id] if forced else []) + self.cue_ids)
            responses.append(
                Response(
                    prompt,
                    thought.tokens,
                    injected,
                    [],
                    self.tokenizer.decode(thought.tokens, skip_special_tokens=False),
                    "",
                    forced,
                    premature,
                    False,
                )
            )
            if not premature:
                answer_rows.append(idx)
                answer_prompts.append(prompt + thought.tokens + injected)
        answers = self.generator.generate_tokens(
            answer_prompts, max_new_tokens=self.cfg.answer_budget, stop_ids=self.eos_ids
        )
        for idx, answer in zip(answer_rows, answers, strict=True):
            responses[idx].answer_tokens = answer.tokens
            responses[idx].answer = self.tokenizer.decode(answer.tokens, skip_special_tokens=True)
            responses[idx].answer_truncated = not answer.stopped
        return responses


def pack_nemo_output(
    prompts: list[list[int]], results: list[TokenGeneration], pad_token_id: int
) -> dict[str, torch.Tensor | dict]:
    """Pack the actual NeMo RL right-padded token and full-sequence logprob contract."""
    if len(prompts) != len(results):
        raise ValueError("Prompt/response count mismatch")
    sizes = [
        len(prompt) + len(result.tokens) for prompt, result in zip(prompts, results, strict=True)
    ]
    width = max(sizes, default=0)
    ids = torch.full((len(prompts), width), pad_token_id, dtype=torch.long)
    logprobs = torch.zeros((len(prompts), width), dtype=torch.float32)
    for row, (prompt, result) in enumerate(zip(prompts, results, strict=True)):
        ids[row, : sizes[row]] = torch.tensor(prompt + result.tokens)
        logprobs[row, len(prompt) : sizes[row]] = torch.tensor(result.logprobs)
    return {
        "output_ids": ids,
        "logprobs": logprobs,
        "generation_lengths": torch.tensor(
            [len(item.tokens) for item in results], dtype=torch.long
        ),
        "unpadded_sequence_lengths": torch.tensor(sizes, dtype=torch.long),
        "truncated": torch.tensor([not item.stopped for item in results], dtype=torch.bool),
        "__extra__": {},
    }
