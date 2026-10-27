from fractions import Fraction

import pytest

from loopsd_search.config import DecodeConfig
from loopsd_search.generation import BudgetDecoder, TokenGeneration
from loopsd_search.scoring import extract_answer, gold_answer, numeric_value, score_answer


@pytest.mark.parametrize(
    "text,expected",
    [
        (r"The result is \boxed{1,250}.", Fraction(1250)),
        (r"\boxed{\frac{3}{4}}", Fraction(3, 4)),
        ("Final answer: -2.5", Fraction(-5, 2)),
        ("#### 100", Fraction(100)),
        ("Worked with 5 and 8.\n13", Fraction(13)),
        ("I should multiply 5 by 8", None),
        (r"\boxed{3} then \boxed{4}", Fraction(4)),
        ("", None),
    ],
)
def test_answer_parser(text, expected):
    assert extract_answer(text) == expected


def test_invalid_values_and_gold():
    assert numeric_value("1/0") is None
    assert numeric_value("nan") is None
    assert numeric_value("__import__('os')") is None
    assert gold_answer("a solution\n#### 1,000") == "1000"
    assert score_answer(r"\boxed{0.75}", "3/4") == (True, "3/4")
    with pytest.raises(ValueError):
        gold_answer("No delimiter")


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text, **kwargs):
        return [50] if text == "</think>" else [60, 61]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_dict"] is False
        return [1, 2]

    def decode(self, tokens, **kwargs):
        return "42" if tokens and tokens[0] == 42 else "reasoning"


class ScriptedGenerator:
    def __init__(self):
        self.calls = []

    def generate_tokens(self, prompts, *, max_new_tokens, stop_ids):
        self.calls.append((prompts, max_new_tokens, stop_ids))
        if len(self.calls) == 1:
            return [
                TokenGeneration([4, 50], [-1, -1], True),
                TokenGeneration([4, 5], [-1, -1], False),
                TokenGeneration([4, 99], [-1, -1], True),
            ]
        return [
            TokenGeneration([42, 99], [-1, -1], True),
            TokenGeneration([42, 43], [-1, -1], False),
        ]


def test_budget_forcing_natural_end_and_premature_eos():
    generator = ScriptedGenerator()
    decoder = BudgetDecoder(
        generator, FakeTokenizer(), DecodeConfig(thinking_budget=2, answer_budget=2)
    )
    rows = decoder.generate(["a", "b", "c"])
    assert generator.calls[1][0] == [[1, 2, 4, 50, 60, 61], [1, 2, 4, 5, 50, 60, 61]]
    assert rows[0].injected_tokens == [60, 61]
    assert not rows[0].forced_think_end
    assert rows[1].forced_think_end and rows[1].answer_truncated
    assert rows[2].premature_eos and rows[2].answer == "" and rows[2].injected_tokens == []
    assert generator.calls[0][2] == {50, 99}
    assert generator.calls[1][2] == {99}
