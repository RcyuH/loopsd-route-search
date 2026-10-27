"""Explicit GSM8K numeric scorer; never take the last number from reasoning text."""

import re
from fractions import Fraction


def numeric_value(text: str) -> Fraction | None:
    value = text.strip().replace(",", "").replace("−", "-")
    value = value.replace("\\,", "").replace(" ", "").strip("$")
    fraction = re.fullmatch(r"([+-]?)\\(?:dfrac|tfrac|frac)\{([+-]?\d+)\}\{([+-]?\d+)\}", value)
    if fraction:
        sign, numerator, denominator = fraction.groups()
        value = f"{sign}{numerator}/{denominator}"
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+|/[+-]?\d+)?", value):
        return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def boxed_values(text: str) -> list[str]:
    values = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        for idx in range(start, len(text)):
            depth += (text[idx] == "{") - (text[idx] == "}")
            if depth == 0:
                values.append(text[start:idx])
                break
    return values


def extract_answer(text: str) -> Fraction | None:
    boxes = boxed_values(text)
    if boxes:
        return numeric_value(boxes[-1])
    marked = re.findall(r"(?:####|final answer\s*:|answer is\s*:?)\s*([^\n]+)", text, re.I)
    if marked:
        return numeric_value(marked[-1].strip().rstrip("."))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return numeric_value(lines[-1].rstrip(".")) if lines else None


def gold_answer(text: str) -> str:
    if "####" not in text:
        raise ValueError("Expected GSM8K gold answer with #### delimiter")
    value = numeric_value(text.rsplit("####", 1)[1])
    if value is None:
        raise ValueError("Could not parse GSM8K gold answer")
    return str(value)


def score_answer(text: str, gold: str) -> tuple[bool, str | None]:
    prediction = extract_answer(text)
    return prediction == Fraction(gold), None if prediction is None else str(prediction)
