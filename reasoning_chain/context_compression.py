"""Deterministic compression helpers for model context payloads."""

import re

_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def normalize_whitespace(text: str) -> str:
    """Apply lossless-for-meaning whitespace normalization to older messages."""
    return " ".join(text.split())


def compress_text(text: str, max_tokens: int, preserve_final_number: bool = False) -> str:
    """Bound text with head/tail context and optionally call out its final numeric value."""
    max_chars = max(120, max_tokens * 3)
    if len(text) <= max_chars:
        return text

    numbers = _NUMBER_PATTERN.findall(text)
    numeric_note = ""
    if preserve_final_number and numbers:
        numeric_note = f"\n[preserved_final_numeric_value: {numbers[-1]}]"

    marker = "\n...[compressed for context budget]...\n"
    available = max(40, max_chars - len(marker) - len(numeric_note))
    head_size = available * 2 // 3
    tail_size = available - head_size
    return f"{text[:head_size]}{marker}{text[-tail_size:]}{numeric_note}"


def compress_step_results(steps: list[dict], max_tokens: int) -> tuple[list[dict], int]:
    """Compress verbose output/error fields without mutating trace source records."""
    compressed_steps = []
    compressed_fields = 0
    for step in steps:
        compact = dict(step)
        for field in ("output", "error"):
            value = compact.get(field)
            if not isinstance(value, str):
                continue
            compressed = compress_text(
                value,
                max_tokens=max_tokens,
                preserve_final_number=field == "output",
            )
            if compressed != value:
                compact[field] = compressed
                compressed_fields += 1
        compressed_steps.append(compact)
    return compressed_steps, compressed_fields
