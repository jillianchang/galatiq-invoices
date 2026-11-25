"""Deterministic keyword backstop for risky free text (D17).

Runs on the raw document regardless of the LLM, so these signals hold even if
the investigator misses them or is unavailable.
"""

from __future__ import annotations

import re

from src.rules.codes import finding
from src.rules.context import RuleContext
from src.schemas import Finding

PATTERNS: dict[str, re.Pattern[str]] = {
    "vendor_identity_change": re.compile(
        r"\b(formerly|now known as|previously known as|acquired by|f/k/a)\b", re.IGNORECASE
    ),
    "payment_pressure": re.compile(
        r"\burgent\b|pay immediately|immediate payment|avoid penalt|\bpenalt(?:y|ies)\b"
        r"|wire transfer|final notice|act now",
        re.IGNORECASE,
    ),
    "prompt_injection": re.compile(
        r"ignore (?:all |any )?(?:previous|prior|above) instructions|pre-?approved"
        r"|skip (?:the )?validation|automated processor"
        r"|disregard (?:the )?(?:rules|policy|instructions)",
        re.IGNORECASE,
    ),
}

MESSAGES = {
    "vendor_identity_change": "The vendor's identity changed; verify it is the same supplier.",
    "payment_pressure": "The invoice pressures for fast payment, a common fraud tactic.",
    "prompt_injection": "The document contains instructions aimed at an automated system.",
}


def _snippet(text: str, start: int, end: int, width: int = 50) -> str:
    return re.sub(r"\s+", " ", text[max(0, start - width) : end + width]).strip()


def check_text_signals(ctx: RuleContext) -> list[Finding]:
    out = []
    for code, pattern in PATTERNS.items():
        matches = list(pattern.finditer(ctx.raw_text))
        if matches:
            out.append(
                finding(
                    code,
                    MESSAGES[code],
                    {
                        "matched": sorted({m.group(0).lower() for m in matches}),
                        "snippet": _snippet(ctx.raw_text, matches[0].start(), matches[0].end()),
                    },
                )
            )
    return out
