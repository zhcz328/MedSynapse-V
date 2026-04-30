"""
Answer Extraction from Generated Text (Appendix §6).

Extracts the first valid option letter (A/B/C/D/E) from VLM output
for closed-ended evaluation. Falls back to fuzzy string matching
against candidate answers when no explicit option is detected.
"""

import re
from typing import Optional, List
from difflib import SequenceMatcher


# Primary pattern: option letter in standard formats
OPTION_PATTERNS = [
    re.compile(r"^\s*([A-E])\s*[.)\]:]", re.MULTILINE),
    re.compile(r"\b(?:answer|option|choice)\s*(?:is|:)\s*\(?([A-E])\)?", re.IGNORECASE),
    re.compile(r"<answer>\s*([A-E])\s*</answer>", re.IGNORECASE),
    re.compile(r"\(([A-E])\)"),
    re.compile(r"^([A-E])$", re.MULTILINE),
]

# Fallback: isolated letter at start of response
FALLBACK_PATTERN = re.compile(r"^[^a-zA-Z]*([A-E])\b")


def extract_option_letter(text: str) -> Optional[str]:
    """
    Extract the first valid option letter from generated text.

    Matching priority:
      1. Structured answer tags (e.g., <answer>B</answer>)
      2. Explicit answer statements (e.g., "The answer is B")
      3. Parenthesized option (e.g., "(B)")
      4. Leading option letter (e.g., "B. ...")
      5. First isolated A-E character
    """
    text = text.strip()

    # Single-character response
    if len(text) == 1 and text.upper() in "ABCDE":
        return text.upper()

    # Try each pattern in priority order
    for pattern in OPTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()

    # Fallback: first A-E character at start
    match = FALLBACK_PATTERN.search(text)
    if match:
        return match.group(1).upper()

    return None


def extract_open_ended_answer(text: str) -> str:
    """
    Normalize open-ended answer for comparison.
    Lowercases, strips punctuation, removes articles.
    """
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b(a|an|the)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fuzzy_match_answer(
    prediction: str,
    candidates: List[str],
    threshold: float = 0.8,
) -> Optional[str]:
    """
    Match prediction against candidate answers using fuzzy matching.
    Returns the best-matching candidate above threshold, or None.
    """
    pred_norm = extract_open_ended_answer(prediction)
    best_match = None
    best_score = 0.0

    for candidate in candidates:
        cand_norm = extract_open_ended_answer(candidate)
        score = SequenceMatcher(None, pred_norm, cand_norm).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate

    if best_score >= threshold:
        return best_match
    return None


def extract_answer(
    text: str,
    task_type: str = "closed_ended",
    options: Optional[List[str]] = None,
    ground_truth: Optional[str] = None,
) -> str:
    """
    Unified answer extraction interface.

    For closed-ended: extract option letter, with fuzzy fallback.
    For open-ended: normalize and return the response.
    """
    if task_type in ("closed_ended", "multi_choice"):
        letter = extract_option_letter(text)
        if letter is not None:
            return letter

        # Fuzzy fallback against options
        if options:
            labels = "ABCDE"
            match = fuzzy_match_answer(text, options)
            if match and match in options:
                idx = options.index(match)
                if idx < len(labels):
                    return labels[idx]

        return ""
    else:
        return extract_open_ended_answer(text)