"""
Shared utilities: logging, GPU info, metrics helpers, text normalisation.
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# GPU utilities
# ---------------------------------------------------------------------------

def get_gpu_info() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        info = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "devices": [],
        }
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / 1024**3
            info["devices"].append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(total_gb, 2),
            })
        return info
    except Exception as e:
        return {"available": False, "error": str(e)}


def log_gpu_info():
    info = get_gpu_info()
    logger = logging.getLogger(__name__)
    if not info.get("available"):
        logger.warning("No GPU available — training will be slow.")
    else:
        for dev in info.get("devices", []):
            logger.info("GPU %d: %s (%.1f GB)", dev["index"], dev["name"], dev["total_memory_gb"])


# ---------------------------------------------------------------------------
# Bangla text normalisation
# ---------------------------------------------------------------------------

# Map half-width digits to Bangla digits
_ASCII_TO_BANGLA_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
_BANGLA_TO_ASCII_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def normalize_bangla(text: str) -> str:
    """Light normalisation for Bangla text."""
    # Unicode NFC normalisation
    text = unicodedata.normalize("NFC", text)
    # Collapse multiple spaces / newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ascii_digits_to_bangla(text: str) -> str:
    return text.translate(_ASCII_TO_BANGLA_DIGITS)


def bangla_digits_to_ascii(text: str) -> str:
    return text.translate(_BANGLA_TO_ASCII_DIGITS)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Simple sentence-level BLEU using sacrebleu."""
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU(effective_order=True)
        scores = [
            bleu.sentence_score(pred, [ref]).score
            for pred, ref in zip(predictions, references)
        ]
        return sum(scores) / len(scores) if scores else 0.0
    except ImportError:
        logging.getLogger(__name__).warning("sacrebleu not installed; skipping BLEU.")
        return 0.0


def _ngrams(tokens: list[str], n: int) -> dict:
    """Return n-gram frequency dict."""
    from collections import Counter
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of longest common subsequence."""
    m, n = len(a), len(b)
    # Space-optimised LCS
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def _rouge_n(pred_tokens: list[str], ref_tokens: list[str], n: int) -> float:
    pred_ng = _ngrams(pred_tokens, n)
    ref_ng  = _ngrams(ref_tokens, n)
    overlap = sum((pred_ng & ref_ng).values())
    ref_total  = sum(ref_ng.values())
    pred_total = sum(pred_ng.values())
    if ref_total == 0 or pred_total == 0:
        return 0.0
    recall    = overlap / ref_total
    precision = overlap / pred_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _rouge_l(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if len(ref_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0
    recall    = lcs / len(ref_tokens)
    precision = lcs / len(pred_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """
    Unicode-safe ROUGE-1 / ROUGE-2 / ROUGE-L.

    Uses whitespace tokenisation so it works correctly with Bangla text,
    where the standard rouge_score ASCII tokeniser produces empty token lists.
    """
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = len(predictions)
    if n == 0:
        return totals
    for pred, ref in zip(predictions, references):
        p_tok = pred.split()
        r_tok = ref.split()
        totals["rouge1"] += _rouge_n(p_tok, r_tok, 1)
        totals["rouge2"] += _rouge_n(p_tok, r_tok, 2)
        totals["rougeL"] += _rouge_l(p_tok, r_tok)
    return {k: round(v / n, 4) for k, v in totals.items()}


def compute_entity_f1(
    pred_entities: list[dict], gold_entities: list[dict], fields: list[str]
) -> dict:
    """
    Token-level F1 per entity field.
    Each entity dict maps field_name → string value.
    """
    results = {}
    for field in fields:
        tp = fp = fn = 0
        for pred, gold in zip(pred_entities, gold_entities):
            pred_toks = set(pred.get(field, "").split())
            gold_toks = set(gold.get(field, "").split())
            tp += len(pred_toks & gold_toks)
            fp += len(pred_toks - gold_toks)
            fn += len(gold_toks - pred_toks)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results[field] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
    return results
