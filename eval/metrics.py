from difflib import SequenceMatcher
from typing import Dict, List, Optional


def compute_metrics(
    predictions: List[str],
    ground_truths: List[str],
    metric_type: str = "accuracy",
    task_types: Optional[List[str]] = None,
) -> Dict:
    correct = 0
    details = []
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        task = task_types[i] if task_types else "closed_ended"
        ok = _matches(pred, gt, task)
        correct += int(ok)
        details.append({"prediction": pred, "ground_truth": gt, "correct": ok})
    total = max(len(predictions), 1)
    score = 100.0 * correct / total
    return {
        "metric": metric_type,
        "score": score,
        "accuracy": score,
        "correct": correct,
        "total": len(predictions),
        "details": details,
    }


def aggregate_results(results: Dict[str, Dict]) -> Dict:
    if not results:
        return {"mean_score": 0.0, "benchmarks": {}}
    scores = [r.get("score", 0.0) for r in results.values()]
    return {
        "mean_score": sum(scores) / len(scores),
        "benchmarks": results,
    }


def _matches(prediction: str, ground_truth: str, task_type: str) -> bool:
    pred = str(prediction).strip().lower()
    gt = str(ground_truth).strip().lower()
    pred_yes_no = _normalize_yes_no(pred)
    gt_yes_no = _normalize_yes_no(gt)
    if gt_yes_no is not None:
        return pred_yes_no == gt_yes_no

    pred_option = _normalize_option(pred)
    gt_option = _normalize_option(gt)
    if task_type == "multi_choice" or gt_option is not None:
        return pred_option == gt_option

    if pred == gt:
        return True
    return SequenceMatcher(None, pred, gt).ratio() >= 0.8


def _normalize_yes_no(value: str) -> Optional[str]:
    value = value.strip().lower()
    if value in {"yes", "true"}:
        return "yes"
    if value in {"no", "false"}:
        return "no"
    words = value.replace(".", " ").replace(",", " ").split()
    for word in words:
        if word in {"yes", "true"}:
            return "yes"
        if word in {"no", "false"}:
            return "no"
    return None


def _normalize_option(value: str) -> Optional[str]:
    value = value.strip().upper()
    if len(value) == 1 and value in "ABCDE":
        return value
    if len(value) >= 2 and value[0] in "ABCDE" and value[1] in {".", ")", ":", "]", " "}:
        return value[0]
    return None
