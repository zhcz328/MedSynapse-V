import re
from typing import List, Optional
from difflib import SequenceMatcher


class AccuracyReward:
    """
    Computes r_acc for GRPO trajectories.
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0.8,
        option_pattern: str = r"[(\s]?([A-E])[)\s.,:]",
    ):
        self.fuzzy_threshold = fuzzy_threshold
        self.option_regex = re.compile(option_pattern)

    def extract_option(self, text: str) -> Optional[str]:
        """Extract first valid option letter from generated text."""
        text = text.strip()

        # Direct single-letter answer
        if len(text) == 1 and text.upper() in "ABCDE":
            return text.upper()

        # Try regex extraction
        match = self.option_regex.search(text)
        if match:
            return match.group(1).upper()

        # Fallback: first occurrence of A-E
        for char in text:
            if char.upper() in "ABCDE":
                return char.upper()

        return None

    def fuzzy_match(self, pred: str, target: str) -> bool:
        """Fuzzy string matching for open-ended evaluation."""
        pred = pred.strip().lower()
        target = target.strip().lower()

        if pred == target:
            return True

        ratio = SequenceMatcher(None, pred, target).ratio()
        return ratio >= self.fuzzy_threshold

    def compute(
        self,
        predictions: List[str],
        ground_truths: List[str],
        task_types: Optional[List[str]] = None,
    ) -> List[float]:
        """
        Compute r_acc for a batch of predictions.

        Args:
            predictions:   list of generated text outputs
            ground_truths: list of reference answers
            task_types:    list of task types ('closed_ended' or 'open_ended')

        Returns:
            rewards: list of binary reward values (0.0 or 1.0)
        """
        rewards = []
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            task_type = task_types[i] if task_types else "closed_ended"

            if task_type == "closed_ended":
                pred_option = self.extract_option(pred)
                gt_option = gt.strip().upper()
                if len(gt_option) > 1:
                    gt_option = self.extract_option(gt) or gt_option[0]
                reward = 1.0 if pred_option == gt_option else 0.0
            else:
                reward = 1.0 if self.fuzzy_match(pred, gt) else 0.0

            rewards.append(reward)

        return rewards

    def __call__(
        self,
        predictions: List[str],
        ground_truths: List[str],
        task_types: Optional[List[str]] = None,
    ) -> List[float]:
        return self.compute(predictions, ground_truths, task_types)
