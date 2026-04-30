"""
Evaluation Pipeline.

Implements the full evaluation protocol:
  1. Load model with appropriate memory pathway (IMT or privileged)
  2. Process each benchmark with corresponding prompt template
  3. Greedy decoding (temperature=0, top-p=1.0), max 128 tokens
  4. Extract predictions via regex and compute metrics
  5. Report per-benchmark and aggregate accuracy

Supports all 7 benchmarks: VQA-RAD, SLAKE, PathVQA, PMC-VQA,
MMMU*, MedXpertQA-MM, GMAI-MMBench.
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
import logging

from core.builder import MedSynapseV
from eval.regex_extractor import AnswerExtractor
from eval.metrics import compute_metrics, aggregate_results

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Unified evaluator for medical VQA benchmarks.

    Performs greedy decoding with diagnostic implicit memory injection
    and evaluates against ground truth using task-appropriate metrics.
    """

    def __init__(
        self,
        model: MedSynapseV,
        benchmark_config: Optional[Dict] = None,
        max_new_tokens: int = 128,
        max_new_tokens_cot: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        batch_size: int = 1,
        num_workers: int = 4,
        output_dir: str = "eval_results",
        use_imt: bool = True,
    ):
        self.model = model
        self.benchmark_config = benchmark_config or {}
        self.max_new_tokens = max_new_tokens
        self.max_new_tokens_cot = max_new_tokens_cot
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.output_dir = output_dir
        self.use_imt = use_imt
        self.extractor = AnswerExtractor()

        os.makedirs(output_dir, exist_ok=True)

    @torch.no_grad()
    def evaluate_benchmark(
        self,
        dataloader: DataLoader,
        benchmark_name: str,
        task_type: str = "closed_ended",
        metric_type: str = "accuracy",
    ) -> Dict[str, Any]:
        """
        Evaluate a single benchmark.

        Returns:
            dict with score, predictions, and per-sample results
        """
        logger.info(f"Evaluating: {benchmark_name} ({task_type})")
        t0 = time.time()

        self.model.vlm.eval()
        if self.model.autonomous_module is not None:
            self.model.autonomous_module.eval()

        all_predictions = []
        all_ground_truths = []
        all_results = []

        for batch_idx, batch in enumerate(dataloader):
            batch = _to_device(batch, self._device)

            # Generate memory via IMT (A_psi) or privileged (E_ana + P_phi) pathway
            if self.use_imt and self.model.autonomous_module is not None:
                memory = self.model.generate_memory_autonomous(
                    input_ids=batch["input_ids"],
                    pixel_values=batch.get("pixel_values"),
                    image_grid_thw=batch.get("image_grid_thw"),
                    attention_mask=batch.get("attention_mask"),
                )
            elif self.model.encoder is not None and self.model.memory_sampler is not None:
                images = batch.get("pixel_values_medsam", batch.get("pixel_values"))
                mem_out = self.model.generate_memory_privileged(images)
                memory = mem_out["memory"]
            else:
                memory = None

            # Prepare inputs with memory injection
            inputs_embeds = self.model.vlm.get_input_embeddings()(batch["input_ids"])
            if memory is not None:
                injected = self.model.injector.inject(
                    inputs_embeds=inputs_embeds,
                    memory=memory,
                    attention_mask=batch["attention_mask"],
                )
                gen_embeds = injected["inputs_embeds"]
                gen_mask = injected["attention_mask"]
            else:
                gen_embeds = inputs_embeds
                gen_mask = batch["attention_mask"]

            # Greedy decoding
            outputs = self.model.vlm.generate(
                inputs_embeds=gen_embeds,
                attention_mask=gen_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=False,
                return_dict_in_generate=True,
            )

            input_len = gen_embeds.shape[1]
            gen_ids = outputs.sequences[:, input_len:]

            for i in range(gen_ids.shape[0]):
                pred_text = self.model.processor.tokenizer.decode(
                    gen_ids[i], skip_special_tokens=True,
                ).strip()

                gt = batch["answer"][i] if isinstance(batch["answer"], list) else batch["answer"]
                sample_task = task_type
                if isinstance(batch.get("task_type"), list):
                    sample_task = batch["task_type"][i]

                pred_answer = self.extractor.extract(pred_text, sample_task)
                all_predictions.append(pred_answer)
                all_ground_truths.append(gt)
                all_results.append({
                    "prediction": pred_answer,
                    "raw_output": pred_text,
                    "ground_truth": gt,
                    "task_type": sample_task,
                    "question": batch["question"][i] if isinstance(batch.get("question"), list) else "",
                })

            if (batch_idx + 1) % 50 == 0:
                logger.info(f"  Processed {(batch_idx+1)*self.batch_size} samples...")

        results = compute_metrics(
            all_predictions, all_ground_truths, metric_type,
            task_types=[r["task_type"] for r in all_results],
        )

        elapsed = time.time() - t0
        results["benchmark"] = benchmark_name
        results["num_samples"] = len(all_predictions)
        results["eval_time_s"] = elapsed

        logger.info(
            f"{benchmark_name}: {results['score']:.2f}% "
            f"({results['num_samples']} samples, {elapsed:.1f}s)"
        )

        save_path = os.path.join(self.output_dir, f"{benchmark_name}_results.json")
        with open(save_path, "w") as f:
            json.dump({"metrics": results, "predictions": all_results}, f, indent=2, ensure_ascii=False)

        return results

    def evaluate_all(
        self,
        benchmark_loaders: Dict[str, DataLoader],
        benchmark_configs: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """Evaluate all benchmarks and produce aggregate summary."""
        all_results = {}
        for name, loader in benchmark_loaders.items():
            cfg = benchmark_configs.get(name, {})
            result = self.evaluate_benchmark(
                dataloader=loader,
                benchmark_name=name,
                task_type=cfg.get("task_type", "closed_ended"),
                metric_type=cfg.get("metric", "accuracy"),
            )
            all_results[name] = result

        summary = aggregate_results(all_results)
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved: {summary_path}")
        return summary

    @property
    def _device(self) -> torch.device:
        return next(self.model.vlm.parameters()).device


def _to_device(batch: Dict, device: torch.device) -> Dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }