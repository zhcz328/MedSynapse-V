import csv
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import functional as TF

logger = logging.getLogger(__name__)


@dataclass
class MedicalSample:
    image: Image.Image
    question: str
    answer: str
    task_type: str = "closed_ended"
    options: Optional[List[str]] = None
    sample_id: Optional[str] = None
    concept_prompt: Optional[str] = None


class MedicalVQADataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict],
        fields: Dict[str, str],
        image_root: Optional[str] = None,
        task_type: str = "closed_ended",
        max_samples: Optional[int] = None,
        name: str = "medical_vqa",
    ):
        self.records = list(records)
        if max_samples is not None:
            self.records = self.records[: int(max_samples)]
        if not self.records:
            raise ValueError(f"{name} has no samples")
        self.fields = fields or {}
        self.image_root = image_root
        self.task_type = task_type
        self.name = name

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = _load_image(
            _field(record, self.fields.get("image", "image"), ["image", "img", "img_name", "image_path", "path"]),
            self.image_root,
        )
        question = str(_field(record, self.fields.get("question", "question"), ["question", "query", "prompt"]))
        answer = _normalize_answer(_field(record, self.fields.get("answer", "answer"), ["answer", "label", "gt"]))
        options = _normalize_options(_field(record, self.fields.get("options", "options"), ["options", "choices", "candidates"], required=False))
        concept_prompt = _field(
            record,
            self.fields.get("concept_prompt", "concept_prompt"),
            ["concept_prompt", "concept", "anatomy", "region", "target"],
            required=False,
        )
        sample_id = str(record.get("id", record.get("sample_id", f"{self.name}-{idx}")))
        return MedicalSample(
            image=image,
            question=question,
            answer=answer,
            task_type=str(record.get("task_type", self.task_type)),
            options=options,
            sample_id=sample_id,
            concept_prompt=str(concept_prompt) if concept_prompt is not None else None,
        )


class MedicalVQACollator:
    def __init__(
        self,
        processor,
        max_length: int = 1024,
        include_answer: bool = True,
        medsam_size: int = 1024,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = int(max_length)
        self.include_answer = include_answer
        self.medsam_size = int(medsam_size)
        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = getattr(self.tokenizer, "eos_token_id", 0) or 0
        self.eos_token = getattr(self.tokenizer, "eos_token", None) or ""

    def __call__(self, samples: Sequence[MedicalSample]) -> Dict:
        images = [sample.image.convert("RGB") for sample in samples]
        prompts = [self._format_prompt(sample) for sample in samples]
        texts = []
        for prompt, sample in zip(prompts, samples):
            if self.include_answer:
                texts.append(prompt + str(sample.answer) + self.eos_token)
            else:
                texts.append(prompt)

        model_inputs = self._processor_call(texts, images)
        labels = self._build_labels(model_inputs["input_ids"], prompts, images)
        model_inputs["labels"] = labels
        model_inputs["pixel_values_medsam"] = torch.stack(
            [_image_to_medsam_tensor(image, self.medsam_size) for image in images],
            dim=0,
        )
        model_inputs["answer"] = [sample.answer for sample in samples]
        model_inputs["question"] = [sample.question for sample in samples]
        model_inputs["task_type"] = [sample.task_type for sample in samples]
        model_inputs["options"] = [sample.options for sample in samples]
        model_inputs["sample_id"] = [sample.sample_id for sample in samples]
        model_inputs["medsam3_prompt"] = [sample.concept_prompt for sample in samples]
        return model_inputs

    def _processor_call(self, texts: Sequence[str], images: Sequence[Image.Image]) -> Dict:
        try:
            encoded = self.processor(
                text=list(texts),
                images=list(images),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        except TypeError:
            encoded = self.processor(
                list(texts),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        return dict(encoded)

    def _build_labels(
        self,
        input_ids: torch.Tensor,
        prompts: Sequence[str],
        images: Sequence[Image.Image],
    ) -> torch.Tensor:
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100
        if not self.include_answer:
            labels[:] = -100
            return labels

        for row, (prompt, image) in enumerate(zip(prompts, images)):
            prompt_len = self._prompt_length(prompt, image)
            labels[row, : min(prompt_len, labels.shape[1])] = -100
        return labels

    def _prompt_length(self, prompt: str, image: Image.Image) -> int:
        try:
            encoded = self.processor(
                text=[prompt],
                images=[image],
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            return int(encoded["input_ids"].shape[1])
        except Exception:
            token_ids = self.tokenizer.encode(
                prompt,
                add_special_tokens=False,
                max_length=self.max_length,
                truncation=True,
            )
            return len(token_ids)

    def _format_prompt(self, sample: MedicalSample) -> str:
        user_text = sample.question.strip()
        if sample.options:
            labels = "ABCDE"
            options = " ".join(
                f"{labels[i]}. {opt}" for i, opt in enumerate(sample.options[:5])
            )
            user_text = f"{user_text}\nOptions: {options}\nAnswer with the option letter only."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        if hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"<image>\nQuestion: {user_text}\nAnswer: "


def build_stage_dataloaders(
    stage_config: Dict,
    data_config: Dict,
    processor,
    stage: int,
    seed: int = 42,
    fold_idx: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    train_cfg = stage_config.get("training", {})
    data_runtime = stage_config.get("data", {})
    loader_cfg = data_config.get("dataloader", {})
    batch_size = int(train_cfg.get("batch_size", train_cfg.get("rollout_batch_size", 1)))
    max_length = int(
        data_runtime.get(
            "max_length",
            data_config.get("preprocessing", {}).get("max_seq_length", 1024),
        )
    )
    dataset = build_dataset_from_config(data_config)

    eval_ratio = float(data_runtime.get("eval_ratio", data_config.get("validation_ratio", 0.0)))
    eval_ds = None
    train_ds = dataset
    if eval_ratio > 0:
        eval_size = max(1, int(len(dataset) * eval_ratio))
        train_size = len(dataset) - eval_size
        train_ds, eval_ds = random_split(
            dataset,
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(seed + fold_idx),
        )

    include_answer = stage == 1
    collator = MedicalVQACollator(
        processor,
        max_length=max_length,
        include_answer=include_answer,
        medsam_size=int(data_runtime.get("medsam_size", 1024)),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=bool(data_config.get("preprocessing", {}).get("shuffle", True)),
        num_workers=int(loader_cfg.get("num_workers", 4)),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        drop_last=bool(loader_cfg.get("drop_last", False)),
        collate_fn=collator,
    )
    eval_loader = None
    if eval_ds is not None:
        eval_loader = DataLoader(
            eval_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
        )
    logger.info("Loaded stage%s dataset: %s train samples", stage, len(train_ds))
    return train_loader, eval_loader


def build_eval_loaders(
    eval_config: Dict,
    processor,
    benchmarks: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, DataLoader], Dict[str, Dict]]:
    selected = set(benchmarks or [])
    loaders, configs = {}, {}
    eval_cfg = eval_config.get("evaluation", {})
    batch_size = int(eval_cfg.get("batch_size", 1))
    max_length = int(eval_cfg.get("max_length", 1024))
    max_samples = eval_cfg.get("max_samples_per_benchmark")

    for bench in eval_config.get("benchmarks", []):
        name = bench["name"]
        if selected and name not in selected:
            continue
        cfg = {
            "dataset": {
                "name": name,
                "type": bench.get("type", "huggingface"),
                "path": bench.get("path"),
                "subset": bench.get("subset"),
                "split": bench.get("split", "test"),
                "image_root": bench.get("image_root"),
            },
            "fields": bench.get("fields", {}),
            "task_type": bench.get("task_type", "closed_ended"),
            "max_samples": bench.get("num_samples", max_samples),
        }
        dataset = build_dataset_from_config(cfg)
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(eval_cfg.get("num_workers", 0)),
            collate_fn=MedicalVQACollator(
                processor,
                max_length=max_length,
                include_answer=False,
                medsam_size=int(eval_cfg.get("medsam_size", 1024)),
            ),
        )
        configs[name] = bench
    return loaders, configs


def build_dataset_from_config(data_config: Dict) -> Dataset:
    if "sources" in data_config:
        datasets = []
        for source in data_config["sources"]:
            source_cfg = {
                "dataset": source,
                "fields": source.get("fields", {}),
                "task_type": source.get("task_type", data_config.get("task_type", "closed_ended")),
                "max_samples": source.get("num_samples"),
            }
            datasets.append(build_dataset_from_config(source_cfg))
        return ConcatDataset(datasets)

    dataset_cfg = {}
    dataset_cfg.update(data_config.get("source", {}))
    dataset_cfg.update(data_config.get("dataset", {}))
    if "path" not in dataset_cfg and "source" in data_config:
        dataset_cfg["path"] = data_config["source"].get("path")
    if "type" not in dataset_cfg and "source" in data_config:
        dataset_cfg["type"] = data_config["source"].get("type")
    if "subset" not in dataset_cfg and "source" in data_config:
        dataset_cfg["subset"] = data_config["source"].get("subset")
    if "split" not in dataset_cfg and "dataset" in data_config:
        dataset_cfg["split"] = data_config["dataset"].get("split")
    source_type = dataset_cfg.get("type", data_config.get("source", {}).get("type", "huggingface"))
    fields = data_config.get("fields", dataset_cfg.get("fields", {}))
    task_type = data_config.get("task_type", dataset_cfg.get("task_type", "closed_ended"))
    max_samples = data_config.get(
        "max_samples",
        dataset_cfg.get("num_samples", dataset_cfg.get("total_samples")),
    )
    name = dataset_cfg.get("name", dataset_cfg.get("path", "medical_vqa"))

    if source_type == "huggingface":
        records = _load_huggingface_records(dataset_cfg, max_samples=max_samples)
        return MedicalVQADataset(
            records=records,
            fields=fields,
            image_root=dataset_cfg.get("image_root"),
            task_type=task_type,
            max_samples=max_samples,
            name=name,
        )
    if source_type in {"local", "json", "jsonl", "csv"}:
        records = _load_local_records(dataset_cfg["path"])
        return MedicalVQADataset(
            records=records,
            fields=fields,
            image_root=dataset_cfg.get("image_root"),
            task_type=task_type,
            max_samples=max_samples,
            name=name,
        )
    raise ValueError(f"Unsupported dataset source type: {source_type}")


def _load_huggingface_records(dataset_cfg: Dict, max_samples=None) -> List[Dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for HuggingFace data loading") from exc

    path = dataset_cfg.get("path")
    if not path:
        raise ValueError("HuggingFace dataset path is required")
    subset = dataset_cfg.get("subset")
    split = dataset_cfg.get("split", "train")
    kwargs = {}
    if subset:
        ds = load_dataset(path, subset, split=split, **kwargs)
    else:
        ds = load_dataset(path, split=split, **kwargs)
    if max_samples is not None:
        ds = ds.select(range(min(int(max_samples), len(ds))))
    return [dict(item) for item in ds]


def _load_local_records(path: str) -> List[Dict]:
    if not path:
        raise ValueError("Local dataset path is required")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    if ext == ".json":
        with open(path) as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = payload.get("data", payload.get("samples", []))
        return list(payload)
    if ext == ".csv":
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported local data extension: {ext}")


def _load_image(value, image_root: Optional[str]) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if "bytes" in value and value["bytes"] is not None:
            from io import BytesIO

            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        value = value.get("path") or value.get("filename")
    if isinstance(value, torch.Tensor):
        if value.ndim == 3:
            return TF.to_pil_image(value).convert("RGB")
        raise ValueError("Image tensor must have shape CxHxW or HxWxC")
    if isinstance(value, str):
        path = value
        if image_root and not os.path.isabs(path):
            path = os.path.join(image_root, path)
        return Image.open(path).convert("RGB")
    raise ValueError(f"Unsupported image value: {type(value)!r}")


def _field(record: Dict, preferred: str, fallbacks: Sequence[str], required: bool = True):
    keys = [preferred] + [key for key in fallbacks if key != preferred]
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    if required:
        raise KeyError(f"Missing required field. Tried: {keys}")
    return None


def _image_to_medsam_tensor(image: Image.Image, size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((size, size), Image.BILINEAR)
    return TF.to_tensor(image) * 255.0


def _normalize_options(value) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, dict):
        return [str(value[k]) for k in sorted(value)]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
            return _normalize_options(parsed)
        except Exception:
            parts = [p.strip() for p in stripped.replace("|", "\n").split("\n")]
            return [p for p in parts if p]
    return [str(value)]


def _normalize_answer(value) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    if isinstance(value, dict):
        for key in ("answer", "label", "text", "value"):
            if key in value:
                return str(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)
