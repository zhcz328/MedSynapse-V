import logging
import os
import shutil
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save(self, payload: Dict[str, Any], step: int, tag: str = "checkpoint") -> str:
        payload = _to_cpu(payload)
        payload.setdefault("step", step)
        payload.setdefault("tag", tag)
        path = os.path.join(self.output_dir, f"checkpoint_{tag}_{step}.pt")
        torch.save(payload, path)
        last_path = os.path.join(self.output_dir, "checkpoint_last.pt")
        shutil.copyfile(path, last_path)
        if "final" in tag or "best" in tag:
            final_path = os.path.join(self.output_dir, "checkpoint_final.pt")
            best_path = os.path.join(self.output_dir, "checkpoint_best.pt")
            legacy_best_path = os.path.join(self.output_dir, "checkpoint_best")
            shutil.copyfile(path, final_path)
            shutil.copyfile(path, best_path)
            shutil.copyfile(path, legacy_best_path)
        logger.info("Saved checkpoint: %s", path)
        return path


class Checkpointer:
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir

    def load(self, model, checkpoint: str, strict: bool = False) -> Optional[Dict[str, Any]]:
        path = self._resolve(checkpoint)
        if path is None:
            logger.warning("Checkpoint not found: %s", checkpoint)
            return None
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)

        if "model" in state:
            _load_module(model, state["model"], strict=strict)
        else:
            loaded_any = False
            for key, module_name in [
                ("memory_sampler", "memory_sampler"),
                ("autonomous_module", "autonomous_module"),
                ("vlm", "vlm"),
                ("vlm_lora", "vlm"),
            ]:
                if key in state and getattr(model, module_name, None) is not None:
                    _load_module(getattr(model, module_name), state[key], strict=False)
                    loaded_any = True
            if not loaded_any:
                _load_module(model, state, strict=strict)

        logger.info("Loaded checkpoint: %s", path)
        return ckpt

    def _resolve(self, checkpoint: str) -> Optional[str]:
        candidates = [checkpoint]
        if not checkpoint.endswith(".pt"):
            candidates.append(f"{checkpoint}.pt")
        if os.path.isdir(checkpoint):
            candidates.extend(
                [
                    os.path.join(checkpoint, "checkpoint_best.pt"),
                    os.path.join(checkpoint, "checkpoint_best"),
                    os.path.join(checkpoint, "checkpoint_final.pt"),
                    os.path.join(checkpoint, "checkpoint_last.pt"),
                ]
            )
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None


def _load_module(module, state_dict: Dict[str, Any], strict: bool):
    if not hasattr(module, "load_state_dict"):
        return
    clean_state = {
        k: v for k, v in state_dict.items()
        if isinstance(v, torch.Tensor)
    }
    missing, unexpected = module.load_state_dict(clean_state, strict=strict)
    if missing:
        logger.debug("Missing keys: %s", missing[:10])
    if unexpected:
        logger.debug("Unexpected keys: %s", unexpected[:10])


def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj
