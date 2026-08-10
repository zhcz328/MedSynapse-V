from typing import Dict

import torch


VLM_TENSOR_KEYS = (
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "second_per_grid_ts",
    "mm_token_type_ids",
)


def batch_vlm_kwargs(batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    kwargs = {}
    for key in VLM_TENSOR_KEYS:
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            kwargs[key] = value.to(device)
    return kwargs


def extract_generated_token_ids(
    sequences: torch.Tensor,
    context_length: int,
    prompt_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    HF generate() may return either full prompt+generation sequences or only
    the newly generated tokens when called with `inputs_embeds`.

    The injected diagnostic memory lives only in embedding space and is not
    represented in `sequences`, so `context_length` must be the textual prompt
    width (for example `input_ids.shape[1]`), not the post-injection embedding
    length.
    """
    if (
        prompt_ids is not None
        and prompt_ids.ndim == 2
        and sequences.ndim == 2
        and sequences.shape[1] >= context_length
    ):
        prompt_prefix = prompt_ids[:, :context_length].to(sequences.device)
        if torch.equal(sequences[:, :context_length], prompt_prefix):
            return sequences[:, context_length:]
        return sequences

    if context_length > 0 and sequences.shape[1] > context_length:
        return sequences[:, context_length:]
    return sequences


def lora_state_dict(module) -> Dict[str, torch.Tensor]:
    try:
        from peft import get_peft_model_state_dict

        state = get_peft_model_state_dict(module)
        if state:
            return state
    except Exception:
        pass

    return {
        name: param.detach()
        for name, param in module.named_parameters()
        if param.requires_grad or "lora_" in name
    }
