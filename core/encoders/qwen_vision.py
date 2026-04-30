"""
Qwen3-VL Vision Encoder Wrapper.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class QwenVisionFeatureExtractor(nn.Module):
    """
    Extracts visual encoding features from a Qwen3-VL model's
    internal vision tower for use by the Autonomous Memory Module.
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_dim: int = 4096,
        pool_strategy: str = "mean",
        extract_layer: int = -1,
    ):
        super().__init__()
        self.model = model
        self.hidden_dim = hidden_dim
        self.pool_strategy = pool_strategy
        self.extract_layer = extract_layer
        self._hooks = []
        self._features = {}

    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""
        self._clear_hooks()

        def hook_fn(name):
            def fn(module, input, output):
                if isinstance(output, tuple):
                    self._features[name] = output[0]
                else:
                    self._features[name] = output
            return fn

        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            target_layer = self.model.model.layers[self.extract_layer]
            h = target_layer.register_forward_hook(hook_fn("hidden_states"))
            self._hooks.append(h)

    def _clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._features = {}

    @torch.no_grad()
    def extract_visual_features(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run the VLM forward pass and extract visual token features.

        Returns:
            visual_features: (B, L_v, d_h) visual encoding features
        """
        self._register_hooks()

        try:
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            if "hidden_states" in self._features:
                hidden = self._features["hidden_states"]
            elif hasattr(outputs, "hidden_states") and outputs.hidden_states:
                hidden = outputs.hidden_states[self.extract_layer]
            else:
                hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else None

            if hidden is None:
                raise RuntimeError("Failed to extract hidden states from VLM")

            return hidden

        finally:
            self._clear_hooks()

    def extract_visual_token_mask(
        self,
        input_ids: torch.Tensor,
        image_token_id: int = 151655,
    ) -> torch.Tensor:
        """
        Create a boolean mask identifying visual token positions.

        Args:
            input_ids: (B, L) token IDs
            image_token_id: Qwen3-VL image placeholder token ID

        Returns:
            mask: (B, L) boolean tensor
        """
        return input_ids == image_token_id

    def pool_visual_features(
        self,
        hidden_states: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool visual token features from hidden states using the visual mask.

        Args:
            hidden_states: (B, L, d_h) full hidden states
            visual_mask:   (B, L) boolean mask for visual tokens

        Returns:
            pooled: (B, L_v, d_h) visual features only
        """
        B, L, D = hidden_states.shape
        results = []

        for i in range(B):
            vis_indices = visual_mask[i].nonzero(as_tuple=True)[0]
            if len(vis_indices) > 0:
                vis_feats = hidden_states[i, vis_indices]  # (L_v_i, D)
            else:
                vis_feats = hidden_states[i, :1]  # fallback
            results.append(vis_feats)

        # Pad to max visual length in batch
        max_len = max(r.shape[0] for r in results)
        padded = torch.zeros(B, max_len, D, dtype=hidden_states.dtype, device=hidden_states.device)
        for i, r in enumerate(results):
            padded[i, :r.shape[0]] = r

        return padded

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_token_id: int = 151655,
    ) -> torch.Tensor:
        """
        Complete pipeline: extract hidden states → isolate visual tokens → return.

        Returns:
            visual_features: (B, L_v, d_h) pooled visual features
        """
        hidden = self.extract_visual_features(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

        vis_mask = self.extract_visual_token_mask(input_ids, image_token_id)
        visual_features = self.pool_visual_features(hidden, vis_mask)

        return visual_features
