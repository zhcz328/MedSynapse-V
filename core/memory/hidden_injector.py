import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List


class HiddenStreamInjector(nn.Module):
    """
    Manages the injection of diagnostic implicit memory tokens into the
    VLM's input embedding sequence.

    """

    def __init__(
        self,
        num_memory_tokens: int = 16,
        d_hidden: int = 4096,
    ):
        super().__init__()
        self.num_memory_tokens = num_memory_tokens
        self.d_hidden = d_hidden

    def compute_injection_position(
        self,
        input_ids: torch.Tensor,
        image_token_id: int,
        pad_token_id: int,
    ) -> torch.Tensor:
        """
        Determine the injection position for each sample in the batch.
        Memory is injected after the last non-pad, non-image token of the
        question encoding and before the answer generation position.

        Returns:
            positions: (B,) tensor of injection indices
        """
        B, L = input_ids.shape
        positions = torch.zeros(B, dtype=torch.long, device=input_ids.device)

        for i in range(B):
            seq = input_ids[i]
            # Find last valid token (non-pad)
            non_pad = (seq != pad_token_id).nonzero(as_tuple=True)[0]
            if len(non_pad) == 0:
                positions[i] = 0
            else:
                positions[i] = non_pad[-1].item() + 1

        return positions

    def inject(
        self,
        inputs_embeds: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        injection_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Insert memory tokens into the embedding sequence.

        Args:
            inputs_embeds: (B, L, d_h) VLM input embeddings
            memory:        (B, N, d_h) diagnostic implicit memory
            attention_mask: (B, L) original attention mask
            position_ids:  (B, L) optional position IDs
            injection_positions: (B,) positions to inject memory

        Returns:
            dict with keys: inputs_embeds, attention_mask, position_ids
        """
        B, L, D = inputs_embeds.shape
        N = memory.shape[1]
        device = inputs_embeds.device

        if injection_positions is None:
            # Default: append after all existing tokens
            injection_positions = attention_mask.sum(dim=1).long()

        new_L = L + N
        new_embeds = torch.zeros(B, new_L, D, dtype=inputs_embeds.dtype, device=device)
        new_mask = torch.zeros(B, new_L, dtype=attention_mask.dtype, device=device)

        for i in range(B):
            pos = injection_positions[i].item()

            # Tokens before injection point
            new_embeds[i, :pos] = inputs_embeds[i, :pos]
            new_mask[i, :pos] = attention_mask[i, :pos]

            # Inject memory tokens
            new_embeds[i, pos:pos + N] = memory[i]
            new_mask[i, pos:pos + N] = 1

            # Tokens after injection point
            remaining = L - pos
            if remaining > 0:
                new_embeds[i, pos + N:pos + N + remaining] = inputs_embeds[i, pos:pos + remaining]
                new_mask[i, pos + N:pos + N + remaining] = attention_mask[i, pos:pos + remaining]

        # Reconstruct position IDs
        new_position_ids = None
        if position_ids is not None:
            new_position_ids = torch.zeros(B, new_L, dtype=position_ids.dtype, device=device)
            for i in range(B):
                pos = injection_positions[i].item()
                new_position_ids[i, :pos] = position_ids[i, :pos]
                # Memory tokens get sequential positions
                if pos > 0:
                    start_pos = position_ids[i, pos - 1].item() + 1
                else:
                    start_pos = 0
                new_position_ids[i, pos:pos + N] = torch.arange(
                    start_pos, start_pos + N, device=device
                )
                remaining = L - pos
                if remaining > 0:
                    new_position_ids[i, pos + N:pos + N + remaining] = torch.arange(
                        start_pos + N,
                        start_pos + N + remaining,
                        device=device,
                    )

        result = {
            "inputs_embeds": new_embeds,
            "attention_mask": new_mask,
        }
        if new_position_ids is not None:
            result["position_ids"] = new_position_ids

        return result

    def create_memory_mask(
        self,
        total_length: int,
        memory_start: int,
        memory_end: int,
        causal: bool = True,
        device: torch.device = None,
    ) -> torch.Tensor:

        mask = torch.ones(total_length, total_length, device=device)
        if causal:
            mask = torch.tril(mask)

        # Memory tokens attend to everything before them and each other
        mask[memory_start:memory_end, :memory_end] = 1

        # Generation tokens attend to memory
        mask[memory_end:, memory_start:memory_end] = 1

        return mask
