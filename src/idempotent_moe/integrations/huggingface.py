import torch
import torch.nn as nn
from ..router import InplaceMoERouter

class HuggingFaceMoEInplaceHook(nn.Module):
    """
    Drop-in In-Situ Token Router Hook for Hugging Face Sparse Mixture-of-Experts (MoE) Models.
    Compatible with Mixtral-8x7B, DeepSeek-MoE, and Qwen-MoE architectures.
    
    Eliminates secondary gather/dispatch buffers in expert feed-forward layers.
    """
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2, block_d: int = 128):
        super().__init__()
        self.router = InplaceMoERouter(hidden_dim=hidden_dim, num_experts=num_experts, block_d=block_d)
        self.top_k = top_k

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor, capacity: int) -> torch.Tensor:
        """
        Routes tokens in-place to expert partitions.

        Args:
            hidden_states: [E, N, D] token activations partitioned by expert
            router_logits: [E, N] gating affinity logits or scores
            capacity:      Capacity limit per expert

        Returns:
            compacted_tokens: [E, capacity, D] contiguous active tensor
        """
        return self.router(hidden_states, router_logits, expert_capacity=capacity)
