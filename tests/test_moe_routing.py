import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from idempotent_moe import InplaceMoERouter, generate_idempotent_moe_map, compact_moe_tokens_inplace

def test_moe_in_situ_routing():
    assert torch.cuda.is_available(), "CUDA required for GPU testing"
    device = torch.device("cuda:0")

    E = 8      # 8 experts
    N = 2048   # 2048 candidate tokens per expert
    D = 512    # 512 hidden dimension
    C = 1024   # 50% capacity factor

    print(f"Testing idempotent-moe on {torch.cuda.get_device_name(0)}...")
    print(f"Shape: Experts={E}, Tokens={N}, HiddenDim={D}, Capacity={C}")

    tokens = torch.randn((E, N, D), dtype=torch.float16, device=device)
    routing_scores = torch.rand((E, N), dtype=torch.float32, device=device)

    # Reference Out-of-Place Gather
    target_map = generate_idempotent_moe_map(routing_scores, C, device=device)
    ref_out = torch.gather(tokens, 1, target_map.unsqueeze(-1).expand_as(tokens))[:, :C, :].clone()

    # In-Situ Compaction via High-Level Router
    router = InplaceMoERouter(hidden_dim=D, num_experts=E, block_d=128)
    in_situ_out = router(tokens.clone(), routing_scores, expert_capacity=C)

    # In-Situ Kernel Zero-Allocation Verification
    tokens_inplace = tokens.clone()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    vram_before = torch.cuda.memory_allocated(device)

    compact_moe_tokens_inplace(tokens_inplace, target_map, block_d=128)

    vram_after = torch.cuda.max_memory_allocated(device)
    aux_memory = vram_after - vram_before

    max_diff = torch.max(torch.abs(in_situ_out - ref_out)).item()
    nan_count = torch.isnan(in_situ_out).sum().item()

    print(f"Kernel Auxiliary Memory Allocated: {aux_memory} bytes")
    print(f"Maximum Parity Difference: {max_diff:.6f}")
    print(f"NaN Detections: {nan_count}")

    assert aux_memory == 0, f"Expected 0 bytes auxiliary memory, got {aux_memory}"
    assert max_diff == 0.0, f"Expected exact numerical parity, got diff={max_diff}"
    assert nan_count == 0, "NaN detected in compacted output"

    print("[SUCCESS] Test passed! In-situ MoE routing is 100% Zero-Copy, Bit-exact and NaN-free.")

if __name__ == "__main__":
    test_moe_in_situ_routing()
