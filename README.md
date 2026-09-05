# idempotent-moe: Zero-Copy Sparse MoE Dynamic Token Router for LLMs

[![Hugging Face RFC](https://img.shields.io/badge/Hugging%20Face%20RFC-%2348548-yellow.svg)](https://github.com/huggingface/transformers/issues/48548)
[![arXiv](https://img.shields.io/badge/arXiv-submit%2F8040718-b31b1b.svg)](https://arxiv.org)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Publication%20414015429-00CCBB.svg)](https://www.researchgate.net/publication/414015429_Zero-Copy_In-Place_Dynamic_Token_Routing_and_Capacity_Compaction_for_Sparse_Mixture-of-Experts_Accelerators)
[![Paper](https://img.shields.io/badge/Research%20Paper-PDF-red.svg)](paper/idempotent_moe_paper.pdf)
[![PyPI](https://img.shields.io/pypi/v/idempotent-moe.svg)](https://pypi.org/project/idempotent-moe/)
[![Patent Pending](https://img.shields.io/badge/Patent-Pending%20(US%2064%2F148%2C668)-blue.svg)](https://uspto.gov)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Hardware](https://img.shields.io/badge/Tested%20on-NVIDIA%20Blackwell%20sm__120-purple.svg)]()
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)]()


> **Eliminate 100% of auxiliary VRAM allocations during Sparse Mixture-of-Experts (MoE) token dispatching and capacity dropping in Mixtral, DeepSeek, Megatron-LM, and Hugging Face models.**

---

## 🚀 The Bottleneck: MoE Dispatch Memory Wall

In modern Sparse Mixture-of-Experts architectures (e.g., Mixtral 8x7B/8x22B, DeepSeek-V2/V3, Grok-1), tokens are dynamically gated to top-$k$ experts. To balance hardware computation across distributed accelerators, expert capacity limits are enforced and excess tokens are dropped or re-routed.

Standard deep learning frameworks (Megatron-LM, DeepSpeed-MoE, PyTorch native) implement token routing via out-of-place tensor gathers (`torch.gather` / `cudaMalloc`):
1. **Auxiliary Buffer Bloat:** Allocating secondary token buffers ($O(E \cdot C \cdot D)$) consumes hundreds of megabytes to gigabytes of transient VRAM across layers.
2. **HBM Bandwidth Saturation:** Redundant global memory read/write cycles congest accelerator high-bandwidth memory (HBM).
3. **Dynamic Memory Fragmentation:** Continuous allocation and deallocation during token routing causes severe heap fragmentation and latency spikes.

---

## ⚡ The Solution: In-Situ Idempotent Token Routing

`idempotent-moe` rearranges high-dimensional token representations **directly within their existing tensor allocations** using **$O(1)$ scalar hardware registers**:
- **Idempotent Invariant:** Enforces $f(f(x)) = f(x)$, locking routed expert tokens into stabilized contiguous sub-tensors $[0, C-1]$.
- **Bitmask-Free Cycle Follower:** Disjoint permutation cycles are resolved in-situ on GPU streaming multiprocessors without auxiliary bitmasks or auxiliary global memory.
- **In-Register 2-Cycle Fast-Path:** Mutually transposed tokens are swapped directly across thread registers with zero memory overhead.
- **100% Zero Auxiliary VRAM:** Exactly **0.00 MB** auxiliary secondary memory allocated.
- **Bit-Exact Numerical Parity:** Zero approximation error ($\Delta = 0.000000$, 0 NaN).

---

## 📊 Benchmark: NVIDIA RTX PRO 500 Blackwell (`sm_120`)

*Workload: 8 Experts, 4,096 Tokens/Expert (32,768 tokens total), HiddenDim=1,024, Capacity=2,048 (50% Token Dropping, float16)*

| Implementation | Latency (ms) | Throughput | Peak Aux VRAM | VRAM Saved | Numerical Diff |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PyTorch Out-of-Place** | 1.631 ms | 20.09 M tok/s | 96.00 MB | Baseline | 0.000000 |
| **`idempotent-moe` (Ours)** | **0.923 ms** | **35.49 M tok/s** | **0.00 MB** | **96.00 MB (100%)** | **0.000000** |
| **Improvement** | **1.77x Faster** | **+76.7% Throughput** | **0.00 MB** | **100% Eliminated** | **Bit-Exact** |

---

## 📦 Installation

```bash
git clone https://github.com/aemre-cetin/idempotent-moe.git
cd idempotent-moe
pip install -e .
```

Requirements: `torch >= 2.0.0`, `triton >= 2.1.0`.

---

## 🛠️ Quickstart

```python
import torch
from idempotent_moe import InplaceMoERouter

# Initialize In-Place MoE Router
router = InplaceMoERouter(hidden_dim=1024, num_experts=8, block_d=128)

# Candidate token representations [Experts, Tokens, HiddenDim]
tokens = torch.randn((8, 4096, 1024), dtype=torch.float16, device="cuda")

# Router gating affinity scores [Experts, Tokens]
routing_scores = torch.rand((8, 4096), dtype=torch.float32, device="cuda")

# In-place compaction: routes top-2048 tokens per expert with 0 bytes aux VRAM
compacted_tokens = router(tokens, routing_scores, expert_capacity=2048)

# Output shape: [8, 2048, 1024] directly contiguous in memory
print("Compacted tokens shape:", compacted_tokens.shape)
```

### Hugging Face MoE Integration Hook

```python
from idempotent_moe.integrations import HuggingFaceMoEInplaceHook

# Attach directly to Mixtral / DeepSeek MoE layer
hook = HuggingFaceMoEInplaceHook(capacity=2048, block_d=128)

# Apply in-place compaction during model forward pass
compacted_expert_inputs = hook(expert_inputs, gating_weights)
```

> 🚀 **Official Hugging Face Upstreaming Proposal:** Track the community RFC and integration discussion at [Hugging Face Transformers Issue #48548](https://github.com/huggingface/transformers/issues/48548).

---

## 🛡️ Patent & Intellectual Property Notice

The mathematical formulations, state-transition architectures, and in-situ hardware compaction kernels implemented in this library are protected under pending patent application with the United States Patent and Trademark Office:

* **U.S. Patent Application Number:** **`64/148,668`**
* **Confirmation Number:** **`5890`**
* **Status:** **PATENT PENDING**
* **First Named Inventor:** **Dr. Ahmet Emre ÇETİN**

Academic evaluation, non-commercial research, and open-source collaboration are permitted under the terms of the Apache 2.0 License. Commercial deployment in proprietary hardware or commercial cloud runtimes is subject to bilateral licensing agreements with the author.

---

## 📜 Academic Citation

```bibtex
@article{cetin2026idempotentmoe,
  title={Zero-Copy In-Place Compaction and Idempotent Associative Routing of Dynamic Key-Value Cache and Sparse Mixture-of-Experts Tensors in Deep Learning Accelerators},
  author={Cetin, A. Emre},
  journal={arXiv preprint},
  year={2026},
  note={U.S. Patent Application No. 64/148,668}
}

@article{cetin2013idempotent,
  title={Idempotent Permutations},
  author={Cetin, A. E.},
  journal={arXiv:1307.3877 [cs.DS]},
  year={2013}
}
```

---

## 📄 License

Licensed under the [Apache License, Version 2.0](LICENSE).
Copyright © 2026 Dr. A. Emre ÇETİN. All Rights Reserved.
