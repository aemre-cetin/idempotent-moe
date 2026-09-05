# arXiv & ResearchGate Submission Metadata: Pillar 3 (idempotent-moe)

## 1. arXiv Metadata

- **Title:**
  Zero-Copy In-Place Dynamic Token Routing and Capacity Compaction for Sparse Mixture-of-Experts Accelerators

- **Authors:**
  Dr. A. Emre ÇETİN (aemre.cetin@gmail.com)

- **Primary Category:**
  `cs.LG` (Machine Learning)

- **Secondary Categories:**
  `cs.DC` (Distributed, Computing, and Cluster Computing), `cs.AI` (Artificial Intelligence), `cs.AR` (Hardware Architecture)

- **Comments:**
  4 pages, 3 figures. Reference implementation and Triton kernels available at https://github.com/aemre-cetin/idempotent-moe. Protected under U.S. Patent Application No. 64/148,668.

- **ACM Classification:**
  C.1.4; I.2.7; B.3.2

- **MSC Classification:**
  68W10; 68M07

- **Submission ID / Tracking:**
  arXiv Submission ID: `submit/8040718`
  ResearchGate Publication: [Publication 414015429](https://www.researchgate.net/publication/414015429_Zero-Copy_In-Place_Dynamic_Token_Routing_and_Capacity_Compaction_for_Sparse_Mixture-of-Experts_Accelerators)

### Abstract (Formatted for arXiv Form):
Sparse Mixture-of-Experts (MoE) architectures, such as Mixtral, DeepSeek-V2/V3, and Switch Transformers, scale model parameter capacity while sustaining constant computational FLOP budgets per token. However, executing dynamic token routing on parallel accelerators incurs severe systems overheads: when tokens are conditionally dispatched to experts, capacity factor constraints necessitate token dropping and dynamic tensor repacking. Conventional MoE frameworks (e.g., Megatron-LM, DeepSpeed-MoE) rely on out-of-place memory gather/scatter passes (O(E * C * D) auxiliary allocations via host cudaMalloc calls) and cause severe memory fragmentation that degrades downstream Tensor Core General Matrix Multiplication (GEMM) efficiency. In this paper, we propose a novel hardware-software co-designed architecture and high-performance GPU execution kernel for zero-copy in-place MoE token routing and capacity compaction. By formulating token dispatch as an algebraic mapping satisfying the idempotence condition (f(f(x)) = f(x)), our method stabilizes retained high-affinity tokens into invariant fixed points and resolves capacity truncations through mutually disjoint permutation cycles. We map the kernel across a 2D accelerator execution grid over parallel experts and hidden dimension tiles, verifying minimal-index cycle leaders with strictly O(1) scalar auxiliary register memory without boolean marking bitmasks. We evaluate our implementation on an enterprise NVIDIA Blackwell GPU accelerator (sm_120) with 8 experts, 32,768 candidate tokens, and a hidden dimension of 1,024 (Float16). Empirical results demonstrate a 100% elimination of peak auxiliary VRAM (dropping from 96.00 MB to exactly 0.00 MB), a 1.73x latency speedup over native PyTorch out-of-place gather (reducing dispatch time from 1.581 ms to 0.915 ms), a throughput of 35.80 Million tokens/second, and guaranteed physical memory contiguity for dense GEMM execution.

---

## 2. ResearchGate Submission Metadata

- **Title:**
  Zero-Copy In-Place Dynamic Token Routing and Capacity Compaction for Sparse Mixture-of-Experts Accelerators

- **Publication Type:**
  Preprint / Research Article (Live at Publication 414015429)

- **Author & Affiliation:**
  Dr. A. Emre ÇETİN (Computational Systems and Cognitive Architectures, Izmir, Turkey)

- **Skills / Topics:**
  Mixture of Experts, LLM, Transformers, Hugging Face, DeepSeek, Mixtral, GPU Acceleration, Triton, Zero-Copy

- **Patent Disclosure:**
  U.S. Patent Application No. 64/148,668 ("Patent Pending", Confirmation No. 5890).

- **Associated Links:**
  - GitHub Repository: https://github.com/aemre-cetin/idempotent-moe
  - PyPI Package: https://pypi.org/project/idempotent-moe/
  - Hugging Face Transformers RFC: https://github.com/huggingface/transformers/issues/48548