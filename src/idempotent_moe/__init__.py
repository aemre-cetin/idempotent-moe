"""
idempotent-moe: Hardware-Accelerated Zero-Copy In-Place Token Router for Sparse Mixture-of-Experts (MoE)
Protected under U.S. Patent Application No. 64/148,668 ("Patent Pending").
Author: Dr. A. Emre ÇETİN
"""

__version__ = "0.1.0"
__author__ = "Dr. A. Emre ÇETİN"
__patent__ = "U.S. Patent Application No. 64/148,668 (Patent Pending)"

from .kernel import compact_moe_tokens_inplace
from .router import InplaceMoERouter, generate_idempotent_moe_map

__all__ = [
    "compact_moe_tokens_inplace",
    "InplaceMoERouter",
    "generate_idempotent_moe_map",
]
