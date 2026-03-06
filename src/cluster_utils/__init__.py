"""
cluster_utils – portable HPC environment & cache management.

A minimal, cluster-agnostic toolkit for:
  • Setting up HuggingFace cache paths (models, datasets, tokens)
  • Detecting the runtime environment (cluster / login / compute / local)
  • Managing offline/online mode for air-gapped compute nodes

Designed to work the same way on Leonardo, LUMI, or a laptop.

Usage:
    from cluster_utils import HFCacheManager

    hf = HFCacheManager()          # auto-detects paths
    hf.download_model("Qwen/Qwen2.5-0.5B")
    hf.setup_environment(offline=True)
"""

__version__ = "0.3.0"

from cluster_utils.cluster import ClusterConfig, detect_cluster
from cluster_utils.hf_cache_manager import HFCacheManager

__all__ = ["ClusterConfig", "HFCacheManager", "detect_cluster"]
