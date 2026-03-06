"""
Cluster detection and configuration.

Auto-detects the current HPC cluster from hostname, environment
variables, and filesystem probes.  Does NOT guess paths — those
are set interactively by setup.sh and stored in .env.

Usage::

    from cluster_utils.cluster import detect_cluster

    cluster = detect_cluster()           # auto-detect
    cluster = detect_cluster("leonardo") # explicit override

    print(cluster.name)                  # "leonardo"
    print(cluster.display_name)          # "Leonardo (Cineca)"
    print(cluster.is_hpc)               # True
    print(cluster.node_type)            # "login"
    print(cluster.slurm)               # ClusterSlurm(...)

    # Shell-friendly dump for setup.sh / env.sh
    cluster.print_shell_exports()
"""

from __future__ import annotations

import os
import platform
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Locate the bundled clusters.toml ──
_REGISTRY_PATH = Path(__file__).parent / "clusters.toml"


# ═══════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════
@dataclass
class ClusterSlurm:
    """SLURM defaults from the registry (user can override in .env)."""
    default_partition: str = ""
    gpu_partition: str = ""
    queue_limit: int = 0
    gpus_per_node: int = 1
    cpus_per_gpu: int = 8


@dataclass
class ClusterContainer:
    """Container runtime settings."""
    runtime: str = ""
    gpu_args: str = ""


@dataclass
class ClusterConfig:
    """Detected cluster info + SLURM/container defaults.

    Does NOT contain filesystem paths — those live in .env
    and are set interactively by setup.sh.
    """
    name: str = "local"
    display_name: str = "Local machine"
    slurm: ClusterSlurm = field(default_factory=ClusterSlurm)
    container: ClusterContainer = field(default_factory=ClusterContainer)

    @property
    def is_hpc(self) -> bool:
        """True if this is a recognised HPC cluster (not local)."""
        return self.name != "local"

    @property
    def is_compute_node(self) -> bool:
        """True if we're inside a SLURM job (compute node)."""
        return bool(os.environ.get("SLURM_JOB_ID"))

    @property
    def node_type(self) -> str:
        if not self.is_hpc:
            return "local"
        return "compute" if self.is_compute_node else "login"

    def summary(self) -> str:
        """Human-readable one-liner."""
        if self.is_hpc:
            return f"{self.display_name} ({self.node_type} node)"
        return self.display_name

    def print_shell_exports(self):
        """Print cluster config as shell variable assignments.

        Intended to be eval'd by setup.sh / env.sh::

            eval "$(python -m cluster_utils.cluster)"
        """
        lines = [
            f'CLUSTER_NAME="{self.name}"',
            f'CLUSTER_DISPLAY="{self.display_name}"',
            f'CLUSTER_NODE_TYPE="{self.node_type}"',
            f'CLUSTER_IS_HPC="{int(self.is_hpc)}"',
            f'CLUSTER_PARTITION="{self.slurm.default_partition}"',
            f'CLUSTER_GPU_PARTITION="{self.slurm.gpu_partition}"',
            f'CLUSTER_QUEUE_LIMIT="{self.slurm.queue_limit}"',
            f'CLUSTER_GPUS_PER_NODE="{self.slurm.gpus_per_node}"',
            f'CLUSTER_CPUS_PER_GPU="{self.slurm.cpus_per_gpu}"',
            f'CLUSTER_CONTAINER_RUNTIME="{self.container.runtime}"',
            f'CLUSTER_CONTAINER_GPU_ARGS="{self.container.gpu_args}"',
        ]
        for line in lines:
            print(line)


# ═══════════════════════════════════════════════════════════════════
#  Registry loading
# ═══════════════════════════════════════════════════════════════════
def _load_registry(path: Optional[Path] = None) -> dict:
    """Load clusters.toml and return as a dict."""
    toml_path = path or _REGISTRY_PATH
    if not toml_path.exists():
        raise FileNotFoundError(f"Cluster registry not found: {toml_path}")

    # Python 3.11+ has tomllib in stdlib
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            raise ImportError(
                "Python <3.11 needs 'tomli' for TOML parsing. "
                "Run: uv add tomli"
            )

    with open(toml_path, "rb") as f:
        return tomllib.load(f)


# ═══════════════════════════════════════════════════════════════════
#  Detection
# ═══════════════════════════════════════════════════════════════════
def _get_hostname() -> str:
    """Best-effort hostname (lowercase)."""
    return (
        os.environ.get("HOSTNAME")
        or os.environ.get("HOST")
        or socket.gethostname()
        or platform.node()
        or ""
    ).lower()


def _matches(cluster_def: dict) -> bool:
    """Check if the current environment matches a cluster's detection rules."""
    detect = cluster_def.get("detect")
    if not detect:
        return False

    hostname = _get_hostname()

    # hostname_contains: any substring match
    for substr in detect.get("hostname_contains", []):
        if substr.lower() in hostname:
            return True

    # slurm_cluster: exact match on $SLURM_CLUSTER_NAME
    slurm_cluster = os.environ.get("SLURM_CLUSTER_NAME", "").lower()
    if detect.get("slurm_cluster") and slurm_cluster == detect["slurm_cluster"]:
        return True

    # env_vars: any of these env vars is set
    for var in detect.get("env_vars", []):
        if os.environ.get(var):
            return True

    # filesystem: any of these paths exist
    for fs_path in detect.get("filesystem", []):
        if Path(fs_path).exists():
            return True

    return False


def _build_config(name: str, cluster_def: dict) -> ClusterConfig:
    """Build a ClusterConfig from a registry entry."""
    raw_slurm = cluster_def.get("slurm", {})
    raw_ctr = cluster_def.get("container", {})

    return ClusterConfig(
        name=name,
        display_name=cluster_def.get("display_name", name),
        slurm=ClusterSlurm(
            default_partition=raw_slurm.get("default_partition", ""),
            gpu_partition=raw_slurm.get("gpu_partition", ""),
            queue_limit=raw_slurm.get("queue_limit", 0),
            gpus_per_node=raw_slurm.get("gpus_per_node", 1),
            cpus_per_gpu=raw_slurm.get("cpus_per_gpu", 8),
        ),
        container=ClusterContainer(
            runtime=raw_ctr.get("runtime", ""),
            gpu_args=raw_ctr.get("gpu_args", ""),
        ),
    )


# ═══════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════
def detect_cluster(
    override: Optional[str] = None,
    registry_path: Optional[Path] = None,
) -> ClusterConfig:
    """Detect the current cluster and return its config.

    Resolution order:
    1. ``override`` argument (explicit cluster name)
    2. ``CLUSTER_NAME`` environment variable
    3. Auto-detect from hostname / env vars / filesystem

    Falls back to a bare "local" config if nothing matches.

    Args:
        override: Force a specific cluster name.
        registry_path: Custom path to clusters.toml.

    Returns:
        :class:`ClusterConfig` with SLURM/container defaults.
        Filesystem paths are NOT included — those come from .env.
    """
    registry = _load_registry(registry_path)

    target = override or os.environ.get("CLUSTER_NAME", "")

    if target:
        if target not in registry:
            available = list(registry.keys())
            raise ValueError(
                f"Unknown cluster '{target}'. "
                f"Registered: {', '.join(available)}. "
                f"Add it to clusters.toml."
            )
        return _build_config(target, registry[target])

    # Auto-detect
    for name, cluster_def in registry.items():
        if _matches(cluster_def):
            return _build_config(name, cluster_def)

    # Fallback: local machine
    return ClusterConfig()


def list_clusters(registry_path: Optional[Path] = None) -> dict[str, str]:
    """Return {name: display_name} for all registered clusters."""
    registry = _load_registry(registry_path)
    return {k: v.get("display_name", k) for k, v in registry.items()}


# ═══════════════════════════════════════════════════════════════════
#  CLI  (called by setup.sh / env.sh via eval)
# ═══════════════════════════════════════════════════════════════════
def main():
    """CLI: detect cluster → output shell variables or JSON."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect HPC cluster and output configuration",
    )
    parser.add_argument(
        "--cluster", "-c",
        help="Force a specific cluster (skip auto-detection)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all registered clusters and exit",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Output as JSON instead of shell variables",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print a one-line summary only",
    )
    args = parser.parse_args()

    if args.list:
        clusters = list_clusters()
        for name, display in clusters.items():
            print(f"  {name:12s}  {display}")
        return

    cluster = detect_cluster(override=args.cluster)

    if args.summary:
        print(cluster.summary())
        return

    if args.as_json:
        import json
        from dataclasses import asdict
        print(json.dumps(asdict(cluster), indent=2))
        return

    # Default: shell variables (for eval)
    cluster.print_shell_exports()


if __name__ == "__main__":
    main()
