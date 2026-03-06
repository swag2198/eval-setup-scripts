#!/usr/bin/env python3
"""Thin wrapper — delegates to cluster_utils.hf_cache_manager.

Prefer using the installed entry point instead:
    $ hf-cache status
    $ hf-cache download-model Qwen/Qwen2.5-0.5B

This script exists so that `python bin/hf_cache_manager.py` keeps
working for anyone who hasn't run `uv sync` yet.
"""

import sys
from pathlib import Path

# Ensure the src/ package is importable even without pip install
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from cluster_utils.hf_cache_manager import main  # noqa: E402

if __name__ == "__main__":
    main()
