#!/usr/bin/env python3
"""
HuggingFace Cache Manager for Leonardo HPC

Unified module for managing HuggingFace models and datasets caching.
Uses the **same cache layout** as oellm-cli so that everything is
interoperable – download once, use everywhere.

Cache layout (mirrors HuggingFace defaults and oellm-cli expectations):

    HF_HOME/               (oellm-evals/hf_data)
    ├── hub/               model snapshots   (HF_HUB_CACHE)
    ├── datasets/          arrow-cached data  (HF_DATASETS_CACHE)
    ├── assets/            HF assets
    └── xet/               xet cache

Usage:
    from hf_cache_manager import HFCacheManager

    # Initialise (auto-detects user, matches oellm-cli HF_HOME)
    hf = HFCacheManager()

    # ---------- on the LOGIN NODE (has internet) ----------
    hf.download_model("Qwen/Qwen2.5-0.5B")
    hf.download_dataset("trl-lib/Capybara", split="train")

    # ---------- on COMPUTE NODES (no internet) ----------
    hf.setup_environment(offline=True)
    # Now from_pretrained / load_dataset will use the cache

    # ---------- with oellm-cli ----------
    # Nothing extra needed – oellm-cli reads the same HF_HOME

    # ---------- your own fine-tuned models ----------
    hf.list_local_models("/leonardo_work/.../checkpoints")
    # Pass the path directly to oellm:
    #   oellm schedule-eval --models /path/to/my-model ...
"""

import getpass
import os
import sys
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
#  Defaults – override via constructor or HF_HOME env var
# ═══════════════════════════════════════════════════════════════════
_DEFAULT_WORK_BASE = "/leonardo_work/{account}/users/{user}"
_DEFAULT_HF_ROOT = _DEFAULT_WORK_BASE + "/oellm-evals/hf_data"


class HFCacheManager:
    """Manages HuggingFace model and dataset caching for Leonardo HPC.

    The directory layout is **identical** to the one oellm-cli uses
    (HF_HOME → hub/ + datasets/ + assets/ + xet/).  This means every
    model or dataset you download here is automatically available to
    ``oellm schedule-eval`` without any extra config.

    For your own fine-tuned models you can simply pass the **local
    directory path** to oellm-cli – it detects safetensors and bind-
    mounts the directory into the container automatically.
    """

    def __init__(self, hf_home: Optional[str] = None):
        """
        Initialise cache manager.

        Resolution order for the HF_HOME root:
        1. ``hf_home`` argument
        2. ``HF_HOME`` environment variable
        3. Default: ``/leonardo_work/$SLURM_ACCOUNT/users/$USER/oellm-evals/hf_data``
           (requires ACCOUNT or SLURM_ACCOUNT env var to be set)

        Args:
            hf_home: Explicit HF_HOME path.  Rarely needed – the default
                     matches the oellm-cli clusters.yaml layout.
        """
        if hf_home:
            self.hf_home = Path(hf_home)
        elif os.environ.get("HF_HOME"):
            self.hf_home = Path(os.environ["HF_HOME"])
        else:
            username = os.environ.get("USER", "unknown")
            account = os.environ.get("ACCOUNT",
                      os.environ.get("SLURM_ACCOUNT", ""))
            if not account:
                raise ValueError(
                    "Cannot determine HF_HOME: set HF_HOME env var, or run "
                    "'source leonardo_env.sh', or pass hf_home= to constructor."
                )
            self.hf_home = Path(
                _DEFAULT_HF_ROOT.format(account=account, user=username)
            )

        # Standard sub-directories (same names as oellm-cli template.sbatch)
        self.hub_cache = self.hf_home / "hub"
        self.datasets_cache = self.hf_home / "datasets"
        self.assets_cache = self.hf_home / "assets"
        self.xet_cache = self.hf_home / "xet"

        # Create directories
        for d in [self.hf_home, self.hub_cache, self.datasets_cache,
                  self.assets_cache, self.xet_cache]:
            d.mkdir(parents=True, exist_ok=True)

        print(f"✅ HFCacheManager initialised")
        print(f"   HF_HOME      : {self.hf_home}")
        print(f"   Models  (hub) : {self.hub_cache}")
        print(f"   Datasets      : {self.datasets_cache}")

    # ───────────────────────────────────────────────────────────────
    #  Environment
    # ───────────────────────────────────────────────────────────────
    def setup_environment(self, offline: bool = False, verbose: bool = True):
        """Set all HuggingFace environment variables.

        After calling this, ``from_pretrained()``, ``load_dataset()``,
        and oellm-cli will all point at the same cache.

        Args:
            offline: Set ``*_OFFLINE=1`` vars (for compute nodes).
            verbose: Print summary.
        """
        env = {
            "HF_HOME": str(self.hf_home),
            "HF_HUB_CACHE": str(self.hub_cache),
            "HF_XET_CACHE": str(self.xet_cache),
            "HF_ASSETS_CACHE": str(self.assets_cache),
            "HUGGINGFACE_HUB_CACHE": str(self.hub_cache),      # legacy
            "HUGGINGFACE_ASSETS_CACHE": str(self.assets_cache), # legacy
            "HF_DATASETS_CACHE": str(self.datasets_cache),
            "TRANSFORMERS_CACHE": str(self.hub_cache),          # some libs still read this
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
        }

        if offline:
            env.update({
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            })

        os.environ.update(env)

        if verbose:
            mode = "OFFLINE" if offline else "ONLINE"
            print(f"\n✅ HuggingFace environment set  [{mode}]")
            print(f"   HF_HOME           = {env['HF_HOME']}")
            print(f"   HF_HUB_CACHE      = {env['HF_HUB_CACHE']}")
            print(f"   HF_DATASETS_CACHE = {env['HF_DATASETS_CACHE']}")
            if offline:
                print(f"   HF_HUB_OFFLINE    = 1")
            print()

    # ───────────────────────────────────────────────────────────────
    #  HuggingFace token management
    # ───────────────────────────────────────────────────────────────
    def _get_token(self) -> Optional[str]:
        """Return the current HF token from env vars or huggingface-cli cache, or None."""
        for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            tok = os.environ.get(var)
            if tok:
                return tok
        # Check the stored token from `huggingface-cli login`
        try:
            from huggingface_hub import HfFolder
            tok = HfFolder.get_token()
            if tok:
                return tok
        except Exception:
            pass
        return None

    def validate_token(self, token: Optional[str] = None) -> bool:
        """Validate a HuggingFace token and display user information.

        Args:
            token: Token to validate. If ``None``, uses the current env/cached token.

        Returns:
            ``True`` if the token is valid.
        """
        from huggingface_hub import HfApi

        tok = token or self._get_token()
        try:
            api = HfApi(token=tok) if tok else HfApi()
            user_info = api.whoami()
            print(f"✅ Token validated successfully!")
            print(f"   Logged in as : {user_info['name']}")
            print(f"   Token type   : {user_info.get('type', 'unknown')}")
            return True
        except Exception as e:
            print(f"❌ Token validation failed.")
            print(f"   Error: {e}")
            print("\n💡 Please check that:")
            print("   1. Your token is correctly set")
            print("   2. Your token has the necessary permissions")
            print("   3. You have accepted the model license on huggingface.co")
            print("   4. Try running: huggingface-cli login")
            return False

    def ensure_token(self) -> Optional[str]:
        """Make sure an HF token is available, prompting interactively if needed.

        Resolution order:
        1. ``HF_TOKEN`` or ``HUGGINGFACE_HUB_TOKEN`` environment variables
        2. Token stored by ``huggingface-cli login``
        3. Interactive prompt (token is then exported for the session)

        Returns:
            The token string, or ``None`` if the user skips.
        """
        token = self._get_token()

        if token:
            # Already have a token – validate silently
            if self.validate_token(token):
                return token
            print("\n⚠️  Existing token is invalid or expired.")
            # Fall through to prompt

        # Interactive prompt
        print("\n🔑 No HuggingFace token found.")
        print("   A token is required to download gated/private models")
        print("   (e.g. Meta-Llama, Mistral, etc.).")
        print("   Get yours at: https://huggingface.co/settings/tokens\n")

        token = getpass.getpass("   Paste your HF token (input hidden): ").strip()

        if not token:
            print("   ⏭️  Skipped – continuing without token (public models only).")
            return None

        # Validate the provided token
        if not self.validate_token(token):
            print("   Continuing anyway – some downloads may fail for gated models.")

        # Persist for this process and child processes
        os.environ["HF_TOKEN"] = token
        print(f"   Token exported as HF_TOKEN for this session.\n")
        return token

    # ───────────────────────────────────────────────────────────────
    #  Download model  (snapshot_download – same as oellm-cli)
    # ───────────────────────────────────────────────────────────────
    def download_model(
        self,
        model_name: str,
        revision: str = "main",
        ignore_patterns: Optional[list[str]] = None,
    ) -> bool:
        """Download a HuggingFace model to the shared hub cache.

        Uses ``snapshot_download`` — the same mechanism oellm-cli uses
        internally — so the cached files are identical.  On compute nodes
        ``from_pretrained()`` will find them via ``HF_HUB_CACHE``.

        This is a **metadata + files** download; it does NOT load the
        model into RAM, so it is safe to run on login nodes.

        Args:
            model_name: Model identifier (e.g. ``'Qwen/Qwen2.5-0.5B'``).
            revision: Git revision / branch / tag.
            ignore_patterns: File patterns to skip (e.g.
                ``['*.bin', '*.gguf']`` to download safetensors only).

        Returns:
            ``True`` on success, ``False`` on error.
        """
        self.setup_environment(offline=False, verbose=False)
        token = self._get_token()

        from huggingface_hub import snapshot_download

        print(f"📥 Downloading model: {model_name}  (revision={revision})")
        if ignore_patterns:
            print(f"   Ignoring: {ignore_patterns}")

        try:
            path = snapshot_download(
                repo_id=model_name,
                revision=revision,
                cache_dir=self.hub_cache,
                ignore_patterns=ignore_patterns,
                token=token,
            )
            print(f"✅ Model cached at {path}")
            return True
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            return False

    # ───────────────────────────────────────────────────────────────
    #  Download dataset  (load_dataset – proper Arrow cache)
    # ───────────────────────────────────────────────────────────────
    def download_dataset(
        self,
        dataset_name: str,
        name: Optional[str] = None,
        split: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> bool:
        """Download a HuggingFace dataset with proper Arrow caching.

        Uses ``load_dataset()`` so the Arrow files end up in
        ``HF_DATASETS_CACHE`` and are usable in offline mode.

        Args:
            dataset_name: Dataset identifier (e.g. ``'trl-lib/Capybara'``).
            name: Configuration / subset name.
            split: Specific split (``'train'``, ``'test'``, …).
            trust_remote_code: Allow remote code execution.

        Returns:
            ``True`` on success, ``False`` on error.
        """
        self.setup_environment(offline=False, verbose=False)
        token = self._get_token()

        from datasets import load_dataset

        print(f"📥 Downloading dataset: {dataset_name}")
        if name:
            print(f"   Config: {name}")
        if split:
            print(f"   Split : {split}")

        try:
            ds = load_dataset(
                dataset_name,
                name=name,
                split=split,
                cache_dir=str(self.datasets_cache),
                trust_remote_code=trust_remote_code,
                token=token,
            )

            if split:
                print(f"✅ Downloaded {len(ds)} examples")
            else:
                print(f"✅ Downloaded splits: {list(ds.keys())}")
                for s, d in ds.items():
                    print(f"   - {s}: {len(d)} examples")

            return True
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return False

    # ───────────────────────────────────────────────────────────────
    #  Batch download from file
    # ───────────────────────────────────────────────────────────────
    def download_from_file(self, filepath: str) -> tuple[int, int]:
        """Download models and datasets listed in a text file.

        File format (one entry per line, comments start with ``#``)::

            # Models (just the repo id)
            Qwen/Qwen2.5-0.5B-Instruct
            Qwen/Qwen2.5-1.5B-Instruct

            # Datasets: name[,config[,split]]
            dataset:hellaswag
            dataset:cais/mmlu,all
            dataset:trl-lib/Capybara,,train

        Lines starting with ``dataset:`` are treated as datasets.
        Everything else is treated as a model.

        Args:
            filepath: Path to the text file.

        Returns:
            Tuple of (successes, failures).
        """
        path = Path(filepath)
        if not path.exists():
            print(f"❌ File not found: {filepath}")
            return (0, 1)

        # Ensure token is available before starting the batch
        self.ensure_token()

        successes, failures = 0, 0

        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("dataset:"):
                # Parse: dataset:name[,config[,split]]
                parts = line[len("dataset:"):].split(",")
                ds_name = parts[0].strip()
                ds_config = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
                ds_split = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
                ok = self.download_dataset(ds_name, name=ds_config, split=ds_split)
            else:
                ok = self.download_model(line)

            if ok:
                successes += 1
            else:
                failures += 1

        print(f"\n{'='*50}")
        print(f"📋 Batch complete: {successes} succeeded, {failures} failed")
        return (successes, failures)

    # ───────────────────────────────────────────────────────────────
    #  Local / fine-tuned models
    # ───────────────────────────────────────────────────────────────
    @staticmethod
    def list_local_models(directory: str) -> list[Path]:
        """Find model directories (containing safetensors) under *directory*.

        Useful for discovering your own fine-tuned checkpoints.  Pass the
        returned paths directly to ``oellm schedule-eval --models <path>``.

        Args:
            directory: Root directory to search.

        Returns:
            List of paths that contain ``*.safetensors`` files.
        """
        root = Path(directory)
        if not root.exists():
            print(f"⚠️  Directory does not exist: {root}")
            return []

        found: list[Path] = []
        for p in sorted(root.rglob("*.safetensors")):
            model_dir = p.parent
            if model_dir not in found:
                found.append(model_dir)

        if found:
            print(f"🔍 Found {len(found)} model(s) under {root}:")
            for m in found:
                print(f"   • {m}")
        else:
            print(f"   No safetensors models found under {root}")

        return found

    # ───────────────────────────────────────────────────────────────
    #  Cache stats
    # ───────────────────────────────────────────────────────────────
    def get_cache_stats(self) -> dict:
        """Get cache usage statistics."""

        def _size(path: Path) -> int:
            total = 0
            if path.exists():
                for entry in path.rglob("*"):
                    if entry.is_file():
                        total += entry.stat().st_size
            return total

        def _fmt(n: int) -> str:
            for unit in ("B", "KB", "MB", "GB", "TB"):
                if n < 1024:
                    return f"{n:.1f} {unit}"
                n /= 1024
            return f"{n:.1f} PB"

        hub = _size(self.hub_cache)
        ds = _size(self.datasets_cache)
        return {
            "hub_size": hub,
            "hub_size_str": _fmt(hub),
            "datasets_size": ds,
            "datasets_size_str": _fmt(ds),
            "total_size": hub + ds,
            "total_size_str": _fmt(hub + ds),
        }

    def print_cache_status(self):
        """Print human-friendly cache summary."""
        import glob

        def _dir_size(path: Path) -> int:
            total = 0
            if path.exists():
                for entry in path.rglob("*"):
                    if entry.is_file():
                        total += entry.stat().st_size
            return total

        def _fmt(n: int) -> str:
            for unit in ("B", "KB", "MB", "GB", "TB"):
                if n < 1024:
                    return f"{n:.1f} {unit}"
                n /= 1024
            return f"{n:.1f} PB"

        stats = self.get_cache_stats()
        print("📊 Cache Status")
        print(f"   HF_HOME  : {self.hf_home}")
        print(f"   Models   : {stats['hub_size_str']}")
        print(f"   Datasets : {stats['datasets_size_str']}")
        print(f"   Total    : {stats['total_size_str']}")

        # List cached model snapshots with individual sizes
        snapshots = sorted(glob.glob(str(self.hub_cache / "models--*")))
        if snapshots:
            print(f"\n   📦 Cached models ({len(snapshots)}):")
            for s in snapshots:
                name = Path(s).name.replace("models--", "").replace("--", "/")
                size = _fmt(_dir_size(Path(s)))
                print(f"     • {name}  ({size})")

        # List cached datasets
        ds_dirs = sorted(glob.glob(str(self.datasets_cache / "*")))
        ds_dirs = [d for d in ds_dirs if Path(d).is_dir()
                   and not Path(d).name.startswith(".")]
        if ds_dirs:
            print(f"\n   📂 Cached datasets ({len(ds_dirs)}):")
            for d in ds_dirs:
                name = Path(d).name.replace("___", "/")
                size = _fmt(_dir_size(Path(d)))
                print(f"     • {name}  ({size})")

        # Check for stale lock files
        locks = list(self.hf_home.rglob("*.lock"))
        if locks:
            print(f"\n   ⚠️  {len(locks)} stale lock file(s) found")
            print(f"      Run: python bin/hf_cache_manager.py clean")

    # ───────────────────────────────────────────────────────────────
    #  Clean cache
    # ───────────────────────────────────────────────────────────────
    def clean(self, dry_run: bool = False) -> int:
        """Remove stale lock files and orphaned incomplete downloads.

        Args:
            dry_run: If True, only print what would be deleted.

        Returns:
            Number of items cleaned.
        """
        cleaned = 0

        # Remove .lock files
        locks = list(self.hf_home.rglob("*.lock"))
        if locks:
            print(f"🔒 Found {len(locks)} lock file(s):")
            for lf in locks:
                print(f"   {'[DRY RUN] ' if dry_run else ''}rm {lf}")
                if not dry_run:
                    lf.unlink()
                cleaned += 1

        # Remove incomplete downloads (.incomplete in hub)
        incompletes = list(self.hub_cache.rglob("*.incomplete"))
        if incompletes:
            print(f"⚠️  Found {len(incompletes)} incomplete download(s):")
            for inc in incompletes:
                print(f"   {'[DRY RUN] ' if dry_run else ''}rm {inc}")
                if not dry_run:
                    inc.unlink()
                cleaned += 1

        # Remove misplaced datasets-- entries in hub/
        import glob as _glob
        misplaced = sorted(_glob.glob(str(self.hub_cache / "datasets--*")))
        if misplaced:
            import shutil
            print(f"🔀 Found {len(misplaced)} misplaced dataset(s) in hub/:")
            for mp in misplaced:
                print(f"   {'[DRY RUN] ' if dry_run else ''}rm -rf {mp}")
                if not dry_run:
                    shutil.rmtree(mp)
                cleaned += 1

        if cleaned == 0:
            print("✅ Cache is clean – nothing to remove.")
        else:
            action = "would be" if dry_run else "were"
            print(f"\n{'='*50}")
            print(f"🧹 {cleaned} item(s) {action} cleaned.")

        return cleaned

    # ───────────────────────────────────────────────────────────────
    #  Verify offline readiness
    # ───────────────────────────────────────────────────────────────
    def verify_offline_ready(
        self,
        model_name: str,
        dataset_name: Optional[str] = None,
    ) -> bool:
        """Check whether a model (and optionally a dataset) can load offline.

        Does NOT load anything into GPU memory.

        Args:
            model_name: HF model id or local path.
            dataset_name: Optional HF dataset id.

        Returns:
            ``True`` if everything looks cached, ``False`` otherwise.
        """
        ok = True

        # --- check model ---
        model_path = Path(model_name)
        if model_path.exists():
            has_weights = (
                any(model_path.glob("*.safetensors"))
                or any(model_path.glob("*.bin"))
            )
            if has_weights:
                print(f"✅ Local model found: {model_path}")
            else:
                print(f"⚠️  Local path exists but no model files: {model_path}")
                ok = False
        else:
            safe_name = model_name.replace("/", "--")
            cached = self.hub_cache / f"models--{safe_name}"
            if cached.exists():
                snap_dir = cached / "snapshots"
                snaps = list(snap_dir.iterdir()) if snap_dir.exists() else []
                if snaps:
                    print(f"✅ Model cached: {model_name}  ({len(snaps)} snapshot(s))")
                else:
                    print(f"⚠️  Cache dir exists but no snapshots: {cached}")
                    ok = False
            else:
                print(f"❌ Model NOT cached: {model_name}")
                print(f"   Run:  python hf_cache_manager.py download-model {model_name}")
                ok = False

        # --- check dataset ---
        if dataset_name:
            safe_ds = dataset_name.replace("/", "___")
            matches = list(self.datasets_cache.glob(f"{safe_ds}*"))
            if matches:
                print(f"✅ Dataset cached: {dataset_name}")
            else:
                print(f"❌ Dataset NOT cached: {dataset_name}")
                print(f"   Run:  python hf_cache_manager.py download-dataset {dataset_name}")
                ok = False

        return ok


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════
def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="HuggingFace Cache Manager for Leonardo HPC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", help="Command to run")

    # ── download-model ──
    p = sub.add_parser("download-model", help="Download a model (snapshot_download)")
    p.add_argument("model_name", help="Model identifier (e.g. Qwen/Qwen2.5-0.5B)")
    p.add_argument("--revision", default="main", help="Git revision")
    p.add_argument(
        "--ignore-patterns", nargs="*",
        help="File patterns to skip (e.g. '*.bin' '*.gguf')",
    )

    # ── download-dataset ──
    p = sub.add_parser("download-dataset", help="Download a dataset (load_dataset)")
    p.add_argument("dataset_name", help="Dataset identifier")
    p.add_argument("--name", help="Configuration / subset name")
    p.add_argument("--split", help="Specific split")
    p.add_argument("--trust-remote-code", action="store_true")

    # ── download-from-file ──
    p = sub.add_parser(
        "download-from-file",
        help="Batch download models & datasets from a text file",
    )
    p.add_argument(
        "filepath",
        help="Path to file listing models/datasets (see examples/)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Only print what would be downloaded, don't actually download",
    )

    # ── status ──
    sub.add_parser("status", help="Show cache status")

    # ── clean ──
    p = sub.add_parser("clean", help="Remove stale locks, incomplete downloads")
    p.add_argument("--dry-run", action="store_true", help="Only show what would be deleted")

    # ── verify ──
    p = sub.add_parser("verify", help="Check if model/dataset are cached for offline use")
    p.add_argument("model_name", help="Model id or local path")
    p.add_argument("--dataset", help="Optional dataset to check")

    # ── list-local ──
    p = sub.add_parser("list-local", help="Find local models (safetensors) in a directory")
    p.add_argument("directory", help="Directory to search")

    # ── login ──
    sub.add_parser("login", help="Check / set HuggingFace token (prompt if missing)")

    # ── setup ──
    p = sub.add_parser("setup", help="Print / export environment variables")
    p.add_argument("--offline", action="store_true", help="Enable offline mode")

    args = parser.parse_args()
    hf = HFCacheManager()

    if args.command == "download-model":
        ok = hf.download_model(args.model_name, args.revision, args.ignore_patterns)
        sys.exit(0 if ok else 1)

    elif args.command == "download-dataset":
        ok = hf.download_dataset(
            args.dataset_name, args.name, args.split, args.trust_remote_code,
        )
        sys.exit(0 if ok else 1)

    elif args.command == "download-from-file":
        if args.dry_run:
            path = Path(args.filepath)
            if not path.exists():
                print(f"❌ File not found: {args.filepath}")
                sys.exit(1)
            print("🔍 Dry run – would download:")
            for raw_line in path.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("dataset:"):
                    print(f"   📂 dataset: {line[len('dataset:'):]}") 
                else:
                    print(f"   📦 model:   {line}")
            sys.exit(0)
        successes, failures = hf.download_from_file(args.filepath)
        sys.exit(0 if failures == 0 else 1)

    elif args.command == "clean":
        hf.clean(dry_run=args.dry_run)

    elif args.command == "status":
        hf.print_cache_status()

    elif args.command == "verify":
        ok = hf.verify_offline_ready(args.model_name, args.dataset)
        sys.exit(0 if ok else 1)

    elif args.command == "list-local":
        hf.list_local_models(args.directory)

    elif args.command == "login":
        token = hf.ensure_token()
        sys.exit(0 if token else 1)

    elif args.command == "setup":
        hf.setup_environment(offline=args.offline)
        # Also print shell-exportable lines for sourcing
        print("# Copy-paste or eval these in your shell:")
        for k in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE",
                   "TRANSFORMERS_CACHE", "HF_HUB_OFFLINE"):
            v = os.environ.get(k, "")
            if v:
                print(f"export {k}={v}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
