# 🖥️ Cluster Utils — HPC LLM Evaluation Toolkit

Portable toolkit for running LLM evaluations on HPC clusters.
Auto-detects the current cluster and provides a unified environment for
**[oellm-cli](https://github.com/OpenEuroLLM/oellm-cli)** and
**[OpenJury](https://github.com/OpenEuroLLM/OpenJury)** with shared HuggingFace caching.

**Supported clusters:** Leonardo (Cineca), JURECA (FZJ), Jupiter (FZJ), LUMI (CSC) — and local machines.
New clusters can be added by editing `clusters.toml`.

## Why?

Many HPC clusters have compute nodes with **no internet access**. You must
pre-download all models and datasets on a login node, then run evaluations
offline. This toolkit handles that seamlessly:

- ✅ One-time setup per user (`bash setup.sh`)
- ✅ Auto-detects your cluster (hostname, SLURM, filesystem probes)
- ✅ All paths are user-configured — no hardcoded assumptions
- ✅ Automatic offline mode on compute nodes
- ✅ Works with oellm-cli, OpenJury, TRL, vLLM, and any HF-based tool
- ✅ Python API (`from cluster_utils import HFCacheManager, detect_cluster`)

## � Installation

### As a standalone project (recommended for HPC)

```bash
git clone <repo-url> scripts
cd scripts
uv sync      # creates .venv with all deps
bash setup.sh  # interactive first-time config
```

### As a dependency in another project

```bash
# From git
uv add git+https://github.com/OpenEuroLLM/cluster-utils
# or
pip install git+https://github.com/OpenEuroLLM/cluster-utils
```

This gives you:
- **CLI tools**: `hf-cache` and `cluster-detect` on your PATH
- **Python API**: `from cluster_utils import HFCacheManager, detect_cluster`

### Editable install (for development)

```bash
git clone <repo-url> scripts
pip install -e ./scripts
# or
uv add --editable ./scripts
```

## 🚀 Quick Start

### 1. Run first-time setup

```bash
cd scripts
bash setup.sh
# or non-interactive:
bash setup.sh --account OELLM_prod2026
# to re-run later:
bash setup.sh --reconfigure
```

This will:
- Auto-detect your HPC cluster (or fall back to local mode)
- Ask for your SLURM account (validates it via `sacctmgr`)
- Ask for your **work directory** (no assumptions — you choose)
- Ask for your **HF cache directory** (default or custom path)
- Pre-fill SLURM defaults from cluster detection (partition, GPUs, etc.)
- Generate your personal `.env` config
- Optionally add auto-sourcing to `~/.bashrc`
- Optionally run `uv sync` to install Python dependencies
### 2. Source the environment

```bash
source env.sh
```

(If you added it to `~/.bashrc` during setup, this happens automatically.)

### 3. Download models & datasets (login node only)

```bash
# Download everything from a file (models + datasets in one go)
hf-cache download-from-file examples/all.txt

# Preview what would be downloaded (dry run)
hf-cache download-from-file examples/all.txt --dry-run

# Or download individually:
hf-cache download-model Qwen/Qwen2.5-0.5B-Instruct
hf-cache download-dataset hellaswag
hf-cache download-dataset cais/mmlu --name all --split test

# Check what's cached (with per-model sizes)
hf-cache status

# Remove stale locks & incomplete downloads
hf-cache clean
hf-cache clean --dry-run   # preview only
```

See [examples/](examples/) for pre-made download lists:
- `examples/models.txt` — common evaluation models
- `examples/datasets.txt` — common evaluation datasets
- `examples/all.txt` — combined list for one-shot download

### 4. Detect your cluster (standalone)

```bash
# Shell-friendly output (eval-able)
cluster-detect

# Human-readable summary
cluster-detect --summary

# JSON output
cluster-detect --json

# List all registered clusters
cluster-detect --list
```

### 5. Run on GPU

```bash
# Get an interactive GPU session
./bin/interactive_gpu.sh                       # 1 hour, 1 GPU (default)
./bin/interactive_gpu.sh 2 4                   # 2 hours, 4 GPUs (positional)
./bin/interactive_gpu.sh --hours 2 --gpus 4    # same, with named args

# Environment auto-loads with HF_HUB_OFFLINE=1
# Your cached models are ready to use
```

## 📁 Directory Layout

After setup, your workspace looks like this:

```
<WORK_DIR>/                     ← chosen during setup.sh
├── scripts/                    ← this repo
│   ├── setup.sh                ← first-time setup (run once)
│   ├── env.sh                  ← environment config (source every session)
│   ├── pyproject.toml          ← Python deps (uv sync to install)
│   ├── .env                    ← your personal config (gitignored)
│   ├── src/cluster_utils/      ← Python package
│   │   ├── __init__.py
│   │   ├── cluster.py          ← cluster detection & config
│   │   ├── clusters.toml       ← cluster registry (detection rules)
│   │   └── hf_cache_manager.py ← HF cache management
│   ├── bin/
│   │   ├── hf_cache_manager.py ← backward-compat wrapper
│   │   └── interactive_gpu.sh  ← quick GPU session
│   ├── slurm/
│   │   ├── eval_single.sbatch  ← SLURM batch template
│   │   ├── build_container.sh  ← build Singularity/Apptainer container
│   │   └── run_in_container.sh ← run inside container
│   └── examples/
│       ├── all.txt             ← combined models + datasets list
│       ├── models.txt          ← example model list
│       └── datasets.txt        ← example dataset list
├── hf_cache/                   ← HuggingFace cache (or custom location)
│   ├── hub/                    ← model snapshots
│   ├── datasets/               ← Arrow-cached datasets
│   ├── assets/
│   └── xet/
├── oellm-evals/
│   └── outputs/                ← evaluation results
├── oellm-cli/                  ← oellm-cli repo (clone separately)
├── OpenJury/                   ← OpenJury repo (clone separately)
├── openjury-eval-data/         ← OpenJury datasets
└── slurm_jobs/
    ├── logs/
    ├── oellm-cli/
    └── openjury/
```

## 🧩 Cluster Registry

Clusters are registered in `src/cluster_utils/clusters.toml`. Each entry has:
- **Detection rules** — hostname substrings, filesystem probes, SLURM vars
- **SLURM defaults** — partition, GPUs/node, queue limit
- **Container settings** — runtime (singularity/apptainer), GPU args

The registry does **not** contain filesystem paths — those vary per project
and user, and are configured interactively by `setup.sh`.

To add a new cluster, add a section to `clusters.toml`:

```toml
[mycluster]
display_name = "MyCluster (MyOrg)"

[mycluster.detect]
hostname_contains = ["mycluster"]
filesystem        = ["/data/mycluster"]

[mycluster.slurm]
default_partition = "gpu"
gpu_partition     = "gpu"
gpus_per_node     = 8

[mycluster.container]
runtime  = "singularity"
gpu_args = "--nv"
```

## 🤖 Using with oellm-cli

oellm-cli reads `HF_HOME` from its own `clusters.yaml`. `setup.sh` can
auto-configure it to match your paths:

```bash
# During setup.sh, say 'Y' when asked to auto-configure clusters.yaml
# Or manually set EVAL_BASE_DIR in oellm-cli/oellm/resources/clusters.yaml
```

Then oellm-cli automatically uses the same cache as `hf-cache`.

```bash
# Pre-download models and task datasets on login node
hf-cache download-model Qwen/Qwen2.5-0.5B-Instruct
cd oellm-cli
oellm schedule-eval --models Qwen/Qwen2.5-0.5B-Instruct --task-groups open-sci-0.01
```

## ⚖️ Using with OpenJury

### Initial setup (one-time, on login node)

```bash
cd $USER_WORK_DIR
git clone https://github.com/OpenEuroLLM/OpenJury
cd OpenJury
uv sync --extra vllm
```

> **Important**: Pin `transformers<5` in OpenJury's `pyproject.toml` under
> `[project.optional-dependencies] vllm` to avoid compatibility issues with
> vLLM. The line should read: `vllm = ["vllm==0.10.2", "transformers>=4.55.2,<5"]`

### Download models & datasets (login node)

```bash
# Batch download
hf-cache download-from-file examples/all.txt

# Or individually:
hf-cache download-model Qwen/Qwen2.5-0.5B-Instruct
hf-cache download-model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8
```

### Run evaluation (compute node)

```bash
# Get a GPU node
./scripts/bin/interactive_gpu.sh 1 1

# On the compute node (offline mode auto-enabled):
cd OpenJury
uv run python openjury/generate_and_evaluate.py \
  --dataset alpaca-eval \
  --model_A VLLM/Qwen/Qwen2.5-0.5B-Instruct \
  --model_B VLLM/Qwen/Qwen2.5-1.5B-Instruct \
  --judge_model VLLM/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 \
  --n_instructions 10
```

## 🐍 Python API

```python
from cluster_utils import detect_cluster, HFCacheManager, ClusterConfig

# Detect current cluster
cluster = detect_cluster()
print(cluster.name)          # "leonardo"
print(cluster.display_name)  # "Leonardo (Cineca)"
print(cluster.is_hpc)        # True
print(cluster.node_type)     # "login" or "compute"
print(cluster.slurm.gpu_partition)  # "boost_usr_prod"

# Manage HF cache (reads HF_HOME from env)
hf = HFCacheManager()
hf.download_model("Qwen/Qwen2.5-0.5B-Instruct")
```

## 🔧 Scripts Reference

| Script | Purpose |
|---|---|
| `setup.sh` | First-time setup — generates `.env` config |
| `env.sh` | Environment loader — source in every session |
| `pyproject.toml` | Python dependencies — `uv sync` to install |
| `bin/hf_cache_manager.py` | Backward-compat wrapper for `hf-cache` |
| `bin/interactive_gpu.sh` | Quick interactive GPU allocation |
| `slurm/eval_single.sbatch` | SLURM batch template for single evaluations |
| `slurm/build_container.sh` | Build Singularity/Apptainer container |
| `slurm/run_in_container.sh` | Run a command inside the container |
| `examples/all.txt` | Combined models + datasets for batch download |

### CLI commands

| Command | Description |
|---|---|
| `hf-cache status` | Show cache summary with per-model sizes |
| `hf-cache download-model <name>` | Download a model |
| `hf-cache download-dataset <name>` | Download a dataset |
| `hf-cache download-from-file <path>` | Batch download from text file |
| `hf-cache clean` | Remove stale locks and incomplete downloads |
| `hf-cache verify <name>` | Check if model is cached for offline use |
| `hf-cache list-local <dir>` | Find local fine-tuned models |
| `hf-cache login` | Check / set HuggingFace token |
| `hf-cache setup` | Print environment variables |
| `cluster-detect` | Detect cluster, output shell variables |
| `cluster-detect --summary` | One-line cluster summary |
| `cluster-detect --json` | Full cluster config as JSON |
| `cluster-detect --list` | List all registered clusters |

### Download file format

Files passed to `download-from-file` use a simple text format:

```text
# Comments start with #
# Plain lines → models
Qwen/Qwen2.5-0.5B-Instruct

# Lines starting with "dataset:" → datasets
# Format: dataset:name[,config[,split]]
dataset:hellaswag
dataset:cais/mmlu,all
dataset:trl-lib/Capybara,,train
```

## ❓ Troubleshooting

### `Network is unreachable` on compute node
Your environment isn't loading correctly. Verify:
```bash
echo $HF_HUB_OFFLINE    # Should print "1" on compute nodes
echo $HF_HOME           # Should print your hf_data path
source env.sh            # Re-source if needed
```

### `Qwen2Tokenizer has no attribute all_special_tokens_extended`
Version mismatch: `transformers>=5` is incompatible with `vllm 0.10.2`.
Fix in OpenJury's `pyproject.toml`:
```toml
vllm = ["vllm==0.10.2", "transformers>=4.55.2,<5"]
```
Then: `cd OpenJury && uv sync --extra vllm`

### Model not found in offline mode
Ensure you downloaded it on the login node first:
```bash
hf-cache verify Qwen/Qwen2.5-0.5B-Instruct
```

### `Repository not found` (gated models like Llama)
1. Accept the license on [huggingface.co](https://huggingface.co)
2. On login node: `hf-cache login`
3. Then download: `hf-cache download-model meta-llama/...`

## 📋 Environment Variables Reference

| Variable | Set by | Purpose |
|---|---|---|
| `HF_HOME` | `env.sh` | Root HF cache directory |
| `HF_HUB_CACHE` | `env.sh` | Model snapshots (`hub/`) |
| `HF_DATASETS_CACHE` | `env.sh` | Arrow datasets (`datasets/`) |
| `HF_HUB_OFFLINE` | `env.sh` | Auto-set to `1` on compute nodes |
| `TRANSFORMERS_OFFLINE` | `env.sh` | Auto-set to `1` on compute nodes |
| `CLUSTER_NAME` | `.env` | Detected cluster identifier |
| `ACCOUNT` | `.env` | SLURM project account |
| `PARTITION` | `.env` | Default SLURM partition |
| `CONTAINER_RUNTIME` | `.env` | Container runtime (singularity/apptainer) |
| `OPENJURY_DATA` | `env.sh` | OpenJury dataset directory |
