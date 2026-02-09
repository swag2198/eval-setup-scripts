# 🖥️ Leonardo LLM Evaluation Scripts

Portable scripts for running LLM evaluations on [Leonardo HPC](https://wiki.u-gov.it/confluence/display/SCAIHPC/UG3.2%3A+LEONARDO+UserGuide) (Cineca).
Provides a unified environment for **[oellm-cli](https://github.com/OpenEuroLLM/oellm-cli)** and **[OpenJury](https://github.com/OpenEuroLLM/OpenJury)** with shared HuggingFace caching.

## Why?

Leonardo compute nodes **have no internet access**. You must pre-download all
models and datasets on a login node, then run evaluations offline. These scripts
handle that seamlessly:

- ✅ One-time setup per user (`bash setup.sh`)
- ✅ Shared or per-user HF cache (choose during setup)
- ✅ Automatic offline mode on compute nodes
- ✅ Works with oellm-cli, OpenJury, TRL, vLLM, and any HF-based tool

## 🚀 Quick Start

### 1. Clone this repository

```bash
# On Leonardo login node
cd /leonardo_work/<YOUR_ACCOUNT>/users/$(whoami)
git clone <repo-url> scripts
cd scripts
```

### 2. Run first-time setup

```bash
bash setup.sh
```

This will:
- Detect your username
- Ask for your SLURM account (e.g. `OELLM_prod2026`)
- Let you choose per-user or shared HF cache
- Generate your personal `.env.leonardo` config
- Optionally add auto-sourcing to `~/.bashrc`

### 3. Install Python dependencies

```bash
uv sync              # base: huggingface-hub, datasets, transformers
uv sync --extra vllm # + vLLM (for OpenJury with local models)
```

This creates a `.venv` with pinned, compatible versions (notably `transformers<5`
to avoid vLLM breakage).

### 4. Source the environment

```bash
source leonardo_env.sh
```

(If you added it to `~/.bashrc` during setup, this happens automatically.)

### 4. Download models & datasets (login node only)

```bash
# Download everything from a file (models + datasets in one go)
python bin/hf_cache_manager.py download-from-file examples/all.txt

# Or download individually:
python bin/hf_cache_manager.py download-model Qwen/Qwen2.5-0.5B-Instruct
python bin/hf_cache_manager.py download-dataset hellaswag
python bin/hf_cache_manager.py download-dataset cais/mmlu --name all --split test

# Check what's cached
python bin/hf_cache_manager.py status
```

See [examples/](examples/) for pre-made download lists:
- `examples/models.txt` — common evaluation models
- `examples/datasets.txt` — common evaluation datasets
- `examples/all.txt` — combined list for one-shot download

### 5. Run on GPU

```bash
# Get an interactive GPU session
./bin/interactive_gpu.sh            # 1 hour, 1 GPU (default)
./bin/interactive_gpu.sh 2 4        # 2 hours, 4 GPUs

# Environment auto-loads with HF_HUB_OFFLINE=1
# Your cached models are ready to use
```

## 📁 Directory Layout

After setup, your workspace looks like this:

```
/leonardo_work/<ACCOUNT>/users/<YOU>/
├── scripts/                    ← this repo
│   ├── setup.sh                ← first-time setup (run once)
│   ├── leonardo_env.sh         ← environment config (source every session)
│   ├── pyproject.toml          ← Python deps (uv sync to install)
│   ├── .env.leonardo           ← your personal config (gitignored)
│   ├── bin/
│   │   ├── hf_cache_manager.py ← download & manage HF cache
│   │   └── interactive_gpu.sh  ← quick GPU session
│   ├── slurm/
│   │   ├── eval_single.sbatch  ← single-eval SLURM template
│   │   ├── build_container.sh  ← build Singularity container
│   │   ├── run_in_container.sh ← run command inside container
│   │   └── ...                 
│   └── examples/
│       ├── all.txt             ← combined models + datasets list
│       ├── models.txt          ← example model list
│       └── datasets.txt        ← example dataset list
├── oellm-evals/
│   ├── hf_data/                ← HuggingFace cache (or shared location)
│   │   ├── hub/                ← model snapshots
│   │   ├── datasets/           ← Arrow-cached datasets
│   │   ├── assets/
│   │   └── xet/
│   └── outputs/                ← evaluation results
├── oellm-cli/                  ← oellm-cli repo (clone separately)
├── OpenJury/                   ← OpenJury repo (clone separately)
├── openjury-eval-data/         ← OpenJury datasets
└── slurm_logs/
```

## 🤖 Using with oellm-cli

oellm-cli reads `HF_HOME` from `clusters.yaml`. Make sure it matches:

```yaml
# In oellm-cli/oellm/resources/clusters.yaml → Leonardo section
EVAL_BASE_DIR: "/leonardo_work/<ACCOUNT>/users/<YOU>/oellm-evals"
```

Then oellm-cli automatically uses the same cache as `hf_cache_manager.py`.

```bash
# Pre-download models and task datasets on login node
python bin/hf_cache_manager.py download-model Qwen/Qwen2.5-0.5B-Instruct
cd oellm-cli
oellm schedule-eval --models Qwen/Qwen2.5-0.5B-Instruct --task-groups open-sci-0.01
```

## ⚖️ Using with OpenJury

### Initial setup (one-time, on login node)

```bash
cd /leonardo_work/<ACCOUNT>/users/$(whoami)
git clone https://github.com/OpenEuroLLM/OpenJury
cd OpenJury
uv sync --extra vllm
```

> **Important**: Pin `transformers<5` in OpenJury's `pyproject.toml` under
> `[project.optional-dependencies] vllm` to avoid compatibility issues with
> vLLM. The line should read: `vllm = ["vllm==0.10.2", "transformers>=4.55.2,<5"]`

### Download models & datasets (login node)

```bash
# Option 1: batch download all OpenJury models at once
python ../scripts/bin/hf_cache_manager.py download-from-file ../scripts/examples/all.txt

# Option 2: download individually
python ../scripts/bin/hf_cache_manager.py download-model Qwen/Qwen2.5-0.5B-Instruct
python ../scripts/bin/hf_cache_manager.py download-model Qwen/Qwen2.5-1.5B-Instruct
python ../scripts/bin/hf_cache_manager.py download-model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8

# Download OpenJury's own datasets (alpaca-eval tables, judge configs, etc.)
uv run python -c "from openjury.utils import download_all; download_all()"
```

### Run evaluation (compute node)

```bash
# Get a GPU node
./scripts/bin/interactive_gpu.sh 1 1    # 1 hour, 1 GPU

# On the compute node (offline mode auto-enabled):
cd OpenJury
uv run python openjury/generate_and_evaluate.py \
  --dataset alpaca-eval \
  --model_A VLLM/Qwen/Qwen2.5-0.5B-Instruct \
  --model_B VLLM/Qwen/Qwen2.5-1.5B-Instruct \
  --judge_model VLLM/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 \
  --n_instructions 10
```

> **Note**: For large judge models (e.g. 32B GPTQ), you may need multiple GPUs:
> `./interactive_gpu.sh 2 4` for 4 GPUs.

## 🔗 Shared Cache

During `setup.sh`, you can choose a **shared cache** so the whole team downloads
each model only once:

```
/leonardo_work/<ACCOUNT>/shared/hf_data/
├── hub/          ← shared models
└── datasets/     ← shared datasets
```

All team members' `HF_HOME` points to this directory. Whoever downloads a model
first makes it available for everyone.

## 🔧 Scripts Reference

| Script | Purpose |
|---|---|
| `setup.sh` | First-time setup — generates `.env.leonardo` config |
| `leonardo_env.sh` | Environment loader — source in every session |
| `pyproject.toml` | Python dependencies — `uv sync` to install |
| `bin/hf_cache_manager.py` | Download models/datasets, check cache status |
| `bin/interactive_gpu.sh` | Quick interactive GPU allocation |
| `slurm/eval_single.sbatch` | SLURM batch template for single evaluations |
| `slurm/build_container.sh` | Build Singularity/Apptainer container |
| `slurm/run_in_container.sh` | Run a command inside the container |
| `examples/all.txt` | Combined models + datasets for batch download |

### hf_cache_manager.py commands

```bash
# Download a single model
python bin/hf_cache_manager.py download-model Qwen/Qwen2.5-0.5B-Instruct

# Download a single dataset (with optional --name config and --split)
python bin/hf_cache_manager.py download-dataset hellaswag
python bin/hf_cache_manager.py download-dataset cais/mmlu --name all --split test

# Batch download from a file (models + datasets together)
python bin/hf_cache_manager.py download-from-file examples/all.txt

# Show cache summary (sizes + cached models)
python bin/hf_cache_manager.py status

# Check if a model is ready for offline use
python bin/hf_cache_manager.py verify Qwen/Qwen2.5-0.5B-Instruct

# Find local fine-tuned models (safetensors)
python bin/hf_cache_manager.py list-local /path/to/checkpoints
```

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
source ~/scripts/leonardo_env.sh   # Re-source if needed
```

### `Qwen2Tokenizer has no attribute all_special_tokens_extended`
Version mismatch: `transformers>=5` is incompatible with `vllm 0.10.2`.
Fix in OpenJury's `pyproject.toml`:
```toml
vllm = ["vllm==0.10.2", "transformers>=4.55.2,<5"]
```
Then: `cd OpenJury && uv sync --extra vllm`

### `ModuleNotFoundError: No module named 'openjury'`
OpenJury uses its own `.venv` managed by `uv`. Run with `uv run`:
```bash
cd OpenJury
uv run python openjury/generate_and_evaluate.py ...
```

### Model not found in offline mode
Ensure you downloaded it on the login node first:
```bash
python scripts/bin/hf_cache_manager.py verify Qwen/Qwen2.5-0.5B-Instruct
```

### `Repository not found` (gated models like Llama)
1. Accept the license on [huggingface.co](https://huggingface.co)
2. On login node: `huggingface-cli login`
3. Then download: `python bin/hf_cache_manager.py download-model meta-llama/...`

## 📋 Environment Variables Reference

| Variable | Set by | Purpose |
|---|---|---|
| `HF_HOME` | `leonardo_env.sh` | Root HF cache directory |
| `HF_HUB_CACHE` | `leonardo_env.sh` | Model snapshots (`hub/`) |
| `HF_DATASETS_CACHE` | `leonardo_env.sh` | Arrow datasets (`datasets/`) |
| `HF_HUB_OFFLINE` | `leonardo_env.sh` | Auto-set to `1` on compute nodes |
| `TRANSFORMERS_OFFLINE` | `leonardo_env.sh` | Auto-set to `1` on compute nodes |
| `OPENJURY_DATA` | `leonardo_env.sh` | OpenJury dataset directory |
| `ACCOUNT` | `.env.leonardo` | SLURM project account |
| `PARTITION` | `.env.leonardo` | SLURM partition |
