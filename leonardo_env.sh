#!/bin/bash
# =============================================================================
# Leonardo LLM Evaluation – Environment Configuration
# =============================================================================
# Sources your personal .env.leonardo config and sets all environment
# variables needed by: oellm-cli, OpenJury, HFCacheManager, TRL, vLLM.
#
# Usage:
#   source leonardo_env.sh          # manual
#   (or add to ~/.bashrc via setup.sh)
#
# First-time setup:
#   bash setup.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/.env.leonardo"

# ═══════════════════════════════════════════════════════════════════
#  Load user config
# ═══════════════════════════════════════════════════════════════════
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "❌ Config file not found: ${CONFIG_FILE}"
    echo "   Run 'bash ${SCRIPT_DIR}/setup.sh' first."
    return 1 2>/dev/null || exit 1
fi

# Source the config (sets LEONARDO_USER, SLURM_ACCOUNT, WORK_DIR, etc.)
source "${CONFIG_FILE}"

# ═══════════════════════════════════════════════════════════════════
#  User-specific paths (derived from config)
# ═══════════════════════════════════════════════════════════════════
export USER_WORK_DIR="${WORK_DIR}"
export SCRIPTS_DIR="${SCRIPT_DIR}"

# ═══════════════════════════════════════════════════════════════════
#  Evaluation paths (oellm-cli)
# ═══════════════════════════════════════════════════════════════════
export EVAL_BASE_DIR="${USER_WORK_DIR}/oellm-evals"
export EVAL_OUTPUT_DIR="${EVAL_BASE_DIR}/outputs"
export EVAL_CONTAINER_IMAGE="eval_env-leonardo.sif"
export EVAL_SIF_PATH="${EVAL_BASE_DIR}/${EVAL_CONTAINER_IMAGE}"

mkdir -p "${EVAL_BASE_DIR}" "${EVAL_OUTPUT_DIR}" 2>/dev/null

# ═══════════════════════════════════════════════════════════════════
#  SLURM settings
# ═══════════════════════════════════════════════════════════════════
export PARTITION="${DEFAULT_PARTITION:-boost_usr_prod}"
export ACCOUNT="${SLURM_ACCOUNT}"
export GPUS_PER_NODE="${DEFAULT_GPUS:-1}"
export QUEUE_LIMIT="${DEFAULT_QUEUE_LIMIT:-1000}"

# ═══════════════════════════════════════════════════════════════════
#  Singularity / Apptainer
# ═══════════════════════════════════════════════════════════════════
export SINGULARITY_ARGS="--nv"
export SINGULARITY_BIND="${EVAL_BASE_DIR}:${EVAL_BASE_DIR},${USER_WORK_DIR}:${USER_WORK_DIR}"

# ═══════════════════════════════════════════════════════════════════
#  HuggingFace cache  (shared or per-user, set by setup.sh)
# ═══════════════════════════════════════════════════════════════════
export HF_HOME="${HF_DATA_DIR}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export HF_ASSETS_CACHE="${HF_HOME}/assets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_ASSETS_CACHE="${HF_HOME}/assets"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1

# ═══════════════════════════════════════════════════════════════════
#  Offline mode (auto-enabled on compute nodes)
# ═══════════════════════════════════════════════════════════════════
if [[ -n "${SLURM_JOB_ID}" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export VLLM_NO_USAGE_STATS=1
fi

# ═══════════════════════════════════════════════════════════════════
#  OpenJury
# ═══════════════════════════════════════════════════════════════════
export OPENJURY_DATA="${USER_WORK_DIR}/openjury-eval-data"
mkdir -p "${OPENJURY_DATA}" 2>/dev/null

# ═══════════════════════════════════════════════════════════════════
#  UV (Python package manager)
# ═══════════════════════════════════════════════════════════════════
export UV_LINK_MODE="copy"

# ═══════════════════════════════════════════════════════════════════
#  SLURM logs
# ═══════════════════════════════════════════════════════════════════
export SLURM_LOGS_DIR="${USER_WORK_DIR}/slurm_logs"
mkdir -p "${SLURM_LOGS_DIR}" 2>/dev/null

# ═══════════════════════════════════════════════════════════════════
#  Time limit
# ═══════════════════════════════════════════════════════════════════
export TIME_LIMIT="${DEFAULT_TIME_LIMIT:-00:30:00}"

# ═══════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════
if [[ -n "${SLURM_JOB_ID}" ]]; then
    echo "✅ Leonardo env loaded (OFFLINE – job ${SLURM_JOB_ID})"
else
    echo "✅ Leonardo env loaded (ONLINE – login node)"
fi
echo "   User:    ${LEONARDO_USER}  Account: ${ACCOUNT}"
echo "   HF_HOME: ${HF_HOME}  (${CACHE_MODE})"