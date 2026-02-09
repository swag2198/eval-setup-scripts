#!/bin/bash
# =============================================================================
# SLURM Job Template for Single Model Evaluation
# =============================================================================
# Submit a single evaluation job to Leonardo GPU cluster.
#
# Usage:
#   sbatch eval_single.sbatch MODEL_NAME TASK N_SHOT
#
# Example:
#   sbatch eval_single.sbatch "Qwen/Qwen3-0.5B" "hellaswag" 10
# =============================================================================

#SBATCH --job-name=oellm-eval-single
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_L01_028
#SBATCH --output=/leonardo_work/OELLM_prod2026/users/bkargi00/slurm_logs/%x-%j.out
#SBATCH --error=/leonardo_work/OELLM_prod2026/users/bkargi00/slurm_logs/%x-%j.err

# =============================================================================
# Configuration - EDIT THESE or pass as arguments
# =============================================================================
MODEL_NAME="${1:-Qwen/Qwen3-0.5B}"
TASK="${2:-hellaswag}"
N_SHOT="${3:-10}"

# =============================================================================
# Environment Setup
# =============================================================================
source /leonardo_work/OELLM_prod2026/users/bkargi00/scripts/leonardo_env.sh

# Enable offline mode (compute nodes have no internet)
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# =============================================================================
# Create output directory
# =============================================================================
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
OUTPUT_DIR="${EVAL_OUTPUT_DIR}/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "  OELLM Evaluation Job"
echo "=============================================="
echo "  Model:      ${MODEL_NAME}"
echo "  Task:       ${TASK}"
echo "  N-shot:     ${N_SHOT}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Container:  ${EVAL_SIF_PATH}"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Node:       ${SLURM_NODELIST}"
echo "=============================================="

# =============================================================================
# Build bind paths
# =============================================================================
BIND_PATHS="${EVAL_BASE_DIR}:${EVAL_BASE_DIR}"
BIND_PATHS="${BIND_PATHS},${USER_WORK_DIR}:${USER_WORK_DIR}"

# If model is a local path, bind its directory too
if [[ -e "${MODEL_NAME}" ]]; then
    MODEL_DIR="${MODEL_NAME}"
    if [[ ! -d "${MODEL_DIR}" ]]; then
        MODEL_DIR="$(dirname "${MODEL_DIR}")"
    fi
    if [[ "${MODEL_DIR}" != ${EVAL_BASE_DIR}* ]]; then
        BIND_PATHS="${BIND_PATHS},${MODEL_DIR}:${MODEL_DIR}"
    fi
fi

# =============================================================================
# Run evaluation inside container
# =============================================================================
echo ""
echo "Starting lm-eval..."
echo ""

singularity exec ${SINGULARITY_ARGS} \
    --cleanenv \
    --bind "${BIND_PATHS}" \
    --env HF_HOME="${HF_HOME}" \
    --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
    --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
    --env HF_HUB_OFFLINE=1 \
    --env HF_DATASETS_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env PYTHONNOUSERSITE=1 \
    "${EVAL_SIF_PATH}" \
    python -m lm_eval \
        --model hf \
        --model_args pretrained="${MODEL_NAME}",trust_remote_code=True \
        --tasks "${TASK}" \
        --num_fewshot "${N_SHOT}" \
        --output_path "${OUTPUT_DIR}/result.json" \
        --trust_remote_code

echo ""
echo "=============================================="
echo "  Evaluation Complete"
echo "  Results: ${OUTPUT_DIR}/result.json"
echo "=============================================="
