#!/bin/bash
# =============================================================================
# Interactive Shell Inside Container on GPU Node
# =============================================================================
# This script requests an interactive GPU allocation AND drops you directly
# into a shell inside the Singularity container.
#
# Perfect for:
# - Testing evaluations interactively
# - Debugging model loading issues
# - Exploring what's installed in the container
# - Running quick experiments
#
# Usage:
#   ./interactive_container.sh              # Default: 1 hour, 1 GPU
#   ./interactive_container.sh 2            # 2 hours
#   ./interactive_container.sh 4 2          # 4 hours, 2 GPUs
# =============================================================================

# Parse arguments
HOURS=${1:-1}
GPUS=${2:-1}

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/leonardo_env.sh"

# Check if container exists
if [[ ! -f "${EVAL_SIF_PATH}" ]]; then
    echo "❌ Container not found: ${EVAL_SIF_PATH}"
    exit 1
fi

echo "=============================================="
echo "  Interactive Container Session"
echo "=============================================="
echo "  Duration:   ${HOURS} hour(s)"
echo "  GPUs:       ${GPUS}"
echo "  Container:  ${EVAL_SIF_PATH}"
echo "  Account:    ${ACCOUNT}"
echo "=============================================="
echo ""
echo "You will be dropped into a bash shell inside the container."
echo "Use 'exit' to leave the container and release the GPU."
echo ""

# Build bind paths
BIND_PATHS="${EVAL_BASE_DIR}:${EVAL_BASE_DIR}"
BIND_PATHS="${BIND_PATHS},${USER_WORK_DIR}:${USER_WORK_DIR}"

# Request interactive session and run container shell
srun --job-name=container_shell \
     --time=${HOURS}:00:00 \
     --nodes=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=8 \
     --gres=gpu:${GPUS} \
     --partition=${PARTITION} \
     --account=${ACCOUNT} \
     --pty \
     singularity exec ${SINGULARITY_ARGS} \
         --cleanenv \
         --bind "${BIND_PATHS}" \
         --env HF_HOME="${HF_HOME}" \
         --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
         --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
         --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
         --env PYTHONNOUSERSITE=1 \
         --env PS1="[container] \u@\h:\w\$ " \
         "${EVAL_SIF_PATH}" \
         bash
