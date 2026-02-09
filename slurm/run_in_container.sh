#!/bin/bash
# =============================================================================
# Run Commands Inside the OELLM Container
# =============================================================================
# Use this script to run any command inside the Singularity/Apptainer container
# that contains all the evaluation dependencies (lm-eval, transformers, etc.)
#
# This ensures you use the container's Python environment, NOT your conda.
#
# Usage:
#   ./run_in_container.sh python --version
#   ./run_in_container.sh python -c "import torch; print(torch.cuda.is_available())"
#   ./run_in_container.sh lm-eval --help
#   ./run_in_container.sh pip list
#
# For an interactive shell inside the container:
#   ./run_in_container.sh bash
# =============================================================================

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/leonardo_env.sh"

# Check if container exists
if [[ ! -f "${EVAL_SIF_PATH}" ]]; then
    echo "❌ Container not found: ${EVAL_SIF_PATH}"
    echo "   The shared container may not be available yet."
    echo "   Contact your admin or check the oellm-cli setup."
    exit 1
fi

# Build bind paths
BIND_PATHS="${EVAL_BASE_DIR}:${EVAL_BASE_DIR}"
BIND_PATHS="${BIND_PATHS},${USER_WORK_DIR}:${USER_WORK_DIR}"

# Add scratch if exists
if [[ -d "/leonardo_scratch" ]]; then
    BIND_PATHS="${BIND_PATHS},/leonardo_scratch:/leonardo_scratch"
fi

# Run command inside container
# --nv enables NVIDIA GPU support
# --cleanenv prevents host environment from leaking in
# PYTHONNOUSERSITE=1 prevents user site-packages from being used
echo "🐳 Running inside container: ${EVAL_SIF_PATH}"
echo "   Command: $@"
echo "================================================"

singularity exec ${SINGULARITY_ARGS} \
    --cleanenv \
    --bind "${BIND_PATHS}" \
    --env HF_HOME="${HF_HOME}" \
    --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
    --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
    --env PYTHONNOUSERSITE=1 \
    "${EVAL_SIF_PATH}" \
    "$@"
