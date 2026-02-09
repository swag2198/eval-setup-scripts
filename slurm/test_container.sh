#!/bin/bash
# =============================================================================
# Quick Test: Verify Container and GPU Setup
# =============================================================================
# Run this on a GPU node to verify everything is working correctly.
#
# Usage (after getting a GPU allocation):
#   ./test_container.sh
#
# Or submit as a job:
#   sbatch --wrap="./test_container.sh" --gres=gpu:1 --time=00:10:00 \
#          --partition=boost_usr_prod --account=AIFAC_L01_028
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/leonardo_env.sh"

echo "=============================================="
echo "  Container & GPU Test"
echo "=============================================="

# Check if we have a GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi available on host"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ nvidia-smi not found (might be on login node)"
fi

echo ""
echo "Testing container..."
echo ""

# Build bind paths
BIND_PATHS="${EVAL_BASE_DIR}:${EVAL_BASE_DIR},${USER_WORK_DIR}:${USER_WORK_DIR}"

# Test Python and PyTorch inside container
singularity exec ${SINGULARITY_ARGS} \
    --cleanenv \
    --bind "${BIND_PATHS}" \
    --env PYTHONNOUSERSITE=1 \
    "${EVAL_SIF_PATH}" \
    python -c "
import sys
print(f'✓ Python version: {sys.version}')
print(f'  Executable: {sys.executable}')

import torch
print(f'✓ PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

import transformers
print(f'✓ Transformers version: {transformers.__version__}')

import lm_eval
print(f'✓ lm-eval version: {lm_eval.__version__}')

print()
print('All tests passed! Container is ready for evaluations.')
"

echo ""
echo "=============================================="
