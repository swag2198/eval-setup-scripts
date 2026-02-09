#!/bin/bash
# =============================================================================
# Interactive GPU Session Script for Leonardo
# =============================================================================
# Use this script to get an interactive shell on a GPU node for debugging,
# testing evaluations, or exploring the container environment.
#
# Usage:
#   ./interactive_gpu.sh              # Default: 1 hour, 1 GPU
#   ./interactive_gpu.sh 2            # 2 hours, 1 GPU  
#   ./interactive_gpu.sh 4 2          # 4 hours, 2 GPUs
# =============================================================================

# Parse arguments
HOURS=${1:-1}
GPUS=${2:-1}

# Load environment (leonardo_env.sh is in the parent directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
source "${REPO_DIR}/leonardo_env.sh"

echo "=============================================="
echo "  Requesting Interactive GPU Session"
echo "=============================================="
echo "  Duration: ${HOURS} hour(s)"
echo "  GPUs:     ${GPUS}"
echo "  Account:  ${ACCOUNT}"
echo "  Partition: ${PARTITION}"
echo "=============================================="

# Request interactive session
srun --job-name=interactive_gpu \
     --time=${HOURS}:00:00 \
     --nodes=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=8 \
     --gres=gpu:${GPUS} \
     --partition=${PARTITION} \
     --account=${ACCOUNT} \
     --pty bash
