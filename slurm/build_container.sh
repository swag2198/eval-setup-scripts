#!/bin/bash
# =============================================================================
# Build the OELLM Evaluation Container for Leonardo
# =============================================================================
# Run this ONCE on a login node to build the Singularity/Apptainer container.
# This may take 15-30 minutes.
#
# Usage:
#   ./build_container.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/leonardo_env.sh"

OELLM_CLI_DIR="${USER_WORK_DIR}/oellm-cli"
DEF_FILE="${OELLM_CLI_DIR}/apptainer/leonardo.def"

echo "=============================================="
echo "  Building OELLM Evaluation Container"
echo "=============================================="
echo "  Definition: ${DEF_FILE}"
echo "  Output:     ${EVAL_SIF_PATH}"
echo "=============================================="

if [[ ! -f "${DEF_FILE}" ]]; then
    echo "❌ Definition file not found: ${DEF_FILE}"
    echo "   Make sure oellm-cli is cloned at ${OELLM_CLI_DIR}"
    exit 1
fi

# Create output directory
mkdir -p "${EVAL_BASE_DIR}"

# Check if container already exists
if [[ -f "${EVAL_SIF_PATH}" ]]; then
    echo "⚠️  Container already exists: ${EVAL_SIF_PATH}"
    read -p "   Rebuild? [y/N]: " response
    if [[ "${response}" != "y" && "${response}" != "Y" ]]; then
        echo "Skipping build."
        exit 0
    fi
    rm -f "${EVAL_SIF_PATH}"
fi

echo ""
echo "Building container (this may take 15-30 minutes)..."
echo ""

# Build the container
# --fakeroot allows building without root privileges
apptainer build --fakeroot "${EVAL_SIF_PATH}" "${DEF_FILE}"

if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ Container built successfully!"
    echo "   Location: ${EVAL_SIF_PATH}"
    echo "   Size: $(du -h "${EVAL_SIF_PATH}" | cut -f1)"
else
    echo ""
    echo "❌ Container build failed!"
    echo "   Check the error messages above."
    exit 1
fi
