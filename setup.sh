#!/bin/bash
# =============================================================================
# First-Time Setup for Leonardo LLM Evaluation Environment
# =============================================================================
# Run this once after cloning the repository. It will:
#   1. Detect your username
#   2. Ask for your SLURM project account
#   3. Generate a personal config file (.env.leonardo)
#   4. Add auto-sourcing to your ~/.bashrc
#
# Usage:
#   bash setup.sh
#   bash setup.sh --account OELLM_prod2026   # non-interactive
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/.env.leonardo"

# ── Colours ──
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

info()  { echo -e "${BLUE}ℹ ${NC}$*"; }
ok()    { echo -e "${GREEN}✓ ${NC}$*"; }
warn()  { echo -e "${YELLOW}⚠ ${NC}$*"; }
err()   { echo -e "${RED}✗ ${NC}$*"; }

# ═══════════════════════════════════════════════════════════════════
#  Parse arguments
# ═══════════════════════════════════════════════════════════════════
ACCOUNT_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --account) ACCOUNT_ARG="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash setup.sh [--account SLURM_ACCOUNT]"
            echo ""
            echo "Options:"
            echo "  --account    SLURM project account (e.g. OELLM_prod2026)"
            echo ""
            echo "If --account is not given, you will be prompted interactively."
            exit 0
            ;;
        *) err "Unknown option: $1"; exit 1 ;;
    esac
done

echo ""
echo -e "${BOLD}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║      Leonardo LLM Evaluation – First-Time Setup         ║${NC}"
echo -e "${BOLD}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════
#  1. Detect username
# ═══════════════════════════════════════════════════════════════════
DETECTED_USER="${USER:-$(whoami)}"
info "Detected username: ${BOLD}${DETECTED_USER}${NC}"

# ═══════════════════════════════════════════════════════════════════
#  2. Get SLURM account
# ═══════════════════════════════════════════════════════════════════
if [[ -n "${ACCOUNT_ARG}" ]]; then
    ACCOUNT="${ACCOUNT_ARG}"
else
    echo ""
    info "Available SLURM accounts (from sacctmgr):"
    # Try to list user's accounts; fall back gracefully
    if command -v sacctmgr &>/dev/null; then
        sacctmgr -n -p show user "${DETECTED_USER}" format=Account 2>/dev/null \
            | tr '|' '\n' | grep -v '^$' | sed 's/^/     /' || true
    else
        warn "sacctmgr not available – enter your account manually."
    fi
    echo ""
    read -rp "$(echo -e ${YELLOW}"Enter your SLURM project account: "${NC})" ACCOUNT
fi

if [[ -z "${ACCOUNT}" ]]; then
    err "Account cannot be empty. Aborting."
    exit 1
fi
ok "Using SLURM account: ${BOLD}${ACCOUNT}${NC}"

# ═══════════════════════════════════════════════════════════════════
#  3. Detect work directory
# ═══════════════════════════════════════════════════════════════════
# Leonardo convention: /leonardo_work/<ACCOUNT>/users/<USERNAME>
WORK_BASE="/leonardo_work/${ACCOUNT}/users/${DETECTED_USER}"

if [[ -d "${WORK_BASE}" ]]; then
    ok "Work directory found: ${WORK_BASE}"
else
    warn "Work directory does not exist yet: ${WORK_BASE}"
    info "It will be created when you first run leonardo_env.sh."
fi

# ═══════════════════════════════════════════════════════════════════
#  4. Ask about shared vs. per-user HF cache
# ═══════════════════════════════════════════════════════════════════
echo ""
info "HuggingFace cache strategy:"
echo "   [1] Per-user cache  (default, each user has own downloads)"
echo "       → ${WORK_BASE}/oellm-evals/hf_data"
echo ""
echo "   [2] Shared cache    (team shares one cache to save disk)"
echo "       → /leonardo_work/${ACCOUNT}/shared/hf_data"
echo ""
read -rp "$(echo -e ${YELLOW}"Choose [1/2] (default: 1): "${NC})" CACHE_CHOICE
CACHE_CHOICE="${CACHE_CHOICE:-1}"

if [[ "${CACHE_CHOICE}" == "2" ]]; then
    HF_DATA_DIR="/leonardo_work/${ACCOUNT}/shared/hf_data"
    CACHE_MODE="shared"
    ok "Using shared HF cache: ${HF_DATA_DIR}"
else
    HF_DATA_DIR="${WORK_BASE}/oellm-evals/hf_data"
    CACHE_MODE="per-user"
    ok "Using per-user HF cache: ${HF_DATA_DIR}"
fi

# ═══════════════════════════════════════════════════════════════════
#  5. SLURM defaults
# ═══════════════════════════════════════════════════════════════════
echo ""
read -rp "$(echo -e ${YELLOW}"Default SLURM partition [boost_usr_prod]: "${NC})" PARTITION
PARTITION="${PARTITION:-boost_usr_prod}"

read -rp "$(echo -e ${YELLOW}"Default GPUs per node [1]: "${NC})" GPUS
GPUS="${GPUS:-1}"

# ═══════════════════════════════════════════════════════════════════
#  6. Write config file
# ═══════════════════════════════════════════════════════════════════
cat > "${CONFIG_FILE}" <<EOF
# =============================================================================
# Leonardo LLM Evaluation – User Configuration
# Generated by setup.sh on $(date '+%Y-%m-%d %H:%M:%S')
# =============================================================================
# Edit this file to change your settings, then re-source leonardo_env.sh.
# DO NOT commit this file to git (it's in .gitignore).
# =============================================================================

# Your Leonardo username
LEONARDO_USER="${DETECTED_USER}"

# SLURM project account
SLURM_ACCOUNT="${ACCOUNT}"

# Base work directory
WORK_DIR="${WORK_BASE}"

# HuggingFace cache directory (per-user or shared)
CACHE_MODE="${CACHE_MODE}"
HF_DATA_DIR="${HF_DATA_DIR}"

# SLURM defaults
DEFAULT_PARTITION="${PARTITION}"
DEFAULT_GPUS="${GPUS}"
DEFAULT_TIME_LIMIT="00:30:00"
DEFAULT_QUEUE_LIMIT=1000
EOF

ok "Config written to: ${CONFIG_FILE}"

# ═══════════════════════════════════════════════════════════════════
#  7. Create directory structure
# ═══════════════════════════════════════════════════════════════════
info "Creating directory structure..."
mkdir -p "${WORK_BASE}/oellm-evals/outputs" 2>/dev/null || true
mkdir -p "${HF_DATA_DIR}"/{hub,datasets,assets,xet} 2>/dev/null || true
mkdir -p "${WORK_BASE}/openjury-eval-data" 2>/dev/null || true
mkdir -p "${WORK_BASE}/slurm_logs" 2>/dev/null || true
ok "Directories created."

# ═══════════════════════════════════════════════════════════════════
#  8. Add to ~/.bashrc
# ═══════════════════════════════════════════════════════════════════
echo ""
BASHRC_LINE="source ${SCRIPT_DIR}/leonardo_env.sh"

if grep -qF "leonardo_env.sh" ~/.bashrc 2>/dev/null; then
    ok "leonardo_env.sh already in ~/.bashrc"
else
    read -rp "$(echo -e ${YELLOW}"Add 'source leonardo_env.sh' to ~/.bashrc? [Y/n]: "${NC})" ADD_BASHRC
    ADD_BASHRC="${ADD_BASHRC:-Y}"
    if [[ "${ADD_BASHRC}" =~ ^[Yy]$ ]]; then
        echo "" >> ~/.bashrc
        echo "# Leonardo LLM Evaluation Environment" >> ~/.bashrc
        echo "${BASHRC_LINE}" >> ~/.bashrc
        ok "Added to ~/.bashrc"
    else
        warn "Skipped. You'll need to run this manually each session:"
        echo "   ${BASHRC_LINE}"
    fi
fi

# ═══════════════════════════════════════════════════════════════════
#  9. Summary
# ═══════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║                    Setup Complete! 🎉                    ║${NC}"
echo -e "${BOLD}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "   User:       ${DETECTED_USER}"
echo "   Account:    ${ACCOUNT}"
echo "   Work dir:   ${WORK_BASE}"
echo "   HF cache:   ${HF_DATA_DIR}  (${CACHE_MODE})"
echo "   Partition:  ${PARTITION}"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo "   1. Source your environment:"
echo "      ${GREEN}source ${SCRIPT_DIR}/leonardo_env.sh${NC}"
echo ""
echo "   2. Download models (on login node):"
echo "      ${GREEN}python ${SCRIPT_DIR}/bin/hf_cache_manager.py download-model Qwen/Qwen2.5-0.5B-Instruct${NC}"
echo ""
echo "   3. Get an interactive GPU node:"
echo "      ${GREEN}${SCRIPT_DIR}/bin/interactive_gpu.sh${NC}"
echo ""
echo "   See ${SCRIPT_DIR}/README.md for full documentation."
echo ""
