#!/bin/bash
# =============================================================================
# First-Time Setup — Cluster-Agnostic LLM Evaluation Environment
# =============================================================================
# Run this once after cloning the repository. It will:
#   1. Auto-detect the current HPC cluster (or fall back to local)
#   2. Ask for your SLURM project account
#   3. Ask for your work directory and HF cache location
#   4. Generate a personal config file (.env)
#   5. Add auto-sourcing to your ~/.bashrc
#   6. Optionally install Python deps via uv
#   7. Optionally configure oellm-cli clusters.yaml
#
# Usage:
#   bash setup.sh                            # interactive first-time setup
#   bash setup.sh --account OELLM_prod2026   # non-interactive account
#   bash setup.sh --reconfigure              # re-run even if already configured
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/.env"
# Backward compat: if only the old config exists, use it
if [[ ! -f "${CONFIG_FILE}" && -f "${SCRIPT_DIR}/.env.leonardo" ]]; then
    CONFIG_FILE="${SCRIPT_DIR}/.env.leonardo"
fi

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
RECONFIGURE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --account) ACCOUNT_ARG="$2"; shift 2 ;;
        --reconfigure) RECONFIGURE=true; shift ;;
        -h|--help)
            echo "Usage: bash setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --account ACCT    SLURM project account (e.g. OELLM_prod2026)"
            echo "  --reconfigure     Re-run setup even if .env exists"
            echo "  -h, --help        Show this help"
            echo ""
            echo "If --account is not given, you will be prompted interactively."
            exit 0
            ;;
        *) err "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Guard: check for existing config ──
if [[ -f "${CONFIG_FILE}" && "${RECONFIGURE}" == "false" ]]; then
    warn "Config already exists: ${CONFIG_FILE}"
    info "To re-run setup, use:  bash setup.sh --reconfigure"
    echo ""
    read -rp "$(echo -e ${YELLOW}"Continue anyway? [y/N]: "${NC})" CONT
    if [[ ! "${CONT}" =~ ^[Yy]$ ]]; then
        info "Aborted. Your existing config is unchanged."
        exit 0
    fi
fi

# ═══════════════════════════════════════════════════════════════════
#  0. Detect cluster
# ═══════════════════════════════════════════════════════════════════
# Try the Python module first; fall back to minimal shell detection
CLUSTER_NAME="local"
CLUSTER_DISPLAY="Local machine"
CLUSTER_IS_HPC="0"
CLUSTER_PARTITION=""
CLUSTER_GPU_PARTITION=""
CLUSTER_QUEUE_LIMIT=""
CLUSTER_GPUS_PER_NODE=""
CLUSTER_CPUS_PER_GPU=""
CLUSTER_CONTAINER_RUNTIME=""
CLUSTER_CONTAINER_GPU_ARGS=""

_detect_via_python() {
    local _py=""
    # Prefer uv run if available, otherwise direct python
    if command -v uv &>/dev/null && [[ -f "${SCRIPT_DIR}/pyproject.toml" ]]; then
        _py="uv run --quiet python -m cluster_utils.cluster"
    elif command -v python3 &>/dev/null; then
        _py="PYTHONPATH=${SCRIPT_DIR}/src python3 -m cluster_utils.cluster"
    elif command -v python &>/dev/null; then
        _py="PYTHONPATH=${SCRIPT_DIR}/src python -m cluster_utils.cluster"
    fi
    if [[ -n "${_py}" ]]; then
        eval "$(eval ${_py} 2>/dev/null)" && return 0
    fi
    return 1
}

if _detect_via_python; then
    : # Variables set by eval above
else
    # Minimal fallback: hostname-based detection
    _hostname="${HOSTNAME:-${HOST:-$(hostname 2>/dev/null || true)}}"
    _hostname_lower="${_hostname,,}"
    if [[ -d "/leonardo_work" ]]; then
        CLUSTER_NAME="leonardo"; CLUSTER_DISPLAY="Leonardo (Cineca)"; CLUSTER_IS_HPC="1"
        CLUSTER_PARTITION="boost_usr_prod"; CLUSTER_GPU_PARTITION="boost_usr_prod"
        CLUSTER_QUEUE_LIMIT="1000"; CLUSTER_GPUS_PER_NODE="4"; CLUSTER_CPUS_PER_GPU="8"
        CLUSTER_CONTAINER_RUNTIME="singularity"; CLUSTER_CONTAINER_GPU_ARGS="--nv"
    elif [[ "${_hostname_lower}" == *"jureca"* ]]; then
        CLUSTER_NAME="jureca"; CLUSTER_DISPLAY="JURECA (FZJ)"; CLUSTER_IS_HPC="1"
        CLUSTER_PARTITION="dc-gpu"; CLUSTER_GPU_PARTITION="dc-gpu"
        CLUSTER_CONTAINER_RUNTIME="singularity"; CLUSTER_CONTAINER_GPU_ARGS="--nv"
    fi
fi

# ── Banner ──
echo ""
echo -e "${BOLD}╔═══════════════════════════════════════════════════════════╗${NC}"
if [[ "${CLUSTER_IS_HPC}" == "1" ]]; then
    _banner="$(printf "║   %s — First-Time Setup" "${CLUSTER_DISPLAY}")"
    _banner="$(printf "%-59s║" "${_banner}")"
    echo -e "${BOLD}${_banner}${NC}"
else
    echo -e "${BOLD}║      LLM Evaluation Environment – First-Time Setup       ║${NC}"
fi
echo -e "${BOLD}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

if [[ "${CLUSTER_IS_HPC}" == "1" ]]; then
    ok "Detected cluster: ${BOLD}${CLUSTER_DISPLAY}${NC}"
else
    info "No known HPC cluster detected — running in local mode."
fi

# ═══════════════════════════════════════════════════════════════════
#  1. Detect username
# ═══════════════════════════════════════════════════════════════════
DETECTED_USER="${USER:-$(whoami)}"
info "Detected username: ${BOLD}${DETECTED_USER}${NC}"

# ═══════════════════════════════════════════════════════════════════
#  2. Get SLURM account (HPC only)
# ═══════════════════════════════════════════════════════════════════
ACCOUNT=""
if [[ "${CLUSTER_IS_HPC}" == "1" ]]; then
    if [[ -n "${ACCOUNT_ARG}" ]]; then
        ACCOUNT="${ACCOUNT_ARG}"
    else
        echo ""
        info "Available SLURM accounts (from sacctmgr):"
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
        err "Account cannot be empty on an HPC cluster. Aborting."
        exit 1
    fi

    # Validate the account exists for this user
    if command -v sacctmgr &>/dev/null; then
        _valid_accounts=$(sacctmgr -n -p show user "${DETECTED_USER}" format=Account 2>/dev/null \
            | tr '|' '\n' | grep -v '^$' || true)
        if [[ -n "${_valid_accounts}" ]]; then
            if ! echo "${_valid_accounts}" | grep -qx "${ACCOUNT}"; then
                warn "Account '${ACCOUNT}' not found in your SLURM associations."
                warn "Your valid accounts: $(echo ${_valid_accounts} | tr '\n' ', ')"
                read -rp "$(echo -e ${YELLOW}"Continue anyway? [y/N]: "${NC})" FORCE
                if [[ ! "${FORCE}" =~ ^[Yy]$ ]]; then
                    err "Aborted."; exit 1
                fi
            fi
        fi
    fi
    ok "Using SLURM account: ${BOLD}${ACCOUNT}${NC}"
fi

# ═══════════════════════════════════════════════════════════════════
#  3. Choose work directory
# ═══════════════════════════════════════════════════════════════════
echo ""
info "Where should the evaluation environment live?"
echo "   This directory will hold eval outputs, SLURM job logs,"
echo "   and tool-specific data (oellm-cli, OpenJury, etc.)."
echo ""

_default_work="$(pwd)"
info "Suggested default (current directory): ${_default_work}"

read -rp "$(echo -e ${YELLOW}"Work directory [${_default_work}]: "${NC})" WORK_BASE
WORK_BASE="${WORK_BASE:-${_default_work}}"

if [[ -d "${WORK_BASE}" ]]; then
    ok "Work directory found: ${WORK_BASE}"
else
    warn "Directory does not exist yet: ${WORK_BASE}"
    info "It will be created when you first source env.sh."
fi

# ═══════════════════════════════════════════════════════════════════
#  4. Ask about HF cache location
# ═══════════════════════════════════════════════════════════════════
echo ""
info "HuggingFace cache directory:"
echo "   This is where models and datasets will be downloaded to."
echo "   It should be on a filesystem with enough space (100+ GB)."
echo ""

_default_hf="${WORK_BASE}/hf_cache"
echo "   [1] Default path: ${_default_hf}"
echo "   [2] Custom path"
echo ""
read -rp "$(echo -e ${YELLOW}"Choose [1/2] (default: 1): "${NC})" CACHE_CHOICE
CACHE_CHOICE="${CACHE_CHOICE:-1}"

case "${CACHE_CHOICE}" in
    2)
        read -rp "$(echo -e ${YELLOW}"Enter full HF cache path: "${NC})" HF_DATA_DIR
        if [[ -z "${HF_DATA_DIR}" ]]; then
            err "Path cannot be empty. Aborting."; exit 1
        fi
        CACHE_MODE="custom"
        ;;
    *)
        HF_DATA_DIR="${_default_hf}"
        CACHE_MODE="per-user"
        ;;
esac
ok "HF cache: ${HF_DATA_DIR}  (${CACHE_MODE})"

# ═══════════════════════════════════════════════════════════════════
#  5. SLURM defaults (HPC only)
# ═══════════════════════════════════════════════════════════════════
if [[ "${CLUSTER_IS_HPC}" == "1" ]]; then
    echo ""
    info "SLURM defaults (press Enter to accept detected values):"

    _def_partition="${CLUSTER_GPU_PARTITION:-${CLUSTER_PARTITION}}"
    read -rp "$(echo -e ${YELLOW}"Default GPU partition [${_def_partition}]: "${NC})" PARTITION
    PARTITION="${PARTITION:-${_def_partition}}"

    _def_gpus="${CLUSTER_GPUS_PER_NODE:-1}"
    read -rp "$(echo -e ${YELLOW}"Default GPUs per node [${_def_gpus}]: "${NC})" GPUS
    GPUS="${GPUS:-${_def_gpus}}"

    _def_cpus="${CLUSTER_CPUS_PER_GPU:-8}"
    read -rp "$(echo -e ${YELLOW}"CPUs per GPU [${_def_cpus}]: "${NC})" CPUS_PER_GPU
    CPUS_PER_GPU="${CPUS_PER_GPU:-${_def_cpus}}"

    _def_limit="${CLUSTER_QUEUE_LIMIT:-1000}"
    read -rp "$(echo -e ${YELLOW}"Queue limit [${_def_limit}]: "${NC})" QUEUE_LIMIT
    QUEUE_LIMIT="${QUEUE_LIMIT:-${_def_limit}}"
else
    PARTITION=""
    GPUS="1"
    CPUS_PER_GPU="8"
    QUEUE_LIMIT=""
fi

# ═══════════════════════════════════════════════════════════════════
#  5b. SLURM job directory
# ═══════════════════════════════════════════════════════════════════
echo ""
info "SLURM job directory (stores logs, scripts, and results):"
_def_slurm_dir="${WORK_BASE}/slurm_jobs"
read -rp "$(echo -e ${YELLOW}"SLURM work dir [${_def_slurm_dir}]: "${NC})" SLURM_WORK_INPUT
SLURM_WORK_BASE="${SLURM_WORK_INPUT:-${_def_slurm_dir}}"

# ═══════════════════════════════════════════════════════════════════
#  6. Write config file
# ═══════════════════════════════════════════════════════════════════
CONFIG_FILE="${SCRIPT_DIR}/.env"  # Always write the new-style config
cat > "${CONFIG_FILE}" <<ENVEOF
# =============================================================================
# LLM Evaluation Environment – User Configuration
# Generated by setup.sh on $(date '+%Y-%m-%d %H:%M:%S')
# Cluster: ${CLUSTER_DISPLAY} (${CLUSTER_NAME})
# =============================================================================
# Edit this file to change your settings, then re-source env.sh.
# DO NOT commit this file to git (it's in .gitignore).
# =============================================================================

# Detected cluster
CLUSTER_NAME="${CLUSTER_NAME}"

# Your username
ENV_USER="${DETECTED_USER}"

# SLURM project account
SLURM_ACCOUNT="${ACCOUNT}"

# Base work directory
WORK_DIR="${WORK_BASE}"

# HuggingFace cache directory
CACHE_MODE="${CACHE_MODE}"
HF_DATA_DIR="${HF_DATA_DIR}"

# SLURM defaults
DEFAULT_PARTITION="${PARTITION}"
DEFAULT_GPUS="${GPUS}"
DEFAULT_CPUS_PER_GPU="${CPUS_PER_GPU}"
DEFAULT_TIME_LIMIT="00:30:00"
DEFAULT_QUEUE_LIMIT="${QUEUE_LIMIT}"

# Container settings (from cluster detection)
CONTAINER_RUNTIME="${CLUSTER_CONTAINER_RUNTIME}"
CONTAINER_GPU_ARGS="${CLUSTER_CONTAINER_GPU_ARGS}"

# Shared SLURM job directories (used by both oellm-cli and OpenJury)
SLURM_WORK_BASE="${SLURM_WORK_BASE}"
ENVEOF

ok "Config written to: ${CONFIG_FILE}"

# Remove old .env.leonardo if .env now exists alongside it
if [[ -f "${SCRIPT_DIR}/.env.leonardo" && -f "${SCRIPT_DIR}/.env" ]]; then
    info "(Old .env.leonardo kept for reference. .env takes priority.)"
fi

# ═══════════════════════════════════════════════════════════════════
#  7. Create directory structure
# ═══════════════════════════════════════════════════════════════════
info "Creating directory structure..."
mkdir -p "${WORK_BASE}/oellm-evals/outputs" 2>/dev/null || true
mkdir -p "${HF_DATA_DIR}"/{hub,datasets,assets,xet} 2>/dev/null || true
mkdir -p "${WORK_BASE}/openjury-eval-data" 2>/dev/null || true
mkdir -p "${SLURM_WORK_BASE}/logs" 2>/dev/null || true
mkdir -p "${SLURM_WORK_BASE}/oellm-cli" 2>/dev/null || true
mkdir -p "${SLURM_WORK_BASE}/openjury" 2>/dev/null || true
ok "Directories created."

# ═══════════════════════════════════════════════════════════════════
#  8. Add to ~/.bashrc
# ═══════════════════════════════════════════════════════════════════
echo ""
ENV_SH_PATH="${SCRIPT_DIR}/env.sh"
BASHRC_LINE="source ${ENV_SH_PATH}"

# Check for either the old or new env script in bashrc
if grep -qF "env.sh" ~/.bashrc 2>/dev/null; then
    ok "env.sh already in ~/.bashrc"
elif grep -qF "leonardo_env.sh" ~/.bashrc 2>/dev/null; then
    read -rp "$(echo -e ${YELLOW}"Update ~/.bashrc to use env.sh instead of leonardo_env.sh? [Y/n]: "${NC})" UPDATE_BASHRC
    UPDATE_BASHRC="${UPDATE_BASHRC:-Y}"
    if [[ "${UPDATE_BASHRC}" =~ ^[Yy]$ ]]; then
        sed -i "s|source .*leonardo_env.sh|source ${ENV_SH_PATH}|g" ~/.bashrc
        ok "Updated ~/.bashrc reference."
    fi
else
    read -rp "$(echo -e ${YELLOW}"Add 'source env.sh' to ~/.bashrc? [Y/n]: "${NC})" ADD_BASHRC
    ADD_BASHRC="${ADD_BASHRC:-Y}"
    if [[ "${ADD_BASHRC}" =~ ^[Yy]$ ]]; then
        echo "" >> ~/.bashrc
        echo "# LLM Evaluation Environment" >> ~/.bashrc
        echo "${BASHRC_LINE}" >> ~/.bashrc
        ok "Added to ~/.bashrc"
    else
        warn "Skipped. You'll need to run this manually each session:"
        echo "   ${BASHRC_LINE}"
    fi
fi

# ═══════════════════════════════════════════════════════════════════
#  9. Install Python dependencies via uv
# ═══════════════════════════════════════════════════════════════════
echo ""
if command -v uv &>/dev/null && [[ -f "${SCRIPT_DIR}/pyproject.toml" ]]; then
    read -rp "$(echo -e ${YELLOW}"Run 'uv sync' to install Python dependencies? [Y/n]: "${NC})" DO_UV
    DO_UV="${DO_UV:-Y}"
    if [[ "${DO_UV}" =~ ^[Yy]$ ]]; then
        info "Running uv sync..."
        (cd "${SCRIPT_DIR}" && uv sync 2>&1 | tail -5)
        ok "Python dependencies installed."
    else
        warn "Skipped. Run manually:  cd ${SCRIPT_DIR} && uv sync"
    fi
else
    if ! command -v uv &>/dev/null; then
        warn "'uv' not found. Install it first, then run:  cd ${SCRIPT_DIR} && uv sync"
    fi
fi

# ═══════════════════════════════════════════════════════════════════
#  10. Summary
# ═══════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║                    Setup Complete! 🎉                    ║${NC}"
echo -e "${BOLD}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "   Cluster:    ${CLUSTER_DISPLAY} (${CLUSTER_NAME})"
echo "   User:       ${DETECTED_USER}"
[[ -n "${ACCOUNT}" ]] && echo "   Account:    ${ACCOUNT}"
echo "   Work dir:   ${WORK_BASE}"
echo "   SLURM jobs: ${SLURM_WORK_BASE}"
echo "   HF cache:   ${HF_DATA_DIR}  (${CACHE_MODE})"
[[ -n "${PARTITION}" ]] && echo "   Partition:  ${PARTITION}"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo "   1. Source your environment:"
echo -e "      ${GREEN}source ${SCRIPT_DIR}/env.sh${NC}"
echo ""
echo "   2. Download models (on login node):"
echo -e "      ${GREEN}hf-cache download-model Qwen/Qwen2.5-0.5B-Instruct${NC}"
echo ""
if [[ "${CLUSTER_IS_HPC}" == "1" ]]; then
    echo "   3. Submit SLURM jobs or get an interactive node"
fi
echo ""
echo "   See ${SCRIPT_DIR}/README.md for full documentation."
echo ""
