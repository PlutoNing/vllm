#!/usr/bin/env bash
# Recreate an editable vLLM v0.16 CPU development environment on x86 AVX512.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"

if [[ "${VENV_DIR}" = /* ]]; then
    VENV_PATH="${VENV_DIR}"
else
    VENV_PATH="${REPO_ROOT}/${VENV_DIR}"
fi

cd "${REPO_ROOT}"

if [[ ! -f pyproject.toml || ! -f requirements/cpu.txt ]]; then
    echo "error: ${REPO_ROOT} is not the expected vLLM source tree" >&2
    exit 1
fi

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "error: this script targets Linux x86_64" >&2
    exit 1
fi

if ! grep -qm1 -w avx512f /proc/cpuinfo; then
    echo "error: AVX512F was not found; use the platform-specific CPU build path" >&2
    exit 1
fi

if [[ ! -f /usr/include/numa.h ]]; then
    echo "error: libnuma development headers are missing (numa.h)" >&2
    exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys

if not ((3, 10) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit(
        f"Python 3.10-3.13 is required, got {sys.version.split()[0]}"
    )
PY

if [[ -e "${VENV_PATH}" ]]; then
    echo "error: ${VENV_PATH} already exists" >&2
    echo "Choose another VENV_DIR or remove/rename it deliberately." >&2
    exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_PATH}"
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip
python -m pip install \
    -r requirements/cpu-build.txt \
    -r requirements/cpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Do not set VLLM_CPU_DISABLE_AVX512 here. CMake detects the host ISA and
# enables the AVX512/oneDNN path available on this machine.
VLLM_TARGET_DEVICE=cpu \
MAX_JOBS="${MAX_JOBS}" \
python -m pip install -e . --no-build-isolation

VLLM_TARGET_DEVICE=cpu python - <<'PY'
import torch
import vllm
import vllm._C
from vllm.platforms import current_platform

assert current_platform.device_type == "cpu"
print(f"vLLM: {vllm.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"platform: {current_platform.device_type}")
print(f"extension: {vllm._C.__file__}")
PY

# Keep provenance beside the disposable environment. This records what was
# installed without pretending that a venv is portable between machines.
{
    echo "repo_root=${REPO_ROOT}"
    echo "git_head=$(git rev-parse HEAD)"
    echo "git_describe=$(git describe --tags --long --dirty)"
    echo "python=$(python --version 2>&1)"
    echo "pip=$(python -m pip --version)"
    echo "compiler=$(c++ --version | sed -n '1p')"
    echo "machine=$(uname -srmo)"
    echo
    echo "[git status]"
    git status --short
    echo
    echo "[installed packages]"
    python -m pip freeze --exclude-editable
} >"${VENV_PATH}/vllm-build-info.txt"

echo
echo "Environment created at: ${VENV_PATH}"
echo "Activate with: source ${VENV_PATH}/bin/activate"
echo "Build record: ${VENV_PATH}/vllm-build-info.txt"
