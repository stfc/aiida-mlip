#!/usr/bin/env bash
# build.sh — Build the aiida-mlip container image(s).
#
# Usage:
#   ./containers/build.sh [options]
#
# Options:
#   --marimo           Build the Marimo image (default: aiida-mlip:latest / aiida-mlip-marimo:latest)
#   --jupyterlab       Build the JupyterLab image (default: aiida-mlip-jupyterlab:latest)
#   --all              Build both Marimo and JupyterLab images
#   --tag <name:tag>   Custom image tag to build
#   --engine <name>    Engine to use: podman or docker (default: podman)
#   --no-cache         Build without docker cache
#   -h, --help         Show this help
set -euo pipefail

ENGINE="${ENGINE:-podman}"
NO_CACHE=""
CUSTOM_TAG=""
BUILD_MARIMO=0
BUILD_JUPYTERLAB=0

usage() {
    awk 'NR > 1 { if (/^#/) { sub(/^# ?/, ""); print } else exit }' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --marimo) BUILD_MARIMO=1; shift ;;
        --jupyterlab) BUILD_JUPYTERLAB=1; shift ;;
        --all) BUILD_MARIMO=1; BUILD_JUPYTERLAB=1; shift ;;
        --tag) CUSTOM_TAG="$2"; shift 2 ;;
        --engine) ENGINE="$2"; shift 2 ;;
        --no-cache) NO_CACHE="--no-cache"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

# If neither is explicitly selected, default to building marimo
if [[ "$BUILD_MARIMO" -eq 0 && "$BUILD_JUPYTERLAB" -eq 0 ]]; then
    BUILD_MARIMO=1
fi

if ! command -v "$ENGINE" >/dev/null 2>&1; then
    if [[ "$ENGINE" == "podman" ]] && command -v docker >/dev/null 2>&1; then
        echo "WARNING: podman not found, falling back to docker." >&2
        ENGINE=docker
    else
        echo "ERROR: container engine '$ENGINE' not found in PATH." >&2
        exit 1
    fi
fi

# Locate repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "$BUILD_MARIMO" -eq 1 ]]; then
    MARIMO_TAG="${CUSTOM_TAG:-aiida-mlip:latest}"
    echo "=== Building Marimo image (${MARIMO_TAG}) using ${ENGINE} ==="
    echo "Context: $REPO_DIR"
    echo "Dockerfile: ${SCRIPT_DIR}/Dockerfile.marimo"
    "$ENGINE" build \
        -f "${SCRIPT_DIR}/Dockerfile.marimo" \
        -t "$MARIMO_TAG" \
        -t "aiida-mlip-marimo:latest" \
        ${NO_CACHE} \
        "$REPO_DIR"
fi

if [[ "$BUILD_JUPYTERLAB" -eq 1 ]]; then
    JUPYTER_TAG="${CUSTOM_TAG:-aiida-mlip-jupyterlab:latest}"
    if [[ "$BUILD_MARIMO" -eq 1 && -n "$CUSTOM_TAG" ]]; then
        JUPYTER_TAG="${CUSTOM_TAG}-jupyterlab"
    fi
    echo "=== Building JupyterLab image (${JUPYTER_TAG}) using ${ENGINE} ==="
    echo "Context: $REPO_DIR"
    echo "Dockerfile: ${SCRIPT_DIR}/Dockerfile.jupyterlab"
    "$ENGINE" build \
        -f "${SCRIPT_DIR}/Dockerfile.jupyterlab" \
        -t "$JUPYTER_TAG" \
        ${NO_CACHE} \
        "$REPO_DIR"
fi

echo "=== Build completed successfully ==="
