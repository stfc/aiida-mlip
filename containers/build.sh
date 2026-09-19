#!/usr/bin/env bash
# build.sh — Build the aiida-mlip container image.
#
# Usage:
#   ./containers/build.sh [options]
#
# Options:
#   --tag <name:tag>   Image tag to build (default: aiida-mlip:latest)
#   --engine <name>    Engine to use: podman or docker (default: podman)
#   --no-cache         Build without docker cache
#   -h, --help         Show this help
set -euo pipefail

TAG="${TAG:-aiida-mlip:latest}"
ENGINE="${ENGINE:-podman}"
NO_CACHE=""

usage() {
    awk 'NR > 1 { if (/^#/) { sub(/^# ?/, ""); print } else exit }' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag) TAG="$2"; shift 2 ;;
        --engine) ENGINE="$2"; shift 2 ;;
        --no-cache) NO_CACHE="--no-cache"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

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

echo "=== Building ${TAG} using ${ENGINE} ==="
echo "Context: $REPO_DIR"
echo "Dockerfile: ${SCRIPT_DIR}/Dockerfile"

exec "$ENGINE" build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t "$TAG" \
    ${NO_CACHE} \
    "$REPO_DIR"
