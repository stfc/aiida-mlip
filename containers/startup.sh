#!/usr/bin/env bash
# startup.sh — Launch the aiida-mlip Marimo container.
#
# Usage:
#   ./containers/startup.sh [options]
#
# Options:
#   --image <name>          Container image (default: aiida-mlip:latest)
#   --engine <podman|docker>
#                           Container engine (default: podman)
#   --port <port>           Host port for Marimo (default: 8842)
#   --restapi-port <port>   Host port for AiiDA REST API (default: 5000, 0 to disable)
#   --bind <path>           Host directory to mount at /app/tutorials
#   --gpu                   Enable NVIDIA GPU acceleration
#   --name <name>           Container name (default: aiida-mlip-marimo)
#   -d, --detach            Run container in background
#   -h, --help              Show this help
set -euo pipefail

IMAGE="${IMAGE:-aiida-mlip:latest}"
ENGINE="${ENGINE:-podman}"
PORT="${PORT:-8842}"
RESTAPI_PORT="${RESTAPI_PORT:-5000}"
BIND="${BIND:-}"
USE_GPU=0
CONTAINER_NAME="aiida-mlip-marimo"
DETACH=0

usage() {
    awk 'NR > 1 { if (/^#/) { sub(/^# ?/, ""); print } else exit }' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image) IMAGE="$2"; shift 2 ;;
        --engine) ENGINE="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --restapi-port) RESTAPI_PORT="$2"; shift 2 ;;
        --bind) BIND="$2"; shift 2 ;;
        --gpu) USE_GPU=1; shift ;;
        --name) CONTAINER_NAME="$2"; shift 2 ;;
        -d|--detach) DETACH=1; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

# Check container engine (defaulting to podman)
if ! command -v "$ENGINE" >/dev/null 2>&1; then
    if [[ "$ENGINE" == "podman" ]] && command -v docker >/dev/null 2>&1; then
        echo "WARNING: podman not found, falling back to docker." >&2
        ENGINE=docker
    else
        echo "ERROR: container engine '$ENGINE' not found in PATH." >&2
        exit 1
    fi
fi

RUN_FLAGS=(-p "${PORT}:8842" --name "$CONTAINER_NAME")

if [[ "$RESTAPI_PORT" != "0" ]]; then
    RUN_FLAGS+=(-p "${RESTAPI_PORT}:5000")
fi

if [[ -n "$BIND" ]]; then
    mkdir -p "$BIND"
    RUN_FLAGS+=(-v "${BIND}:/app/tutorials:Z")
fi

if [[ "$DETACH" -eq 1 ]]; then
    RUN_FLAGS+=(-d)
else
    RUN_FLAGS+=(--rm -it)
fi

if [[ "$ENGINE" == "podman" ]]; then
    RUN_FLAGS+=(--security-opt seccomp=unconfined)
    if [[ "$USE_GPU" -eq 1 ]]; then
        RUN_FLAGS+=(--device nvidia.com/gpu=all)
    fi
elif [[ "$ENGINE" == "docker" ]]; then
    if [[ "$USE_GPU" -eq 1 ]]; then
        RUN_FLAGS+=(--gpus all)
    fi
fi

echo "================================================================="
echo "  Starting aiida-mlip Marimo Container ($ENGINE)"
echo "  Image:       $IMAGE"
echo "  Marimo URL:  http://localhost:${PORT}"
if [[ "$RESTAPI_PORT" != "0" ]]; then
    echo "  REST API:    http://localhost:${RESTAPI_PORT}"
fi
if [[ -n "$BIND" ]]; then
    echo "  Bind mount:  ${BIND} -> /app/tutorials"
fi
echo "================================================================="

exec "$ENGINE" run "${RUN_FLAGS[@]}" "$IMAGE"
