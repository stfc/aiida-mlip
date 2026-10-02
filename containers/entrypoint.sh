#!/usr/bin/env bash
# entrypoint.sh — Container entrypoint for aiida-mlip container.
# Starts PostgreSQL, RabbitMQ, initializes AiiDA profile and daemon,
# then launches Marimo or executes passed command.
set -euo pipefail

CONTAINER_USER="${CONTAINER_USER:-aiida}"
PORT="${PORT:-8842}"
HOST="${HOST:-0.0.0.0}"
TUTORIALS_DIR="${TUTORIALS_DIR:-/app/tutorials}"

# ── 1. Start background services if running as root ──────────────────────────
if [ "$(id -u)" = "0" ]; then
    echo "=== Initializing services as root ==="

    # PostgreSQL startup
    if [ -x "/etc/init.d/postgresql" ]; then
        if ! service postgresql status >/dev/null 2>&1; then
            echo "Starting PostgreSQL..."
            service postgresql start
        fi

        # Ensure database and user exist
        su - postgres -c "psql -tc \"SELECT 1 FROM pg_roles WHERE rolname='${AIIDA_DB_USER:-aiida}'\"" | grep -q 1 || \
            su - postgres -c "psql -c \"CREATE USER ${AIIDA_DB_USER:-aiida} WITH PASSWORD '${AIIDA_DB_PASS:-aiida}'; CREATE DATABASE ${AIIDA_DB_NAME:-aiidadb} OWNER ${AIIDA_DB_USER:-aiida};\""
    fi

    # RabbitMQ startup
    if [ -x "/etc/init.d/rabbitmq-server" ]; then
        if ! service rabbitmq-server status >/dev/null 2>&1; then
            echo "Starting RabbitMQ..."
            service rabbitmq-server start
        fi
    fi

    # Ensure ownership of home directory and tutorial directory
    chown -R "${CONTAINER_USER}:${CONTAINER_USER}" "/home/${CONTAINER_USER}" 2>/dev/null || true
    if [ -d "$TUTORIALS_DIR" ]; then
        chown -R "${CONTAINER_USER}:${CONTAINER_USER}" "$TUTORIALS_DIR" 2>/dev/null || true
    fi

    # Run AiiDA profile setup as non-root container user
    if [ "${AUTO_SETUP_AIIDA:-true}" = "true" ]; then
        su - "$CONTAINER_USER" -c "/usr/local/bin/setup-aiida.sh"
    fi

    # Determine command to run
    if [ "$#" -eq 0 ] || [ "$1" = "marimo" ]; then
        echo "=== Launching Marimo on http://${HOST}:${PORT} ==="
        exec su - "$CONTAINER_USER" -c "exec marimo edit --no-token -p $PORT --host $HOST $TUTORIALS_DIR"
    else
        exec su - "$CONTAINER_USER" -c "exec $*"
    fi
else
    # Running directly as non-root user (e.g. rootless container with services already external or mocked)
    if [ "${AUTO_SETUP_AIIDA:-true}" = "true" ]; then
        /usr/local/bin/setup-aiida.sh || true
    fi

    if [ "$#" -eq 0 ] || [ "$1" = "marimo" ]; then
        echo "=== Launching Marimo on http://${HOST}:${PORT} ==="
        exec marimo edit --no-token -p "$PORT" --host "$HOST" "$TUTORIALS_DIR"
    else
        exec "$@"
    fi
fi
