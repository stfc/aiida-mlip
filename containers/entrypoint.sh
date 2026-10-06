#!/usr/bin/env bash
# entrypoint.sh — Container entrypoint for aiida-mlip container.
# Starts PostgreSQL, RabbitMQ, initializes AiiDA profile and daemon,
# then launches Marimo or executes passed command.
set -euo pipefail

CONTAINER_USER="${CONTAINER_USER:-aiida}"
PORT="${PORT:-8842}"
HOST="${HOST:-0.0.0.0}"
TUTORIALS_DIR="${TUTORIALS_DIR:-/app/tutorials}"
NOTEBOOKS_DIR="${NOTEBOOKS_DIR:-/app/notebooks}"

# ── 1. Start background services if running as root ──────────────────────────
if [ "$(id -u)" = "0" ]; then
    echo "=== Initializing services as root ==="

    # PostgreSQL startup
    if [ -x "/etc/init.d/postgresql" ]; then
        if ! service postgresql status >/dev/null 2>&1; then
            echo "Starting PostgreSQL..."
            service postgresql start
        fi

        # Ensure PostgreSQL is ready
        local_attempts=15
        until su - postgres -c "pg_isready -q" >/dev/null 2>&1 || [ "$local_attempts" -le 0 ]; do
            sleep 1
            local_attempts=$((local_attempts - 1))
        done

        # Ensure database user exists
        if ! su - postgres -c "psql -tc \"SELECT 1 FROM pg_roles WHERE rolname='${AIIDA_DB_USER:-aiida}'\"" | grep -q 1; then
            echo "Creating PostgreSQL user '${AIIDA_DB_USER:-aiida}'..."
            su - postgres -c "psql -c \"CREATE USER ${AIIDA_DB_USER:-aiida} WITH PASSWORD '${AIIDA_DB_PASS:-aiida}';\""
        fi

        # Ensure database exists (CREATE DATABASE cannot run inside a multi-statement transaction block)
        if ! su - postgres -c "psql -tc \"SELECT 1 FROM pg_database WHERE datname='${AIIDA_DB_NAME:-aiidadb}'\"" | grep -q 1; then
            echo "Creating PostgreSQL database '${AIIDA_DB_NAME:-aiidadb}'..."
            su - postgres -c "psql -c \"CREATE DATABASE ${AIIDA_DB_NAME:-aiidadb} OWNER ${AIIDA_DB_USER:-aiida};\""
            su - postgres -c "psql -c \"GRANT ALL PRIVILEGES ON DATABASE ${AIIDA_DB_NAME:-aiidadb} TO ${AIIDA_DB_USER:-aiida};\""
        fi
        su - postgres -c "psql -d ${AIIDA_DB_NAME:-aiidadb} -c \"ALTER SCHEMA public OWNER TO ${AIIDA_DB_USER:-aiida}; GRANT ALL ON SCHEMA public TO ${AIIDA_DB_USER:-aiida};\"" 2>/dev/null || true
    fi

    # RabbitMQ startup
    if [ -x "/etc/init.d/rabbitmq-server" ]; then
        if ! service rabbitmq-server status >/dev/null 2>&1; then
            echo "Starting RabbitMQ..."
            service rabbitmq-server start
        fi
    fi

    # Ensure tutorials directory exists and seed default tutorials if missing
    mkdir -p "$TUTORIALS_DIR" 2>/dev/null || true
    if [ ! -f "${TUTORIALS_DIR}/tutorial_marimo.py" ]; then
        if [ -d "/opt/aiida-mlip/containers/assets/tutorials" ]; then
            cp -r /opt/aiida-mlip/containers/assets/tutorials/* "$TUTORIALS_DIR/" 2>/dev/null || true
        elif [ -d "/app/tutorials" ] && [ "$TUTORIALS_DIR" != "/app/tutorials" ]; then
            cp -r /app/tutorials/* "$TUTORIALS_DIR/" 2>/dev/null || true
        fi
    fi

    # Ensure notebooks directory exists and seed default notebooks if missing
    mkdir -p "$NOTEBOOKS_DIR" 2>/dev/null || true
    if [ -z "$(ls -A "$NOTEBOOKS_DIR" 2>/dev/null)" ]; then
        if [ -d "/opt/aiida-mlip/examples" ]; then
            cp -r /opt/aiida-mlip/examples/* "$NOTEBOOKS_DIR/" 2>/dev/null || true
        elif [ -d "/app/examples" ]; then
            cp -r /app/examples/* "$NOTEBOOKS_DIR/" 2>/dev/null || true
        fi
    fi

    # Ensure ownership of home directory and workspace directories
    chown -R "${CONTAINER_USER}:${CONTAINER_USER}" "/home/${CONTAINER_USER}" 2>/dev/null || true
    if [ -d "$TUTORIALS_DIR" ]; then
        chown -R "${CONTAINER_USER}:${CONTAINER_USER}" "$TUTORIALS_DIR" 2>/dev/null || true
    fi
    if [ -d "$NOTEBOOKS_DIR" ]; then
        chown -R "${CONTAINER_USER}:${CONTAINER_USER}" "$NOTEBOOKS_DIR" 2>/dev/null || true
    fi

    # Run AiiDA profile setup as non-root container user
    if [ "${AUTO_SETUP_AIIDA:-true}" = "true" ]; then
        su - "$CONTAINER_USER" -c "/usr/local/bin/setup-aiida.sh"
    fi

    # Determine command to run
    if [ "$#" -eq 0 ] || [ "$1" = "marimo" ]; then
        echo "=== Launching Marimo on http://${HOST}:${PORT} ==="
        exec sudo -u "$CONTAINER_USER" -H -- marimo edit --no-token -p "$PORT" --host "$HOST" "$TUTORIALS_DIR"
    elif [ "$1" = "jupyter" ] || [ "$1" = "jupyter-lab" ] || [ "$1" = "jupyterlab" ]; then
        JUPYTER_ROOT="${NOTEBOOKS_DIR}"
        if [ ! -d "$JUPYTER_ROOT" ]; then
            JUPYTER_ROOT="/app"
        fi
        echo "=== Launching JupyterLab on http://${HOST}:${PORT}/lab ==="
        exec sudo -u "$CONTAINER_USER" -H -- jupyter lab \
            --ip="$HOST" \
            --port="$PORT" \
            --no-browser \
            --ServerApp.token='' \
            --ServerApp.password='' \
            --ServerApp.root_dir="$JUPYTER_ROOT" \
            --allow-root
    else
        exec sudo -u "$CONTAINER_USER" -H -- "$@"
    fi
else
    # Running directly as non-root user (e.g. rootless container with services already external or mocked)
    mkdir -p "$TUTORIALS_DIR" 2>/dev/null || true
    if [ ! -f "${TUTORIALS_DIR}/tutorial_marimo.py" ]; then
        if [ -d "/opt/aiida-mlip/containers/assets/tutorials" ]; then
            cp -r /opt/aiida-mlip/containers/assets/tutorials/* "$TUTORIALS_DIR/" 2>/dev/null || true
        elif [ -d "/app/tutorials" ] && [ "$TUTORIALS_DIR" != "/app/tutorials" ]; then
            cp -r /app/tutorials/* "$TUTORIALS_DIR/" 2>/dev/null || true
        fi
    fi

    mkdir -p "$NOTEBOOKS_DIR" 2>/dev/null || true
    if [ -z "$(ls -A "$NOTEBOOKS_DIR" 2>/dev/null)" ]; then
        if [ -d "/opt/aiida-mlip/examples" ]; then
            cp -r /opt/aiida-mlip/examples/* "$NOTEBOOKS_DIR/" 2>/dev/null || true
        elif [ -d "/app/examples" ]; then
            cp -r /app/examples/* "$NOTEBOOKS_DIR/" 2>/dev/null || true
        fi
    fi

    if [ "${AUTO_SETUP_AIIDA:-true}" = "true" ]; then
        /usr/local/bin/setup-aiida.sh || true
    fi

    if [ "$#" -eq 0 ] || [ "$1" = "marimo" ]; then
        echo "=== Launching Marimo on http://${HOST}:${PORT} ==="
        exec marimo edit --no-token -p "$PORT" --host "$HOST" "$TUTORIALS_DIR"
    elif [ "$1" = "jupyter" ] || [ "$1" = "jupyter-lab" ] || [ "$1" = "jupyterlab" ]; then
        JUPYTER_ROOT="${NOTEBOOKS_DIR}"
        if [ ! -d "$JUPYTER_ROOT" ]; then
            JUPYTER_ROOT="/app"
        fi
        echo "=== Launching JupyterLab on http://${HOST}:${PORT}/lab ==="
        exec jupyter lab \
            --ip="$HOST" \
            --port="$PORT" \
            --no-browser \
            --ServerApp.token='' \
            --ServerApp.password='' \
            --ServerApp.root_dir="$JUPYTER_ROOT" \
            --allow-root
    else
        exec "$@"
    fi
fi
