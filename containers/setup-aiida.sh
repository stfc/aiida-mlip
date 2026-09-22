#!/usr/bin/env bash
# setup-aiida.sh — Configure AiiDA profile, computer, codes, and daemon for aiida-mlip.
# This script is idempotent and safe to run multiple times.
set -euo pipefail

USER_NAME="${USER:-aiida}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
AIIDA_PROFILE_NAME="${AIIDA_PROFILE_NAME:-default}"
AIIDA_DB_HOST="${AIIDA_DB_HOST:-localhost}"
AIIDA_DB_PORT="${AIIDA_DB_PORT:-5432}"
AIIDA_DB_NAME="${AIIDA_DB_NAME:-aiidadb}"
AIIDA_DB_USER="${AIIDA_DB_USER:-aiida}"
AIIDA_DB_PASS="${AIIDA_DB_PASS:-aiida}"
AIIDA_BROKER_HOST="${AIIDA_BROKER_HOST:-localhost}"
AIIDA_BROKER_PORT="${AIIDA_BROKER_PORT:-5672}"
AIIDA_BROKER_USER="${AIIDA_BROKER_USER:-guest}"
AIIDA_BROKER_PASS="${AIIDA_BROKER_PASS:-guest}"
AIIDA_USER_EMAIL="${AIIDA_USER_EMAIL:-aiida@localhost}"
AIIDA_USER_FIRST_NAME="${AIIDA_USER_FIRST_NAME:-AiiDA}"
AIIDA_USER_LAST_NAME="${AIIDA_USER_LAST_NAME:-User}"
AIIDA_USER_INSTITUTION="${AIIDA_USER_INSTITUTION:-STFC}"
AIIDA_REPO_DIR="${AIIDA_REPO_DIR:-/home/${USER_NAME}/.aiida/repository}"
AIIDA_WORK_DIR="${AIIDA_WORK_DIR:-/home/${USER_NAME}/aiida_run}"

echo "=== Configuring AiiDA environment for aiida-mlip ==="

# ── 1. Locate the janus executable ──────────────────────────────────────────
JANUS_EXE="${JANUS_EXE:-}"
if [ -z "$JANUS_EXE" ]; then
    if command -v janus >/dev/null 2>&1; then
        JANUS_EXE="$(command -v janus)"
    else
        JANUS_EXE="$(python3 -c 'import shutil, sys, os; exe = shutil.which("janus") or os.path.join(os.path.dirname(sys.executable), "janus"); print(exe if os.path.isfile(exe) else "")')"
    fi
fi

if [ -z "$JANUS_EXE" ] || [ ! -x "$JANUS_EXE" ]; then
    echo "ERROR: janus executable not found or not executable (checked '$JANUS_EXE')" >&2
    exit 1
fi
echo "✓ Found janus executable: $JANUS_EXE"

# ── 2. Wait for PostgreSQL and RabbitMQ readiness ───────────────────────────
wait_for_service() {
    local host="$1"
    local port="$2"
    local name="$3"
    local attempts=30
    echo "Checking connectivity to $name ($host:$port)..."
    for i in $(seq 1 "$attempts"); do
        if python3 -c "import socket; s = socket.create_connection(('$host', int($port)), timeout=2); s.close()" >/dev/null 2>&1; then
            echo "✓ $name is ready."
            return 0
        fi
        sleep 1
    done
    echo "WARNING: Timeout waiting for $name at $host:$port (attempted $attempts times)" >&2
    return 1
}

wait_for_service "$AIIDA_DB_HOST" "$AIIDA_DB_PORT" "PostgreSQL" || true
wait_for_service "$AIIDA_BROKER_HOST" "$AIIDA_BROKER_PORT" "RabbitMQ" || true

# ── 3. Configure AiiDA Profile ──────────────────────────────────────────────
mkdir -p "$AIIDA_REPO_DIR" "$AIIDA_WORK_DIR"

if ! verdi profile show "$AIIDA_PROFILE_NAME" >/dev/null 2>&1; then
    echo "Creating AiiDA profile '$AIIDA_PROFILE_NAME' (core.psql_dos)..."
    verdi profile setup core.psql_dos \
        --profile-name "$AIIDA_PROFILE_NAME" \
        --set-as-default \
        --non-interactive \
        --database-hostname "$AIIDA_DB_HOST" \
        --database-port "$AIIDA_DB_PORT" \
        --database-name "$AIIDA_DB_NAME" \
        --database-username "$AIIDA_DB_USER" \
        --database-password "$AIIDA_DB_PASS" \
        --use-rabbitmq \
        --broker-protocol amqp \
        --broker-host "$AIIDA_BROKER_HOST" \
        --broker-port "$AIIDA_BROKER_PORT" \
        --broker-username "$AIIDA_BROKER_USER" \
        --broker-password "$AIIDA_BROKER_PASS" \
        --email "$AIIDA_USER_EMAIL" \
        --first-name "$AIIDA_USER_FIRST_NAME" \
        --last-name "$AIIDA_USER_LAST_NAME" \
        --institution "$AIIDA_USER_INSTITUTION" \
        --repository-uri "file://${AIIDA_REPO_DIR}"
    echo "✓ Profile '$AIIDA_PROFILE_NAME' successfully created."
else
    echo "✓ Profile '$AIIDA_PROFILE_NAME' already exists."
fi

verdi profile setdefault "$AIIDA_PROFILE_NAME" >/dev/null 2>&1 || true

# ── 4. Set up localhost computer ────────────────────────────────────────────
if ! verdi computer show localhost >/dev/null 2>&1; then
    echo "Configuring localhost computer..."
    verdi computer setup \
        --label localhost \
        --hostname localhost \
        --transport core.local \
        --scheduler core.direct \
        --work-dir "$AIIDA_WORK_DIR" \
        --mpirun-command "" \
        --non-interactive
    verdi computer configure core.local localhost --non-interactive --safe-interval 0
    echo "✓ Computer 'localhost' configured."
else
    echo "✓ Computer 'localhost' already configured."
fi

# ── 5. Register janus@localhost code ────────────────────────────────────────
if ! verdi code show "janus@localhost" >/dev/null 2>&1; then
    echo "Registering janus@localhost code..."
    verdi code create core.code.installed \
        --non-interactive \
        --label janus \
        --computer localhost \
        --filepath-executable "$JANUS_EXE" \
        --description "Janus MLIP executable from janus-core"
    echo "✓ Code 'janus@localhost' registered."
else
    echo "✓ Code 'janus@localhost' already registered."
fi

# ── 6. Register python3@localhost code ──────────────────────────────────────
if ! verdi code show "python3@localhost" >/dev/null 2>&1; then
    echo "Registering python3@localhost code..."
    verdi code create core.code.installed \
        --non-interactive \
        --label python3 \
        --computer localhost \
        --filepath-executable "$PYTHON_BIN" \
        --description "Python 3 interpreter"
    echo "✓ Code 'python3@localhost' registered."
else
    echo "✓ Code 'python3@localhost' already registered."
fi

# ── 7. Ensure AiiDA Daemon is running ───────────────────────────────────────
if ! verdi daemon status 2>/dev/null | grep -q "Daemon is running"; then
    echo "Starting AiiDA daemon..."
    verdi daemon start 2 || echo "WARNING: Failed to start AiiDA daemon" >&2
else
    echo "✓ AiiDA daemon is running."
fi

echo "=== AiiDA setup complete for aiida-mlip ==="
verdi status || true
