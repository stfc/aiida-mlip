#!/usr/bin/env bash
# run_setup_and_test.sh: Automate local setup and execution test for janus-core with aiida-mlip
set -euo pipefail

CLEANUP=false
for arg in "$@"; do
    case "$arg" in
        --cleanup)
            CLEANUP=true
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Configuration variables
CONDA_PREFIX="${CONDA_PREFIX:-$(python3 -c 'import sys; print(sys.prefix)')}"
PGDATA="${PGDATA:-${SCRIPT_DIR}/.pg_data}"
PGLOG="${PGDATA}/logfile"
DB_NAME="${DB_NAME:-aiida_janus_db}"
DB_USER="${DB_USER:-aiida_janus}"
DB_PASS="${DB_PASS:-aiida_janus_password}"
PROFILE_NAME="${PROFILE_NAME:-janus_local}"
AIIDA_REPO="${AIIDA_REPO:-${SCRIPT_DIR}/.aiida_repo}"
AIIDA_WORK_DIR="${AIIDA_WORK_DIR:-${SCRIPT_DIR}/.aiida_work}"

echo "=========================================================="
echo " 1. RabbitMQ Configuration and Startup"
echo "=========================================================="
mkdir -p "${CONDA_PREFIX}/etc/rabbitmq"
if [ ! -f "${CONDA_PREFIX}/etc/rabbitmq/advanced.config" ]; then
    cat << 'EOF' > "${CONDA_PREFIX}/etc/rabbitmq/advanced.config"
%% advanced.config
[
  {rabbit, [
    {consumer_timeout, undefined}
  ]}
].
EOF
fi

if ! rabbitmqctl status >/dev/null 2>&1; then
    echo "Starting RabbitMQ server..."
    rabbitmq-server -detached
    # Wait for RabbitMQ port 5672
    for i in $(seq 1 30); do
        if python3 -c "import socket; s = socket.create_connection(('localhost', 5672), timeout=1); s.close()" >/dev/null 2>&1; then
            echo "RabbitMQ is up and running."
            break
        fi
        sleep 1
    done
else
    echo "RabbitMQ is already running."
fi

echo "=========================================================="
echo " 2. PostgreSQL Configuration and Startup"
echo "=========================================================="
if [ ! -d "${PGDATA}" ]; then
    echo "Initializing PostgreSQL cluster at ${PGDATA}..."
    initdb -D "${PGDATA}" --auth=trust
fi

if ! pg_ctl -D "${PGDATA}" status >/dev/null 2>&1; then
    echo "Starting PostgreSQL server..."
    pg_ctl -D "${PGDATA}" -l "${PGLOG}" start
    sleep 2
else
    echo "PostgreSQL is already running."
fi

# Ensure user and database exist
if ! psql -U "$(whoami)" -d postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" 2>/dev/null | grep -q 1; then
    echo "Creating database user '${DB_USER}'..."
    psql -U "$(whoami)" -d postgres -c "CREATE USER \"${DB_USER}\" WITH PASSWORD '${DB_PASS}' CREATEDB;"
fi

if ! psql -U "$(whoami)" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" 2>/dev/null | grep -q 1; then
    echo "Creating database '${DB_NAME}'..."
    createdb -U "$(whoami)" --owner="${DB_USER}" "${DB_NAME}"
fi

echo "=========================================================="
echo " 3. Installing Dependencies via uv"
echo "=========================================================="
python3 -m pip install -q uv
uv pip install -q aiida-core
uv pip install -q -e "${REPO_DIR}[mace]"
uv pip install -q "janus-core[mace]@git+https://github.com/stfc/janus-core.git"

JANUS_EXE="$(which janus || python3 -c 'import shutil, sys, os; exe = shutil.which("janus") or os.path.join(os.path.dirname(sys.executable), "janus"); print(exe if os.path.isfile(exe) else "")')"
if [ -z "${JANUS_EXE}" ] || [ ! -x "${JANUS_EXE}" ]; then
    echo "ERROR: janus executable not found or not executable!" >&2
    exit 1
fi
echo "Using janus executable: ${JANUS_EXE}"

echo "=========================================================="
echo " 4. AiiDA Profile Configuration"
echo "=========================================================="
mkdir -p "${AIIDA_REPO}" "${AIIDA_WORK_DIR}"

if ! verdi profile show "${PROFILE_NAME}" >/dev/null 2>&1; then
    echo "Creating profile '${PROFILE_NAME}'..."
    verdi profile setup core.psql_dos \
        --profile-name "${PROFILE_NAME}" \
        --set-as-default \
        --non-interactive \
        --database-hostname localhost \
        --database-port 5432 \
        --database-name "${DB_NAME}" \
        --database-username "${DB_USER}" \
        --database-password "${DB_PASS}" \
        --use-rabbitmq \
        --broker-protocol amqp \
        --broker-host localhost \
        --broker-port 5672 \
        --broker-username guest \
        --broker-password guest \
        --email "user@localhost" \
        --first-name "Janus" \
        --last-name "User" \
        --institution "STFC" \
        --repository-uri "file://${AIIDA_REPO}"
fi

verdi profile setdefault "${PROFILE_NAME}" >/dev/null 2>&1 || true
verdi profile configure-rabbitmq || true

if ! verdi daemon status 2>/dev/null | grep -q "Daemon is running"; then
    echo "Starting AiiDA daemon..."
    verdi daemon start 2
fi

verdi status

echo "=========================================================="
echo " 5. Computer Setup (localhost)"
echo "=========================================================="
if ! verdi computer show localhost >/dev/null 2>&1; then
    echo "Registering localhost computer..."
    verdi computer setup \
        --label localhost \
        --hostname localhost \
        --transport core.local \
        --scheduler core.direct \
        --work-dir "${AIIDA_WORK_DIR}" \
        --mpirun-command "" \
        --non-interactive
    verdi computer configure core.local localhost --non-interactive --safe-interval 0
else
    echo "Computer 'localhost' is already registered."
fi

verdi computer test localhost

echo "=========================================================="
echo " 6. Janus Code Setup (janus@localhost)"
echo "=========================================================="
if ! verdi code show janus@localhost >/dev/null 2>&1; then
    echo "Registering janus@localhost code..."
    verdi code create core.code.installed \
        --non-interactive \
        --label janus \
        --computer localhost \
        --filepath-executable "${JANUS_EXE}" \
        --default-calc-job-plugin mlip.sp \
        --description "janus-core MLIP calculator"
else
    echo "Code 'janus@localhost' is already registered."
fi

verdi code show janus@localhost

echo "=========================================================="
echo " 7. Executing Test Calculation"
echo "=========================================================="
python3 "${SCRIPT_DIR}/test_janus.py" --codelabel janus@localhost --arch mace_mp --device cpu

verdi process list -a

echo "=========================================================="
echo " All tests passed successfully!"
echo "=========================================================="

if [ "${CLEANUP}" = true ]; then
    echo "Cleaning up services..."
    verdi daemon stop >/dev/null 2>&1 || true
    pg_ctl -D "${PGDATA}" stop >/dev/null 2>&1 || true
    rabbitmqctl stop >/dev/null 2>&1 || true
    echo "Cleanup complete."
fi
