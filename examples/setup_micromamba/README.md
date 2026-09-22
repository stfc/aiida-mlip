# Local Environment Setup with Micromamba for `janus-core` and `aiida-mlip`

This guide demonstrates how to set up and run `janus-core` calculations locally via `aiida-mlip` using `micromamba`, PostgreSQL, RabbitMQ, and AiiDA.

## Quick Start (Automated)

An automated script is provided that performs the full configuration and executes a test calculation:

```bash
# 1. Create and activate the micromamba environment
micromamba create -n aiida-janus -c conda-forge python=3.12 postgresql rabbitmq-server -y
micromamba activate aiida-janus

# 2. Run the automated setup and calculation test
bash examples/setup_micromamba/run_setup_and_test.sh
```

---

## Step-by-Step Manual Setup

### 1. Create the Micromamba Environment

```bash
micromamba create -n aiida-janus -c conda-forge python=3.12 postgresql rabbitmq-server -y
micromamba activate aiida-janus
```

### 2. Configure and Start RabbitMQ

RabbitMQ requires setting the consumer timeout to `undefined` so that long calculations do not prematurely disconnect:

```bash
mkdir -p "$CONDA_PREFIX/etc/rabbitmq"
cat << 'EOF' > "$CONDA_PREFIX/etc/rabbitmq/advanced.config"
%% advanced.config
[
  {rabbit, [
    {consumer_timeout, undefined}
  ]}
].
EOF

# Start RabbitMQ daemon
rabbitmq-server -detached
```

### 3. Initialize and Start PostgreSQL

```bash
# Initialize data directory
initdb -D "$HOME/.aiida_janus_pg"

# Start PostgreSQL server
pg_ctl -D "$HOME/.aiida_janus_pg" -l "$HOME/.aiida_janus_pg/logfile" start

# Create database user and database
createuser --encrypted --pwprompt aiida_janus
# (When prompted, set password to: aiida_janus_password)

createdb --owner=aiida_janus aiida_janus_db
```

### 4. Install Dependencies with `uv`

```bash
python3 -m pip install uv
uv pip install aiida-core
uv pip install -e ".[mace]"
uv pip install "janus-core[mace,chgnet]@git+https://github.com/stfc/janus-core.git"
```

Verify that both `verdi` and `janus` are available:
```bash
which janus
verdi --version
verdi plugin list aiida.calculations
```

### 5. Setup AiiDA Profile

A template profile configuration is available in `profile.yaml`:

```bash
verdi profile setup core.psql_dos --config examples/setup_micromamba/profile.yaml
verdi profile setdefault janus_local
verdi profile configure-rabbitmq
verdi daemon start
verdi status
```

### 6. Setup Localhost Computer

A template computer configuration is available in `localhost.yaml`:

```bash
verdi computer setup --config examples/setup_micromamba/localhost.yaml
verdi computer configure core.local localhost --non-interactive --safe-interval 0
verdi computer test localhost
```

### 7. Register the `janus` Code

Locate the `janus` executable and create the code:

```bash
JANUS_PATH=$(which janus)

verdi code create core.code.installed \
  --non-interactive \
  --label janus \
  --computer localhost \
  --filepath-executable "$JANUS_PATH" \
  --default-calc-job-plugin mlip.sp \
  --description "janus-core MLIP calculator"

verdi code show janus@localhost
```

### 8. Run a Test Calculation

Execute the included singlepoint energy calculation on bulk silicon using MACE:

```bash
python examples/setup_micromamba/test_janus.py --codelabel janus@localhost --arch mace_mp --device cpu
```

Check the process status in AiiDA:
```bash
verdi process list -a
```

---

## Service Management

| Service | Start Command | Stop Command |
| :--- | :--- | :--- |
| **PostgreSQL** | `pg_ctl -D "$HOME/.aiida_janus_pg" -l "$HOME/.aiida_janus_pg/logfile" start` | `pg_ctl -D "$HOME/.aiida_janus_pg" stop` |
| **RabbitMQ** | `rabbitmq-server -detached` | `rabbitmqctl stop` |
| **AiiDA Daemon** | `verdi daemon start` | `verdi daemon stop` |
