# aiida-mlip Container Environment

A self-contained Docker/Podman container providing an interactive **[Marimo](https://marimo.io/)** environment for running machine learning interatomic potentials (MLIPs) with **[AiiDA](https://www.aiida.net/)** and **[janus-core](https://github.com/stfc/janus-core)**.

---

## Features

- **Automated AiiDA Profile & Services**:
  - Automatically starts **PostgreSQL** and **RabbitMQ** services.
  - Idempotently creates the `default` AiiDA profile (`core.psql_dos`).
  - Configures the `localhost` computer (`core.local` / `core.direct`).
  - Registers `janus@localhost` (`janus-core`) and `python3@localhost` installed codes.
  - Automatically starts and monitors the AiiDA background daemon (`verdi daemon start`).
- **Interactive Marimo Interface**:
  - Web-based reactive notebooks running at `http://localhost:8842`.
  - Includes interactive tutorials demonstrating single-point calculations, geometry optimization, phonons, and WorkGraphs.
- **Universal Engine Support**:
  - Works seamlessly with both **Docker** and **Podman** (including rootless Podman).
  - Supports NVIDIA GPU passthrough for hardware acceleration.

---

## Quick Start

### 1. Build the Image

Using the build helper script (defaults to `podman`):
```bash
./containers/build.sh
```
Or directly with Podman:
```bash
podman build -f containers/Dockerfile -t aiida-mlip:latest .
```

### 2. Launch the Container

Using the startup helper script (defaults to `podman`):
```bash
./containers/startup.sh
```

Or directly with Podman:
```bash
podman run -it --rm -p 8842:8842 aiida-mlip:latest
```

Or using Podman Compose:
```bash
cd containers
podman compose up -d
```

### 3. Open Marimo

Open your browser and navigate to:
```
http://localhost:8842
```
Select `tutorial_marimo.py` to start interacting with `aiida-mlip`!

---

## Startup Script Options

The `./containers/startup.sh` script provides several useful options:

| Option | Description | Default |
|---|---|---|
| `--image <name>` | Container image to run | `aiida-mlip:latest` |
| `--engine <engine>` | Container engine (`podman`, `docker`) | `podman` |
| `--port <port>` | Host port for the Marimo notebook server | `8842` |
| `--restapi-port <port>` | Host port for the AiiDA REST API (`0` to disable) | `5000` |
| `--bind <path>` | Host directory to mount at `/app/tutorials` for persistence | (none) |
| `--gpu` | Enable NVIDIA GPU acceleration | disabled |
| `--name <name>` | Container name | `aiida-mlip-marimo` |
| `-d, --detach` | Run container in background | interactive |
| `-h, --help` | Show command usage and options | |

### Examples

**Persist your work to a local folder:**
```bash
./containers/startup.sh --bind ~/my_mlip_projects
```

**Run with NVIDIA GPU passthrough:**
```bash
./containers/startup.sh --gpu
```

**Run in the background on custom port:**
```bash
./containers/startup.sh --port 9000 -d
```

---

## Interacting with AiiDA via CLI

You can execute `verdi` commands inside the running container at any time:

```bash
# Check status of AiiDA services and daemon
podman exec -it aiida-mlip-marimo su - aiida -c "verdi status"

# List running or completed calculations
podman exec -it aiida-mlip-marimo su - aiida -c "verdi process list -a"

# Open an interactive IPython AiiDA shell
podman exec -it aiida-mlip-marimo su - aiida -c "verdi shell"

# Inspect calculation node
podman exec -it aiida-mlip-marimo su - aiida -c "verdi node show <PK>"
```

---

## Architecture & Design Inspiration

- **Default Profile & Service Setup**: Inspired by [aiidateam/aiida-core](https://github.com/aiidateam/aiida-core) and [stfc/aiidalab-mlip](https://github.com/stfc/aiidalab-mlip).
- **Marimo Server & Lean Build**: Inspired by [stfc/janus-core/containers](https://github.com/stfc/janus-core/tree/main/containers).
