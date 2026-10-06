# aiida-mlip Container Environments

Self-contained Docker/Podman containers providing interactive **[Marimo](https://marimo.io/)** and **[JupyterLab](https://jupyter.org/)** environments for running machine learning interatomic potentials (MLIPs) with **[AiiDA](https://www.aiida.net/)** and **[janus-core](https://github.com/stfc/janus-core)**.

---

## Features

- **Automated AiiDA Profile & Services**:
  - Automatically starts **PostgreSQL** and **RabbitMQ** services.
  - Idempotently creates the `default` AiiDA profile (`core.psql_dos`).
  - Configures the `localhost` computer (`core.local` / `core.direct`).
  - Registers `janus@localhost` (`janus-core`) and `python3@localhost` installed codes.
  - Automatically starts and monitors the AiiDA background daemon (`verdi daemon start`).
- **Interactive Notebook Interfaces**:
  - **Marimo** (`Dockerfile.marimo`): Web-based reactive notebooks running at `http://localhost:8842`. Includes interactive tutorials demonstrating single-point calculations, geometry optimization, phonons, and WorkGraphs.
  - **JupyterLab** (`Dockerfile.jupyterlab` / `Dockerfile.jupyter`): Classic JupyterLab environment running at `http://localhost:8888/lab`. Fully integrated with `ipykernel`, the `jupyter-ai` extension for AI-assisted workflows, and AiiDA's native greenback event loop portal. Includes sample Jupyter notebooks.
- **Universal Engine Support**:
  - Works seamlessly with both **Docker** and **Podman** (including rootless Podman).
  - Supports NVIDIA GPU passthrough for hardware acceleration.

---

## Quick Start

### 1. Build the Images

Using the build helper script (defaults to `podman`):

```bash
# Build Marimo image (default)
./containers/build.sh --marimo

# Build JupyterLab image
./containers/build.sh --jupyterlab

# Build both images
./containers/build.sh --all
```

Or directly with Podman / Docker:

```bash
# Build Marimo
podman build -f containers/Dockerfile.marimo -t aiida-mlip:latest .

# Build JupyterLab
podman build -f containers/Dockerfile.jupyterlab -t aiida-mlip-jupyterlab:latest .
```

### 2. Launch the Container

Using the startup helper script:

```bash
# Launch Marimo on port 8842
./containers/startup.sh --marimo

# Launch JupyterLab on port 8888
./containers/startup.sh --jupyterlab
```

Or directly with Podman / Docker:

```bash
# Marimo
podman run -it --rm -p 8842:8842 -p 5000:5000 aiida-mlip:latest

# JupyterLab
podman run -it --rm -p 8888:8888 -p 5000:5000 aiida-mlip-jupyterlab:latest
```

### 3. Using Docker Compose / Podman Compose

To run both services (or select one):

```bash
cd containers

# Launch both Marimo (port 8842) and JupyterLab (port 8888)
podman compose up -d

# Or launch only JupyterLab
podman compose up -d jupyterlab

# Or launch only Marimo
podman compose up -d marimo
```

### 4. Access the Interfaces

- **Marimo**: Navigate to `http://localhost:8842`
  - Select `tutorial_marimo.py` to start the interactive reactive tutorial.
- **JupyterLab**: Navigate to `http://localhost:8888/lab`
  - Open `/app/notebooks` for tutorial notebooks with interactive AiiDA execution.
- **AiiDA REST API**:
  - Marimo container: `http://localhost:5000`
  - JupyterLab container (via compose): `http://localhost:5001` (or `5000` if standalone)

---

## Startup Script Options

The `./containers/startup.sh` script provides several useful options:

| Option | Description | Default |
|---|---|---|
| `--marimo` | Launch the Marimo container | default |
| `--jupyterlab, --jupyter` | Launch the JupyterLab container | |
| `--image <name>` | Container image to run | `aiida-mlip:latest` (Marimo) or `aiida-mlip-jupyterlab:latest` (JupyterLab) |
| `--engine <engine>` | Container engine (`podman`, `docker`) | `podman` |
| `--port <port>` | Host port for the web interface | `8842` (Marimo), `8888` (JupyterLab) |
| `--restapi-port <port>` | Host port for the AiiDA REST API (`0` to disable) | `5000` |
| `--bind <path>` | Host directory to mount for persistence (`/app/tutorials` for Marimo, `/app/notebooks` for JupyterLab) | (none) |
| `--gpu` | Enable NVIDIA GPU acceleration | disabled |
| `--name <name>` | Container name | `aiida-mlip-marimo` or `aiida-mlip-jupyterlab` |
| `-d, --detach` | Run container in background | interactive |
| `-h, --help` | Show command usage and options | |

### Examples

**Persist your notebooks/tutorials to a local folder:**
```bash
# For JupyterLab
./containers/startup.sh --jupyterlab --bind ~/my_notebooks

# For Marimo
./containers/startup.sh --marimo --bind ~/my_mlip_projects
```

**Run with NVIDIA GPU passthrough:**
```bash
./containers/startup.sh --jupyterlab --gpu
```

**Run in the background on custom port:**
```bash
./containers/startup.sh --jupyterlab --port 9000 -d
```

---

## Interacting with AiiDA via CLI

You can execute `verdi` commands inside the running container at any time:

```bash
# Check status of AiiDA services and daemon
podman exec -it aiida-mlip-jupyterlab su - aiida -c "verdi status"
# or
podman exec -it aiida-mlip-marimo su - aiida -c "verdi status"

# List running or completed calculations
podman exec -it aiida-mlip-jupyterlab su - aiida -c "verdi process list -a"

# Open an interactive IPython AiiDA shell
podman exec -it aiida-mlip-jupyterlab su - aiida -c "verdi shell"

# Inspect calculation node
podman exec -it aiida-mlip-jupyterlab su - aiida -c "verdi node show <PK>"
```

---

## Architecture & Design Inspiration

- **Default Profile & Service Setup**: Inspired by [aiidateam/aiida-core](https://github.com/aiidateam/aiida-core) and [stfc/aiidalab-mlip](https://github.com/stfc/aiidalab-mlip).
- **Marimo Server & Lean Build**: Inspired by [stfc/janus-core/containers](https://github.com/stfc/janus-core/tree/main/containers).
