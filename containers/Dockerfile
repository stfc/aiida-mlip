# Dockerfile for the aiida-mlip Marimo deployment image.
#
# Build:
#   podman build -f containers/Dockerfile -t aiida-mlip:latest .
#   (or run ./containers/build.sh)
#
# Run:
#   podman run -it --rm -p 8842:8842 aiida-mlip:latest
#   (or use ./containers/startup.sh)

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

LABEL maintainer="STFC <support@stfc.ac.uk>"
LABEL description="Container environment for aiida-mlip with automated AiiDA profile and Marimo interface"

ARG UID=1000
ARG GID=100
ARG CONTAINER_USER=aiida
ARG MARIMO_VERSION=latest

ENV CONTAINER_USER=${CONTAINER_USER}
ENV DEBIAN_FRONTEND=noninteractive
ENV UV_SYSTEM_PYTHON=1
ENV MARIMO_SKIP_UPDATE_CHECK=1
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

# 1. Install system utilities, build tools, PostgreSQL, and RabbitMQ
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
        ca-certificates \
        locales \
        sudo \
        tini \
        wget \
        bzip2 \
        curl \
        git \
        gcc \
        g++ \
        make \
        cmake \
        graphviz \
        postgresql \
        postgresql-contrib \
        rabbitmq-server && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

# 2. Configure system user with sudo access
RUN useradd -m -u ${UID} -g ${GID} -s /bin/bash ${CONTAINER_USER} && \
    echo "${CONTAINER_USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-${CONTAINER_USER} && \
    chmod 0440 /etc/sudoers.d/90-${CONTAINER_USER}

WORKDIR /app

# 3. Install latest Marimo and core visualization packages
RUN if [ "$MARIMO_VERSION" = "latest" ] || [ -z "$MARIMO_VERSION" ]; then \
        uv pip install --no-cache-dir "marimo" "marimo[recommended,lsp,sql]"; \
    else \
        uv pip install --no-cache-dir "marimo==${MARIMO_VERSION}" "marimo[recommended,lsp,sql]"; \
    fi && \
    uv pip install --no-cache-dir \
        altair \
        pandas \
        python-lsp-server \
        websockets \
        duckdb \
        sqlglot \
        vegafusion \
        vl-convert-python \
        ruff \
        pytest

# 4. Install MLIP and AiiDA dependencies
RUN uv pip install --no-cache-dir \
    "aiida-core~=2.7.4" \
    "janus-core[all]" \
    "aiida-workgraph<0.8,>=0.7.6" \
    "ase<4.0,>=3.24" \
    "pymatgen" \
    "matplotlib" \
    "seaborn"

# 5. Install aiida-mlip package
COPY . /opt/aiida-mlip
RUN uv pip install --no-cache-dir /opt/aiida-mlip[mace]

# 6. Copy tutorials, examples, and setup scripts
COPY containers/assets/tutorials /app/tutorials
COPY examples /app/examples
COPY containers/setup-aiida.sh /usr/local/bin/setup-aiida.sh
COPY containers/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/setup-aiida.sh /usr/local/bin/entrypoint.sh && \
    mkdir -p /home/${CONTAINER_USER}/.aiida /home/${CONTAINER_USER}/aiida_run && \
    chown -R ${CONTAINER_USER}:${GID} /app /home/${CONTAINER_USER} /opt/aiida-mlip

ENV PORT=8842
ENV HOST=0.0.0.0
EXPOSE 8842 5000

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["marimo"]
