==========
MCP Server
==========

``aiida-mlip`` includes an optional Model Context Protocol (MCP) server
(``aiida-mlip[mcp]``) that exposes all six AiiDA calculation entry points,
``aiida-workgraph`` workflows, lifecycle management, and failure diagnostics
over ``stdio``.

Installation
============

Install the optional ``mcp`` extra alongside ``aiida-mlip``:

.. code-block:: bash

    uv pip install "aiida-mlip[mcp]"

Running the Server
==================

Ensure an AiiDA profile is configured (for example via ``verdi presto``) and
that the daemon is running (``verdi daemon start``), then launch the server:

.. code-block:: bash

    aiida-mlip-mcp --profile <profile-name>

Optional flags:

- ``--profile <name>``: Default AiiDA profile name to load.
- ``--presets <path>``: Path to a YAML file defining named HPC
  ``metadata.options`` presets (defaults to ``$AIIDA_MLIP_MCP_PRESETS`` or
  ``~/.config/aiida-mlip-mcp/presets.yml``).

Available Tools
===============

Calculation tools (all enforce ``tracker: false`` on ``JanusConfigfile`` by
default, except ``mlip_train`` whose ``mlip_config`` is passed directly to the
MLIP trainer):

- ``mlip_sp`` — Single-point energy, forces, and stress (``mlip.sp``).
- ``mlip_opt`` — Geometry and cell optimization with ``FrechetCellFilter``
  default (``mlip.opt``).
- ``mlip_md`` — NVE, NVT, and NPT molecular dynamics (``mlip.md``).
- ``mlip_ph`` — Harmonic phonons, DOS, PDOS, and band structures (``mlip.ph``).
- ``mlip_descriptors`` — MLIP invariant descriptors (``mlip.descriptors``).
- ``mlip_train`` — MLIP model training and fine-tuning (``mlip.train``).

Workflow, lifecycle, and diagnostic tools:

- ``mlip_ht_screen`` — High-throughput folder screening via ``build_ht_calc``.
- ``mlip_chain`` — Sequential multi-stage ``WorkGraph`` (e.g. ``opt`` → ``ph``).
- ``mlip_status`` — Query node status and bounded Physics Digest (``<= 2 KB``).
- ``mlip_cancel`` — Request termination of a running process.
- ``diagnose_node`` — Classify failures from exit codes and logs with retry-cap
  tracking.
- ``list_profiles`` / ``list_codes`` — Discover AiiDA profiles, daemon status,
  and registered codes.

Resources & Prompts
===================

Read-only MCP resources:

- ``aiida://codes``
- ``aiida://node/{pk}/digest``
- ``aiida://node/{pk}/logs``
- ``aiida://node/{pk}/structure``

Workflow prompt templates: ``sp_workflow``, ``opt_workflow``, ``md_workflow``,
``phonons_workflow``, ``descriptors_workflow``, and ``train_workflow``.
