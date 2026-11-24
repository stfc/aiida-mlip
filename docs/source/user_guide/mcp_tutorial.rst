===================
MCP Server Tutorial
===================

This tutorial walks through configuring and using the ``aiida-mlip`` Model
Context Protocol (MCP) server (``aiida-mlip[mcp]``) to run provenance-tracked
MLIP calculations, ``aiida-workgraph`` pipelines, and diagnostics from an
MCP-compatible AI assistant (such as Antigravity, Claude Desktop, or Gemini
CLI) or directly via Python.

Prerequisites
=============

1. Install ``aiida-mlip`` with the optional ``mcp`` extra:

   .. code-block:: bash

       uv pip install "aiida-mlip[mcp]"

2. Ensure an AiiDA profile is configured, the daemon is running, and a
   ``janus`` code is registered (for example ``janus@localhost``):

   .. code-block:: bash

       verdi profile list
       verdi daemon status   # start with: verdi daemon start
       verdi code show janus@localhost

3. *(Optional, for HPC)* Create a presets file at
   ``~/.config/aiida-mlip-mcp/presets.yml`` with named ``metadata.options``
   bundles:

   .. code-block:: yaml

       hpc_gpu:
         resources:
           num_machines: 1
           num_mpiprocs_per_machine: 4
         max_wallclock_seconds: 3600
         queue_name: gpu

Connecting an MCP Client
========================

Configure your MCP client to launch ``aiida-mlip-mcp`` over ``stdio``. For
example, in an MCP client JSON configuration:

.. code-block:: json

    {
      "mcpServers": {
        "aiida-mlip": {
          "command": "aiida-mlip-mcp",
          "args": [
            "--profile", "default",
            "--presets", "~/.config/aiida-mlip-mcp/presets.yml"
          ]
        }
      }
    }

If ``aiida-mlip`` is installed inside a ``micromamba`` or ``uv`` environment,
point ``command`` at the environment's ``aiida-mlip-mcp`` binary (or run via
``uv run aiida-mlip-mcp``).

Step 1: Verify Your Profile, Daemon, and Codes
==============================================

Ask your assistant to check AiiDA readiness, or invoke ``list_profiles`` and
``list_codes``:

.. code-block:: json

    // Tool call: list_profiles()
    {
      "status": "ok",
      "active_profile": "default",
      "profiles": ["default"],
      "daemon": {"running": true, "workers": 1}
    }

.. code-block:: json

    // Tool call: list_codes(plugin="mlip")
    {
      "status": "ok",
      "profile": "default",
      "codes": [
        {
          "label": "janus",
          "full_label": "janus@localhost",
          "computer": "localhost",
          "plugin": "mlip.sp",
          "pk": 2
        }
      ]
    }

The same code list is also exposed as the read-only MCP resource
``aiida://codes``.

Step 2: Dry-Run and Execute a Single-Point Calculation
======================================================

All calculation tools accept **tagged reference strings** for structures and
models:

- ``file:/path/to/NaCl.cif`` — load from a local file
- ``pk:42`` or ``uuid:...`` — reuse an existing AiiDA ``StructureData`` or
  ``ModelData`` node
- ``uri:https://.../mace_mp_small.model`` — download and cache a ``ModelData``
  node (requires ``arch``)

Start with ``dry_run=True`` to validate all inputs without storing nodes in
AiiDA or downloading remote models:

.. code-block:: json

    // Tool call: mlip_sp(...)
    {
      "struct": "file:./tests/calculations/structures/NaCl.cif",
      "model": "uri:https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model",
      "arch": "mace_mp",
      "properties": ["energy", "forces", "stress"],
      "code": "janus@localhost",
      "dry_run": true
    }

Once ``status`` returns ``"dry_run"``, submit by setting ``dry_run=False``.
Every ``BaseJanus`` tool automatically enforces ``tracker: false`` in its
synthesized ``JanusConfigfile`` and waits up to ``wait_seconds`` (default
``15.0`` s) before detaching:

.. code-block:: json

    // Response (when finished within the wait lease):
    {
      "pk": 105,
      "uuid": "8c3e...",
      "entry_point": "mlip.sp",
      "status": "finished",
      "cached": false,
      "intent_hash": "9a1f...",
      "digest": {
        "pk": 105,
        "process_type": "mlip.sp",
        "energy": -27.14,
        "n_atoms": 8,
        "max_force": 0.0001,
        "rms_force": 0.0001,
        "stress_voigt": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "outputs": ["log_output", "results_dict", "std_output", "xyz_output"]
      },
      "diagnosis": null,
      "resources": {
        "digest": "aiida://node/105/digest",
        "logs": "aiida://node/105/logs",
        "structure": "aiida://node/105/structure"
      }
    }

Step 3: Geometry Optimization and Provenance Deduplication
==========================================================

Run a full cell relaxation with ``mlip_opt``. When ``opt_cell_fully=True`` or
``opt_cell_lengths=True``, ``FrechetCellFilter`` is added to
``minimize_kwargs`` automatically:

.. code-block:: json

    // Tool call: mlip_opt(...)
    {
      "struct": "file:./NaCl.cif",
      "model": "pk:101",
      "fmax": 0.01,
      "steps": 500,
      "opt_cell_fully": true,
      "filter_func": "FrechetCellFilter",
      "code": "janus@localhost"
    }

The returned ``digest`` includes ``final_structure_pk`` and
``volume_change_pct``. If you (or an agent retrying after a timeout) submit the
exact same inputs again under the default ``dedupe="reuse"`` mode, the server
finds the matching ``intent_hash`` in AiiDA provenance and returns the existing
PK immediately with ``"cached": true`` instead of launching a duplicate job.
Use ``dedupe="strict"`` to block only in-flight retries while allowing fresh
re-runs, or ``dedupe="force"`` to bypass deduplication.

Step 4: Long-Running MD and Asynchronous Monitoring
===================================================

For longer calculations such as ``mlip_md``, ``mlip_ph``, or ``mlip_train`` on
an HPC cluster, pass an ``hpc_preset`` (or ``hpc_overrides``) and a short
``wait_seconds``:

.. code-block:: json

    // Tool call: mlip_md(...)
    {
      "struct": "pk:112",
      "model": "pk:101",
      "ensemble": "npt",
      "temp": 300.0,
      "pressure": 0.0,
      "steps": 50000,
      "timestep": 1.0,
      "code": "janus@archer2",
      "device": "cuda",
      "hpc_preset": "hpc_gpu",
      "wait_seconds": 5.0
    }

When the calculation exceeds ``wait_seconds``, the tool returns immediately
with an asynchronous handle:

.. code-block:: json

    {
      "pk": 210,
      "uuid": "3b9d...",
      "entry_point": "mlip.md",
      "status": "running",
      "cached": false,
      "message": "Running asynchronously (exceeded 5.0s lease). Poll with mlip_status(pk=210).",
      "resources": {
        "digest": "aiida://node/210/digest",
        "logs": "aiida://node/210/logs",
        "structure": "aiida://node/210/structure"
      }
    }

Poll progress with ``mlip_status(pk=210)``, inspect trailing logs via the
``aiida://node/210/logs`` resource, inspect geometry via
``aiida://node/210/structure``, or stop the run with ``mlip_cancel(pk=210)``.

Step 5: Chained WorkGraphs and High-Throughput Screening
========================================================

Chain a relaxation directly into a harmonic phonon calculation using
``mlip_chain`` (stage 0's ``final_structure`` socket is automatically wired
into stage 1's ``struct`` input):

.. code-block:: json

    // Tool call: mlip_chain(...)
    {
      "struct": "file:./NaCl.cif",
      "model": "pk:101",
      "stages": [
        {"calc": "opt", "params": {"fmax": 0.005, "opt_cell_fully": true}},
        {"calc": "ph", "params": {"supercell": "2 2 2", "dos": true, "bands": true}}
      ],
      "code": "janus@localhost"
    }

Or screen an entire directory of structure files with ``mlip_ht_screen``
(wrapping ``build_ht_calc`` with concurrency control via ``max_number_jobs``):

.. code-block:: json

    // Tool call: mlip_ht_screen(...)
    {
      "calc": "opt",
      "folder": "./structures_batch",
      "model": "pk:101",
      "calc_params": {"fmax": 0.01, "opt_cell_fully": true},
      "max_number_jobs": 4,
      "code": "janus@localhost"
    }

.. note::
   Every readable structure filename stem in ``folder`` must be unique and a
   valid Python identifier (e.g. ``NaCl_1.cif`` rather than ``NaCl-1.cif``) so
   AiiDA WorkGraph link labels remain valid.

Step 6: Training and Fine-Tuning Models
=======================================

Use ``mlip_train`` to submit a provenance-tracked ``mlip.train`` calculation.
Unlike ``BaseJanus`` calculations, ``mlip_train`` forwards ``mlip_config``
directly to the underlying MLIP trainer and therefore does **not** inject
``tracker: false``:

.. code-block:: json

    // Tool call: mlip_train(...)
    {
      "name": "nacl_finetuned",
      "train_file": "./data/train.xyz",
      "valid_file": "./data/valid.xyz",
      "test_file": "./data/test.xyz",
      "arch": "mace_mp",
      "fine_tune": true,
      "foundation_model": "pk:101",
      "train_overrides": {"max_num_epochs": 20, "batch_size": 4},
      "code": "janus@archer2",
      "hpc_preset": "hpc_gpu",
      "dry_run": true
    }

Step 7: Diagnosing Failed Calculations
======================================

If any calculation or WorkGraph finishes with a non-zero exit status (for
example ``305 ERROR_MISSING_OUTPUT_FILES``) or enters ``excepted`` state,
``mlip_status`` automatically attaches a diagnosis, or you can invoke
``diagnose_node(pk=...)`` directly:

.. code-block:: json

    // Tool call: diagnose_node(pk=314)
    {
      "pk": 314,
      "process_state": "finished",
      "exit_status": 305,
      "exit_message": "Some output files missing or cannot be read",
      "category": "cuda_oom",
      "log_tail": "... RuntimeError: CUDA out of memory ...",
      "suggested_action": "Reduce supercell/batch size or retry with device='cpu'.",
      "retry_depth": 1,
      "retry_cap_exceeded": false
    }

When ``retry_depth >= 3`` for the same ``intent_hash``,
``retry_cap_exceeded`` becomes ``true`` to prevent agents from entering
infinite retry loops.

Using Built-In Workflow Prompts
===============================

MCP clients that support prompts expose six guided slash-command templates:

- ``sp_workflow``
- ``opt_workflow``
- ``md_workflow``
- ``phonons_workflow``
- ``descriptors_workflow``
- ``train_workflow``

Selecting a prompt pre-fills the recommended workflow steps (checking
``aiida://codes``, validating with ``dry_run=True``, and submitting with safe
defaults).
