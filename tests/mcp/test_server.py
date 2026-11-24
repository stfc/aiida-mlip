"""Tests for ``aiida_mlip.mcp.server`` and optional-extra isolation."""

from __future__ import annotations

import builtins

import pytest

import aiida_mlip
from aiida_mlip.mcp import server as mcp_server


def test_import_without_mcp_and_cli_exit_code(monkeypatch, capsys):
    """Verify aiida_mlip imports cleanly and CLI exits 1 if mcp is absent."""
    assert hasattr(aiida_mlip, "__version__")

    real_import = builtins.__import__

    def _block_mcp(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("No module named 'mcp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_mcp)
    with pytest.raises(SystemExit) as exc:
        mcp_server.main([])
    assert exc.value.code == 1
    assert "aiida-mlip[mcp]" in capsys.readouterr().err


def test_create_server_registers_tools_resources_and_prompts():
    """Verify FastMCP registers all 13 tools, 4 resources, and 6 prompts."""
    pytest.importorskip("mcp")
    mcp = mcp_server.create_server()

    tool_names = set(mcp._tool_manager._tools.keys())
    expected_tools = {
        "mlip_sp",
        "mlip_opt",
        "mlip_md",
        "mlip_ph",
        "mlip_descriptors",
        "mlip_train",
        "mlip_ht_screen",
        "mlip_chain",
        "mlip_status",
        "mlip_cancel",
        "diagnose_node",
        "list_profiles",
        "list_codes",
    }
    assert expected_tools.issubset(tool_names)

    prompt_names = set(mcp._prompt_manager._prompts.keys())
    expected_prompts = {
        "sp_workflow",
        "opt_workflow",
        "md_workflow",
        "phonons_workflow",
        "descriptors_workflow",
        "train_workflow",
    }
    assert expected_prompts.issubset(prompt_names)


def test_server_tool_dry_run_end_to_end(
    mcp_code, structure_folder, workflow_structure_folder, config_folder
):
    """Invoke registered tools and resources end-to-end."""
    pytest.importorskip("mcp")
    import json

    from aiida.orm import CalcJobNode

    mcp = mcp_server.create_server()
    tools = mcp._tool_manager._tools

    # mlip_sp dry_run
    sp_res = tools["mlip_sp"].fn(
        struct=f"file:{structure_folder / 'NaCl.cif'}",
        model=f"file:{config_folder / 'test.model'}",
        arch="mace_mp",
        code="janus@localhost",
        dry_run=True,
    )
    assert sp_res["status"] == "dry_run"
    assert sp_res["entry_point"] == "mlip.sp"

    # mlip_train dry_run
    train_res = tools["mlip_train"].fn(
        name="demo",
        train_file=str(structure_folder / "mlip_train.xyz"),
        valid_file=str(structure_folder / "mlip_valid.xyz"),
        test_file=str(structure_folder / "mlip_test.xyz"),
        arch="mace_mp",
        code="janus@localhost",
        dry_run=True,
    )
    assert train_res["status"] == "dry_run"
    assert train_res["entry_point"] == "mlip.train"

    # mlip_ht_screen dry_run & max_number_jobs validation
    ht_res = tools["mlip_ht_screen"].fn(
        calc="opt",
        folder=str(workflow_structure_folder),
        model=f"file:{config_folder / 'test.model'}",
        arch="mace_mp",
        code="janus@localhost",
        dry_run=True,
    )
    assert ht_res["status"] == "dry_run"
    assert ht_res["validated_inputs"]["n_structures"] == 2

    bad_ht = tools["mlip_ht_screen"].fn(
        calc="opt",
        folder=str(workflow_structure_folder),
        model=f"file:{config_folder / 'test.model'}",
        arch="mace_mp",
        max_number_jobs=0,
        code="janus@localhost",
        dry_run=True,
    )
    assert bad_ht["status"] == "error"
    assert bad_ht["error"] == "validation_error"

    # mlip_chain dry_run & _safe_call error envelope on invalid stage
    chain_res = tools["mlip_chain"].fn(
        struct=f"file:{structure_folder / 'NaCl.cif'}",
        model=f"file:{config_folder / 'test.model'}",
        arch="mace_mp",
        stages=[{"calc": "opt"}, {"calc": "ph"}],
        code="janus@localhost",
        dry_run=True,
    )
    assert chain_res["status"] == "dry_run"

    bad_chain = tools["mlip_chain"].fn(
        struct=f"file:{structure_folder / 'NaCl.cif'}",
        model=f"file:{config_folder / 'test.model'}",
        arch="mace_mp",
        stages=[{"calc": None}],
        code="janus@localhost",
        dry_run=True,
    )
    assert bad_chain["status"] == "error"
    assert bad_chain["error"] == "validation_error"

    # diagnose_node + MCP resources
    failed_node = CalcJobNode(process_type="aiida.calculations:mlip.sp")
    failed_node.set_process_state("finished")
    failed_node.set_exit_status(305)
    failed_node.store()
    diag = tools["diagnose_node"].fn(pk=failed_node.pk)
    assert diag["category"] == "missing_output_files"

    codes_res = mcp._resource_manager._resources["aiida://codes"].fn()
    assert json.loads(codes_res)["status"] == "ok"

    templates = mcp._resource_manager._templates
    digest_json = templates["aiida://node/{pk}/digest"].fn(pk=failed_node.pk)
    assert json.loads(digest_json)["process_type"] == "mlip.sp"

    logs_txt = templates["aiida://node/{pk}/logs"].fn(pk=failed_node.pk)
    assert isinstance(logs_txt, str)

    struct_json = templates["aiida://node/{pk}/structure"].fn(pk=failed_node.pk)
    assert json.loads(struct_json)["error"] == "no_structure_found"
