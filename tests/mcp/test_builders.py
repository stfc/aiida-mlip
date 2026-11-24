"""Tests for ``aiida_mlip.mcp.builders``."""

from __future__ import annotations

import shutil

import pytest

from aiida_mlip.data.model import ModelData
from aiida_mlip.mcp.builders import (
    build_chain_workgraph,
    build_descriptors_inputs,
    build_ht_workgraph,
    build_md_inputs,
    build_opt_inputs,
    build_ph_inputs,
    build_sp_inputs,
    build_train_inputs,
    validate_chain_stages,
    validate_ht_folder,
)
from aiida_mlip.mcp.refs import MCPToolError, resolve_structure


def _load_test_model(config_folder):
    return ModelData.from_local(config_folder / "test.model", architecture="mace_mp")


def _prepare_calc_job(folder, entry_point_name, inputs):
    """Instantiate a CalcJob and run prepare_for_submission without RabbitMQ."""
    from aiida.engine.runners import Runner
    from aiida.engine.utils import instantiate_process
    from aiida.plugins import CalculationFactory

    runner = Runner(communicator=None)
    process_cls = CalculationFactory(entry_point_name)
    process = instantiate_process(runner, process_cls, **inputs)
    return process.prepare_for_submission(folder)


def test_all_six_calc_builders_prepare_for_submission(
    fixture_sandbox,
    mcp_code,
    structure_folder,
    config_folder,
    tmp_path,
):
    """Validate all 6 calc builders through CalcJob.prepare_for_submission."""
    struct = resolve_structure(f"file:{structure_folder / 'NaCl.cif'}")
    model = _load_test_model(config_folder)

    sp_inputs = build_sp_inputs(
        code=mcp_code,
        model=model,
        struct=struct,
        properties=["energy", "forces"],
        config_cache_dir=tmp_path,
    )
    sp_info = _prepare_calc_job(fixture_sandbox, "mlip.sp", sp_inputs)
    assert sp_info.codes_info[0].cmdline_params[0] == "singlepoint"
    assert sp_inputs["config"].as_dictionary["tracker"] is False

    opt_inputs = build_opt_inputs(
        code=mcp_code,
        model=model,
        struct=struct,
        opt_cell_fully=True,
        config_cache_dir=tmp_path,
    )
    opt_info = _prepare_calc_job(fixture_sandbox, "mlip.opt", opt_inputs)
    assert opt_info.codes_info[0].cmdline_params[0] == "geomopt"
    assert (
        opt_inputs["minimize_kwargs"].get_dict()["filter_func"] == "FrechetCellFilter"
    )

    md_inputs = build_md_inputs(
        code=mcp_code,
        model=model,
        struct=struct,
        ensemble="nvt",
        temp=300.0,
        steps=50,
        config_cache_dir=tmp_path,
    )
    md_info = _prepare_calc_job(fixture_sandbox, "mlip.md", md_inputs)
    assert md_info.codes_info[0].cmdline_params[0] == "md"
    assert md_inputs["md_kwargs"].get_dict()["temp"] == 300.0

    md_nve = build_md_inputs(
        code=mcp_code,
        model=model,
        struct=struct,
        ensemble="nve",
        config_cache_dir=tmp_path,
    )
    assert "temp" not in md_nve["md_kwargs"].get_dict()

    ph_inputs = build_ph_inputs(
        code=mcp_code,
        model=model,
        struct=struct,
        supercell="2 2 2",
        dos=True,
        config_cache_dir=tmp_path,
    )
    ph_info = _prepare_calc_job(fixture_sandbox, "mlip.ph", ph_inputs)
    assert ph_info.codes_info[0].cmdline_params[0] == "phonons"

    desc_inputs = build_descriptors_inputs(
        code=mcp_code,
        model=model,
        struct=struct,
        calc_per_atom=True,
        config_cache_dir=tmp_path,
    )
    desc_info = _prepare_calc_job(fixture_sandbox, "mlip.descriptors", desc_inputs)
    assert desc_info.codes_info[0].cmdline_params[0] == "descriptors"

    train_inputs = build_train_inputs(
        code=mcp_code,
        name="demo_mace",
        train_file=str(structure_folder / "mlip_train.xyz"),
        valid_file=str(structure_folder / "mlip_valid.xyz"),
        test_file=str(structure_folder / "mlip_test.xyz"),
        arch="mace_mp",
        fine_tune=True,
        foundation_model=model,
        config_cache_dir=tmp_path,
    )
    assert "tracker" not in train_inputs["mlip_config"].as_dictionary
    train_info = _prepare_calc_job(fixture_sandbox, "mlip.train", train_inputs)
    assert train_info.codes_info[0].cmdline_params[0] == "train"


def test_opt_and_train_validation_errors(mcp_code, structure_folder, config_folder):
    """Test mutual exclusion in mlip_opt and duplicate foundation_model in train."""
    struct = resolve_structure(f"file:{structure_folder / 'NaCl.cif'}")
    model = _load_test_model(config_folder)

    with pytest.raises(MCPToolError):
        build_opt_inputs(
            code=mcp_code,
            model=model,
            struct=struct,
            opt_cell_fully=True,
            opt_cell_lengths=True,
        )

    with pytest.raises(MCPToolError):
        build_train_inputs(
            code=mcp_code,
            name="dup",
            train_file=str(structure_folder / "mlip_train.xyz"),
            valid_file=str(structure_folder / "mlip_valid.xyz"),
            test_file=str(structure_folder / "mlip_test.xyz"),
            arch="mace_mp",
            fine_tune=True,
            foundation_model=model,
            train_overrides={"foundation_model": "small"},
        )


def test_validate_ht_folder_and_workgraph(
    mcp_code, workflow_structure_folder, config_folder, tmp_path
):
    """Test duplicate stem rejection, invalid identifier rejection, and shallow copy."""
    files = validate_ht_folder(workflow_structure_folder)
    assert {f.stem for f in files} == {"H2O", "methane"}

    # Duplicate stems across subdirectories
    sub_a = tmp_path / "dup" / "a"
    sub_b = tmp_path / "dup" / "b"
    sub_a.mkdir(parents=True)
    sub_b.mkdir(parents=True)
    shutil.copy(workflow_structure_folder / "H2O.xyz", sub_a / "mol.xyz")
    shutil.copy(workflow_structure_folder / "H2O.xyz", sub_b / "mol.xyz")
    with pytest.raises(MCPToolError, match="Duplicate structure stem"):
        validate_ht_folder(tmp_path / "dup", recursive=True)

    # Invalid Python identifier stem
    bad_dir = tmp_path / "bad_id"
    bad_dir.mkdir()
    shutil.copy(workflow_structure_folder / "H2O.xyz", bad_dir / "NaCl-1.xyz")
    with pytest.raises(MCPToolError, match="valid AiiDA link label"):
        validate_ht_folder(bad_dir)

    from aiida_workgraph import WorkGraph

    # Shallow copy safety and normalized alias
    model = _load_test_model(config_folder).store()
    shared = build_opt_inputs(
        code=mcp_code, model=model, struct=None, config_cache_dir=tmp_path
    )
    wg = build_ht_workgraph(
        entry_point="mlip.opt",
        folder=workflow_structure_folder,
        calc_inputs=shared,
        max_number_jobs=2,
    )
    assert isinstance(wg, WorkGraph)
    assert wg.max_number_jobs == 2
    assert "struct" not in shared


def test_chain_workgraph_validation_and_build(
    mcp_code, structure_folder, config_folder, tmp_path
):
    """Test valid opt -> ph chain and rejection of invalid chains."""
    from aiida_workgraph import WorkGraph

    struct = resolve_structure(f"file:{structure_folder / 'NaCl.cif'}").store()
    model = _load_test_model(config_folder).store()

    stages = validate_chain_stages(
        [
            {"calc": "opt", "params": {"fmax": 0.05}},
            {"calc": "ph", "params": {"dos": True}},
        ],
        code=mcp_code,
        model=model,
        struct=struct,
        config_cache_dir=tmp_path,
    )
    wg = build_chain_workgraph(stages)
    assert isinstance(wg, WorkGraph)
    assert "stage_0_opt" in wg.tasks
    assert "stage_1_ph" in wg.tasks

    with pytest.raises(MCPToolError, match="non-empty"):
        validate_chain_stages([], code=mcp_code, model=model, struct=struct)

    with pytest.raises(MCPToolError, match="final_structure"):
        validate_chain_stages(
            [{"calc": "sp"}, {"calc": "ph"}],
            code=mcp_code,
            model=model,
            struct=struct,
        )

    with pytest.raises(MCPToolError, match="Unsupported keys"):
        validate_chain_stages(
            [{"calc": "sp", "params": {"code": "other@host"}}],
            code=mcp_code,
            model=model,
            struct=struct,
        )
