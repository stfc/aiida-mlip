"""Tests for ``aiida_mlip.mcp.refs``."""

from __future__ import annotations

from pathlib import Path

from aiida.orm import Dict, InstalledCode, StructureData
import pytest

from aiida_mlip.data.model import ModelData
from aiida_mlip.mcp.refs import (
    DryRunModelRef,
    MCPToolError,
    build_config_node,
    build_metadata_options,
    resolve_code,
    resolve_model,
    resolve_structure,
)


def test_resolve_structure_variants(structure_folder):
    """Test resolving structures via file:, bare path, pk:, and uuid:."""
    nacl_path = structure_folder / "NaCl.cif"
    from_file = resolve_structure(f"file:{nacl_path}")
    assert isinstance(from_file, StructureData)
    assert from_file.get_formula() == "Cl4Na4"

    from_bare = resolve_structure(str(nacl_path))
    assert from_bare.get_formula() == "Cl4Na4"

    stored = from_file.store()
    assert resolve_structure(f"pk:{stored.pk}").pk == stored.pk
    assert resolve_structure(str(stored.pk)).pk == stored.pk
    assert resolve_structure(f"uuid:{stored.uuid}").uuid == stored.uuid


def test_resolve_structure_errors(tmp_path):
    """Test error classification for missing, wrong-type, and unreadable refs."""
    with pytest.raises(MCPToolError) as exc_missing:
        resolve_structure("pk:999999")
    assert exc_missing.value.error == "node_not_found"

    not_a_struct = Dict({"a": 1}).store()
    with pytest.raises(MCPToolError) as exc_type:
        resolve_structure(f"pk:{not_a_struct.pk}")
    assert exc_type.value.error == "validation_error"

    bad_file = tmp_path / "broken.unknownext"
    bad_file.write_text("not a structure", encoding="utf-8")
    with pytest.raises(MCPToolError) as exc_read:
        resolve_structure(f"file:{bad_file}")
    assert exc_read.value.error == "validation_error"


def test_resolve_model_variants_and_dry_run_uri(config_folder, monkeypatch):
    """Test pk:, uuid:, file:, arch mismatch, and uri: dry_run (zero network)."""
    model_file = config_folder / "test.model"
    local_model = resolve_model(f"file:{model_file}", arch="mace_mp")
    assert isinstance(local_model, ModelData)
    assert local_model.architecture == "mace_mp"

    stored = local_model.store()
    assert resolve_model(f"pk:{stored.pk}").pk == stored.pk
    assert resolve_model(f"uuid:{stored.uuid}", arch="mace_mp").uuid == stored.uuid

    with pytest.raises(MCPToolError) as exc_arch:
        resolve_model(f"pk:{stored.pk}", arch="sevennet")
    assert exc_arch.value.error == "validation_error"

    def _forbid_download(*args, **kwargs):
        raise AssertionError("from_uri should not be called during dry_run=True")

    monkeypatch.setattr(ModelData, "from_uri", classmethod(_forbid_download))
    dry_ref = resolve_model(
        "uri:https://example.org/mace.model", arch="mace_mp", dry_run=True
    )
    assert isinstance(dry_ref, DryRunModelRef)
    assert dry_ref.architecture == "mace_mp"
    assert dry_ref.uri == "https://example.org/mace.model"


def test_resolve_code_not_found_and_ambiguous(mcp_code, aiida_computer_local):
    """Test resolve_code success, NotExistent, and MultipleObjectsError."""
    assert resolve_code("janus@localhost").pk == mcp_code.pk

    with pytest.raises(MCPToolError) as exc_missing:
        resolve_code("missing_code@localhost")
    assert exc_missing.value.error == "code_not_found"
    assert "janus@localhost" in (exc_missing.value.hint or "")

    other_comp = aiida_computer_local(label="other_host")
    InstalledCode(
        label="janus",
        computer=other_comp,
        filepath_executable="/usr/bin/true",
        default_calc_job_plugin="mlip.sp",
    ).store()
    with pytest.raises(MCPToolError) as exc_multi:
        resolve_code("janus")
    assert exc_multi.value.error == "code_not_found"


def test_build_config_node_persistent_and_tracker(tmp_path):
    """Test persistent JanusConfigfile creation and force_no_tracker toggle."""
    cfg = build_config_node({"fmax": 0.05}, force_no_tracker=True, cache_dir=tmp_path)
    assert Path(cfg.filepath).is_file()
    parsed = cfg.as_dictionary
    assert parsed["tracker"] is False
    assert parsed["fmax"] == 0.05

    train_cfg = build_config_node(
        {"name": "mace_test"}, force_no_tracker=False, cache_dir=tmp_path
    )
    assert "tracker" not in train_cfg.as_dictionary
    assert train_cfg.as_dictionary["name"] == "mace_test"


def test_build_metadata_options_defaults_and_presets(tmp_path):
    """Test default resources, named presets, deep merge, and invalid YAML."""
    default_opts = build_metadata_options()
    assert default_opts["resources"] == {"num_machines": 1}

    presets_file = tmp_path / "presets.yml"
    presets_file.write_text(
        "gpu_small:\n"
        "  resources:\n"
        "    num_machines: 1\n"
        "    num_mpiprocs_per_machine: 4\n"
        "  max_wallclock_seconds: 3600\n",
        encoding="utf-8",
    )
    merged = build_metadata_options(
        hpc_preset="gpu_small",
        hpc_overrides={"resources": {"num_mpiprocs_per_machine": 8}},
        presets_path=presets_file,
    )
    assert merged["resources"] == {"num_machines": 1, "num_mpiprocs_per_machine": 8}
    assert merged["max_wallclock_seconds"] == 3600

    bad_yaml = tmp_path / "bad.yml"
    bad_yaml.write_text("gpu: [unclosed", encoding="utf-8")
    with pytest.raises(MCPToolError) as exc_bad:
        build_metadata_options(hpc_preset="gpu", presets_path=bad_yaml)
    assert exc_bad.value.error == "validation_error"
