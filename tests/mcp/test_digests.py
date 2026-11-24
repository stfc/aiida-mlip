"""Tests for ``aiida_mlip.mcp.digests``."""

from __future__ import annotations

import json

from aiida.common.links import LinkType
from aiida.orm import CalcJobNode, Dict, SinglefileData, StructureData
from ase import Atoms
from ase.build import bulk
import pytest

from aiida_mlip.mcp.digests import (
    MAX_DIGEST_BYTES,
    build_digest,
    diagnose_node_impl,
    read_node_logs,
)
from aiida_mlip.mcp.provenance import tag_process_intent
from aiida_mlip.mcp.refs import MCPToolError


def _make_calc_node(process_type: str) -> CalcJobNode:
    node = CalcJobNode(process_type=f"aiida.calculations:{process_type}")
    node.set_process_state("finished")
    node.set_exit_status(0)
    return node


def test_digest_sp_opt_molecule_and_descriptors(tmp_path):
    """Test architecture-prefixed Atoms.todict() extraction and zero-volume guard."""
    sp = _make_calc_node("mlip.sp")
    sp.store()
    sp_res = Dict(
        {
            "numbers": [11, 17],
            "mace_mp_forces": [[0.1, 0.0, 0.0], [-0.2, 0.0, 0.0]],
            "info": {"mace_mp_energy": -7.5, "mace_mp_stress": [0.0] * 6},
        }
    ).store()
    sp_res.base.links.add_incoming(
        sp, link_type=LinkType.CREATE, link_label="results_dict"
    )
    digest_sp = build_digest(sp)
    assert digest_sp["energy"] == -7.5
    assert digest_sp["n_atoms"] == 2
    assert digest_sp["max_force"] == 0.2
    assert len(json.dumps(digest_sp).encode("utf-8")) <= MAX_DIGEST_BYTES

    # Non-periodic molecule in mlip.opt -> initial_volume == 0.0 -> None pct
    mol = StructureData(ase=Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])).store()
    opt = _make_calc_node("mlip.opt")
    opt.base.links.add_incoming(mol, link_type=LinkType.INPUT_CALC, link_label="struct")
    opt.store()
    final_mol = StructureData(
        ase=Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]])
    ).store()
    final_mol.base.links.add_incoming(
        opt, link_type=LinkType.CREATE, link_label="final_structure"
    )
    digest_opt = build_digest(opt)
    assert digest_opt["volume_change_pct"] is None
    assert digest_opt["final_structure_pk"] == final_mol.pk

    # Descriptors in both info and top-level
    desc = _make_calc_node("mlip.descriptors")
    desc.store()
    desc_res = Dict(
        {
            "numbers": [11, 17],
            "positions": [[0, 0, 0], [1, 1, 1]],
            "mace_mp_descriptors": [[1.0, 2.0], [3.0, 4.0]],
            "info": {"mace_mp_descriptor": [0.5, 1.5]},
        }
    ).store()
    desc_res.base.links.add_incoming(
        desc, link_type=LinkType.CREATE, link_label="results_dict"
    )
    digest_desc = build_digest(desc)
    assert "info.mace_mp_descriptor" in digest_desc["descriptor_keys"]
    assert "mace_mp_descriptors" in digest_desc["descriptor_keys"]
    assert "positions" not in digest_desc["descriptor_keys"]
    assert digest_desc["summary_stats"]["info.mace_mp_descriptor"]["mean"] == 1.0


def test_digest_md_nan_detection_and_size_cap(tmp_path):
    """Test NaN detection via stats_file and <= 2048 byte cap."""
    md = _make_calc_node("mlip.md")
    md.store()
    stats_path = tmp_path / "aiida-stats.dat"
    stats_path.write_text("step temp energy\n1 300.0 NaN\n", encoding="utf-8")
    stats_node = SinglefileData(file=stats_path).store()
    stats_node.base.links.add_incoming(
        md, link_type=LinkType.CREATE, link_label="stats_file"
    )
    big_res = Dict({f"metric_{i}": "x" * 200 for i in range(30)}).store()
    big_res.base.links.add_incoming(
        md, link_type=LinkType.CREATE, link_label="results_dict"
    )
    digest = build_digest(md)
    assert len(json.dumps(digest).encode("utf-8")) <= MAX_DIGEST_BYTES
    assert digest.get("truncated") is True


def test_diagnose_node_categories_and_retry_cap(tmp_path):
    """Test CUDA OOM, 305 exit code, retry cap, and non-ProcessNode guard."""
    intent = "deadbeef"
    for _ in range(3):
        prev = CalcJobNode()
        prev.set_process_state("finished")
        prev.set_exit_status(305)
        prev.store()
        tag_process_intent(prev, intent)

    log_path = tmp_path / "aiida.log"
    log_path.write_text("RuntimeError: CUDA out of memory.\n", encoding="utf-8")
    log_node = SinglefileData(file=log_path).store()
    log_node.base.links.add_incoming(
        prev, link_type=LinkType.CREATE, link_label="log_output"
    )

    assert "CUDA out of memory" in read_node_logs(prev)
    diag = diagnose_node_impl(prev)
    assert diag["category"] == "cuda_oom"
    assert diag["retry_depth"] == 3
    assert diag["retry_cap_exceeded"] is True

    non_proc = StructureData(ase=bulk("NaCl", "rocksalt", 5.63)).store()
    with pytest.raises(MCPToolError) as exc:
        diagnose_node_impl(non_proc)
    assert exc.value.error == "validation_error"
