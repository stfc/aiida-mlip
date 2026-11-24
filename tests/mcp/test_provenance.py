"""Tests for ``aiida_mlip.mcp.provenance``."""

from __future__ import annotations

from aiida.orm import CalcJobNode, StructureData
from ase.build import bulk

from aiida_mlip.mcp.provenance import (
    compute_intent_hash,
    count_failed_attempts,
    find_matching_process,
    structure_fingerprint,
    tag_process_intent,
)


def test_structure_fingerprint_matches_file_and_pk_and_respects_pbc(structure_folder):
    """Identical structures via file and stored PK produce the same fingerprint."""
    from aiida_mlip.mcp.refs import resolve_structure

    nacl_path = structure_folder / "NaCl.cif"
    unstored = resolve_structure(f"file:{nacl_path}")
    fp_file = structure_fingerprint(unstored)

    stored = unstored.store()
    fp_pk = structure_fingerprint(resolve_structure(f"pk:{stored.pk}"))
    assert fp_file == fp_pk

    non_periodic = bulk("NaCl", "rocksalt", 5.63)
    non_periodic.set_pbc([False, False, False])
    fp_np = structure_fingerprint(StructureData(ase=non_periodic))
    assert fp_np != fp_file


def test_dedupe_modes_and_failed_attempt_count():
    """Test reuse, strict, and force modes against real QueryBuilder nodes."""
    intent = compute_intent_hash("mlip.sp", {"k": "v"})

    finished_ok = CalcJobNode()
    finished_ok.set_process_state("finished")
    finished_ok.set_exit_status(0)
    finished_ok.store()
    tag_process_intent(finished_ok, intent)

    assert find_matching_process(intent, dedupe="reuse").pk == finished_ok.pk
    assert find_matching_process(intent, dedupe="strict") is None
    assert find_matching_process(intent, dedupe="force") is None

    running = CalcJobNode()
    running.set_process_state("running")
    running.store()
    tag_process_intent(running, intent)
    assert find_matching_process(intent, dedupe="strict").pk == running.pk

    for status in (305, 400):
        failed = CalcJobNode()
        failed.set_process_state("finished")
        failed.set_exit_status(status)
        failed.store()
        tag_process_intent(failed, intent)

    excepted = CalcJobNode()
    excepted.set_process_state("excepted")
    excepted.store()
    tag_process_intent(excepted, intent)

    assert count_failed_attempts(intent) == 3
    assert count_failed_attempts(None) == 0
