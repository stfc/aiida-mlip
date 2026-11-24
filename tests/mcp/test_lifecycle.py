"""Tests for ``aiida_mlip.mcp.lifecycle``."""

from __future__ import annotations

from aiida.orm import CalcJobNode, Node, QueryBuilder
from aiida.plugins import CalculationFactory
import pytest

from aiida_mlip.data.model import ModelData
from aiida_mlip.mcp import lifecycle
from aiida_mlip.mcp.builders import build_sp_inputs
from aiida_mlip.mcp.provenance import compute_intent_hash, tag_process_intent
from aiida_mlip.mcp.refs import MCPToolError, resolve_structure


def test_dry_run_stores_zero_nodes_and_validates(
    mcp_code, structure_folder, config_folder, tmp_path
):
    """Verify dry_run=True mutates zero DB nodes and catches InputValidationError."""
    struct = resolve_structure(f"file:{structure_folder / 'NaCl.cif'}")
    model = ModelData.from_local(config_folder / "test.model", architecture="mace_mp")
    inputs = build_sp_inputs(
        code=mcp_code, model=model, struct=struct, config_cache_dir=tmp_path
    )
    before = QueryBuilder().append(Node).count()
    res = lifecycle.submit_with_lease(
        CalculationFactory("mlip.sp"),
        inputs,
        entry_point="mlip.sp",
        intent_hash="abc123",
        dry_run=True,
        validated_summary={"ok": True},
    )
    after = QueryBuilder().append(Node).count()
    assert res["status"] == "dry_run"
    assert after == before

    # Missing model without skip_model_port_check must raise validation_error
    bad_inputs = dict(inputs)
    bad_inputs.pop("model")
    with pytest.raises(MCPToolError) as exc:
        lifecycle.submit_with_lease(
            CalculationFactory("mlip.sp"),
            bad_inputs,
            entry_point="mlip.sp",
            intent_hash="abc123",
            dry_run=True,
        )
    assert exc.value.error == "validation_error"


def test_dedupe_and_daemon_guard_defer_factory(monkeypatch):
    """Factory must not be invoked on dedupe hit or daemon-stopped error."""
    called = False

    def _factory():
        nonlocal called
        called = True
        raise AssertionError("Factory should not have been called")

    # Daemon stopped
    monkeypatch.setattr(
        lifecycle, "check_daemon_running", lambda: {"running": False, "workers": 0}
    )
    with pytest.raises(MCPToolError) as exc:
        lifecycle.submit_with_lease(
            _factory,
            None,
            entry_point="workgraph.ht.mlip.sp",
            intent_hash="hash_daemon_off",
        )
    assert exc.value.error == "daemon_not_running"
    assert called is False

    # Dedupe hit
    existing = CalcJobNode(process_type="aiida.calculations:mlip.sp")
    existing.set_process_state("finished")
    existing.set_exit_status(0)
    existing.store()
    intent = compute_intent_hash("mlip.sp", {"x": 1})
    tag_process_intent(existing, intent)

    res = lifecycle.submit_with_lease(
        _factory,
        None,
        entry_point="mlip.sp",
        intent_hash=intent,
        dedupe="reuse",
    )
    assert res["cached"] is True
    assert res["pk"] == existing.pk
    assert called is False


def test_sync_and_async_lease_and_cancel(monkeypatch):
    """Test sync completion within lease, async detachment, status, and cancel."""
    monkeypatch.setattr(
        lifecycle, "check_daemon_running", lambda: {"running": True, "workers": 1}
    )

    finished_node = CalcJobNode(process_type="aiida.calculations:mlip.sp")
    finished_node.set_process_state("finished")
    finished_node.set_exit_status(0)
    finished_node.store()
    monkeypatch.setattr(lifecycle, "submit", lambda cls, **kw: finished_node)

    res_sync = lifecycle.submit_with_lease(
        CalculationFactory("mlip.sp"),
        {},
        entry_point="mlip.sp",
        intent_hash="sync_hash",
        wait_seconds=1.0,
    )
    assert res_sync["status"] == "finished"
    assert res_sync["cached"] is False
    assert res_sync["digest"]["process_type"] == "mlip.sp"

    running_node = CalcJobNode(process_type="aiida.calculations:mlip.md")
    running_node.set_process_state("running")
    running_node.store()
    monkeypatch.setattr(lifecycle, "submit", lambda cls, **kw: running_node)

    res_async = lifecycle.submit_with_lease(
        CalculationFactory("mlip.md"),
        {},
        entry_point="mlip.md",
        intent_hash="async_hash",
        wait_seconds=0.0,
    )
    assert res_async["status"] == "running"
    assert "mlip_status" in res_async["message"]

    status = lifecycle.get_process_status(finished_node.pk)
    assert status["status"] == "finished"

    # Cancel already terminated
    cancel_done = lifecycle.cancel_process(finished_node.pk)
    assert cancel_done["status"] == "already_terminated"

    # Cancel running
    killed: list[int] = []
    monkeypatch.setattr(
        lifecycle, "kill_processes", lambda nodes: killed.extend(n.pk for n in nodes)
    )
    cancel_run = lifecycle.cancel_process(running_node.pk)
    assert cancel_run["status"] == "kill_requested"
    assert killed == [running_node.pk]


def test_list_profiles_and_list_codes(mcp_code):
    """Test list_profiles and list_codes filtering."""
    profiles = lifecycle.list_profiles_impl()
    assert profiles["status"] == "ok"
    assert isinstance(profiles["profiles"], list)

    all_codes = lifecycle.list_codes_impl()
    assert any(c["full_label"] == "janus@localhost" for c in all_codes["codes"])

    filtered = lifecycle.list_codes_impl(plugin="mlip")
    assert any(c["full_label"] == "janus@localhost" for c in filtered["codes"])

    empty = lifecycle.list_codes_impl(plugin="nonexistent.plugin")
    assert empty["codes"] == []
