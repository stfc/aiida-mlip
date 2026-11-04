"""Tests for the Marimo tutorial notebook."""

from __future__ import annotations

import asyncio
import types

import pytest


def test_marimo_tutorial_app():
    """Test importing and running tutorial_marimo.py."""
    pytest.importorskip("marimo")
    from containers.assets.tutorials import tutorial_marimo

    outputs, defs = tutorial_marimo.app.run()
    assert len(outputs) > 0
    assert "mo" in defs
    assert "janus_code" in defs
    assert "structure_node" in defs
    assert "model" in defs


def test_marimo_tutorial_calculation():
    """Test executing the singlepoint calculation cell in tutorial_marimo.py."""
    pytest.importorskip("marimo")
    from containers.assets.tutorials import tutorial_marimo

    outputs, defs = tutorial_marimo.app.run()

    # Locate the calculation cell
    calc_cell = None
    for cell in tutorial_marimo.app._cell_manager.cells():
        if "CalculationFactory" in getattr(cell, "defs", set()):
            calc_cell = cell
            break
    assert calc_cell is not None

    mock_button = types.SimpleNamespace(value=True)

    async def _run():
        return await calc_cell.run(
            janus_code=defs["janus_code"],
            mo=defs["mo"],
            model=defs["model"],
            run_button=mock_button,
            structure_node=defs["structure_node"],
        )

    output, cell_defs = asyncio.run(_run())
    res_text = cell_defs["res_output"].text
    assert "Calculation Summary" in res_text
    assert "Finished OK" in res_text or "finished" in res_text


if __name__ == "__main__":
    print("Running test_marimo_tutorial_app()...")
    test_marimo_tutorial_app()
    print("Running test_marimo_tutorial_calculation()...")
    test_marimo_tutorial_calculation()
    print("test_marimo.py: ALL TESTS PASSED SUCCESSFULLY!")
