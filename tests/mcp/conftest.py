"""Fixtures for ``aiida_mlip.mcp`` unit and integration tests."""

from __future__ import annotations

from aiida.common import exceptions
from aiida.orm import InstalledCode, load_code
import pytest


@pytest.fixture
def mcp_code(fixture_localhost):
    """
    Return a stored dummy ``InstalledCode`` requiring no live ``janus`` binary.

    Parameters
    ----------
    fixture_localhost : fixture
        AiiDA localhost ``Computer`` fixture.

    Returns
    -------
    InstalledCode
        Stored ``InstalledCode`` named ``janus`` on ``localhost``.
    """
    try:
        return load_code("janus@localhost")
    except exceptions.NotExistent:
        return InstalledCode(
            label="janus",
            computer=fixture_localhost,
            filepath_executable="/usr/bin/true",
            default_calc_job_plugin="mlip.sp",
        ).store()
