"""pytest fixtures for simplified testing."""

import pytest

pytest_plugins = ["aiida.manage.tests.pytest_fixtures"]


@pytest.fixture(scope="function", autouse=True)
def clear_database_auto(clear_database):  # pylint: disable=unused-argument
    """Automatically clear database in between tests."""


@pytest.fixture(scope="function")
def mlip_code(aiida_local_code_factory):
    """Get a mlip code."""
    return aiida_local_code_factory(executable="diff", entry_point="mlip")
