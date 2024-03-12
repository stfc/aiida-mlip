# pylint: disable=redefined-outer-name,too-many-statements
"""Initialise a text database and profile for pytest."""

from pathlib import Path

import pytest

from collections.abc import Mapping
from aiida.common import exceptions
from aiida.common.folders import SandboxFolder
from aiida.engine.utils import instantiate_process
from aiida.manage.manager import get_manager
from aiida.orm import InstalledCode, load_code
from aiida.plugins import CalculationFactory

pytest_plugins = ["aiida.manage.tests.pytest_fixtures"]  # pylint: disable=invalid-name


@pytest.fixture(scope="function", autouse=True)
def clear_database_auto(aiida_profile_clean):  # pylint: disable=unused-argument
    """Automatically clear database in between tests."""


@pytest.fixture(scope="session")
def filepath_tests():
    """Return the absolute filepath of the `tests` folder.

    .. warning:: if this file moves with respect to the `tests` folder, the
    implementation should change.

    :return: absolute filepath of `tests` folder which is the basepath for all test
    resources.
    """
    return Path(__file__).resolve()


@pytest.fixture
def filepath_fixtures(filepath_tests):
    """Return the absolute filepath to the directory containing the file `fixtures`."""
    return Path(filepath_tests / "fixtures")


@pytest.fixture(scope="function")
def fixture_sandbox():
    """Return a `SandboxFolder`."""

    with SandboxFolder() as folder:
        yield folder


@pytest.fixture
def fixture_localhost(aiida_localhost):
    """Return a localhost `Computer`."""
    localhost = aiida_localhost
    localhost.set_default_mpiprocs_per_machine(1)
    return localhost


@pytest.fixture(scope="function")
def janus_code(aiida_local_code_factory):
    """Get the janus code."""
    return aiida_local_code_factory(executable="janus", entry_point="janus.sp")


@pytest.fixture
def fixture_code(fixture_localhost):
    """
    Return an ``InstalledCode`` instance configured to run calculations of given
    entry point on localhost.
    """

    def _fixture_code(entry_point_name):

        label = f"test.{entry_point_name}"

        try:
            return load_code(label=label)
        except exceptions.NotExistent:
            return InstalledCode(
                label=label,
                computer=fixture_localhost,
                filepath_executable="/bin/true",
                default_calc_job_plugin=entry_point_name,
            )

    return _fixture_code


@pytest.fixture
def generate_calc_job():
    """
    Fixture to construct a new `CalcJob` instance
    and call `prepare_for_submission` for testing `CalcJob` classes.

    The fixture will return the `CalcInfo` returned by `prepare_for_submission`
    and the temporary folder that was passed to it,
    into which the raw input files will have been written.
    """

    def _generate_calc_job(folder, entry_point_name, inputs=None):
        """Fixture to generate a mock `CalcInfo` for testing calculation jobs."""

        manager = get_manager()
        runner = manager.get_runner()

        process_class = CalculationFactory(entry_point_name)
        process = instantiate_process(runner, process_class, **inputs)

        calc_info = process.prepare_for_submission(folder)

        return calc_info

    return _generate_calc_job

