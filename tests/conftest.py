# pylint: disable=redefined-outer-name,too-many-statements
"""Initialise a text database and profile for pytest."""

import os
from pathlib import Path

import pytest

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
    """
    Return the absolute filepath of the `tests` folder.
    .. warning: If this file moves with respect to the `tests` folder,
        the implementation should change.

    Parameters
    ----------
    None

    Returns
    -------
    Path
        Absolute filepath of `tests` folder.
    """
    return Path(__file__).resolve()


@pytest.fixture(scope="function")
def fixture_sandbox():
    """
    Return a `SandboxFolder` fixture.

    This fixture returns a `SandboxFolder` instance for temporary file operations
    within a test function.

    Yields
    ------
    SandboxFolder
        A `SandboxFolder` instance for temporary file operations.
    """

    with SandboxFolder() as folder:
        yield folder


@pytest.fixture
def fixture_localhost(aiida_localhost):
    """
    Return a localhost `Computer` fixture.

    Parameters
    ----------
    aiida_localhost : fixture
        A fixture providing a localhost `Computer` instance.

    Returns
    -------
    `Computer`
        A localhost `Computer` instance.
    """
    localhost = aiida_localhost
    localhost.set_default_mpiprocs_per_machine(1)
    return localhost


@pytest.fixture(scope="function")
def janus_code(aiida_local_code_factory):
    """
    Fixture to get the janus code.

    Parameters
    ----------
    aiida_local_code_factory : fixture
        A fixture providing a factory for creating local codes.

    Returns
    -------
    `Code`
        The janus code instance.
    """
    janus_path = os.environ.get("JANUS_PATH")
    return aiida_local_code_factory(executable=janus_path, entry_point="janus.sp")


@pytest.fixture
def fixture_code(fixture_localhost):
    """
    Return an `InstalledCode` instance configured to run calculations of a given
    entry point on localhost.

    Parameters
    ----------
    fixture_localhost : fixture
        A fixture providing a localhost `Computer` instance.

    Notes
    -----
    This fixture returns a function that can be called with the entry point name.
    If the code with the specified label already exists, it loads and returns it.
    Otherwise, it creates a new `InstalledCode` instance with the provided
    parameters.
    """

    def _fixture_code(entry_point_name):
        """
        Create an `InstalledCode` to run calculations of a given entry point.

        Parameters
        ----------
        entry_point_name : str
            The entry point name for the calculation plugin.

        Returns
        -------
        aiida.orm.nodes.data.code.Code
            An `InstalledCode` instance.
        """
        label = f"test.{entry_point_name}"
        janus_path = os.environ.get("JANUS_PATH")
        try:
            return load_code(label=label)
        except exceptions.NotExistent:
            return InstalledCode(
                label=label,
                computer=fixture_localhost,
                filepath_executable=janus_path,
                default_calc_job_plugin=entry_point_name,
            )

    return _fixture_code


@pytest.fixture
def generate_calc_job():
    """
    Fixture to construct a new `CalcJob` instance and prepare it for submission.

    This fixture returns a function that constructs a new `CalcJob` instance
    for testing purposes.
    """

    def _generate_calc_job(folder, entry_point_name, inputs=None):
        """
        Generate a mock `CalcInfo` for testing calculation jobs.

        Parameters
        ----------
        folder : SandboxFolder
            The temporary folder for storing raw input files.

        entry_point_name : str
            The entry point name of the `CalcJob` class to be instantiated.

        inputs : Optional[Dict[str, Any]], optional
            A dictionary of inputs for the calculation job, by default None.

        Returns
        -------
        CalcInfo
            The `CalcInfo` object returned by `prepare_for_submission`.

        Notes
        -----
        This function constructs a new instance of the specified `CalcJob` class
        using the provided inputs, and calls `prepare_for_submission`.
        The resulting `CalcInfo` object is returned.
        """

        manager = get_manager()
        runner = manager.get_runner()

        process_class = CalculationFactory(entry_point_name)
        process = instantiate_process(runner, process_class, **inputs)

        calc_info = process.prepare_for_submission(folder)

        return calc_info

    return _generate_calc_job


# Fixture to provide the path to the example file
@pytest.fixture
def example_path():
    """
    Fixture to provide the path to the example file.

    Returns:
        Path: The path to the example file.
    """
    return Path(__file__).resolve().parent.parent / "examples" / "calculations"
