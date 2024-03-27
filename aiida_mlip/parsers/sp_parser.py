"""
Parsers provided by aiida_mlip.
"""

from pathlib import Path

from ase.io import read
import numpy as np

from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import Dict, SinglefileData
from aiida.orm.nodes.process.process import ProcessNode
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory

singlePointCalculation = CalculationFactory("janus.sp")


def convert_numpy(dictionary: dict) -> dict:
    """
    A function to convert numpy ndarrays in dictionary into lists.

    Parameters
    ----------
    dictionary : dict
        A dictionary with numpy array values to be converted into lists.

    Returns
    -------
    dict
        Converted dictionary.
    """
    for key, value in dictionary.items():
        if isinstance(value, np.ndarray):
            dictionary[key] = value.tolist()
    return dictionary


class SPParser(Parser):
    """
    Parser class for parsing output of calculation.

    Parameters
    ----------
    node : aiida.orm.nodes.process.process.ProcessNode
        ProcessNode of calculation.

    Methods
    -------
    __init__(node: aiida.orm.nodes.process.process.ProcessNode)
        Initialize the SPParser instance.

    parse(**kwargs: Any) -> int:
        Parse outputs, store results in the database.

    Returns
    -------
    int
        An exit code.

    Raises
    ------
    exceptions.ParsingError
        If the ProcessNode being passed was not produced by a singlePointCalculation.
    """

    def __init__(self, node: ProcessNode):
        """
        Check that the ProcessNode being passed was produced by a `Singlepoint`.

        Parameters
        ----------
        node : aiida.orm.nodes.process.process.ProcessNode
            ProcessNode of calculation.
        """
        super().__init__(node)

        if not issubclass(node.process_class, singlePointCalculation):
            raise exceptions.ParsingError("Can only parse `Singlepoint` calculations")

    def parse(self, **kwargs) -> int:
        """
        Parse outputs, store results in the database.

        Parameters
        ----------
        **kwargs : Any
            Any keyword arguments.

        Returns
        -------
        int
            An exit code.
        """
        output_filename = self.node.get_option("output_filename")
        xyzoutput = (self.node.inputs.xyz_output_name).value
        logoutput = (self.node.inputs.log_filename).value

        # Check that folder content is as expected
        files_retrieved = self.retrieved.list_object_names()

        files_expected = {xyzoutput, logoutput}
        # Note: set(A) <= set(B) checks whether A is a subset of B
        if not set(files_expected) <= set(files_retrieved):
            self.logger.error(
                f"Found files '{files_retrieved}', expected to find '{files_expected}'"
            )
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        # Add output file to the outputs
        self.logger.info(f"Parsing '{xyzoutput}'")

        with self.retrieved.open(logoutput, "rb") as handle:
            self.out("log_output", SinglefileData(file=handle))
        with self.retrieved.open(xyzoutput, "rb") as handle:
            self.out("xyz_output", SinglefileData(file=handle))
        with self.retrieved.open(output_filename, "rb") as handle:
            self.out("std_output", SinglefileData(file=handle))

        content = read(Path(self.node.get_remote_workdir(), xyzoutput))
        results = convert_numpy(content.todict())
        results_node = Dict(results)
        self.out("results_dict", results_node)

        return ExitCode(0)
