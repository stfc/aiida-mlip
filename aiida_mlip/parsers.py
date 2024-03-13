"""
Parsers provided by aiida_mlip.

Register parsers via the "aiida.parsers" entry point in setup.json.
"""

from pathlib import Path

from ase.io import read
import numpy as np

from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import Dict, SinglefileData
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory

Singlepointcalc = CalculationFactory("janus.sp")


def convert_numpy(dictionary: dict) -> dict:
    """
    A function to convert numpy `ndarrays` in `dictionary` into `list`s.

    Parameters
    ----------
    dictionary : dict
            A dictionary.

    Returns
    -------
    dict
        A dictionary.
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
    """

    def __init__(self, node):
        """
        Check that the ProcessNode being passed was produced by a Singlepointcalc.

        Parameters
        ----------
        node : aiida.orm.nodes.process.process.ProcessNode
            ProcessNode of calculation.
        """
        super().__init__(node)
        if not issubclass(node.process_class, Singlepointcalc):
            raise exceptions.ParsingError("Can only parse Singlepointcalc")

    def parse(self, **kwargs):
        """
        Parse outputs, store results in the database.

        Parameters
        ----------
        **kwargs : Any
            Other arguments.

        Returns
        -------
        int
            An exit code.
        """
        output_filename = self.node.get_option("output_filename")
        xyzoutput_node = self.node.inputs.xyzoutput
        xyzoutput = xyzoutput_node.value

        remote_folder = self.node.get_remote_workdir()

        # Check that folder content is as expected
        files_retrieved = self.retrieved.list_object_names()
        print(files_retrieved)
        files_expected = [output_filename, xyzoutput]
        # Note: set(A) <= set(B) checks whether A is a subset of B
        if not set(files_expected) <= set(files_retrieved):
            self.logger.error(
                f"Found files '{files_retrieved}', expected to find '{files_expected}'"
            )
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        # Add output file to the outpus
        self.logger.info(f"Parsing '{xyzoutput}'")

        print("reading outputs")
        with self.retrieved.open(output_filename, "rb") as handle:
            print(handle)
            output_node = SinglefileData(file=handle)

        self.out("log_output", output_node)

        print("reading outputs 2")

        with self.retrieved.open(xyzoutput, "rb") as handle:
            print(handle)
            output_2 = SinglefileData(file=handle)

        self.out("xyz_output", output_2)

        output_path = Path(remote_folder, xyzoutput)
        content = read(output_path)
        results = content.todict()
        results = convert_numpy(results)
        results_node = Dict(results)
        self.out("results_dict", results_node)

        return ExitCode(0)
