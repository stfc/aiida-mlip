"""
Parsers provided by aiida_mlip.
"""

from pathlib import Path

from ase.io import read

from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import Dict, SinglefileData
from aiida.orm.nodes.process.process import ProcessNode
from aiida.plugins import CalculationFactory

from aiida_mlip.helpers.converters import convert_numpy
from aiida_mlip.parsers.base_parser import BaseParser

singlePointCalculation = CalculationFactory("janus.sp")


class SPParser(BaseParser):
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

        exit_code = super().parse(**kwargs)

        if exit_code != ExitCode(0):
            return exit_code

        xyz_output = (self.node.inputs.out).value

        # Check that folder content is as expected
        files_retrieved = self.retrieved.list_object_names()

        files_expected = {xyz_output}
        if not files_expected.issubset(files_retrieved):
            self.logger.error(
                f"Found files '{files_retrieved}', expected to find '{files_expected}'"
            )
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        # Add output file to the outputs
        self.logger.info(f"Parsing '{xyz_output}'")

        with self.retrieved.open(xyz_output, "rb") as handle:
            self.out("xyz_output", SinglefileData(file=handle))

        content = read(Path(self.node.get_remote_workdir(), xyz_output))
        results = convert_numpy(content.todict())
        results_node = Dict(results)
        self.out("results_dict", results_node)

        return ExitCode(0)
