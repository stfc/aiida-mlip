"""
Parsers provided by aiida_mlip.
"""

from aiida.engine import ExitCode
from aiida.orm import SinglefileData
from aiida.orm.nodes.process.process import ProcessNode
from aiida.parsers.parser import Parser


class BaseParser(Parser):
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
        log_output = (self.node.inputs.log_filename).value

        # Check that folder content is as expected
        files_retrieved = self.retrieved.list_object_names()

        files_expected = {output_filename, log_output}
        if not files_expected.issubset(files_retrieved):
            self.logger.error(
                f"Found files '{files_retrieved}', expected to find '{files_expected}'"
            )
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        # Add output file to the outputs

        with (
            self.retrieved.open(log_output, "rb") as log,
            self.retrieved.open(output_filename, "rb") as output,
        ):
            self.out("log_output", SinglefileData(file=log))
            self.out("std_output", SinglefileData(file=output))

        return ExitCode(0)
