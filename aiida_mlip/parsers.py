"""
Parsers provided by aiida_mlip.

Register parsers via the "aiida.parsers" entry point in setup.json.
"""

from ase.io import read
from numpy import ndarray

from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import Dict, SinglefileData
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory

Singlepointcalc = CalculationFactory("janus.sp")


class SPParser(Parser):
    """
    Parser class for parsing output of calculation.
    """

    def __init__(self, node):
        """
        Initialize Parser instance

        Checks that the ProcessNode being passed was produced by a Singlepointcalc.

        :param node: ProcessNode of calculation
        :param type node: :class:`aiida.orm.nodes.process.process.ProcessNode`
        """
        super().__init__(node)
        if not issubclass(node.process_class, Singlepointcalc):
            raise exceptions.ParsingError("Can only parse Singlepointcalc")

    def parse(self, **kwargs):
        """
        Parse outputs, store results in database.

        :returns: an exit code, if parsing fails (or nothing if parsing succeeds)
        """
        output_filename = self.node.get_option("output_filename")
        xyzoutput = self.node.get_option("xyzoutput")

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

        # add output file
        self.logger.info(f"Parsing '{output_filename}'")

        with self.retrieved.open(output_filename, "r") as file:
            content = read(file)
            results = content.todict()
            with self.retrieved.open(output_filename, "rb") as handle:
                print(handle)
                output_node = SinglefileData(file=handle)

        self.out("outputfile", output_node)

        self.out("results_dict", Dict(results))

        return ExitCode(0)
