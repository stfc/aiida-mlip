"""
MD parser.
"""

from pathlib import Path

import yaml

from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import Dict, SinglefileData
from aiida.orm.nodes.process.process import ProcessNode
from aiida.plugins import CalculationFactory

from aiida_mlip.calculations.md import MD
from aiida_mlip.helpers.converters import xyz_to_aiida_traj
from aiida_mlip.parsers.base_parser import BaseParser

MDCalculation = CalculationFactory("janus.md")


class MDParser(BaseParser):
    """
    Parser class for parsing output of molecular dynamics simulation.

    Inherits from SPParser.

    Parameters
    ----------
    node : aiida.orm.nodes.process.process.ProcessNode
        ProcessNode of calculation.

    Methods
    -------
    parse(**kwargs: Any) -> int:
        Parse outputs, store results in the database.

    Returns
    -------
    int
        An exit code.

    Raises
    ------
    exceptions.ParsingError
        If the ProcessNode being passed was not produced by a `MD`.
    """

    def __init__(self, node: ProcessNode):
        """
        Check that the ProcessNode being passed was produced by a `MD`.

        Parameters
        ----------
        node : aiida.orm.nodes.process.process.ProcessNode
            ProcessNode of calculation.
        """
        super().__init__(node)

        if not issubclass(node.process_class, MDCalculation):
            raise exceptions.ParsingError("Can only parse `MD` calculations")

    def parse(self, **kwargs) -> ExitCode:
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
        # Call the parent parse method to handle common parsing logic
        exit_code = super().parse(**kwargs)

        if exit_code != ExitCode(0):
            return exit_code

        md_dictionary = self.node.inputs.md_kwargs.get_dict()

        # Process trajectory file saving both the file and trajectory as aiida data
        traj_filepath = md_dictionary.get("traj-file", MD.DEFAULT_TRAJ_FILE)
        with self.retrieved.open(traj_filepath, "rb") as handle:
            self.out("traj_file", SinglefileData(file=handle))
        final_str, traj_output = xyz_to_aiida_traj(
            Path(self.node.get_remote_workdir(), traj_filepath)
        )
        self.out("traj_output", traj_output)
        self.out("final_structure", final_str)

        # Process stats file as singlefiledata
        stats_filepath = md_dictionary.get("stats-file", MD.DEFAULT_STATS_FILE)
        with self.retrieved.open(stats_filepath, "rb") as handle:
            self.out("stats_file", SinglefileData(file=handle))

        # Process summary as both singlefiledata and results dictionary
        summary_filepath = md_dictionary.get("summary", MD.DEFAULT_SUMMARY_FILE)
        print(self.node.get_remote_workdir(), summary_filepath)
        with self.retrieved.open(summary_filepath, "rb") as handle:
            self.out("summary", SinglefileData(file=handle))

        with self.retrieved.open(summary_filepath, "r") as handle:
            try:
                res_dict = yaml.safe_load(handle.read())
            except yaml.YAMLError as exc:
                print("Error loading YAML:", exc)
            if res_dict is None:
                self.logger.error("Results dictionary empty")
                return self.exit_codes.ERROR_MISSING_OUTPUT_FILES
            results_node = Dict(res_dict)
            self.out("results_dict", results_node)
        return ExitCode(0)
