"""
Geom optimisation parser.
"""

from ase.io.trajectory import Trajectory

from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import SinglefileData, StructureData, TrajectoryData
from aiida.orm.nodes.process.process import ProcessNode
from aiida.plugins import CalculationFactory

from aiida_mlip.parsers.sp_parser import SPParser

geomoptCalculation = CalculationFactory("janus.opt")


class GeomOptParser(SPParser):
    """
    Parser class for parsing output of geometry optimization calculation.

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
        If the ProcessNode being passed was not produced by a `GeomOpt`.
    """

    def __init__(self, node: ProcessNode):
        """
        Check that the ProcessNode being passed was produced by a `GeomOpt`.

        Parameters
        ----------
        node : aiida.orm.nodes.process.process.ProcessNode
            ProcessNode of calculation.
        """
        super().__init__(node)

        if not issubclass(node.process_class, geomoptCalculation):
            raise exceptions.ParsingError("Can only parse `GeomOpt` calculations")

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

        traj_file = (self.node.inputs.traj).value

        # Parse the trajectory file and save it as `SingleFileData`
        with self.retrieved.open(traj_file, "rb") as handle:
            self.out("log_output", SinglefileData(file=handle))
        # Parse trajectory and save it as `TrajectoryData`
        traj = Trajectory(traj_file)
        traj_output = TrajectoryData(traj)
        self.out("traj_output", traj_output)

        # Parse the final structure of the trajectory to obtain the optimized structure
        final_structure = StructureData(traj[-1])
        self.out("final_structure", final_structure)

        return ExitCode(0)
