"""
Geom optimisation parser.
"""

from pathlib import Path
from typing import Union

from ase.io import read

from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import SinglefileData, StructureData, TrajectoryData
from aiida.orm.nodes.process.process import ProcessNode
from aiida.plugins import CalculationFactory

from aiida_mlip.parsers.sp_parser import SPParser

geomoptCalculation = CalculationFactory("janus.opt")


def xyz_to_aiida_traj(
    traj_file: Union[str, Path]
) -> tuple[StructureData, TrajectoryData]:
    """
    A function to convert xyz trajectory file to `TrajectoryData` data type.

    Parameters
    ----------
    traj_file : Union[str, Path]
        The path to the XYZ file.

    Returns
    -------
    Tuple[StructureData, TrajectoryData]
        A tuple containing the last structure in the trajectory and a `TrajectoryData`
        object containing all structures from the trajectory.
    """
    # Read the XYZ file using ASE
    struct_list = read(traj_file, index=":")

    # Create a TrajectoryData object
    traj = [StructureData(ase=struct) for struct in struct_list]

    return traj[-1], TrajectoryData(traj)


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

        if exit_code == ExitCode(0):
            traj_file = (self.node.inputs.traj).value

            # Parse the trajectory file and save it as `SingleFileData`
            with self.retrieved.open(traj_file, "rb") as handle:
                self.out("traj_file", SinglefileData(file=handle))
            # Parse trajectory and save it as `TrajectoryData`
            opt, traj_output = xyz_to_aiida_traj(
                Path(self.node.get_remote_workdir(), traj_file)
            )
            self.out("traj_output", traj_output)

            # Parse the final structure of the trajectory to obtain the opt structure
            self.out("final_structure", opt)

        return exit_code
