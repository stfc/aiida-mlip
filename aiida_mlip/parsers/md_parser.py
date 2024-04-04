"""
MD parser.
"""

from pathlib import Path
from typing import Union

from ase.io import read

from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import SinglefileData, StructureData, TrajectoryData
from aiida.orm.nodes.process.process import ProcessNode
from aiida.plugins import CalculationFactory

from aiida_mlip.parsers.base_parser import BaseParser

MDCalculation = CalculationFactory("janus.md")


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


class MDParser(BaseParser):
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
        if exit_code == ExitCode(0):
            
            md_dictionary = self.node.inputs.md_dict.get_dict()
           
            if "traj-file" in md_dictionary:
                with self.retrieved.open(md_dictionary["traj-file"], "rb") as handle:
                    self.out("traj_file", SinglefileData(file=handle))
                fin, traj_output = xyz_to_aiida_traj(
                    Path(self.node.get_remote_workdir(), md_dictionary["traj-file"])
                )
                self.out("traj_output", traj_output)
            else:
                with self.retrieved.open("aiida-traj.xyz", "rb") as handle:
                    self.out("traj_file", SinglefileData(file=handle))
                fin, traj_output = xyz_to_aiida_traj(
                    Path(self.node.get_remote_workdir(), "aiida-traj.xyz")
                )
                self.out("traj_output", traj_output)

            if "stats-file" in md_dictionary:
                with self.retrieved.open(md_dictionary["stats-file"], "rb") as handle:
                    self.out("stats_file", SinglefileData(file=handle))
            else:
                with self.retrieved.open("aiida-stats.dat", "rb") as handle:
                    self.out("stats_file", SinglefileData(file=handle))
                # Parse trajectory and save it as `TrajectoryData`

                # Parse the final structure of the trajectory
                self.out("final_structure", fin)

        return exit_code
