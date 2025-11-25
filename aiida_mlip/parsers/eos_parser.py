from __future__ import annotations

from aiida.common import exceptions
from aiida.orm import SinglefileData
from aiida.orm.nodes.process.process import ProcessNode
from aiida.engine import ExitCode
from aiida.plugins import CalculationFactory

from ase.io import read
from pathlib import Path

from aiida_mlip.calculations.eos import EOS
from aiida_mlip.parsers.base_parser import BaseParser

EosCalc = CalculationFactory("mlip.eos")

class EOSParser(BaseParser):

    def __init__(self, node: ProcessNode):
        super().__init__(node)

        if not issubclass(node.process_class, EosCalc):
            raise exceptions.ParsingError("Can only parse `EOS` calculations")
        
    def parse(self, **kwargs) -> ExitCode:

        exit_code = super().parse(**kwargs)

        if exit_code == ExitCode(0):
            
            

            files_retrieved = self.retrieved.list_object_names()
            print("retrieved", files_retrieved)
            print("outputs_folder",self.node.get_remote_workdir())
            print("node", self.node.get_retrieve_list())
            print("option", self.node.get_options())
            # with self.retrieved.open(default_output_file, "rb") as handle:
            #     self.out("traj_file", SinglefileData(file=handle, filename=default_output_file))

            default_output_folder = self.node.get_remote_workdir()

            content = [x for x in Path(default_output_folder).iterdir() if x.is_dir()]
            print(content)

            # self.out("xyz_output", )
    
        return ExitCode(0)