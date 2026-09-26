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


            default_output_folder = f"{self.node.get_remote_workdir()}/janus_results"


            content = [x for x in Path(default_output_folder).iterdir()]
            for file_path in content:
                filename = str(file_path)[len(default_output_folder)+1:]
                if filename == "aiida-generated.extxyz":
                    print(filename)
                    self.out("xyz_output", SinglefileData(file_path, filename="xyz_output"))
                else:
                    self.out(filename, SinglefileData(file_path))
            
            # self.out("xyz_output", structs)
            
            # for files in content:
            #     if str(files k) == f"{default_output_folder}/aiida-generated.extxyz":
            #         print(read(files, format="ext"))

            # self.out("xyz_output", )
    
        return ExitCode(0)