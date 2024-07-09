from aiida_workgraph import Workgraph, task

from aiida.engine import submit
from aiida.orm import load_node
from aiida.plugins import CalculationFactory, WorkflowFactory

from aiida_mlip.helpers.help_load import load_structure


# define DFT task
@task.calcfunction()
def submit_DFT(child, dft_inputs, group):
        print(child.name)
        dft_inputs['metadata']['label']=f"{child.name}"
    optcalculation = WorkflowFactory("quantumespresso.pw.relax")
    struc = load_structure(child)
    dft_inputs['struct']=struc
    result = submit(optcalculation, **inputs)
    group.add_nodes(load_node(result.pk))
    return group

#syntax of this wrong
@task.calcfunction()
def create_input(group):
    with open("input_file") as input_file:
        for node in group:
            #get the output structure
            structure = node.outputs.structure
            #convert it to extxyz
            structure.to_ase()
            # add to file
            input_file.writelines(structure)
    return input_file


# define traning task
@task.calcfunction()
def training(input_file, train_inputs):
    training = CalculationFactory("mlip.train")
    #check name of input file in training
    train_inputs['xyz_input'] = input_file
    future = submit(training, **train_inputs)
    return future



wg = WorkGraph("training_workflow")

for child in folder.glob('**/*'):
        if child.name.endswith("cif"):
            submitdft_task = wg.tasks.new(submit_DFT, name="submission")

# link the output of the `add` task to one of the `x` input of the `multiply` task.
create_file_task = wg.tasks.new(create_input, name="createinput", group = submitdft_task.outputs["result"])

train_task = wg.tasks.new(training, name="training", input_file=create_file_task.outputs['input_file'])

# export the workgraph to html file so that it can be visualized in a browser
wg.to_html()
# comment out the following line to visualize the workgraph in jupyter-notebook
# wg

# Set the maximum number of running jobs inside the WorkGraph
wg.max_number_jobs = 10
wg.submit(wait=True)
