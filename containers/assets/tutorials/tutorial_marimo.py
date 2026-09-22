from __future__ import annotations

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    mo.md(
        r"""
        # 🚀 Welcome to `aiida-mlip` in Marimo!

        This container provides a fully integrated, interactive environment for running
        Machine Learning Interatomic Potentials (MLIPs) using **AiiDA** and **janus-core**.

        ### Highlights
        - **AiiDA profile**: Automatically configured with PostgreSQL and RabbitMQ.
        - **MLIP Engine**: `janus-core` registered as `janus@localhost`.
        - **Calculations supported**: Single-point (`mlip.sp`), Geometry Optimization (`mlip.opt`),
          Phonons (`mlip.ph`), Molecular Dynamics (`mlip.md`), and WorkGraphs!
        """
    )
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 1. Verify AiiDA Environment & Profile
        First, let's load the AiiDA profile and verify the status of the daemon and registered codes.
        """
    )
    return


@app.cell
def __(mo):
    from aiida import load_profile
    from aiida.orm import load_code

    profile = load_profile()
    janus_code = load_code("janus@localhost")
    python_code = load_code("python3@localhost")

    mo.md(
        f"""
        - **Loaded Profile**: `{profile.name}` ({profile.storage_backend})
        - **Janus Code**: `{janus_code.full_label}` (executable: `{janus_code.filepath_executable}`)
        - **Python Code**: `{python_code.full_label}` (executable: `{python_code.filepath_executable}`)
        """
    )
    return load_code, load_profile, profile, python_code, janus_code


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 2. Prepare an Atomic Structure
        We can construct an atomic structure using **ASE** (`ase.build.bulk`) and convert it to an AiiDA `StructureData` node.
        """
    )
    return


@app.cell
def __(mo):
    from aiida.orm import StructureData
    from ase.build import bulk

    element_selector = mo.ui.dropdown(
        options=["NaCl", "Si", "Cu", "Fe", "Al"],
        value="NaCl",
        label="Select material:",
    )
    element_selector
    return StructureData, bulk, element_selector


@app.cell
def __(StructureData, bulk, element_selector, mo):
    material = element_selector.value
    if material == "NaCl":
        atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64)
    elif material == "Si":
        atoms = bulk("Si", crystalstructure="diamond", a=5.43)
    else:
        atoms = bulk(material, a=3.6)

    structure_node = StructureData(ase=atoms)

    mo.md(
        f"""
        ### Structure Details: **{material}**
        - **Formula**: `{atoms.get_chemical_formula()}`
        - **Number of atoms**: `{len(atoms)}`
        - **Cell vectors (Å)**:
        ```
        {atoms.cell[:]}
        ```
        - **Positions (Å)**:
        ```
        {atoms.positions}
        ```
        """
    )
    return atoms, material, structure_node


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 3. Configure and Launch an `aiida-mlip` Calculation
        You can choose the MLIP architecture (e.g. MACE, M3GNet, CHGNet) and calculate properties directly.
        """
    )
    return


@app.cell
def __(mo):
    arch_dropdown = mo.ui.dropdown(
        options=["mace_mp", "m3gnet", "chgnet"],
        value="mace_mp",
        label="Select MLIP Architecture:",
    )
    run_button = mo.ui.run_button(label="Execute Single-Point Calculation")
    mo.hstack([arch_dropdown, run_button])
    return arch_dropdown, run_button


@app.cell
def __(arch_dropdown, janus_code, mo, run_button, structure_node):
    from aiida.engine import run_get_node
    from aiida.orm import Dict, Str
    from aiida.plugins import CalculationFactory

    if not run_button.value:
        res_output = mo.md(
            "*Click the button above to run the singlepoint calculation using `aiida-mlip`.*"
        )
    else:
        with mo.status.spinner("Running singlepoint calculation via AiiDA..."):
            Singlepoint = CalculationFactory("mlip.sp")
            inputs = {
                "code": janus_code,
                "structure": structure_node,
                "arch": Str(arch_dropdown.value),
                "device": Str("cpu"),
                "precision": Str("float64"),
                "metadata": {
                    "options": {"resources": {"num_machines": 1, "tot_num_mpiprocs": 1}}
                },
            }
            try:
                results, node = run_get_node(Singlepoint, **inputs)
                res_output = mo.md(
                    f"""
                    ### Calculation Completed!
                    - **PK**: `{node.pk}`
                    - **State**: `{node.process_state.value}`
                    - **Exit status**: `{node.exit_status}`
                    - **Results dict**:
                    ```json
                    {results["results_dict"].get_dict()}
                    ```
                    """
                )
            except Exception as exc:
                res_output = mo.md(f"⚠️ Calculation run encountered: `{exc}`")

    res_output
    return CalculationFactory, Dict, Singlepoint, Str, inputs, res_output, run_get_node


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ### Next Steps
        - Explore the tutorials and workgraphs in `/app/examples` or `/app/tutorials`.
        - You can launch `verdi` commands from terminal inside the container:
          `verdi process list`, `verdi status`, `verdi node show <PK>`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
