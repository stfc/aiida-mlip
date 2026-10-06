import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import marimo as mo

    return Path, mo


@app.cell
def _(mo):
    mo.md(r"""
    # 🚀 Single Point Calculation with `aiida-mlip`

    This interactive tutorial demonstrates how to run a single-point energy calculation using
    **`aiida-mlip`**, **`janus-core`**, and **AiiDA**, following the same logic flow as the
    `singlepoint.ipynb` tutorial notebook.

    To run a single point using `aiida-mlip`, we define inputs as AiiDA data types:
    1. **AiiDA Profile**: Load and verify the active database & services.
    2. **Structure (`StructureData`)**: Target crystal or molecule geometry.
    3. **Model & Architecture (`ModelData`)**: Machine learning potential (MACE).
    4. **Code (`InstalledCode`)**: MLIP calculation engine (`janus@localhost`).
    5. **Calculation Inputs**: Configure options and resources.
    6. **Execution (`mlip.sp`)**: Run via `CalculationFactory("mlip.sp")` and `run_get_node`.
    7. **Results Inspection**: Inspect energy, forces, stress, and AiiDA process status.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Load AiiDA Profile
    First, we load the AiiDA profile to connect to PostgreSQL, RabbitMQ, and the storage repository.
    """)
    return


@app.cell
def _(mo):
    from aiida import load_profile

    profile = load_profile()
    mo.md(
        f"""
        - **Active Profile**: `{profile.name}`
        - **Storage Backend**: `{profile.storage_backend}`
        """
    )
    return (profile,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Prepare Atomic Structure (`StructureData`)
    First of all, we need a structure on which to perform the calculation. By default, we define
    a rocksalt **NaCl** structure using ASE (`ase.build.bulk`), just like in `singlepoint.ipynb`.
    You can also select alternative elemental bulk crystals below:
    """)
    return


@app.cell
def _(mo):
    element_selector = mo.ui.dropdown(
        options=["NaCl", "Si", "Cu", "Fe", "Al"],
        value="NaCl",
        label="Select structure:",
    )
    element_selector
    return (element_selector,)


@app.cell
def _(element_selector, mo):
    from aiida.orm import StructureData
    from ase.build import bulk

    material = element_selector.value
    match material:
        case "NaCl":
            atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.63)
            material_label = "NaCl (rocksalt, a=5.63 Å)"
        case "Si":
            atoms = bulk("Si", crystalstructure="diamond", a=5.43)
            material_label = "Si (diamond, a=5.43 Å)"
        case _:
            atoms = bulk(material, a=3.6)
            material_label = f"{material} (bulk, a=3.6 Å)"

    structure_node = StructureData(ase=atoms)

    mo.md(
        f"""
        ### Structure: **{material_label}**
        - **Chemical Formula**: `{atoms.get_chemical_formula()}`
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
    return StructureData, atoms, material, material_label, structure_node


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Choose Model and Architecture (`ModelData`)
    Then we need to choose a model and architecture to be used for the calculation and save it
    as a `ModelData` type, a specific data type of this plugin.

    In this example, we use MACE with a model that we download from this URL:
    `"https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model"`,
    and we save the file in the cache folder (`default="~/.cache/mlips/"`):
    """)
    return


@app.cell
def _(mo):
    from aiida_mlip.data.model import ModelData

    uri = "https://github.com/stfc/janus-core/raw/main/tests/models/mace_mp_small.model"
    model = ModelData.from_uri(uri, architecture="mace_mp", cache_dir="mlips")

    mo.md(
        f"""
        - **Model URI**: `{uri}`
        - **Architecture**: `{model.architecture}`
        - **ModelData Node UUID**: `{model.uuid}`
        """
    )
    return ModelData, model, uri


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Load the Calculation Code (`janus@localhost`)
    Another parameter that we need to define as an AiiDA type is the code. Assuming the code is
    saved as `janus` in the `localhost` computer, the code can be loaded using `load_code`:
    """)
    return


@app.cell
def _(Path, mo):
    from aiida.orm import load_code

    try:
        janus_code = load_code("janus@localhost")
    except Exception:
        import shutil
        from aiida.orm import Computer, InstalledCode, load_computer

        janus_path = shutil.which("janus") or "janus"
        try:
            comp = load_computer("localhost")
        except Exception:
            comp = Computer(
                label="localhost",
                hostname="localhost",
                transport_type="core.local",
                scheduler_type="core.direct",
                workdir=Path("/tmp/aiida_work").as_posix(),
            ).store()
            comp.configure()
        janus_code = InstalledCode(
            label="janus",
            computer=comp,
            filepath_executable=janus_path,
            default_calc_job_plugin="mlip.sp",
        ).store()

    mo.md(
        f"""
        - **Loaded Code**: `{janus_code.full_label}`
        - **Executable**: `{janus_code.filepath_executable}`
        - **Default Plugin**: `{janus_code.default_calc_job_plugin or 'mlip.sp'}`
        """
    )
    return (janus_code,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Prepare Calculation Inputs & Launch
    The entry point for single-point calculations is `mlip.sp`. We assemble the `inputs`
    dictionary with the code, model, structure, architecture, and resource options:
    """)
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="🚀 Run Single Point Calculation")
    run_button
    return (run_button,)


@app.cell
async def _(janus_code, mo, model, run_button, structure_node):
    from aiida.engine import run_get_node
    from aiida.orm import Dict, Str
    from aiida.plugins import CalculationFactory
    from plumpy import ensure_portal

    if not run_button.value:
        res_output = mo.md(
            "*Click the button above to execute `singlepointCalc` via AiiDA `run_get_node`.*"
        )
    else:
        with mo.status.spinner("Running singlepoint calculation via AiiDA..."):
            try:
                await ensure_portal()
                singlepointCalc = CalculationFactory("mlip.sp")

                inputs = {
                    "code": janus_code,
                    "model": model,
                    "struct": structure_node,
                    "arch": Str(model.architecture),
                    "device": Str("cpu"),
                    "calc_kwargs": Dict({"dispersion": True}),
                    "metadata": {
                        "options": {
                            "resources": {
                                "num_machines": 1,
                                "tot_num_mpiprocs": 1,
                            }
                        }
                    },
                }

                result, node = run_get_node(singlepointCalc, inputs)

                # Extract primary observables matching singlepoint.ipynb
                results_dict = result["results_dict"].get_dict()
                info = results_dict.get("info", {})
                energy_val = (
                    info.get("mace_mp_energy")
                    or info.get("mace_mp_d3_energy")
                    or results_dict.get("energy")
                )
                forces = results_dict.get("mace_mp_forces") or results_dict.get("forces")

                status_badge = (
                    "✅ Finished OK"
                    if node.is_finished_ok
                    else f"⚠️ Exit status {node.exit_status}"
                )

                res_output = mo.md(
                    f"""
                    ### 📊 Calculation Summary
                    - **Status**: {status_badge}
                    - **Node PK**: `{node.pk}`
                    - **Process State**: `{node.process_state.value}`
                    - **Exit Status**: `{node.exit_status}`

                    ### 🔬 Physical Outputs
                    - **Total Energy**: `{energy_val} eV`
                    - **Atomic Forces**: `{len(forces) if forces else 0} forces computed`
                    - **Cell Vectors**:
                    ```
                    {results_dict.get('cell')}
                    ```

                    <details>
                    <summary><b>View Complete results_dict</b></summary>

                    ```json
                    {results_dict}
                    ```
                    </details>
                    """
                )
            except Exception as exc:
                res_output = mo.md(f"⚠️ Calculation encountered error: `{exc}`")

    res_output
    return (
        CalculationFactory,
        Dict,
        Str,
        energy_val,
        ensure_portal,
        forces,
        info,
        inputs,
        node,
        res_output,
        result,
        results_dict,
        run_get_node,
        singlepointCalc,
        status_badge,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 6. Inspect Calculations via Command Line
    You can inspect the processes and results using `verdi` commands in the terminal:

    ```bash
    # List all processes
    verdi process list -a

    # Show calculation details and inputs/outputs
    verdi node show <PK>

    # View parsed calculation results
    verdi calcjob res <PK>
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
