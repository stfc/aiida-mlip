"""AiiDA input-dict and WorkGraph builders for all ``aiida-mlip`` calculations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiida.orm import Bool, Code, Dict, Float, Int, List, Str, StructureData
from aiida.plugins import CalculationFactory
import ase.io

from aiida_mlip.calculations.md import MD
from aiida_mlip.data.model import ModelData
from aiida_mlip.mcp.refs import (
    DryRunModelRef,
    MCPToolError,
    build_config_node,
    build_metadata_options,
)

if TYPE_CHECKING:
    from aiida_workgraph import WorkGraph

CALC_ALIASES: dict[str, str] = {
    "sp": "mlip.sp",
    "mlip.sp": "mlip.sp",
    "opt": "mlip.opt",
    "mlip.opt": "mlip.opt",
    "md": "mlip.md",
    "mlip.md": "mlip.md",
    "ph": "mlip.ph",
    "mlip.ph": "mlip.ph",
    "descriptors": "mlip.descriptors",
    "mlip.descriptors": "mlip.descriptors",
    "train": "mlip.train",
    "mlip.train": "mlip.train",
}

STRUCTURE_PRODUCING_CALCS = frozenset({"mlip.opt", "mlip.md"})
ALLOWED_STAGE_KEYS = frozenset({"calc", "params", "name"})
FORBIDDEN_STAGE_PARAMS = frozenset(
    {"struct", "model", "arch", "code", "device", "hpc_preset", "hpc_overrides"}
)


def normalize_calc_alias(calc: str) -> str:
    """
    Normalize a short or full calculation name to an AiiDA entry point.

    Parameters
    ----------
    calc : str
        Calculation alias (e.g. ``"opt"`` or ``"mlip.opt"``).

    Returns
    -------
    str
        Canonical entry point string (e.g. ``"mlip.opt"``).

    Raises
    ------
    MCPToolError
        If ``calc`` is not a known ``aiida-mlip`` calculation alias.
    """
    if not isinstance(calc, str) or not calc.strip():
        raise MCPToolError(
            "validation_error",
            f"Calculation name must be a non-empty string, got {type(calc).__name__}.",
        )
    key = calc.strip().lower()
    if key not in CALC_ALIASES:
        raise MCPToolError(
            "validation_error",
            f"Unknown calculation '{calc}'. Supported: sp, opt, md, ph, "
            "descriptors, train.",
        )
    return CALC_ALIASES[key]


def _base_janus_inputs(
    *,
    code: Code,
    struct: StructureData | None,
    model: ModelData | DryRunModelRef,
    device: str,
    calc_kwargs: dict[str, Any] | None,
    janus_overrides: dict[str, Any] | None,
    hpc_preset: str | None,
    hpc_overrides: dict[str, Any] | None,
    presets_path: Path | str | None,
    config_cache_dir: Path | None,
) -> dict[str, Any]:
    """
    Build the shared ``BaseJanus`` input dictionary.

    Parameters
    ----------
    code : Code
        Registered AiiDA ``Code`` node.
    struct : StructureData | None
        Input structure, or ``None`` when omitted for WorkGraph socket wiring.
    model : ModelData | DryRunModelRef
        Resolved model node or dry-run URI placeholder.
    device : str
        Compute device (``cpu``, ``cuda``, or ``mps``).
    calc_kwargs : dict[str, Any] | None
        Optional calculator keyword arguments.
    janus_overrides : dict[str, Any] | None
        Optional YAML overrides compiled into ``JanusConfigfile`` with
        ``tracker: false`` enforced by default.
    hpc_preset : str | None
        Optional named HPC preset.
    hpc_overrides : dict[str, Any] | None
        Optional ``metadata.options`` overrides.
    presets_path : Path | str | None
        Optional path to HPC presets YAML file.
    config_cache_dir : Path | None
        Optional cache directory for synthesized ``JanusConfigfile`` YAML.

    Returns
    -------
    dict[str, Any]
        Shared ``BaseJanus`` inputs dictionary.
    """
    inputs: dict[str, Any] = {
        "code": code,
        "arch": Str(model.architecture),
        "device": Str(device),
        "config": build_config_node(
            janus_overrides,
            force_no_tracker=True,
            cache_dir=config_cache_dir,
        ),
        "metadata": {
            "options": build_metadata_options(
                hpc_preset=hpc_preset,
                hpc_overrides=hpc_overrides,
                presets_path=presets_path,
            )
        },
    }
    if struct is not None:
        inputs["struct"] = struct
    if isinstance(model, ModelData):
        inputs["model"] = model
    if calc_kwargs:
        inputs["calc_kwargs"] = Dict(calc_kwargs)
    return inputs


def build_sp_inputs(
    *,
    code: Code,
    model: ModelData | DryRunModelRef,
    struct: StructureData | None = None,
    properties: list[str] | None = None,
    calc_kwargs: dict[str, Any] | None = None,
    device: str = "cpu",
    janus_overrides: dict[str, Any] | None = None,
    hpc_preset: str | None = None,
    hpc_overrides: dict[str, Any] | None = None,
    presets_path: Path | str | None = None,
    config_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Build AiiDA inputs for ``mlip.sp`` (``Singlepoint``).

    Parameters
    ----------
    code : Code
        Registered AiiDA ``Code`` node.
    model : ModelData | DryRunModelRef
        Resolved model node or dry-run placeholder.
    struct : StructureData | None
        Input structure, or ``None`` for WorkGraph templates.
    properties : list[str] | None
        Optional list of properties to calculate (e.g. ``["energy", "forces"]``).
    calc_kwargs : dict[str, Any] | None
        Optional calculator keyword arguments.
    device : str
        Compute device (``cpu``, ``cuda``, or ``mps``).
    janus_overrides : dict[str, Any] | None
        Optional ``JanusConfigfile`` overrides (``tracker: false`` default).
    hpc_preset : str | None
        Optional named HPC preset.
    hpc_overrides : dict[str, Any] | None
        Optional ``metadata.options`` overrides.
    presets_path : Path | str | None
        Optional path to HPC presets YAML file.
    config_cache_dir : Path | None
        Optional directory for synthesized YAML config files.

    Returns
    -------
    dict[str, Any]
        AiiDA inputs dictionary for ``Singlepoint``.
    """
    inputs = _base_janus_inputs(
        code=code,
        struct=struct,
        model=model,
        device=device,
        calc_kwargs=calc_kwargs,
        janus_overrides=janus_overrides,
        hpc_preset=hpc_preset,
        hpc_overrides=hpc_overrides,
        presets_path=presets_path,
        config_cache_dir=config_cache_dir,
    )
    if properties is not None:
        inputs["properties"] = List(list(properties))
    return inputs


def build_opt_inputs(
    *,
    code: Code,
    model: ModelData | DryRunModelRef,
    struct: StructureData | None = None,
    fmax: float = 0.01,
    steps: int = 1000,
    opt_cell_fully: bool = True,
    opt_cell_lengths: bool = False,
    filter_func: str = "FrechetCellFilter",
    pressure: float | None = None,
    minimize_kwargs: dict[str, Any] | None = None,
    calc_kwargs: dict[str, Any] | None = None,
    device: str = "cpu",
    janus_overrides: dict[str, Any] | None = None,
    hpc_preset: str | None = None,
    hpc_overrides: dict[str, Any] | None = None,
    presets_path: Path | str | None = None,
    config_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Build AiiDA inputs for ``mlip.opt`` (``GeomOpt``).

    Parameters
    ----------
    code : Code
        Registered AiiDA ``Code`` node.
    model : ModelData | DryRunModelRef
        Resolved model node or dry-run placeholder.
    struct : StructureData | None
        Input structure, or ``None`` for WorkGraph templates.
    fmax : float
        Force convergence threshold in eV/A.
    steps : int
        Maximum number of optimization steps.
    opt_cell_fully : bool
        Fully optimize cell vectors, angles, and atomic positions.
    opt_cell_lengths : bool
        Optimize cell vector lengths and atomic positions.
    filter_func : str
        Cell filter name added to ``minimize_kwargs`` when cell optimization
        is enabled. Defaults to ``"FrechetCellFilter"``.
    pressure : float | None
        Optional scalar target pressure in GPa.
    minimize_kwargs : dict[str, Any] | None
        Optional extra optimizer keyword arguments.
    calc_kwargs : dict[str, Any] | None
        Optional calculator keyword arguments.
    device : str
        Compute device.
    janus_overrides : dict[str, Any] | None
        Optional ``JanusConfigfile`` overrides.
    hpc_preset : str | None
        Optional named HPC preset.
    hpc_overrides : dict[str, Any] | None
        Optional ``metadata.options`` overrides.
    presets_path : Path | str | None
        Optional path to HPC presets YAML file.
    config_cache_dir : Path | None
        Optional directory for synthesized YAML config files.

    Returns
    -------
    dict[str, Any]
        AiiDA inputs dictionary for ``GeomOpt``.

    Raises
    ------
    MCPToolError
        If both ``opt_cell_fully`` and ``opt_cell_lengths`` are ``True``.
    """
    if opt_cell_fully and opt_cell_lengths:
        raise MCPToolError(
            "validation_error",
            "Cannot set both 'opt_cell_fully=True' and 'opt_cell_lengths=True'.",
        )

    inputs = _base_janus_inputs(
        code=code,
        struct=struct,
        model=model,
        device=device,
        calc_kwargs=calc_kwargs,
        janus_overrides=janus_overrides,
        hpc_preset=hpc_preset,
        hpc_overrides=hpc_overrides,
        presets_path=presets_path,
        config_cache_dir=config_cache_dir,
    )
    inputs["fmax"] = Float(fmax)
    inputs["steps"] = Int(steps)
    inputs["opt_cell_fully"] = Bool(opt_cell_fully)
    inputs["opt_cell_lengths"] = Bool(opt_cell_lengths)
    if pressure is not None:
        inputs["pressure"] = Float(pressure)

    mk = dict(minimize_kwargs or {})
    if opt_cell_fully or opt_cell_lengths:
        mk.setdefault("filter_func", filter_func)
    if mk:
        inputs["minimize_kwargs"] = Dict(mk)
    return inputs


def build_md_inputs(
    *,
    code: Code,
    model: ModelData | DryRunModelRef,
    struct: StructureData | None = None,
    ensemble: str = "nvt",
    temp: float | None = 300.0,
    steps: int = 1000,
    timestep: float = 1.0,
    pressure: float | None = None,
    md_kwargs: dict[str, Any] | None = None,
    calc_kwargs: dict[str, Any] | None = None,
    device: str = "cpu",
    janus_overrides: dict[str, Any] | None = None,
    hpc_preset: str | None = None,
    hpc_overrides: dict[str, Any] | None = None,
    presets_path: Path | str | None = None,
    config_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Build AiiDA inputs for ``mlip.md`` (``MD``).

    Parameters
    ----------
    code : Code
        Registered AiiDA ``Code`` node.
    model : ModelData | DryRunModelRef
        Resolved model node or dry-run placeholder.
    struct : StructureData | None
        Input structure, or ``None`` for WorkGraph templates.
    ensemble : str
        Thermodynamic ensemble: ``"nve"``, ``"nvt"``, or ``"npt"``.
    temp : float | None
        Temperature in K. Omitted for ``"nve"`` unless provided inside
        ``md_kwargs``.
    steps : int
        Number of MD steps.
    timestep : float
        Integration timestep in fs.
    pressure : float | None
        Optional target pressure in GPa (for ``npt``).
    md_kwargs : dict[str, Any] | None
        Additional ``janus md`` keyword arguments (take precedence over
        shorthand arguments).
    calc_kwargs : dict[str, Any] | None
        Optional calculator keyword arguments.
    device : str
        Compute device.
    janus_overrides : dict[str, Any] | None
        Optional ``JanusConfigfile`` overrides.
    hpc_preset : str | None
        Optional named HPC preset.
    hpc_overrides : dict[str, Any] | None
        Optional ``metadata.options`` overrides.
    presets_path : Path | str | None
        Optional path to HPC presets YAML file.
    config_cache_dir : Path | None
        Optional directory for synthesized YAML config files.

    Returns
    -------
    dict[str, Any]
        AiiDA inputs dictionary for ``MD``.

    Raises
    ------
    MCPToolError
        If ``ensemble`` is not one of ``nve``, ``nvt``, or ``npt``.
    """
    ens = ensemble.strip().lower()
    if ens not in ("nve", "nvt", "npt"):
        raise MCPToolError(
            "validation_error",
            f"Unsupported MD ensemble '{ensemble}'. Expected 'nve', 'nvt', or 'npt'.",
        )
    if pressure is not None and ens != "npt":
        raise MCPToolError(
            "validation_error",
            f"'pressure' is only supported for ensemble='npt', got '{ens}'.",
        )

    inputs = _base_janus_inputs(
        code=code,
        struct=struct,
        model=model,
        device=device,
        calc_kwargs=calc_kwargs,
        janus_overrides=janus_overrides,
        hpc_preset=hpc_preset,
        hpc_overrides=hpc_overrides,
        presets_path=presets_path,
        config_cache_dir=config_cache_dir,
    )
    inputs["ensemble"] = Str(ens)

    md_dict: dict[str, Any] = {
        "traj-file": MD.DEFAULT_TRAJ_FILE,
        "stats-file": MD.DEFAULT_STATS_FILE,
        "steps": steps,
        "timestep": timestep,
    }
    if temp is not None and ens != "nve":
        md_dict["temp"] = temp
    if pressure is not None:
        md_dict["pressure"] = pressure
    if md_kwargs:
        normalized_md = {
            (
                "traj-file"
                if k == "traj_file"
                else "stats-file"
                if k == "stats_file"
                else k
            ): v
            for k, v in md_kwargs.items()
        }
        md_dict.update(normalized_md)

    inputs["md_kwargs"] = Dict(md_dict)
    return inputs


def build_ph_inputs(
    *,
    code: Code,
    model: ModelData | DryRunModelRef,
    struct: StructureData | None = None,
    supercell: str = "2 2 2",
    displacement: float = 0.01,
    n_qpoints: int = 100,
    dos: bool = False,
    pdos: bool = False,
    bands: bool = False,
    symmetrize: bool = False,
    no_hdf5: bool = False,
    calc_kwargs: dict[str, Any] | None = None,
    device: str = "cpu",
    janus_overrides: dict[str, Any] | None = None,
    hpc_preset: str | None = None,
    hpc_overrides: dict[str, Any] | None = None,
    presets_path: Path | str | None = None,
    config_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Build AiiDA inputs for ``mlip.ph`` (``Phonons``).

    Populates all eight runtime-required ``Phonons`` ports unconditionally so
    ``Phonons.prepare_for_submission`` and ``PhononParser.parse`` never fail
    with missing-attribute errors.

    Parameters
    ----------
    code : Code
        Registered AiiDA ``Code`` node.
    model : ModelData | DryRunModelRef
        Resolved model node or dry-run placeholder.
    struct : StructureData | None
        Input structure, or ``None`` for WorkGraph templates.
    supercell : str
        Supercell specification string (e.g. ``"2 2 2"``).
    displacement : float
        Atomic displacement in A for finite differences.
    n_qpoints : int
        Number of q-points along band path.
    dos : bool
        Whether to compute total phonon DOS.
    pdos : bool
        Whether to compute projected phonon DOS.
    bands : bool
        Whether to compute phonon band structure.
    symmetrize : bool
        Whether to symmetrize force constants.
    no_hdf5 : bool
        Write force constants to YAML instead of HDF5.
    calc_kwargs : dict[str, Any] | None
        Optional calculator keyword arguments.
    device : str
        Compute device.
    janus_overrides : dict[str, Any] | None
        Optional ``JanusConfigfile`` overrides.
    hpc_preset : str | None
        Optional named HPC preset.
    hpc_overrides : dict[str, Any] | None
        Optional ``metadata.options`` overrides.
    presets_path : Path | str | None
        Optional path to HPC presets YAML file.
    config_cache_dir : Path | None
        Optional directory for synthesized YAML config files.

    Returns
    -------
    dict[str, Any]
        AiiDA inputs dictionary for ``Phonons``.
    """
    inputs = _base_janus_inputs(
        code=code,
        struct=struct,
        model=model,
        device=device,
        calc_kwargs=calc_kwargs,
        janus_overrides=janus_overrides,
        hpc_preset=hpc_preset,
        hpc_overrides=hpc_overrides,
        presets_path=presets_path,
        config_cache_dir=config_cache_dir,
    )
    inputs.update(
        {
            "supercell": Str(supercell),
            "displacement": Float(displacement),
            "n_qpoints": Int(n_qpoints),
            "no_hdf5": Bool(no_hdf5),
            "dos": Bool(dos),
            "pdos": Bool(pdos),
            "bands": Bool(bands),
            "symmetrize": Bool(symmetrize),
        }
    )
    return inputs


def build_descriptors_inputs(
    *,
    code: Code,
    model: ModelData | DryRunModelRef,
    struct: StructureData | None = None,
    invariants_only: bool = True,
    calc_per_element: bool = False,
    calc_per_atom: bool = False,
    calc_kwargs: dict[str, Any] | None = None,
    device: str = "cpu",
    janus_overrides: dict[str, Any] | None = None,
    hpc_preset: str | None = None,
    hpc_overrides: dict[str, Any] | None = None,
    presets_path: Path | str | None = None,
    config_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Build AiiDA inputs for ``mlip.descriptors`` (``Descriptors``).

    Parameters
    ----------
    code : Code
        Registered AiiDA ``Code`` node.
    model : ModelData | DryRunModelRef
        Resolved model node or dry-run placeholder.
    struct : StructureData | None
        Input structure, or ``None`` for WorkGraph templates.
    invariants_only : bool
        Compute only invariant descriptors.
    calc_per_element : bool
        Compute mean descriptors per chemical element.
    calc_per_atom : bool
        Compute per-atom descriptors.
    calc_kwargs : dict[str, Any] | None
        Optional calculator keyword arguments.
    device : str
        Compute device.
    janus_overrides : dict[str, Any] | None
        Optional ``JanusConfigfile`` overrides.
    hpc_preset : str | None
        Optional named HPC preset.
    hpc_overrides : dict[str, Any] | None
        Optional ``metadata.options`` overrides.
    presets_path : Path | str | None
        Optional path to HPC presets YAML file.
    config_cache_dir : Path | None
        Optional directory for synthesized YAML config files.

    Returns
    -------
    dict[str, Any]
        AiiDA inputs dictionary for ``Descriptors``.
    """
    inputs = _base_janus_inputs(
        code=code,
        struct=struct,
        model=model,
        device=device,
        calc_kwargs=calc_kwargs,
        janus_overrides=janus_overrides,
        hpc_preset=hpc_preset,
        hpc_overrides=hpc_overrides,
        presets_path=presets_path,
        config_cache_dir=config_cache_dir,
    )
    inputs.update(
        {
            "invariants_only": Bool(invariants_only),
            "calc_per_element": Bool(calc_per_element),
            "calc_per_atom": Bool(calc_per_atom),
        }
    )
    return inputs


def build_train_inputs(
    *,
    code: Code,
    name: str,
    train_file: str,
    valid_file: str,
    test_file: str,
    arch: str | None = None,
    fine_tune: bool = False,
    foundation_model: ModelData | DryRunModelRef | None = None,
    train_overrides: dict[str, Any] | None = None,
    hpc_preset: str | None = None,
    hpc_overrides: dict[str, Any] | None = None,
    presets_path: Path | str | None = None,
    config_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Build AiiDA inputs for ``mlip.train`` (``Train``).

    Does NOT inject ``tracker: false`` into ``mlip_config`` because
    ``janus train`` forwards ``mlip_config`` directly to the underlying MLIP
    trainer (e.g. ``mace_run_train``).

    Parameters
    ----------
    code : Code
        Registered AiiDA ``Code`` node.
    name : str
        Model output name.
    train_file : str
        Path to training dataset XYZ file.
    valid_file : str
        Path to validation dataset XYZ file.
    test_file : str
        Path to test dataset XYZ file.
    arch : str | None
        MLIP architecture to train (inferred from ``foundation_model`` if
        omitted).
    fine_tune : bool
        Whether to fine-tune a foundation model.
    foundation_model : ModelData | DryRunModelRef | None
        Optional resolved foundation model for fine-tuning.
    train_overrides : dict[str, Any] | None
        Additional trainer YAML keys (explicit arguments take precedence).
    hpc_preset : str | None
        Optional named HPC preset.
    hpc_overrides : dict[str, Any] | None
        Optional ``metadata.options`` overrides.
    presets_path : Path | str | None
        Optional path to HPC presets YAML file.
    config_cache_dir : Path | None
        Optional directory for synthesized YAML config files.

    Returns
    -------
    dict[str, Any]
        AiiDA inputs dictionary for ``Train``.

    Raises
    ------
    MCPToolError
        If required dataset files do not exist, ``arch`` cannot be determined,
        or ``foundation_model`` is duplicated in both argument and overrides.
    """
    if not name or not name.strip():
        raise MCPToolError("validation_error", "'name' must be a non-empty string.")

    overrides = dict(train_overrides or {})
    if foundation_model is not None and "foundation_model" in overrides:
        raise MCPToolError(
            "validation_error",
            "Specify 'foundation_model' either as the tool argument or inside "
            "'train_overrides', not both.",
        )

    resolved_paths: dict[str, str] = {}
    for key, raw_path in (
        ("train_file", train_file),
        ("valid_file", valid_file),
        ("test_file", test_file),
    ):
        expanded = Path(raw_path).expanduser().resolve()
        if not expanded.is_file():
            raise MCPToolError(
                "validation_error",
                f"Training dataset '{key}' does not exist: '{expanded}'.",
            )
        resolved_paths[key] = str(expanded)

    resolved_arch = arch or (
        foundation_model.architecture if foundation_model is not None else None
    )
    if not resolved_arch:
        raise MCPToolError(
            "validation_error",
            "'arch' must be specified when 'foundation_model' is not provided.",
        )
    if (
        foundation_model is not None
        and arch is not None
        and arch != foundation_model.architecture
    ):
        raise MCPToolError(
            "validation_error",
            f"Architecture mismatch: foundation_model has "
            f"'{foundation_model.architecture}', but arch='{arch}' was given.",
        )
    if fine_tune and foundation_model is None and "foundation_model" not in overrides:
        raise MCPToolError(
            "validation_error",
            "'fine_tune=True' requires 'foundation_model' (as argument or in "
            "'train_overrides').",
        )

    # Explicit tool arguments take precedence over conflicting keys in overrides.
    cfg = {**overrides, "name": name.strip(), **resolved_paths}
    mlip_config = build_config_node(
        cfg,
        force_no_tracker=False,
        cache_dir=config_cache_dir,
    )

    inputs: dict[str, Any] = {
        "code": code,
        "mlip_config": mlip_config,
        "arch": Str(resolved_arch),
        "fine_tune": Bool(fine_tune),
        "metadata": {
            "options": build_metadata_options(
                hpc_preset=hpc_preset,
                hpc_overrides=hpc_overrides,
                presets_path=presets_path,
            )
        },
    }
    if isinstance(foundation_model, ModelData):
        inputs["foundation_model"] = foundation_model
    return inputs


def build_calc_inputs_for_alias(
    calc: str,
    *,
    code: Code,
    model: ModelData | DryRunModelRef,
    struct: StructureData | None = None,
    params: dict[str, Any] | None = None,
    device: str = "cpu",
    hpc_preset: str | None = None,
    hpc_overrides: dict[str, Any] | None = None,
    presets_path: Path | str | None = None,
    config_cache_dir: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Dispatch to the appropriate ``BaseJanus`` input builder for an alias.

    Parameters
    ----------
    calc : str
        Calculation alias (``sp``, ``opt``, ``md``, ``ph``, ``descriptors``).
    code : Code
        Registered AiiDA ``Code`` node.
    model : ModelData | DryRunModelRef
        Resolved model node or dry-run placeholder.
    struct : StructureData | None
        Input structure, or ``None`` for WorkGraph templates.
    params : dict[str, Any] | None
        Calculation-specific keyword arguments.
    device : str
        Compute device.
    hpc_preset : str | None
        Optional named HPC preset.
    hpc_overrides : dict[str, Any] | None
        Optional ``metadata.options`` overrides.
    presets_path : Path | str | None
        Optional path to HPC presets YAML file.
    config_cache_dir : Path | None
        Optional directory for synthesized YAML config files.

    Returns
    -------
    tuple[str, dict[str, Any]]
        Tuple of canonical entry point name and AiiDA inputs dictionary.

    Raises
    ------
    MCPToolError
        If ``calc`` is ``train``, unknown, or ``params`` contains unexpected
        keyword arguments.
    """
    entry_point = normalize_calc_alias(calc)
    if entry_point == "mlip.train":
        raise MCPToolError(
            "validation_error",
            "'mlip.train' is not supported in high-throughput or chained "
            "structure WorkGraphs; use 'mlip_train' directly.",
        )

    dispatch = {
        "mlip.sp": build_sp_inputs,
        "mlip.opt": build_opt_inputs,
        "mlip.md": build_md_inputs,
        "mlip.ph": build_ph_inputs,
        "mlip.descriptors": build_descriptors_inputs,
    }
    builder = dispatch[entry_point]
    extra = dict(params or {})
    forbidden = FORBIDDEN_STAGE_PARAMS.intersection(extra.keys())
    if forbidden:
        raise MCPToolError(
            "validation_error",
            f"Unsupported keys in calculation params: {sorted(forbidden)}. "
            "Pass them at the top level instead.",
        )

    try:
        inputs = builder(
            code=code,
            model=model,
            struct=struct,
            device=device,
            hpc_preset=hpc_preset,
            hpc_overrides=hpc_overrides,
            presets_path=presets_path,
            config_cache_dir=config_cache_dir,
            **extra,
        )
    except TypeError as exc:
        raise MCPToolError(
            "validation_error",
            f"Invalid parameter for '{entry_point}': {exc}",
        ) from exc
    return entry_point, inputs


def validate_ht_folder(folder: str | Path, *, recursive: bool = True) -> list[Path]:
    """
    Validate a structure folder for ``build_ht_calc`` without storing nodes.

    Parameters
    ----------
    folder : str | Path
        Path to directory containing structure files.
    recursive : bool
        Whether to scan subdirectories recursively.

    Returns
    -------
    list[Path]
        Sorted list of readable structure file paths.

    Raises
    ------
    MCPToolError
        If the folder does not exist, contains no readable structure files,
        has duplicate file stems, or has stems that are invalid AiiDA link
        labels.
    """
    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.is_dir():
        raise MCPToolError(
            "validation_error",
            f"Structure folder does not exist or is not a directory: '{folder_path}'.",
        )

    pattern = "**/*" if recursive else "*"
    readable: list[Path] = []
    seen_stems: dict[str, Path] = {}
    for candidate in sorted(filter(Path.is_file, folder_path.glob(pattern))):
        try:
            ase.io.read(candidate)
        except Exception:
            continue
        stem = candidate.stem
        if not stem.isidentifier():
            raise MCPToolError(
                "validation_error",
                f"Structure filename stem '{stem}' ('{candidate.name}') is not a "
                "valid AiiDA link label (must be a valid Python identifier).",
            )
        if stem in seen_stems:
            raise MCPToolError(
                "validation_error",
                f"Duplicate structure stem '{stem}' found in '{seen_stems[stem]}' "
                f"and '{candidate}'. Rename files to have unique stems.",
            )
        seen_stems[stem] = candidate
        readable.append(candidate)

    if not readable:
        raise MCPToolError(
            "validation_error",
            f"Folder '{folder_path}' is empty or contains no readable structure files.",
        )
    return readable


def build_ht_workgraph(
    *,
    entry_point: str,
    folder: str | Path,
    calc_inputs: dict[str, Any],
    recursive: bool = True,
    max_number_jobs: int = 4,
) -> WorkGraph:
    """
    Build an ``aiida-workgraph`` high-throughput screening ``WorkGraph``.

    Parameters
    ----------
    entry_point : str
        Normalized ``mlip.*`` calculation entry point (excluding ``mlip.train``).
    folder : str | Path
        Validated directory containing structure files.
    calc_inputs : dict[str, Any]
        Shared inputs dictionary (without ``struct``); copied before passing
        to ``build_ht_calc``.
    recursive : bool
        Whether to scan ``folder`` recursively.
    max_number_jobs : int
        Maximum concurrent jobs allowed by the WorkGraph.

    Returns
    -------
    WorkGraph
        Configured ``WorkGraph`` ready for ``wg.submit()``.
    """
    if max_number_jobs < 1:
        raise MCPToolError(
            "validation_error",
            f"'max_number_jobs' must be >= 1, got {max_number_jobs}.",
        )

    from aiida_mlip.workflows.ht_workgraph import build_ht_calc

    calc_cls = CalculationFactory(entry_point)
    final_key = (
        "final_structure"
        if entry_point in STRUCTURE_PRODUCING_CALCS
        else "results_dict"
    )
    wg = build_ht_calc(
        calc=calc_cls,
        folder=Path(folder).expanduser().resolve(),
        calc_inputs=dict(calc_inputs),
        final_struct_key=final_key,
        recursive=recursive,
    )
    wg.max_number_jobs = max_number_jobs
    return wg


def validate_chain_stages(
    stages: list[dict[str, Any]],
    *,
    code: Code,
    model: ModelData | DryRunModelRef,
    struct: StructureData,
    device: str = "cpu",
    hpc_preset: str | None = None,
    hpc_overrides: dict[str, Any] | None = None,
    presets_path: Path | str | None = None,
    config_cache_dir: Path | None = None,
) -> list[tuple[str, str, dict[str, Any]]]:
    """
    Validate and build input dicts for every stage in ``mlip_chain``.

    Parameters
    ----------
    stages : list[dict[str, Any]]
        Non-empty list of stage dicts ``{"calc": str, "params": dict, "name": str}``.
    code : Code
        Registered AiiDA ``Code`` node.
    model : ModelData | DryRunModelRef
        Resolved model node or dry-run placeholder.
    struct : StructureData
        Initial structure for stage 0.
    device : str
        Compute device.
    hpc_preset : str | None
        Optional named HPC preset.
    hpc_overrides : dict[str, Any] | None
        Optional ``metadata.options`` overrides.
    presets_path : Path | str | None
        Optional path to HPC presets YAML file.
    config_cache_dir : Path | None
        Optional directory for synthesized YAML config files.

    Returns
    -------
    list[tuple[str, str, dict[str, Any]]]
        List of ``(task_name, entry_point, inputs_dict)`` tuples.

    Raises
    ------
    MCPToolError
        If ``stages`` is empty, contains unknown keys, uses ``mlip.train``,
        or places a non-structure-producing calculation before another stage.
    """
    if not stages:
        raise MCPToolError(
            "validation_error", "'stages' must be a non-empty list of stage dicts."
        )

    built: list[tuple[str, str, dict[str, Any]]] = []
    used_names: set[str] = set()
    for idx, stage in enumerate(stages):
        if not isinstance(stage, dict) or "calc" not in stage:
            raise MCPToolError(
                "validation_error",
                f"Stage {idx} must be a dict containing a 'calc' key.",
            )
        unknown_keys = set(stage.keys()) - ALLOWED_STAGE_KEYS
        if unknown_keys:
            raise MCPToolError(
                "validation_error",
                f"Stage {idx} contains unsupported keys: {sorted(unknown_keys)}. "
                f"Allowed keys: {sorted(ALLOWED_STAGE_KEYS)}.",
            )
        params = stage.get("params") or {}
        if not isinstance(params, dict):
            raise MCPToolError(
                "validation_error", f"Stage {idx} 'params' must be a dictionary."
            )

        entry_point, stage_inputs = build_calc_inputs_for_alias(
            stage["calc"],
            code=code,
            model=model,
            struct=struct if idx == 0 else None,
            params=params,
            device=device,
            hpc_preset=hpc_preset,
            hpc_overrides=hpc_overrides,
            presets_path=presets_path,
            config_cache_dir=config_cache_dir,
        )
        if idx < len(stages) - 1 and entry_point not in STRUCTURE_PRODUCING_CALCS:
            raise MCPToolError(
                "validation_error",
                f"Stage {idx} ('{entry_point}') does not produce a 'final_structure' "
                "output and cannot precede another stage. Only 'opt' and 'md' can "
                "feed downstream stages.",
            )

        short = entry_point.split(".", maxsplit=1)[-1]
        task_name = stage.get("name") or f"stage_{idx}_{short}"
        if not isinstance(task_name, str) or not task_name.isidentifier():
            raise MCPToolError(
                "validation_error",
                f"Stage {idx} name '{task_name}' must be a valid Python identifier.",
            )
        if task_name in used_names:
            raise MCPToolError(
                "validation_error", f"Duplicate stage name '{task_name}' in chain."
            )
        used_names.add(task_name)
        built.append((task_name, entry_point, stage_inputs))

    return built


def build_chain_workgraph(
    validated_stages: list[tuple[str, str, dict[str, Any]]],
) -> WorkGraph:
    """
    Construct a chained ``WorkGraph`` from validated stage specifications.

    Parameters
    ----------
    validated_stages : list[tuple[str, str, dict[str, Any]]]
        Output of ``validate_chain_stages``.

    Returns
    -------
    WorkGraph
        Configured ``WorkGraph`` with sequential ``final_structure -> struct``
        links.
    """
    from aiida_workgraph import WorkGraph

    with WorkGraph("mlip_chain") as wg:
        prev_task = None
        last_entry_point = ""
        for task_name, entry_point, stage_inputs in validated_stages:
            calc_cls = CalculationFactory(entry_point)
            task_kwargs = dict(stage_inputs)
            if prev_task is not None:
                task_kwargs["struct"] = prev_task.outputs.final_structure
            task = wg.add_task(calc_cls, name=task_name, **task_kwargs)
            prev_task = task
            last_entry_point = entry_point

        if prev_task is not None:
            wg.outputs.final_result = prev_task.outputs.results_dict
            if last_entry_point in STRUCTURE_PRODUCING_CALCS:
                wg.outputs.final_structure = prev_task.outputs.final_structure
        return wg
