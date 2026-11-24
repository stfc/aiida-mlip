"""FastMCP server, tools, resources, prompts, and CLI for ``aiida-mlip``."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Literal

from aiida.common import exceptions as aiida_exc
from aiida.orm import ProcessNode, StructureData, load_node
from aiida.plugins import CalculationFactory
import yaml

from aiida_mlip.mcp.builders import (
    build_calc_inputs_for_alias,
    build_chain_workgraph,
    build_descriptors_inputs,
    build_ht_workgraph,
    build_md_inputs,
    build_opt_inputs,
    build_ph_inputs,
    build_sp_inputs,
    build_train_inputs,
    validate_chain_stages,
    validate_ht_folder,
)
from aiida_mlip.mcp.digests import build_digest, diagnose_node_impl, read_node_logs
from aiida_mlip.mcp.lifecycle import (
    cancel_process,
    ensure_profile,
    get_process_status,
    list_codes_impl,
    list_profiles_impl,
    submit_with_lease,
)
from aiida_mlip.mcp.provenance import (
    compute_intent_hash,
    file_sha256,
    model_fingerprint,
    structure_fingerprint,
)
from aiida_mlip.mcp.refs import (
    DryRunModelRef,
    MCPToolError,
    resolve_code,
    resolve_model,
    resolve_structure,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def _safe_call(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """
    Execute a tool handler and convert domain/AiiDA errors into error dicts.

    Parameters
    ----------
    fn : Callable[[], dict[str, Any]]
        Zero-argument callable executing the tool logic.

    Returns
    -------
    dict[str, Any]
        Tool result dictionary or structured error envelope.
    """
    try:
        return fn()
    except MCPToolError as exc:
        return exc.to_dict()
    except (
        aiida_exc.InputValidationError,
        ValueError,
        FileNotFoundError,
        OSError,
        yaml.YAMLError,
    ) as exc:
        return MCPToolError("validation_error", str(exc)).to_dict()
    except aiida_exc.AiidaException as exc:
        return MCPToolError("submission_error", str(exc)).to_dict()


def _register_calc_tools(
    mcp: FastMCP,
    default_profile: str | None,
    presets_path: str | None,
) -> None:
    """Register the 6 explicit ``mlip.*`` calculation tools on ``mcp``."""

    def _run_base_calc(
        *,
        entry_point: str,
        builder: Callable[..., dict[str, Any]],
        struct_ref: str,
        model_ref: str,
        arch: str | None,
        code_label: str,
        device: str,
        hpc_preset: str | None,
        hpc_overrides: dict[str, Any] | None,
        janus_overrides: dict[str, Any] | None,
        profile: str | None,
        wait_seconds: float,
        dedupe: Literal["reuse", "strict", "force"],
        dry_run: bool,
        extra_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        ensure_profile(profile or default_profile)
        struct_node = resolve_structure(struct_ref)
        model_node = resolve_model(model_ref, arch=arch, dry_run=dry_run)
        code_node = resolve_code(code_label)
        inputs = builder(
            code=code_node,
            model=model_node,
            struct=struct_node,
            device=device,
            janus_overrides=janus_overrides,
            hpc_preset=hpc_preset,
            hpc_overrides=hpc_overrides,
            presets_path=presets_path,
            **extra_kwargs,
        )
        intent_hash = compute_intent_hash(
            entry_point,
            {
                "code_label": code_node.full_label,
                "device": device,
                "structure": structure_fingerprint(struct_node),
                "model": model_fingerprint(model_node),
                "config": inputs["config"].as_dictionary,
                "extra": extra_kwargs,
                "metadata_options": inputs["metadata"]["options"],
            },
        )
        return submit_with_lease(
            CalculationFactory(entry_point),
            inputs,
            entry_point=entry_point,
            intent_hash=intent_hash,
            dedupe=dedupe,
            wait_seconds=wait_seconds,
            dry_run=dry_run,
            validated_summary={
                "code": code_node.full_label,
                "arch": model_node.architecture,
                "device": device,
                "formula": struct_node.get_formula(),
            },
            skip_model_port_check=isinstance(model_node, DryRunModelRef),
        )

    @mcp.tool()
    def mlip_sp(
        struct: str,
        model: str,
        arch: str | None = None,
        properties: list[str] | None = None,
        calc_kwargs: dict[str, Any] | None = None,
        code: str = "janus@localhost",
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        hpc_preset: str | None = None,
        hpc_overrides: dict[str, Any] | None = None,
        janus_overrides: dict[str, Any] | None = None,
        profile: str | None = None,
        wait_seconds: float = 15.0,
        dedupe: Literal["reuse", "strict", "force"] = "reuse",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Run a provenance-tracked single-point calculation (``mlip.sp``)."""
        return _safe_call(
            lambda: _run_base_calc(
                entry_point="mlip.sp",
                builder=build_sp_inputs,
                struct_ref=struct,
                model_ref=model,
                arch=arch,
                code_label=code,
                device=device,
                hpc_preset=hpc_preset,
                hpc_overrides=hpc_overrides,
                janus_overrides=janus_overrides,
                profile=profile,
                wait_seconds=wait_seconds,
                dedupe=dedupe,
                dry_run=dry_run,
                extra_kwargs={"properties": properties, "calc_kwargs": calc_kwargs},
            )
        )

    @mcp.tool()
    def mlip_opt(
        struct: str,
        model: str,
        arch: str | None = None,
        fmax: float = 0.01,
        steps: int = 1000,
        opt_cell_fully: bool = True,
        opt_cell_lengths: bool = False,
        filter_func: str = "FrechetCellFilter",
        pressure: float | None = None,
        minimize_kwargs: dict[str, Any] | None = None,
        calc_kwargs: dict[str, Any] | None = None,
        code: str = "janus@localhost",
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        hpc_preset: str | None = None,
        hpc_overrides: dict[str, Any] | None = None,
        janus_overrides: dict[str, Any] | None = None,
        profile: str | None = None,
        wait_seconds: float = 15.0,
        dedupe: Literal["reuse", "strict", "force"] = "reuse",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Run a provenance-tracked geometry optimization (``mlip.opt``)."""
        return _safe_call(
            lambda: _run_base_calc(
                entry_point="mlip.opt",
                builder=build_opt_inputs,
                struct_ref=struct,
                model_ref=model,
                arch=arch,
                code_label=code,
                device=device,
                hpc_preset=hpc_preset,
                hpc_overrides=hpc_overrides,
                janus_overrides=janus_overrides,
                profile=profile,
                wait_seconds=wait_seconds,
                dedupe=dedupe,
                dry_run=dry_run,
                extra_kwargs={
                    "fmax": fmax,
                    "steps": steps,
                    "opt_cell_fully": opt_cell_fully,
                    "opt_cell_lengths": opt_cell_lengths,
                    "filter_func": filter_func,
                    "pressure": pressure,
                    "minimize_kwargs": minimize_kwargs,
                    "calc_kwargs": calc_kwargs,
                },
            )
        )

    @mcp.tool()
    def mlip_md(
        struct: str,
        model: str,
        ensemble: Literal["nve", "nvt", "npt"] = "nvt",
        arch: str | None = None,
        temp: float | None = 300.0,
        steps: int = 1000,
        timestep: float = 1.0,
        pressure: float | None = None,
        md_kwargs: dict[str, Any] | None = None,
        calc_kwargs: dict[str, Any] | None = None,
        code: str = "janus@localhost",
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        hpc_preset: str | None = None,
        hpc_overrides: dict[str, Any] | None = None,
        janus_overrides: dict[str, Any] | None = None,
        profile: str | None = None,
        wait_seconds: float = 15.0,
        dedupe: Literal["reuse", "strict", "force"] = "reuse",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Run a provenance-tracked molecular dynamics simulation (``mlip.md``)."""
        return _safe_call(
            lambda: _run_base_calc(
                entry_point="mlip.md",
                builder=build_md_inputs,
                struct_ref=struct,
                model_ref=model,
                arch=arch,
                code_label=code,
                device=device,
                hpc_preset=hpc_preset,
                hpc_overrides=hpc_overrides,
                janus_overrides=janus_overrides,
                profile=profile,
                wait_seconds=wait_seconds,
                dedupe=dedupe,
                dry_run=dry_run,
                extra_kwargs={
                    "ensemble": ensemble,
                    "temp": temp,
                    "steps": steps,
                    "timestep": timestep,
                    "pressure": pressure,
                    "md_kwargs": md_kwargs,
                    "calc_kwargs": calc_kwargs,
                },
            )
        )

    @mcp.tool()
    def mlip_ph(
        struct: str,
        model: str,
        arch: str | None = None,
        supercell: str = "2 2 2",
        displacement: float = 0.01,
        n_qpoints: int = 100,
        dos: bool = False,
        pdos: bool = False,
        bands: bool = False,
        symmetrize: bool = False,
        no_hdf5: bool = False,
        calc_kwargs: dict[str, Any] | None = None,
        code: str = "janus@localhost",
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        hpc_preset: str | None = None,
        hpc_overrides: dict[str, Any] | None = None,
        janus_overrides: dict[str, Any] | None = None,
        profile: str | None = None,
        wait_seconds: float = 15.0,
        dedupe: Literal["reuse", "strict", "force"] = "reuse",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Run a provenance-tracked harmonic phonon calculation (``mlip.ph``)."""
        return _safe_call(
            lambda: _run_base_calc(
                entry_point="mlip.ph",
                builder=build_ph_inputs,
                struct_ref=struct,
                model_ref=model,
                arch=arch,
                code_label=code,
                device=device,
                hpc_preset=hpc_preset,
                hpc_overrides=hpc_overrides,
                janus_overrides=janus_overrides,
                profile=profile,
                wait_seconds=wait_seconds,
                dedupe=dedupe,
                dry_run=dry_run,
                extra_kwargs={
                    "supercell": supercell,
                    "displacement": displacement,
                    "n_qpoints": n_qpoints,
                    "dos": dos,
                    "pdos": pdos,
                    "bands": bands,
                    "symmetrize": symmetrize,
                    "no_hdf5": no_hdf5,
                    "calc_kwargs": calc_kwargs,
                },
            )
        )

    @mcp.tool()
    def mlip_descriptors(
        struct: str,
        model: str,
        arch: str | None = None,
        invariants_only: bool = True,
        calc_per_element: bool = False,
        calc_per_atom: bool = False,
        calc_kwargs: dict[str, Any] | None = None,
        code: str = "janus@localhost",
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        hpc_preset: str | None = None,
        hpc_overrides: dict[str, Any] | None = None,
        janus_overrides: dict[str, Any] | None = None,
        profile: str | None = None,
        wait_seconds: float = 15.0,
        dedupe: Literal["reuse", "strict", "force"] = "reuse",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Extract provenance-tracked MLIP descriptors (``mlip.descriptors``)."""
        return _safe_call(
            lambda: _run_base_calc(
                entry_point="mlip.descriptors",
                builder=build_descriptors_inputs,
                struct_ref=struct,
                model_ref=model,
                arch=arch,
                code_label=code,
                device=device,
                hpc_preset=hpc_preset,
                hpc_overrides=hpc_overrides,
                janus_overrides=janus_overrides,
                profile=profile,
                wait_seconds=wait_seconds,
                dedupe=dedupe,
                dry_run=dry_run,
                extra_kwargs={
                    "invariants_only": invariants_only,
                    "calc_per_element": calc_per_element,
                    "calc_per_atom": calc_per_atom,
                    "calc_kwargs": calc_kwargs,
                },
            )
        )

    @mcp.tool()
    def mlip_train(
        name: str,
        train_file: str,
        valid_file: str,
        test_file: str,
        arch: str | None = None,
        fine_tune: bool = False,
        foundation_model: str | None = None,
        train_overrides: dict[str, Any] | None = None,
        code: str = "janus@localhost",
        hpc_preset: str | None = None,
        hpc_overrides: dict[str, Any] | None = None,
        profile: str | None = None,
        wait_seconds: float = 15.0,
        dedupe: Literal["reuse", "strict", "force"] = "reuse",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Train or fine-tune an MLIP model with provenance (``mlip.train``)."""

        def _impl() -> dict[str, Any]:
            ensure_profile(profile or default_profile)
            code_node = resolve_code(code)
            fmodel = (
                resolve_model(foundation_model, arch=arch, dry_run=dry_run)
                if foundation_model is not None
                else None
            )
            inputs = build_train_inputs(
                code=code_node,
                name=name,
                train_file=train_file,
                valid_file=valid_file,
                test_file=test_file,
                arch=arch,
                fine_tune=fine_tune,
                foundation_model=fmodel,
                train_overrides=train_overrides,
                hpc_preset=hpc_preset,
                hpc_overrides=hpc_overrides,
                presets_path=presets_path,
            )
            cfg_dict = inputs["mlip_config"].as_dictionary
            dataset_hashes = {
                key: file_sha256(Path(cfg_dict[key]))
                for key in ("train_file", "valid_file", "test_file")
            }
            intent_hash = compute_intent_hash(
                "mlip.train",
                {
                    "code_label": code_node.full_label,
                    "arch": inputs["arch"].value,
                    "fine_tune": fine_tune,
                    "foundation_model": (
                        model_fingerprint(fmodel) if fmodel is not None else None
                    ),
                    "mlip_config": cfg_dict,
                    "dataset_hashes": dataset_hashes,
                    "metadata_options": inputs["metadata"]["options"],
                },
            )
            return submit_with_lease(
                CalculationFactory("mlip.train"),
                inputs,
                entry_point="mlip.train",
                intent_hash=intent_hash,
                dedupe=dedupe,
                wait_seconds=wait_seconds,
                dry_run=dry_run,
                validated_summary={
                    "code": code_node.full_label,
                    "name": name,
                    "arch": inputs["arch"].value,
                    "fine_tune": fine_tune,
                },
                skip_model_port_check=isinstance(fmodel, DryRunModelRef),
            )

        return _safe_call(_impl)


def _register_workflow_tools(
    mcp: FastMCP,
    default_profile: str | None,
    presets_path: str | None,
) -> None:
    """Register ``mlip_ht_screen`` and ``mlip_chain`` WorkGraph tools."""

    @mcp.tool()
    def mlip_ht_screen(
        calc: str,
        folder: str,
        model: str,
        arch: str | None = None,
        calc_params: dict[str, Any] | None = None,
        recursive: bool = True,
        max_number_jobs: int = 4,
        code: str = "janus@localhost",
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        hpc_preset: str | None = None,
        hpc_overrides: dict[str, Any] | None = None,
        profile: str | None = None,
        wait_seconds: float = 15.0,
        dedupe: Literal["reuse", "strict", "force"] = "reuse",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Run a high-throughput WorkGraph across a folder of structures."""

        def _impl() -> dict[str, Any]:
            ensure_profile(profile or default_profile)
            if max_number_jobs < 1:
                raise MCPToolError(
                    "validation_error",
                    f"'max_number_jobs' must be >= 1, got {max_number_jobs}.",
                )
            files = validate_ht_folder(folder, recursive=recursive)
            model_node = resolve_model(model, arch=arch, dry_run=dry_run)
            code_node = resolve_code(code)
            entry_point, shared_inputs = build_calc_inputs_for_alias(
                calc,
                code=code_node,
                model=model_node,
                struct=None,
                params=calc_params,
                device=device,
                hpc_preset=hpc_preset,
                hpc_overrides=hpc_overrides,
                presets_path=presets_path,
            )
            file_hashes = [(f.stem, file_sha256(f)) for f in files]
            intent_hash = compute_intent_hash(
                f"workgraph.ht.{entry_point}",
                {
                    "entry_point": entry_point,
                    "files": file_hashes,
                    "model": model_fingerprint(model_node),
                    "calc_params": calc_params or {},
                    "code_label": code_node.full_label,
                    "device": device,
                    "metadata_options": shared_inputs["metadata"]["options"],
                },
            )
            return submit_with_lease(
                lambda: build_ht_workgraph(
                    entry_point=entry_point,
                    folder=folder,
                    calc_inputs=shared_inputs,
                    recursive=recursive,
                    max_number_jobs=max_number_jobs,
                ),
                None,
                entry_point=f"workgraph.ht.{entry_point}",
                intent_hash=intent_hash,
                dedupe=dedupe,
                wait_seconds=wait_seconds,
                dry_run=dry_run,
                validated_summary={
                    "entry_point": entry_point,
                    "n_structures": len(files),
                    "stems": [f.stem for f in files[:20]],
                    "max_number_jobs": max_number_jobs,
                },
            )

        return _safe_call(_impl)

    @mcp.tool()
    def mlip_chain(
        struct: str,
        model: str,
        stages: list[dict[str, Any]],
        arch: str | None = None,
        code: str = "janus@localhost",
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        hpc_preset: str | None = None,
        hpc_overrides: dict[str, Any] | None = None,
        profile: str | None = None,
        wait_seconds: float = 15.0,
        dedupe: Literal["reuse", "strict", "force"] = "reuse",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Chain multiple calculations in an ``aiida-workgraph`` pipeline."""

        def _impl() -> dict[str, Any]:
            ensure_profile(profile or default_profile)
            struct_node = resolve_structure(struct)
            model_node = resolve_model(model, arch=arch, dry_run=dry_run)
            code_node = resolve_code(code)
            validated = validate_chain_stages(
                stages,
                code=code_node,
                model=model_node,
                struct=struct_node,
                device=device,
                hpc_preset=hpc_preset,
                hpc_overrides=hpc_overrides,
                presets_path=presets_path,
            )
            normalized_stages = [
                {
                    "name": name,
                    "entry_point": ep,
                    "params": stage.get("params") or {},
                }
                for (name, ep, _), stage in zip(validated, stages, strict=True)
            ]
            intent_hash = compute_intent_hash(
                "workgraph.chain",
                {
                    "structure": structure_fingerprint(struct_node),
                    "model": model_fingerprint(model_node),
                    "stages": normalized_stages,
                    "code_label": code_node.full_label,
                    "device": device,
                    "metadata_options": validated[0][2]["metadata"]["options"],
                },
            )
            return submit_with_lease(
                lambda: build_chain_workgraph(validated),
                None,
                entry_point="workgraph.chain",
                intent_hash=intent_hash,
                dedupe=dedupe,
                wait_seconds=wait_seconds,
                dry_run=dry_run,
                validated_summary={
                    "stages": [
                        {"name": name, "entry_point": ep} for name, ep, _ in validated
                    ],
                    "formula": struct_node.get_formula(),
                },
            )

        return _safe_call(_impl)


def _register_lifecycle_tools(
    mcp: FastMCP,
    default_profile: str | None,
) -> None:
    """Register status, cancel, diagnose, profile, and code discovery tools."""

    @mcp.tool()
    def mlip_status(pk: int, profile: str | None = None) -> dict[str, Any]:
        """Get status, bounded digest, and diagnostics for a process PK."""

        def _impl() -> dict[str, Any]:
            ensure_profile(profile or default_profile)
            return get_process_status(pk)

        return _safe_call(_impl)

    @mcp.tool()
    def mlip_cancel(pk: int, profile: str | None = None) -> dict[str, Any]:
        """Request cancellation of a running AiiDA process by PK."""

        def _impl() -> dict[str, Any]:
            ensure_profile(profile or default_profile)
            return cancel_process(pk)

        return _safe_call(_impl)

    @mcp.tool()
    def diagnose_node(pk: int, profile: str | None = None) -> dict[str, Any]:
        """Diagnose a failed or excepted AiiDA process node by PK."""

        def _impl() -> dict[str, Any]:
            ensure_profile(profile or default_profile)
            try:
                node = load_node(pk)
            except aiida_exc.NotExistent as exc:
                raise MCPToolError(
                    "node_not_found", f"Node PK {pk} was not found."
                ) from exc
            if not isinstance(node, ProcessNode):
                raise MCPToolError(
                    "validation_error",
                    f"Node PK {pk} is {type(node).__name__}, expected ProcessNode.",
                )
            return diagnose_node_impl(node)

        return _safe_call(_impl)

    @mcp.tool()
    def list_profiles() -> dict[str, Any]:
        """List configured AiiDA profiles, active profile, and daemon status."""
        return _safe_call(list_profiles_impl)

    @mcp.tool()
    def list_codes(
        plugin: str | None = None,
        profile: str | None = None,
    ) -> dict[str, Any]:
        """List registered AiiDA codes in the active profile."""

        def _impl() -> dict[str, Any]:
            ensure_profile(profile or default_profile)
            return list_codes_impl(plugin=plugin)

        return _safe_call(_impl)


def _serialize_structure(struct: StructureData) -> dict[str, Any]:
    """Serialize an AiiDA ``StructureData`` to a bounded JSON dictionary."""
    atoms = struct.get_ase()
    sites = [
        {
            "symbol": sym,
            "position": [round(float(c), 6) for c in pos],
        }
        for sym, pos in zip(
            atoms.get_chemical_symbols()[:50],
            atoms.get_positions().tolist()[:50],
            strict=False,
        )
    ]
    return {
        "pk": struct.pk,
        "formula": struct.get_formula(),
        "n_atoms": len(atoms),
        "pbc": [bool(x) for x in atoms.get_pbc()],
        "cell": [
            [round(float(v), 6) for v in row] for row in atoms.get_cell().tolist()
        ],
        "sites": sites,
        "truncated": len(atoms) > 50,
    }


def _register_resources(mcp: FastMCP, default_profile: str | None) -> None:
    """Register the 4 read-only ``aiida://`` MCP resources."""

    @mcp.resource("aiida://codes", mime_type="application/json")
    def resource_codes() -> str:
        """Return JSON list of registered AiiDA codes."""
        res = _safe_call(
            lambda: (ensure_profile(default_profile), list_codes_impl())[1]
        )
        return json.dumps(res)

    @mcp.resource("aiida://node/{pk}/digest", mime_type="application/json")
    def resource_digest(pk: int) -> str:
        """Return JSON bounded Physics Digest for a process node."""

        def _impl() -> dict[str, Any]:
            ensure_profile(default_profile)
            try:
                node = load_node(pk)
            except aiida_exc.NotExistent as exc:
                raise MCPToolError(
                    "node_not_found", f"Node PK {pk} was not found."
                ) from exc
            return build_digest(node)

        return json.dumps(_safe_call(_impl))

    @mcp.resource("aiida://node/{pk}/logs", mime_type="text/plain")
    def resource_logs(pk: int) -> str:
        """Return plain-text trailing logs for a process node."""
        try:
            ensure_profile(default_profile)
            node = load_node(pk)
            return read_node_logs(node, max_lines=100)
        except (MCPToolError, aiida_exc.AiidaException) as exc:
            return f"Error reading logs for PK {pk}: {exc}"

    @mcp.resource("aiida://node/{pk}/structure", mime_type="application/json")
    def resource_structure(pk: int) -> str:
        """Return compact JSON geometry for a structure or calculation node."""

        def _impl() -> dict[str, Any]:
            ensure_profile(default_profile)
            try:
                node = load_node(pk)
            except aiida_exc.NotExistent as exc:
                raise MCPToolError(
                    "node_not_found", f"Node PK {pk} was not found."
                ) from exc
            if isinstance(node, StructureData):
                return _serialize_structure(node)
            if isinstance(node, ProcessNode):
                if "final_structure" in node.outputs:
                    return _serialize_structure(node.outputs.final_structure)
                if "struct" in node.inputs:
                    return _serialize_structure(node.inputs.struct)
            return {"error": "no_structure_found", "pk": pk}

        return json.dumps(_safe_call(_impl))


def _register_prompts(mcp: FastMCP) -> None:
    """Register the 6 workflow prompt templates on ``mcp``."""

    @mcp.prompt()
    def sp_workflow(
        struct: str = "",
        model: str = "",
        code: str = "janus@localhost",
    ) -> str:
        """Guide an agent through a single-point MLIP calculation."""
        return (
            f"Run an aiida-mlip single-point calculation using mlip_sp.\n"
            f"- Structure: {struct or '<provide pk:, uuid:, or file:>'}\n"
            f"- Model: {model or '<provide pk:, file:, or uri:>'}\n"
            f"- Code: {code} (verify via aiida://codes)\n"
            "Start with dry_run=True to validate inputs, then submit."
        )

    @mcp.prompt()
    def opt_workflow(
        struct: str = "",
        model: str = "",
        code: str = "janus@localhost",
    ) -> str:
        """Guide an agent through a geometry optimization with FrechetCellFilter."""
        return (
            f"Run an aiida-mlip geometry optimization using mlip_opt.\n"
            f"- Structure: {struct or '<provide pk:, uuid:, or file:>'}\n"
            f"- Model: {model or '<provide pk:, file:, or uri:>'}\n"
            f"- Code: {code} (verify via aiida://codes)\n"
            "Prefer filter_func='FrechetCellFilter' for cell relaxation and "
            "validate with dry_run=True before submitting."
        )

    @mcp.prompt()
    def md_workflow(
        struct: str = "",
        model: str = "",
        ensemble: str = "nvt",
        code: str = "janus@localhost",
    ) -> str:
        """Guide an agent through an MD simulation."""
        return (
            f"Run an aiida-mlip MD simulation ({ensemble}) using mlip_md.\n"
            f"- Structure: {struct or '<provide pk:, uuid:, or file:>'}\n"
            f"- Model: {model or '<provide pk:, file:, or uri:>'}\n"
            f"- Code: {code} (verify via aiida://codes)\n"
            "Validate with dry_run=True first, then monitor with mlip_status."
        )

    @mcp.prompt()
    def phonons_workflow(
        struct: str = "",
        model: str = "",
        supercell: str = "2 2 2",
        code: str = "janus@localhost",
    ) -> str:
        """Guide an agent through a harmonic phonon calculation."""
        return (
            f"Run an aiida-mlip phonon calculation (supercell='{supercell}') "
            "using mlip_ph (or mlip_chain with opt -> ph).\n"
            f"- Structure: {struct or '<provide relaxed pk: or file:>'}\n"
            f"- Model: {model or '<provide pk:, file:, or uri:>'}\n"
            f"- Code: {code} (verify via aiida://codes)\n"
            "Validate with dry_run=True before submitting."
        )

    @mcp.prompt()
    def descriptors_workflow(
        struct: str = "",
        model: str = "",
        code: str = "janus@localhost",
    ) -> str:
        """Guide an agent through MLIP descriptor extraction."""
        return (
            f"Extract MLIP descriptors using mlip_descriptors.\n"
            f"- Structure: {struct or '<provide pk:, uuid:, or file:>'}\n"
            f"- Model: {model or '<provide pk:, file:, or uri:>'}\n"
            f"- Code: {code} (verify via aiida://codes)\n"
            "Validate with dry_run=True before submitting."
        )

    @mcp.prompt()
    def train_workflow(
        name: str = "",
        train_file: str = "",
        valid_file: str = "",
        test_file: str = "",
        code: str = "janus@localhost",
    ) -> str:
        """Guide an agent through MLIP model training or fine-tuning."""
        return (
            f"Train or fine-tune an MLIP model using mlip_train.\n"
            f"- Name: {name or '<model name>'}\n"
            f"- Datasets: train={train_file or '<path>'}, "
            f"valid={valid_file or '<path>'}, test={test_file or '<path>'}\n"
            f"- Code: {code} (verify via aiida://codes)\n"
            "Always run with dry_run=True first to verify dataset paths and arch."
        )


def create_server(
    default_profile: str | None = None,
    presets_path: str | None = None,
) -> FastMCP:
    """
    Create and configure the ``aiida-mlip`` FastMCP server.

    Parameters
    ----------
    default_profile : str | None
        Default AiiDA profile name to load when a tool call omits ``profile``.
    presets_path : str | None
        Optional path to a YAML file defining named HPC ``metadata.options``
        presets.

    Returns
    -------
    FastMCP
        Configured FastMCP server instance.

    Raises
    ------
    ImportError
        If the optional ``mcp`` package is not installed.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "The 'mcp' extra is required. Install with: "
            "uv pip install 'aiida-mlip[mcp]'"
        ) from exc

    mcp = FastMCP("aiida-mlip")
    _register_calc_tools(mcp, default_profile, presets_path)
    _register_workflow_tools(mcp, default_profile, presets_path)
    _register_lifecycle_tools(mcp, default_profile)
    _register_resources(mcp, default_profile)
    _register_prompts(mcp)
    return mcp


def main(argv: list[str] | None = None) -> None:
    """
    Run the ``aiida-mlip`` MCP server over ``stdio``.

    Parameters
    ----------
    argv : list[str] | None
        Optional command-line argument list (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(
        prog="aiida-mlip-mcp",
        description="Model Context Protocol (MCP) server for aiida-mlip.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Default AiiDA profile name to load.",
    )
    parser.add_argument(
        "--presets",
        default=None,
        help="Path to YAML file with named HPC metadata.options presets.",
    )
    args = parser.parse_args(argv)

    try:
        server = create_server(
            default_profile=args.profile,
            presets_path=args.presets,
        )
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    server.run(transport="stdio")
