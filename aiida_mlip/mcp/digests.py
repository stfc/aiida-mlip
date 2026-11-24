"""Bounded physics digests, log readers, and failure diagnostics for MCP."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from aiida.orm import ProcessNode
import numpy as np

from aiida_mlip.mcp.provenance import INTENT_HASH_EXTRA, count_failed_attempts
from aiida_mlip.mcp.refs import MCPToolError

MAX_DIGEST_BYTES = 2048
STRUCTURAL_ATOMS_KEYS = frozenset({"numbers", "positions", "cell", "pbc", "info"})

_FAILURE_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "cuda_oom",
        re.compile(r"CUDA out of memory|OutOfMemoryError", re.IGNORECASE),
        "Reduce supercell/batch size or retry with device='cpu'.",
    ),
    (
        "walltime_exceeded",
        re.compile(r"DUE TO TIME LIMIT|SIGTERM|walltime", re.IGNORECASE),
        "Increase 'max_wallclock_seconds' in hpc_overrides or reduce steps.",
    ),
    (
        "tracker_error",
        re.compile(r"codecarbon|pynvml|emissions.*tracker", re.IGNORECASE),
        "Ensure 'tracker: false' is set in janus_overrides.",
    ),
    (
        "invalid_cli_option",
        re.compile(
            r"No such option|unrecognized arguments|Invalid value", re.IGNORECASE
        ),
        "Check janus_overrides, md_kwargs, or minimize_kwargs key names.",
    ),
    (
        "nan_blowup",
        re.compile(r"\bnan\b|explosion|forces.*\binf\b", re.IGNORECASE),
        "Reduce MD timestep or relax the input structure with mlip_opt first.",
    ),
)


def build_resource_uris(node: ProcessNode) -> dict[str, str]:
    """
    Build standard ``aiida://`` MCP resource URIs for a ``ProcessNode``.

    Parameters
    ----------
    node : ProcessNode
        Stored AiiDA process node.

    Returns
    -------
    dict[str, str]
        Mapping of resource role names to ``aiida://`` URIs.
    """
    pk = node.pk
    return {
        "digest": f"aiida://node/{pk}/digest",
        "logs": f"aiida://node/{pk}/logs",
        "structure": f"aiida://node/{pk}/structure",
    }


def _find_suffix(mapping: dict[str, Any], suffix: str) -> Any:
    """
    Return the first value in ``mapping`` whose key ends with ``suffix``.

    Parameters
    ----------
    mapping : dict[str, Any]
        Dictionary to search.
    suffix : str
        Key suffix to match.

    Returns
    -------
    Any
        Matching value, or ``None`` if not found.
    """
    for key, val in mapping.items():
        if key.endswith(suffix):
            return val
    return None


def _force_norms(forces: Any) -> tuple[float | None, float | None]:
    """
    Compute ``(max_force, rms_force)`` from a nested list of force vectors.

    Parameters
    ----------
    forces : Any
        Sequence of 3-component force vectors.

    Returns
    -------
    tuple[float | None, float | None]
        Maximum and root-mean-square force magnitudes, or ``(None, None)``.
    """
    if not isinstance(forces, list) or not forces:
        return None, None
    norms: list[float] = []
    for vec in forces:
        if isinstance(vec, list) and len(vec) == 3:
            comps = [float(c) for c in vec]
            if not all(math.isfinite(c) for c in comps):
                return None, None
            norms.append(math.sqrt(sum(c * c for c in comps)))
    if not norms:
        return None, None
    rms = math.sqrt(sum(n * n for n in norms) / len(norms))
    return round(max(norms), 6), round(rms, 6)


def _flatten_numeric(value: Any) -> list[float]:
    """
    Flatten nested numeric lists/scalars into a list of finite floats.

    Parameters
    ----------
    value : Any
        Scalar or nested list of numbers.

    Returns
    -------
    list[float]
        Flattened list of finite float values.
    """
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        fval = float(value)
        return [fval] if math.isfinite(fval) else []
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_numeric(item))
        return out
    return []


def _sanitize_json(value: Any) -> Any:
    """Replace non-finite floats with ``None`` for strict RFC 8259 JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(v) for v in value]
    return value


def _has_non_finite(value: Any) -> bool:
    """
    Return ``True`` if any numeric leaf in ``value`` is NaN or infinite.

    Parameters
    ----------
    value : Any
        Scalar, list, or dict to inspect.

    Returns
    -------
    bool
        ``True`` if NaN or Inf is detected.
    """
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    if isinstance(value, dict):
        return any(_has_non_finite(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_non_finite(v) for v in value)
    return False


def _enforce_size_cap(digest: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure JSON-serialized digest stays within ``MAX_DIGEST_BYTES``.

    Parameters
    ----------
    digest : dict[str, Any]
        Candidate digest dictionary.

    Returns
    -------
    dict[str, Any]
        Bounded digest dictionary (with ``truncated=True`` if trimmed).
    """
    digest = _sanitize_json(digest)
    raw = json.dumps(digest, default=str, allow_nan=False).encode("utf-8")
    if len(raw) <= MAX_DIGEST_BYTES:
        return digest

    trimmed = dict(digest)
    trimmed["truncated"] = True
    for drop_key in (
        "summary_stats",
        "results_keys",
        "summary",
        "final_metrics",
        "failed_children",
        "child_pks",
        "outputs",
        "descriptor_keys",
    ):
        if drop_key in trimmed:
            val = trimmed[drop_key]
            if isinstance(val, list) and len(val) > 5:
                trimmed[drop_key] = val[:5]
            elif isinstance(val, dict) and len(val) > 5:
                trimmed[drop_key] = dict(list(val.items())[:5])
            else:
                trimmed.pop(drop_key)
            if (
                len(json.dumps(trimmed, default=str).encode("utf-8"))
                <= MAX_DIGEST_BYTES
            ):
                return trimmed

    return {
        "pk": digest.get("pk"),
        "process_type": digest.get("process_type"),
        "truncated": True,
    }


def _outputs_dict(node: ProcessNode) -> dict[str, Any]:
    """Return a plain dict mapping output link labels to output nodes."""
    return {key: node.outputs[key] for key in node.outputs}


def _digest_sp(node: ProcessNode) -> dict[str, Any]:
    """Build bounded digest for ``mlip.sp``."""
    outputs = _outputs_dict(node)
    res = outputs["results_dict"].get_dict() if "results_dict" in outputs else {}
    info = res.get("info", {}) if isinstance(res.get("info"), dict) else {}
    max_f, rms_f = _force_norms(_find_suffix(res, "_forces"))
    return {
        "pk": node.pk,
        "process_type": "mlip.sp",
        "energy": _find_suffix(info, "_energy"),
        "n_atoms": len(res.get("numbers", [])),
        "max_force": max_f,
        "rms_force": rms_f,
        "stress_voigt": _find_suffix(info, "_stress"),
        "outputs": sorted(outputs.keys()),
    }


def _digest_opt(node: ProcessNode) -> dict[str, Any]:
    """Build bounded digest for ``mlip.opt``."""
    outputs = _outputs_dict(node)
    res = outputs["results_dict"].get_dict() if "results_dict" in outputs else {}
    info = res.get("info", {}) if isinstance(res.get("info"), dict) else {}
    max_f, _ = _force_norms(_find_suffix(res, "_forces"))
    n_steps = outputs["traj_output"].numsteps if "traj_output" in outputs else None
    final_struct = outputs.get("final_structure")
    init_struct = node.inputs.struct if "struct" in node.inputs else None

    init_vol = float(init_struct.get_cell_volume()) if init_struct is not None else None
    final_vol = (
        float(final_struct.get_cell_volume()) if final_struct is not None else None
    )
    vol_pct = (
        round(((final_vol - init_vol) / init_vol) * 100.0, 4)
        if init_vol is not None and final_vol is not None and init_vol > 0.0
        else None
    )
    return {
        "pk": node.pk,
        "process_type": "mlip.opt",
        "final_energy": _find_suffix(info, "_energy"),
        "max_force": max_f,
        "n_steps": n_steps,
        "final_structure_pk": final_struct.pk if final_struct is not None else None,
        "initial_volume": round(init_vol, 4) if init_vol is not None else None,
        "final_volume": round(final_vol, 4) if final_vol is not None else None,
        "volume_change_pct": vol_pct,
    }


def _digest_md(node: ProcessNode) -> dict[str, Any]:
    """Build bounded digest for ``mlip.md``."""
    outputs = _outputs_dict(node)
    res = outputs["results_dict"].get_dict() if "results_dict" in outputs else {}
    scalars = {k: v for k, v in res.items() if isinstance(v, (int, float, str, bool))}
    nan_detected = _has_non_finite(scalars)
    if not nan_detected and "traj_output" in outputs:
        traj = outputs["traj_output"]
        for getter in ("get_positions", "get_cells"):
            if hasattr(traj, getter):
                try:
                    arr = np.asarray(getattr(traj, getter)())
                    if not np.isfinite(arr).all():
                        nan_detected = True
                        break
                except Exception:
                    pass
    if not nan_detected and "stats_file" in outputs:
        try:
            text = outputs["stats_file"].get_content()
            if re.search(r"\b(nan|inf|-inf)\b", text, re.IGNORECASE):
                nan_detected = True
        except Exception:
            pass

    final_struct = outputs.get("final_structure")
    return {
        "pk": node.pk,
        "process_type": "mlip.md",
        "n_frames": outputs["traj_output"].numsteps
        if "traj_output" in outputs
        else None,
        "final_structure_pk": final_struct.pk if final_struct is not None else None,
        "summary": scalars,
        "nan_detected": nan_detected,
    }


def _digest_ph(node: ProcessNode) -> dict[str, Any]:
    """Build bounded digest for ``mlip.ph``."""
    outputs = _outputs_dict(node)
    res = outputs["results_dict"].get_dict() if "results_dict" in outputs else {}
    return {
        "pk": node.pk,
        "process_type": "mlip.ph",
        "supercell": node.inputs.supercell.value
        if "supercell" in node.inputs
        else None,
        "has_dos": "dos" in outputs,
        "has_pdos": "pdos" in outputs,
        "has_bands": "band_structure" in outputs,
        "has_force_constants": "force_constants" in outputs,
        "results_keys": sorted(res.keys())[:20],
    }


def _digest_descriptors(node: ProcessNode) -> dict[str, Any]:
    """Build bounded digest for ``mlip.descriptors``."""
    outputs = _outputs_dict(node)
    res = outputs["results_dict"].get_dict() if "results_dict" in outputs else {}
    info = res.get("info", {}) if isinstance(res.get("info"), dict) else {}

    candidates: dict[str, Any] = {}
    for key, val in info.items():
        if key.endswith("_descriptor"):
            candidates[f"info.{key}"] = val
    for key, val in res.items():
        if key not in STRUCTURAL_ATOMS_KEYS and key.endswith("_descriptors"):
            candidates[key] = val

    stats: dict[str, dict[str, float | int]] = {}
    for key, val in candidates.items():
        nums = _flatten_numeric(val)
        if nums:
            stats[key] = {
                "count": len(nums),
                "min": round(min(nums), 6),
                "max": round(max(nums), 6),
                "mean": round(sum(nums) / len(nums), 6),
            }

    return {
        "pk": node.pk,
        "process_type": "mlip.descriptors",
        "n_atoms": len(res.get("numbers", [])),
        "descriptor_keys": sorted(candidates.keys()),
        "summary_stats": stats,
    }


def _digest_train(node: ProcessNode) -> dict[str, Any]:
    """Build bounded digest for ``mlip.train``."""
    outputs = _outputs_dict(node)
    res = outputs["results_dict"].get_dict() if "results_dict" in outputs else {}
    metrics = {k: v for k, v in res.items() if isinstance(v, (int, float, str, bool))}
    return {
        "pk": node.pk,
        "process_type": "mlip.train",
        "trained_model_pk": outputs["model"].pk if "model" in outputs else None,
        "compiled_model_pk": outputs["compiled_model"].pk
        if "compiled_model" in outputs
        else None,
        "final_metrics": metrics,
    }


def _digest_workgraph(node: ProcessNode) -> dict[str, Any]:
    """Build bounded digest for a ``WorkGraphNode`` or caller process."""
    called = list(node.called)
    state_counts: dict[str, int] = {}
    failed_children: list[dict[str, Any]] = []
    for child in called:
        state = child.process_state.value if child.process_state else "unknown"
        state_counts[state] = state_counts.get(state, 0) + 1
        if child.is_failed or child.is_excepted or child.is_killed:
            failed_children.append(
                {
                    "pk": child.pk,
                    "exit_status": child.exit_status,
                    "exit_message": child.exit_message,
                }
            )
    return {
        "pk": node.pk,
        "process_type": node.process_type or "workgraph",
        "called_count": len(called),
        "state_counts": state_counts,
        "child_pks": [c.pk for c in called[:20]],
        "failed_children": failed_children[:10],
    }


def build_digest(node: ProcessNode) -> dict[str, Any]:
    """
    Build a bounded (<= 2048 bytes) physics digest for a ``ProcessNode``.

    Parameters
    ----------
    node : ProcessNode
        AiiDA ``CalcJobNode`` or ``WorkGraphNode``.

    Returns
    -------
    dict[str, Any]
        Compact JSON-serializable summary dictionary.

    Raises
    ------
    MCPToolError
        If ``node`` is not an AiiDA ``ProcessNode``.
    """
    if not isinstance(node, ProcessNode):
        raise MCPToolError(
            "validation_error",
            f"Node PK {getattr(node, 'pk', '?')} is {type(node).__name__}, "
            "expected a ProcessNode.",
        )

    ptype = node.process_type or ""
    dispatch = {
        "mlip.sp": _digest_sp,
        "mlip.opt": _digest_opt,
        "mlip.md": _digest_md,
        "mlip.ph": _digest_ph,
        "mlip.descriptors": _digest_descriptors,
        "mlip.train": _digest_train,
    }
    for suffix, handler in dispatch.items():
        if ptype.endswith(f":{suffix}") or ptype == suffix:
            return _enforce_size_cap(handler(node))

    if "workgraph" in ptype.lower() or list(node.called):
        return _enforce_size_cap(_digest_workgraph(node))

    return _enforce_size_cap(
        {
            "pk": node.pk,
            "process_type": ptype,
            "outputs": sorted(_outputs_dict(node).keys()),
        }
    )


def _read_retrieved_text(node: ProcessNode, filenames: tuple[str, ...]) -> list[str]:
    """Read available text files from ``node.outputs.retrieved``."""
    lines: list[str] = []
    if "retrieved" not in node.outputs:
        return lines
    retrieved = node.outputs.retrieved
    available = set(retrieved.list_object_names())
    for fname in filenames:
        if fname in available:
            try:
                content = retrieved.get_object_content(fname, mode="r")
                lines.append(f"--- {fname} ---")
                lines.extend(content.splitlines())
            except Exception:
                continue
    return lines


def read_node_logs(node: ProcessNode, *, max_lines: int = 100) -> str:
    """
    Read tail logs for ``BaseJanus``, ``Train``, failed, or ``WorkGraph`` nodes.

    Parameters
    ----------
    node : ProcessNode
        Target AiiDA process node.
    max_lines : int
        Maximum number of trailing lines to return.

    Returns
    -------
    str
        Combined log tail string.

    Raises
    ------
    MCPToolError
        If ``node`` is not a ``ProcessNode``.
    """
    if not isinstance(node, ProcessNode):
        raise MCPToolError(
            "validation_error",
            f"Node PK {getattr(node, 'pk', '?')} is {type(node).__name__}, "
            "expected a ProcessNode.",
        )

    lines: list[str] = []
    outputs = _outputs_dict(node)

    for key in ("log_output", "std_output"):
        if key in outputs:
            try:
                lines.append(f"--- {key} ---")
                lines.extend(outputs[key].get_content().splitlines())
            except Exception:
                pass

    if "logs" in outputs and hasattr(outputs["logs"], "list_object_names"):
        folder_node = outputs["logs"]
        for fname in sorted(folder_node.list_object_names())[:3]:
            try:
                lines.append(f"--- logs/{fname} ---")
                lines.extend(
                    folder_node.get_object_content(fname, mode="r").splitlines()
                )
            except Exception:
                pass

    if not lines:
        lines.extend(_read_retrieved_text(node, ("aiida.log", "aiida-stdout.txt")))
    lines.extend(_read_retrieved_text(node, ("_scheduler-stderr.txt",)))

    try:
        for record in node.base.logs.get_logs_for(node):
            lines.append(f"[AiiDA {record.levelname}] {record.message}")
    except Exception:
        pass

    for child in node.called:
        if child.is_failed or child.is_excepted:
            child_tail = read_node_logs(child, max_lines=20)
            if child_tail:
                lines.append(f"--- child PK {child.pk} ---")
                lines.extend(child_tail.splitlines())

    return "\n".join(lines[-max_lines:]) if max_lines > 0 else ""


def diagnose_node_impl(node: ProcessNode) -> dict[str, Any]:
    """
    Classify a failed/excepted ``ProcessNode`` and compute retry depth.

    Parameters
    ----------
    node : ProcessNode
        Target AiiDA process node.

    Returns
    -------
    dict[str, Any]
        Structured diagnosis report including ``category``, ``log_tail``,
        ``suggested_action``, ``retry_depth``, and ``retry_cap_exceeded``.

    Raises
    ------
    MCPToolError
        If ``node`` is not a ``ProcessNode``.
    """
    if not isinstance(node, ProcessNode):
        raise MCPToolError(
            "validation_error",
            f"Node PK {getattr(node, 'pk', '?')} is {type(node).__name__}, "
            "expected a ProcessNode.",
        )

    log_tail = read_node_logs(node, max_lines=60)
    state = node.process_state.value if node.process_state else "unknown"
    exit_status = node.exit_status
    exit_message = node.exit_message

    category = "unknown_failure"
    suggested_action = (
        f"Inspect full logs via aiida://node/{node.pk}/logs and check inputs."
    )

    search_blob = f"{exit_message or ''}\n{log_tail}"
    for rule_cat, pattern, action in _FAILURE_RULES:
        if pattern.search(search_blob):
            category = rule_cat
            suggested_action = action
            break
    else:
        if exit_status == 305:
            category = "missing_output_files"
            suggested_action = (
                f"Expected janus output files were missing (exit 305). Inspect "
                f"aiida://node/{node.pk}/logs for upstream janus errors."
            )
        elif state == "excepted":
            category = "excepted"
            suggested_action = (
                "Process raised an internal Python exception. Inspect the "
                "AiiDA exception traceback in logs."
            )

    intent_hash = node.base.extras.get(INTENT_HASH_EXTRA, None)
    retry_depth = count_failed_attempts(intent_hash)
    return {
        "pk": node.pk,
        "process_state": state,
        "exit_status": exit_status,
        "exit_message": exit_message,
        "category": category,
        "log_tail": log_tail,
        "suggested_action": suggested_action,
        "retry_depth": retry_depth,
        "retry_cap_exceeded": retry_depth >= 3,
    }
