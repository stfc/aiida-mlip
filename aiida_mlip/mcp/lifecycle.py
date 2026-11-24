"""AiiDA profile loading, daemon checks, and adaptive sync-to-async lease."""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import TYPE_CHECKING, Any, Literal

import aiida
from aiida.common import exceptions as aiida_exc
from aiida.engine import CalcJob, submit
from aiida.engine.daemon.client import get_daemon_client
from aiida.engine.processes.control import kill_processes
from aiida.manage.configuration import get_config, get_profile
from aiida.orm import Code, ProcessNode, QueryBuilder, load_node

if TYPE_CHECKING:
    from aiida_workgraph import WorkGraph

from aiida_mlip.mcp.digests import (
    build_digest,
    build_resource_uris,
    diagnose_node_impl,
)
from aiida_mlip.mcp.provenance import find_matching_process, tag_process_intent
from aiida_mlip.mcp.refs import MCPToolError

_POLL_INTERVAL_SECONDS = 0.5


def ensure_profile(profile: str | None = None) -> Any:
    """
    Load the requested or default AiiDA profile with switching enabled.

    Parameters
    ----------
    profile : str | None
        Profile name to load, or ``None`` for the default profile.

    Returns
    -------
    Any
        Loaded AiiDA ``Profile`` instance.

    Raises
    ------
    MCPToolError
        With ``error="profile_not_found"`` if no profile is configured.
    """
    try:
        return aiida.load_profile(profile, allow_switch=True)
    except (
        aiida_exc.ProfileConfigurationError,
        aiida_exc.MissingConfigurationError,
    ) as exc:
        raise MCPToolError(
            "profile_not_found",
            f"Could not load AiiDA profile '{profile or 'default'}': {exc}",
            hint="Run 'verdi presto' or pass --profile <name>.",
        ) from exc


def check_daemon_running() -> dict[str, Any]:
    """
    Check whether the AiiDA daemon for the active profile is running.

    Returns
    -------
    dict[str, Any]
        Dictionary ``{"running": bool, "workers": int}``.
    """
    try:
        client = get_daemon_client()
        status = client.get_status()
    except Exception:
        return {"running": False, "workers": 0}
    running = bool(status.get("status") == "running" or client.is_daemon_running)
    workers = len(status.get("workers", {})) if isinstance(status, dict) else 0
    return {"running": running, "workers": workers}


def _node_status(node: ProcessNode) -> str:
    """Map an AiiDA ``ProcessNode`` state to a concise status string."""
    if node.is_finished_ok:
        return "finished"
    if node.is_terminated:
        return "failed"
    return node.process_state.value if node.process_state else "unknown"


def _validate_calc_inputs(
    calc_cls: type[CalcJob],
    inputs: dict[str, Any],
    *,
    skip_model_port_check: bool,
) -> None:
    """
    Validate ``CalcJob`` inputs during ``dry_run=True``.

    Handles both return-value ``PortValidationError`` from ``plumpy`` and
    raised ``InputValidationError`` from ``BaseJanus`` / ``Train`` validators.

    Parameters
    ----------
    calc_cls : type[CalcJob]
        Target ``CalcJob`` class.
    inputs : dict[str, Any]
        Candidate inputs dictionary.
    skip_model_port_check : bool
        When ``True``, ignores only missing-model errors caused by a
        ``DryRunModelRef`` placeholder.

    Raises
    ------
    MCPToolError
        With ``error="validation_error"`` if validation fails.
    """
    try:
        err = calc_cls.spec().inputs.validate(inputs)
    except aiida_exc.InputValidationError as exc:
        msg = str(exc)
        if skip_model_port_check and (
            "'model' must be specified" in msg or "Undefined Model to fine-tune" in msg
        ):
            if "arch" not in inputs:
                raise MCPToolError(
                    "validation_error",
                    "'arch' must be specified for dry-run URI models.",
                ) from exc
            return
        raise MCPToolError("validation_error", msg) from exc

    if err is not None:
        raise MCPToolError("validation_error", str(err))


def submit_with_lease(
    target_or_factory: type[CalcJob] | WorkGraph | Callable[[], WorkGraph],
    inputs: dict[str, Any] | None,
    *,
    entry_point: str,
    intent_hash: str,
    dedupe: Literal["reuse", "strict", "force"] = "reuse",
    wait_seconds: float = 15.0,
    dry_run: bool = False,
    validated_summary: dict[str, Any] | None = None,
    skip_model_port_check: bool = False,
) -> dict[str, Any]:
    """
    Validate, deduplicate, submit, and poll an AiiDA process with a wait lease.

    Parameters
    ----------
    target_or_factory : type[CalcJob] | WorkGraph | Callable[[], WorkGraph]
        Either a ``CalcJob`` subclass, a ``WorkGraph``, or a zero-argument
        factory returning a ``WorkGraph`` (deferred until after ``dry_run``,
        dedupe, and daemon checks pass).
    inputs : dict[str, Any] | None
        AiiDA inputs dictionary for ``CalcJob`` targets, or ``None`` for
        ``WorkGraph`` targets.
    entry_point : str
        Canonical calculation or workflow identifier.
    intent_hash : str
        Canonical SHA-256 intent hash.
    dedupe : Literal["reuse", "strict", "force"]
        Provenance deduplication mode.
    wait_seconds : float
        Maximum seconds to wait synchronously before returning an async handle.
    dry_run : bool
        When ``True``, validates without storing nodes or submitting.
    validated_summary : dict[str, Any] | None
        Human-readable summary returned in ``dry_run`` responses.
    skip_model_port_check : bool
        Set ``True`` when a ``DryRunModelRef`` omitted ``model`` /
        ``foundation_model``.

    Returns
    -------
    dict[str, Any]
        Standard MCP submission envelope.
    """
    if dry_run:
        if (
            inputs is not None
            and isinstance(target_or_factory, type)
            and issubclass(target_or_factory, CalcJob)
        ):
            _validate_calc_inputs(
                target_or_factory,
                inputs,
                skip_model_port_check=skip_model_port_check,
            )
        return {
            "status": "dry_run",
            "entry_point": entry_point,
            "intent_hash": intent_hash,
            "validated_inputs": validated_summary or {},
        }

    existing = find_matching_process(intent_hash, dedupe=dedupe)
    if existing is not None:
        return {
            "pk": existing.pk,
            "uuid": existing.uuid,
            "entry_point": entry_point,
            "status": _node_status(existing),
            "cached": True,
            "intent_hash": intent_hash,
            "digest": build_digest(existing) if existing.is_terminated else None,
            "diagnosis": None,
            "resources": build_resource_uris(existing),
        }

    daemon = check_daemon_running()
    if not daemon["running"]:
        raise MCPToolError(
            "daemon_not_running",
            "AiiDA daemon is not running.",
            hint="Run 'verdi daemon start'.",
        )

    is_calc_cls = isinstance(target_or_factory, type) and issubclass(
        target_or_factory, CalcJob
    )
    target = (
        target_or_factory()
        if callable(target_or_factory) and not is_calc_cls
        else target_or_factory
    )
    if not is_calc_cls and hasattr(target, "submit"):
        node = target.submit()
    else:
        node = submit(target, **(inputs or {}))

    tag_process_intent(node, intent_hash)

    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    while not node.is_terminated and time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_SECONDS)

    if node.is_terminated:
        is_ok = bool(node.is_finished_ok)
        return {
            "pk": node.pk,
            "uuid": node.uuid,
            "entry_point": entry_point,
            "status": "finished" if is_ok else "failed",
            "cached": False,
            "intent_hash": intent_hash,
            "digest": build_digest(node),
            "diagnosis": None if is_ok else diagnose_node_impl(node),
            "resources": build_resource_uris(node),
        }

    return {
        "pk": node.pk,
        "uuid": node.uuid,
        "entry_point": entry_point,
        "status": node.process_state.value if node.process_state else "running",
        "cached": False,
        "intent_hash": intent_hash,
        "message": (
            f"Running asynchronously (exceeded {wait_seconds}s lease). "
            f"Poll with mlip_status(pk={node.pk})."
        ),
        "digest": None,
        "diagnosis": None,
        "resources": build_resource_uris(node),
    }


def _load_process_node(pk: int) -> ProcessNode:
    """Load an AiiDA ``ProcessNode`` by PK, raising ``MCPToolError`` on error."""
    try:
        node = load_node(pk)
    except aiida_exc.NotExistent as exc:
        raise MCPToolError(
            "node_not_found", f"Node PK {pk} was not found in the active profile."
        ) from exc
    if not isinstance(node, ProcessNode):
        raise MCPToolError(
            "validation_error",
            f"Node PK {pk} is {type(node).__name__}, expected a ProcessNode.",
        )
    return node


def get_process_status(pk: int) -> dict[str, Any]:
    """
    Inspect status, bounded digest, and optional diagnosis for a ``ProcessNode``.

    Parameters
    ----------
    pk : int
        Primary key of the AiiDA ``ProcessNode``.

    Returns
    -------
    dict[str, Any]
        Structured status dictionary.
    """
    node = _load_process_node(pk)
    is_terminated = bool(node.is_terminated)
    is_ok = bool(node.is_finished_ok)
    return {
        "pk": node.pk,
        "uuid": node.uuid,
        "process_type": node.process_type or "",
        "status": _node_status(node),
        "exit_status": node.exit_status,
        "exit_message": node.exit_message,
        "digest": build_digest(node) if is_terminated else None,
        "diagnosis": (
            diagnose_node_impl(node) if (is_terminated and not is_ok) else None
        ),
        "resources": build_resource_uris(node),
    }


def cancel_process(pk: int) -> dict[str, Any]:
    """
    Request termination of a running AiiDA process.

    Parameters
    ----------
    pk : int
        Primary key of the AiiDA ``ProcessNode``.

    Returns
    -------
    dict[str, Any]
        Cancellation status dictionary.
    """
    node = _load_process_node(pk)
    state = node.process_state.value if node.process_state else "unknown"
    if node.is_terminated:
        return {
            "pk": node.pk,
            "status": "already_terminated",
            "process_state": state,
        }
    if not check_daemon_running()["running"]:
        raise MCPToolError(
            "daemon_not_running",
            "AiiDA daemon is not running; cannot send kill request.",
            hint="Run 'verdi daemon start'.",
        )
    kill_processes([node])
    return {
        "pk": node.pk,
        "status": "kill_requested",
        "process_state": state,
    }


def list_profiles_impl() -> dict[str, Any]:
    """
    List configured AiiDA profiles, active profile, and daemon status.

    Returns
    -------
    dict[str, Any]
        Profile and daemon summary dictionary.
    """
    try:
        profiles = list(get_config().profile_names)
    except Exception:
        profiles = []
    active = get_profile()
    active_name = active.name if active is not None else None
    daemon = (
        check_daemon_running()
        if active_name is not None
        else {"running": False, "workers": 0}
    )
    return {
        "status": "ok",
        "active_profile": active_name,
        "profiles": profiles,
        "daemon": daemon,
    }


def list_codes_impl(plugin: str | None = None) -> dict[str, Any]:
    """
    List registered AiiDA ``Code`` nodes in the active profile.

    Parameters
    ----------
    plugin : str | None
        Optional plugin name or prefix filter (e.g. ``"mlip"`` or ``"mlip.sp"``).

    Returns
    -------
    dict[str, Any]
        Dictionary containing active profile name and list of code metadata.
    """
    active = get_profile()
    profile_name = active.name if active is not None else ""
    qb = QueryBuilder().append(Code)
    codes: list[dict[str, Any]] = []
    for (code,) in qb.all():
        default_plugin = getattr(code, "default_calc_job_plugin", None)
        if plugin and (
            not default_plugin
            or (default_plugin != plugin and not default_plugin.startswith(plugin))
        ):
            continue
        computer_label = code.computer.label if code.computer is not None else ""
        codes.append(
            {
                "label": code.label,
                "full_label": code.full_label,
                "computer": computer_label,
                "plugin": default_plugin,
                "pk": code.pk,
            }
        )
    return {
        "status": "ok",
        "profile": profile_name,
        "codes": codes,
    }
