"""Model Context Protocol (MCP) server for ``aiida-mlip``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


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
        Configured FastMCP server instance with tools, resources, and prompts.
    """
    from aiida_mlip.mcp.server import create_server as _create_server

    return _create_server(default_profile=default_profile, presets_path=presets_path)


__all__ = ["create_server"]
