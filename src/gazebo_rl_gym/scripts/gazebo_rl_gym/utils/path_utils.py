"""Utility helpers for resolving package-relative paths.

These helpers keep all runtime path handling in one place and avoid hardcoded
relative traversals sprinkled through the codebase.
"""
from __future__ import annotations

import os
try:
    import rospkg
except ImportError:  # pragma: no cover - during testing without ROS
    rospkg = None


def get_package_root(package_name: str = "gazebo_rl_gym") -> str:
    """Return the absolute path to the ROS package.

    We prefer `rospkg` when available so paths also resolve correctly once the
    package is installed into an overlay. Falling back to the workspace layout
    keeps development workflows working even when ROS is not sourced.
    """
    if rospkg is not None:
        try:
            return rospkg.RosPack().get_path(package_name)
        except rospkg.ResourceNotFound:
            pass
    # development fallback: package lives under src/<package_name>
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def resolve_relative(*parts: str, package_name: str = "gazebo_rl_gym") -> str:
    """Resolve a path relative to the package root."""
    root = get_package_root(package_name)
    return os.path.join(root, *parts)


def ensure_dir(path: str) -> str:
    """Create `path` if it is missing and return it for fluent usage."""
    os.makedirs(path, exist_ok=True)
    return path
