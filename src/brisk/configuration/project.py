"""Project configuration utilities for Brisk.

This module provides utilities for locating and managing project-level
configuration files and directory structures. Supports both the new
``.brisk/`` directory format and the legacy ``.briskconfig`` file.
"""

import json
import pathlib
import functools
import contextvars
import sqlite3
import warnings
from typing import Optional, Tuple

_project_root_override: contextvars.ContextVar[Optional[pathlib.Path]] = (
    contextvars.ContextVar("project_root_override", default=None)
)

_legacy_warning_issued = False


class ProjectRootContext:
    """Context manager for temporarily overriding project root.

    This is used for testing where project markers do not exist and are
    not relevant to the test case.

    Parameters
    ----------
    project_root: pathlib.Path
        The project root to use within this context

    Examples
    --------
    >>> with ProjectRootContext(Path("/tmp/project_root/")):
    ...     root = find_project_root()
    ...     # Write the test code
    """
    def __init__(self, project_root: pathlib.Path):
        self.project_root = project_root
        self.token = None

    def __enter__(self):
        self.token = _project_root_override.set(self.project_root)
        find_project_root.cache_clear()
        return self.project_root

    def __exit__(self, exc_type, exc_val, exc_tb):
        _project_root_override.reset(self.token)
        find_project_root.cache_clear()


def _parse_briskconfig(config_path: pathlib.Path) -> dict:
    """Parse a legacy .briskconfig file into a dict of key=value pairs."""
    result = {}
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                result[key.strip()] = value.strip()
    return result


def _check_legacy_config(project_root: pathlib.Path) -> None:
    """Issue a deprecation warning when a legacy .briskconfig is detected.

    Called by ``find_project_root`` when the project uses the old format.
    Warns once per process and offers the user the option to migrate
    immediately or run ``brisk migrate`` later.
    """
    global _legacy_warning_issued  # noqa: PLW0603
    if _legacy_warning_issued:
        return
    _legacy_warning_issued = True

    warnings.warn(
        "The .briskconfig file is deprecated and will be removed in "
        "Brisk 1.3. Use `brisk migrate` to convert to the new "
        ".brisk/ directory format.",
        DeprecationWarning,
        stacklevel=4,
    )

    try:
        response = input(
            "Your project uses the deprecated .briskconfig format.\n"
            "Would you like to migrate to the new .brisk/ format now? "
            "[y/N]: "
        )
    except (EOFError, KeyboardInterrupt):
        response = ""

    if response.strip().lower() == "y":
        migrate_project(project_root)
        print("Migration complete.")
    else:
        print(
            "Continuing with legacy format. Run `brisk migrate` to "
            "update when ready."
        )


def migrate_project(project_root: pathlib.Path) -> None:
    """Migrate a project from legacy .briskconfig to the .brisk/ format.

    Creates the ``.brisk/`` directory with ``brisk.sqlite`` and
    ``project.json``, then removes the old ``.briskconfig`` file.

    Parameters
    ----------
    project_root : pathlib.Path
        Path to the project root containing ``.briskconfig``
    """
    config_path = project_root / ".briskconfig"
    brisk_dir = project_root / ".brisk"

    config_data = {}
    if config_path.exists():
        config_data = _parse_briskconfig(config_path)

    project_name = config_data.get("project_name", project_root.name)

    brisk_dir.mkdir(exist_ok=True)

    db_path = brisk_dir / "brisk.sqlite"
    if not db_path.exists():
        conn = sqlite3.connect(str(db_path))
        conn.close()

    project_json_path = brisk_dir / "project.json"
    if not project_json_path.exists():
        project_json = {
            "project_name": project_name,
            "project_path": str(project_root.resolve()),
            "project_description": "",
            "project_type": "classification",
            "datasets": [],
        }
        with open(project_json_path, "w", encoding="utf-8") as f:
            json.dump(project_json, f, indent=2)

    if config_path.exists():
        config_path.unlink()

    find_project_root.cache_clear()


def _search_for_project_root() -> Tuple[
    Optional[pathlib.Path], bool, bool
]:
    """Walk up from cwd looking for .brisk/ or .briskconfig.

    Returns
    -------
    tuple of (pathlib.Path or None, bool, bool)
        (project_root, has_new_format, has_legacy_format)
    """
    current = pathlib.Path.cwd()
    while current != current.parent:
        has_new = (current / ".brisk").is_dir()
        has_legacy = (current / ".briskconfig").exists()
        if has_new or has_legacy:
            return current, has_new, has_legacy
        current = current.parent
    return None, False, False


@functools.lru_cache
def find_project_root() -> pathlib.Path:
    """Find the project root directory.

    Searches current directory and parent directories for a ``.brisk/``
    directory (new format) or a ``.briskconfig`` file (legacy). The new
    format takes precedence. When only the legacy format is found, a
    deprecation warning is issued and the user is offered an interactive
    migration prompt.

    Can be overridden in tests using ProjectRootContext.

    Returns
    -------
    pathlib.Path
        Path to project root directory

    Raises
    ------
    FileNotFoundError
        If no project marker can be found in any parent directory
    """
    override = _project_root_override.get()
    if override is not None:
        return override

    root, has_new, has_legacy = _search_for_project_root()

    if root is None:
        raise FileNotFoundError(
            "Could not find a brisk project (.brisk/ directory or "
            ".briskconfig) in any parent directory"
        )

    if has_legacy and not has_new:
        _check_legacy_config(root)

    return root
