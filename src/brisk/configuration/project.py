"""Project configuration utilities for Brisk.

This module provides utilities for locating and managing project-level
configuration files and directory structures.
"""

import pathlib
import functools
import contextvars
from typing import Optional

_project_root_override: contextvars.ContextVar[Optional[pathlib.Path]] = (
    contextvars.ContextVar("project_root_override", default=None)
)

class ProjectRootContext:
    """Context manager for temporarily overriding project root.

    This is used for testing where .briskconfig does not exist and is not 
    relevant to the test case.

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


@functools.lru_cache
def find_project_root() -> pathlib.Path:
    """Find the project root directory containing .briskconfig.

    Searches current directory and parent directories for .briskconfig file.
    Result is cached to avoid repeated filesystem operations.

    Can be overridden in tests using ProjectRootContext.

    Returns
    -------
    pathlib.Path
        Path to project root directory

    Raises
    ------
    FileNotFoundError
        If .briskconfig cannot be found in any parent directory

    Examples
    --------
    >>> root = find_project_root()
    >>> datasets_dir = root / 'datasets'
    >>> config_file = root / '.briskconfig'
    """
    override = _project_root_override.get()
    if override is not None:
        return override
    
    current = pathlib.Path.cwd()
    while current != current.parent:
        if (current / ".briskconfig").exists():
            return current
        current = current.parent
    raise FileNotFoundError(
        "Could not find .briskconfig in any parent directory"
    )
