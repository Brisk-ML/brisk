"""Root conftest.py to ensure proper test imports.

Top level testing configuration. Anything defined here should be minimal and
lightweight. Only define here if absolutely necessary, otherwise define in a
conftest.py scoped to the tests that require the data.
"""
import sys
from pathlib import Path

# Add the workspace root to Python path
workspace_root = Path(__file__).parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))
