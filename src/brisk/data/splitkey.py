"""The split key module"""
import dataclasses
from typing import Optional

@dataclasses.dataclass(frozen=True)
class SplitKey:
    """Frozen to make hashable."""
    group_name: str
    dataset_name: str
    table_name: Optional[str] = None
