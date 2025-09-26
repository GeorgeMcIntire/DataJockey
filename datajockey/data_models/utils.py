import numpy as np
from typing import Any
from sqlalchemy.types import TypeDecorator, JSON
from sqlalchemy.dialects.postgresql import JSONB

def _to_native(obj: Any) -> Any:
    """Recursively convert NumPy types into JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    return obj

class NumpyArrayJSON(TypeDecorator):
    """
    Store JSON/JSONB, optionally return np.ndarray on read.
    Automatically converts NumPy scalars/arrays to JSON-safe types on write.
    """
    cache_ok = True
    impl = JSON  # default; we'll switch to JSONB at runtime for Postgres

    def __init__(self, *, as_array: bool = True, dtype=np.float32):
        super().__init__()
        self._as_array = as_array
        self._dtype = dtype

    def load_dialect_impl(self, dialect):
        # Use JSONB on Postgres, JSON elsewhere
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        # Called when writing to DB
        if value is None:
            return None
        return _to_native(value)

    def process_result_value(self, value, dialect):
        # Called when reading from DB
        if value is None:
            return None
        if not self._as_array:
            return value
        # Convert back to np.ndarray with a consistent dtype
        return np.asarray(value, dtype=self._dtype)