from cocoindex_code_mcp_server import LOGGER
from typing import Any, Dict
from collections.abc import Iterable, Sized

def is_empty_iterable(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes)) and isinstance(value, Sized) and len(value) == 0

def update_defaults(d: Dict[str, Any], defaults: Dict[str, Any]) -> None:
    for k, v in defaults.items():
        if isinstance(v, Iterable) and not isinstance(v, (str, bytes)) and isinstance(v, Sized) and len(v) > 0:
            current_value = d.get(k)
            if current_value is None or is_empty_iterable(current_value):
                d[k] = v
        else:
            d.setdefault(k, v)

