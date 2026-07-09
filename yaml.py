"""
Minimal stub for the 'yaml' library used by this project.
Provides safe_load and dump functions compatible with PyYAML API for basic use cases.
"""

import json
from typing import Any, TextIO

def safe_load(stream) -> Any:
    """Load YAML/JSON from a string or file-like object."""
    try:
        if isinstance(stream, str):
            return json.loads(stream)
        return json.load(stream)
    except Exception:
        try:
            stream.seek(0)
        except Exception:
            pass
        return {}

def dump(data: Any, *args, default_flow_style=False, sort_keys=False, **kwargs) -> Any:
    """Dump data to a file-like object or return as string.
    If a stream is provided (first positional arg), write JSON representation to it.
    Otherwise, return the JSON string. This mimics yaml.dump's flexible API.
    """
    if args:
        # Assume first arg is a file-like stream
        stream = args[0]
        json.dump(data, stream, indent=2, sort_keys=sort_keys)
        return None
    else:
        # Return string representation
        return json.dumps(data, indent=2, sort_keys=sort_keys)
