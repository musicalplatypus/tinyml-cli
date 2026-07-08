"""Output formatting helpers for programmatic use."""

import json
from typing import Any, Dict
import csv
import io


def format_json(data: Dict[str, Any]) -> str:
    """Format data as JSON."""
    return json.dumps(data, indent=2)


def format_csv(data: Dict[str, Any], keys: list = None) -> str:
    """Format data as CSV."""
    if not data:
        return ""

    # Flatten nested dict for CSV
    def flatten(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flattened = flatten(data)
    keys = keys or list(flattened.keys())

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=keys)
    writer.writeheader()
    writer.writerow(flattened)

    return output.getvalue()


def format_yaml(data: Dict[str, Any]) -> str:
    """Format data as YAML."""
    try:
        import yaml
        return yaml.dump(data, default_flow_style=False)
    except ImportError:
        # Fallback to JSON if PyYAML not available
        return json.dumps(data, indent=2)


def format_table(data: Dict[str, Any]) -> str:
    """Format data as a simple text table."""
    lines = []
    for key, value in data.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)