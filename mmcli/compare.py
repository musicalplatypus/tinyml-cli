"""Model comparison across task types via the tinyml-modelmaker registry."""

import json
import os
import subprocess
import sys
import textwrap
from typing import Any, Dict, List, Optional

_QUANT_NAMES = {0: "float", 1: "int8", 2: "TI-NPU"}

_QUERY_SCRIPT = textwrap.dedent(r'''
import json, sys

module_name = {module!r}
task_type   = {task_type!r}
target_device = {target_device!r}

try:
    if module_name == "timeseries":
        from tinyml_modelmaker.ai_modules.timeseries import constants, training
    elif module_name == "vision":
        from tinyml_modelmaker.ai_modules.vision import constants, training
    elif module_name == "audio":
        from tinyml_modelmaker.ai_modules.audio import constants, training
    else:
        print(json.dumps({{"error": f"Unknown module: {{module_name}}"}}))
        sys.exit(0)
except ImportError as e:
    print(json.dumps({{"error": f"Cannot import tinyml_modelmaker: {{e}}"}}))
    sys.exit(0)

if task_type and task_type not in constants.TASK_DESCRIPTIONS:
    print(json.dumps({{"error": f"Unknown task type: {{task_type}}"}}))
    sys.exit(0)

kwargs = {{"task_type": task_type}} if task_type else {{}}
if target_device:
    kwargs["target_device"] = target_device

models_raw = training.get_model_descriptions(**kwargs)

models = {{}}
for name, desc in models_raw.items():
    t = desc.get("training", {{}}) if hasattr(desc, "get") else {{}}
    td = t.get("target_devices", {{}}) if hasattr(t, "get") else {{}}
    devices = list(td.keys()) if hasattr(td, "keys") else (td or [])
    quant = t.get("quantization", 0) if hasattr(t, "get") else 0
    models[name] = {{
        "quantization": quant,
        "learning_rate": t.get("learning_rate") if hasattr(t, "get") else None,
        "batch_size": t.get("batch_size") if hasattr(t, "get") else None,
        "device_count": len(devices),
        "devices": devices,
    }}

print(json.dumps({{"module": module_name, "task_type": task_type, "models": models}}))
''')


def _get_python_exe() -> str:
    return os.environ.get("MMCLI_PYTHON", sys.executable)


def _query_models(module: str, task_type: str,
                  device: Optional[str] = None) -> Dict[str, Any]:
    """Query modelmaker for real model data for *task_type*."""
    script = _QUERY_SCRIPT.format(
        module=module,
        task_type=task_type,
        target_device=device or "",
    )
    python_exe = _get_python_exe()
    result = subprocess.run(
        [python_exe, "-c", script],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {"error": f"Query failed: {result.stderr.strip()[:200]}"}
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        return {"error": f"Bad JSON from query: {exc}"}


def compare_models(
    module_type: str,
    task_types: List[str],
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare model availability and attributes across task types.

    Args:
        module_type: AI module (timeseries, vision, audio)
        task_types: List of task-type strings to compare
        device: Optional device filter applied to all queries

    Returns:
        Comparison dict with per-task model info and a cross-task summary.
    """
    comparison: Dict[str, Any] = {
        "module": module_type,
        "device": device,
        "tasks": {},
    }

    for task in task_types:
        data = _query_models(module_type, task, device)
        if "error" in data:
            comparison["tasks"][task] = {"error": data["error"]}
            continue
        models = data.get("models", {})
        quant_types = sorted({m["quantization"] for m in models.values()})
        all_devices: set = set()
        for m in models.values():
            all_devices.update(m.get("devices", []))

        comparison["tasks"][task] = {
            "model_count": len(models),
            "models": models,
            "quantization_types": quant_types,
            "supported_device_count": len(all_devices),
        }

    return comparison


def format_comparison(comparison: Dict[str, Any]) -> str:
    """Format comparison result as a human-readable table."""
    lines: List[str] = []
    SEP = "=" * 64

    lines.append(SEP)
    lines.append("TASK TYPE COMPARISON")
    lines.append(SEP)

    if comparison.get("device"):
        lines.append(f"Device filter: {comparison['device']}")
        lines.append("")

    tasks = comparison.get("tasks", {})
    if not tasks:
        lines.append("No task data.")
        return "\n".join(lines)

    # Summary table
    col_w = max(30, max(len(t) for t in tasks) + 2)
    header = f"{'Task':{col_w}} {'Models':>8} {'Quant types':<20} {'Devices':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for task, info in tasks.items():
        if "error" in info:
            lines.append(f"{task:{col_w}} ERROR: {info['error']}")
            continue
        q_names = "/".join(_QUANT_NAMES.get(q, str(q))
                           for q in info.get("quantization_types", []))
        lines.append(
            f"{task:{col_w}} {info['model_count']:>8} {q_names:<20} "
            f"{info['supported_device_count']:>8}"
        )

    # Per-task model list
    for task, info in tasks.items():
        if "error" in info:
            continue
        models = info.get("models", {})
        if not models:
            continue
        lines.append("")
        lines.append(f"  {task} ({info['model_count']} models):")
        for name, m in models.items():
            q = _QUANT_NAMES.get(m.get("quantization", 0), "?")
            dc = m.get("device_count", 0)
            lines.append(f"    {name:<30} quant={q:<8} devices={dc}")

    lines.append("")
    return "\n".join(lines)
