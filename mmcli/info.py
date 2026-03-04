"""
Query the tinyml-modelmaker registry and display available options.

The registry lives in the tinyml-modelmaker package (which may be installed
in a different Python environment).  We query it by running a small Python
snippet via MMCLI_PYTHON subprocess and parsing the JSON output.
"""

import json
import os
import subprocess
import sys
import textwrap

# ---------------------------------------------------------------------------
# Query script template — executed in the MMCLI_PYTHON environment
# ---------------------------------------------------------------------------

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
    else:
        print(json.dumps({{"error": f"Unknown module: {{module_name}}"}}))
        sys.exit(0)
except ImportError as e:
    print(json.dumps({{"error": f"Cannot import tinyml_modelmaker: {{e}}"}}))
    sys.exit(0)

result = {{"module": module_name}}

# --- Task descriptions ---
task_descriptions = {{}}
for tt, desc in constants.TASK_DESCRIPTIONS.items():
    task_descriptions[tt] = {{
        "task_name": desc.get("task_name", tt),
        "target_devices": desc.get("target_devices", []),
    }}
result["task_descriptions"] = task_descriptions

if task_type:
    if task_type not in constants.TASK_DESCRIPTIONS:
        result["error"] = f"Unknown task type: {{task_type}}"
        print(json.dumps(result))
        sys.exit(0)

    # --- Models ---
    kwargs = {{"task_type": task_type}}
    if target_device:
        kwargs["target_device"] = target_device
    models_raw = training.get_model_descriptions(**kwargs)

    models = {{}}
    for name, desc in models_raw.items():
        info = {{"name": name, "devices": []}}
        # desc can be a dict or a ConfigDict (attribute-accessible dict)
        if hasattr(desc, "get"):
            td = desc.get("training", {{}}).get("target_devices", {{}})
        else:
            td = getattr(getattr(desc, "training", None), "target_devices", {{}}) or {{}}
        if isinstance(td, dict):
            info["devices"] = list(td.keys())
        models[name] = info
    result["models"] = models

    # --- Feature extraction presets ---
    fe_presets = []
    for name, desc in constants.FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.items():
        if hasattr(desc, "get"):
            common = desc.get("common", {{}})
        else:
            common = getattr(desc, "common", {{}}) or {{}}

        if hasattr(common, "get"):
            preset_task = common.get("task_type", None)
        else:
            preset_task = getattr(common, "task_type", None)

        if preset_task is None:
            fe_presets.append(name)
        elif isinstance(preset_task, list):
            if task_type in preset_task:
                fe_presets.append(name)
        elif preset_task == task_type:
            fe_presets.append(name)
    result["fe_presets"] = fe_presets

print(json.dumps(result))
''')


def _build_query_script(module: str, task_type: str | None,
                        target_device: str | None) -> str:
    """Build the Python snippet to run in MMCLI_PYTHON."""
    return _QUERY_SCRIPT.format(
        module=module,
        task_type=task_type or "",
        target_device=target_device or "",
    )


def _run_query(python_exe: str, script: str) -> dict:
    """Run *script* via *python_exe* and return parsed JSON."""
    result = subprocess.run(
        [python_exe, "-c", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            f"ERROR: Query failed (exit {result.returncode}).\n"
            f"stderr: {result.stderr.strip()}\n\n"
            "Ensure MMCLI_PYTHON points to a Python with "
            "tinyml_modelmaker installed.",
            file=sys.stderr,
        )
        sys.exit(1)

    stdout = result.stdout.strip()
    if not stdout:
        print(
            "ERROR: No output from registry query.\n"
            f"stderr: {result.stderr.strip()}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        print(
            f"ERROR: Could not parse query output as JSON: {exc}\n"
            f"stdout: {stdout[:500]}",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Device grouping (for display)
# ---------------------------------------------------------------------------

_DEVICE_FAMILIES = {
    "C2000": [
        "F280013", "F280015", "F28003", "F28004", "F2837",
        "F28P55", "F28P65", "F29H85", "F29P58", "F29P32",
    ],
    "MSPM0": [
        "MSPM0G3507", "MSPM0G3519", "MSPM0G5187",
        "MSPM33C32", "MSPM33C34",
    ],
    "SimpleLink": ["CC2755", "CC1352", "CC1354", "CC35X1"],
    "Sitara": ["AM263", "AM263P", "AM261", "AM13E2"],
}

# Reverse lookup: device → family
_DEVICE_TO_FAMILY = {}
for _family, _devs in _DEVICE_FAMILIES.items():
    for _dev in _devs:
        _DEVICE_TO_FAMILY[_dev] = _family


def _group_devices(devices: list[str]) -> dict[str, list[str]]:
    """Group device list by family, preserving order."""
    groups: dict[str, list[str]] = {}
    for dev in devices:
        family = _DEVICE_TO_FAMILY.get(dev, "Other")
        groups.setdefault(family, []).append(dev)
    return groups


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _print_task_list(data: dict) -> None:
    """Print a table of all task types for the module."""
    module = data.get("module", "?")
    tasks = data.get("task_descriptions", {})

    print(f"\n{module.title()} Task Types:\n")

    if not tasks:
        print("  (none found)")
        return

    # Column widths
    name_width = max(len(tt) for tt in tasks) + 2
    print(f"  {'Task Type':<{name_width}}  {'Name':<25}  Devices")
    print(f"  {'─' * name_width}  {'─' * 25}  {'─' * 10}")

    for tt, desc in sorted(tasks.items()):
        task_name = desc.get("task_name", tt)
        n_devices = len(desc.get("target_devices", []))
        print(f"  {tt:<{name_width}}  {task_name:<25}  {n_devices} devices")

    print(f"\nUse 'mmcli info -m {module} -t <task>' for details.")


def _print_task_details(data: dict, task_type: str,
                        target_device: str | None) -> None:
    """Print devices, models, and FE presets for a specific task."""
    tasks = data.get("task_descriptions", {})
    task_desc = tasks.get(task_type, {})
    task_name = task_desc.get("task_name", task_type)
    devices = task_desc.get("target_devices", [])

    print(f"\nTask: {task_name} ({task_type})")

    # --- Devices ---
    print(f"\nSupported Devices ({len(devices)}):")
    grouped = _group_devices(devices)
    for family, devs in grouped.items():
        dev_str = "  ".join(devs)
        print(f"  {family + ':':<12} {dev_str}")

    # --- Models ---
    models = data.get("models", {})
    if target_device:
        print(f"\nModels for {target_device} ({len(models)} available):")
    else:
        print(f"\nModels ({len(models)} available):")

    if models:
        name_width = max(len(n) for n in models) + 2
        name_width = max(name_width, 6)  # min header width
        print(f"  {'Name':<{name_width}}  Supported Devices")
        print(f"  {'─' * name_width}  {'─' * 40}")

        for name in sorted(models.keys()):
            info = models[name]
            model_devs = info.get("devices", [])
            if len(model_devs) <= 5:
                dev_str = ", ".join(model_devs)
            else:
                dev_str = ", ".join(model_devs[:4]) + f", ... (+{len(model_devs) - 4})"
            print(f"  {name:<{name_width}}  {dev_str}")
    else:
        print("  (none found)")

    # --- FE Presets ---
    fe_presets = data.get("fe_presets", [])
    print(f"\nFeature Extraction Presets ({len(fe_presets)} available):")
    if fe_presets:
        for p in sorted(fe_presets):
            print(f"  {p}")
    else:
        print("  (none found)")

    # --- Example Datasets ---
    from mmcli.datasets import list_datasets
    datasets = list_datasets(task_type=task_type)
    print(f"Example Datasets ({len(datasets)} available):")
    if datasets:
        name_width = max(len(d["name"]) for d in datasets) + 2
        name_width = max(name_width, 6)
        print(f"  {'Name':<{name_width}}  Description")
        print(f"  {'─' * name_width}  {'─' * 40}")
        for d in sorted(datasets, key=lambda x: x["name"]):
            desc = d.get("description", "")
            print(f"  {d['name']:<{name_width}}  {desc}")
    else:
        print("  (none available)")
    print()
    print(f"To create a project from a dataset:")
    print(f"  mmcli init -t {task_type} --dataset <DATASET_NAME> -p <PROJECT_DIR>")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_info(args, python_exe: str) -> None:
    """Run the info subcommand."""
    module = args.module
    task_type = getattr(args, "task", None)
    target_device = getattr(args, "device", None)

    script = _build_query_script(module, task_type, target_device)
    data = _run_query(python_exe, script)

    # Check for errors from the query script
    if "error" in data:
        print(f"ERROR: {data['error']}", file=sys.stderr)
        sys.exit(1)

    if task_type:
        _print_task_details(data, task_type, target_device)
    else:
        _print_task_list(data)
