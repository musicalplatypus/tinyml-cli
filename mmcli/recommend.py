"""
Model and feature-extraction preset recommendations for mmcli.

Reads the tinyml-modelzoo examples directory (config.yaml / config_*.yaml)
and scores each example against the user's task/device/variables to recommend
the best-matching model name and feature extraction preset.

Also reads the bundled schema.yaml for task-specific FE preset guidance.
"""

import os
import re
import sys
import yaml
from typing import Any, Optional

# Import output formatting functions
from mmcli.output import format_json, format_csv, format_yaml, format_table


# ---------------------------------------------------------------------------
# Schema loader (bundled schema.yaml)
# ---------------------------------------------------------------------------

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "data", "schema.yaml")
_SCHEMA: dict | None = None


def _load_schema() -> dict:
    global _SCHEMA
    if _SCHEMA is None:
        try:
            with open(_SCHEMA_PATH) as fh:
                _SCHEMA = yaml.safe_load(fh) or {}
        except Exception:
            _SCHEMA = {}
    return _SCHEMA


def _task_fe_recommendations(task_type: str) -> dict:
    """Return the task_recommendations entry for task_type (or empty dict)."""
    return _load_schema().get("task_recommendations", {}).get(task_type, {})


# ---------------------------------------------------------------------------
# Task-type → module mapping (keep in sync with agent-skills constants)
# ---------------------------------------------------------------------------

TASK_TYPE_TO_MODULE: dict[str, str] = {
    "generic_timeseries_classification":    "timeseries",
    "generic_timeseries_regression":        "timeseries",
    "generic_timeseries_forecasting":       "timeseries",
    "generic_timeseries_anomalydetection":  "timeseries",
    "motor_fault":                          "timeseries",
    "ecg_classification":                   "timeseries",
    "arc_fault":                            "timeseries",
    "blower_imbalance":                     "timeseries",
    "pir_detection":                        "timeseries",
    "image_classification":                 "vision",
    "audio_classification":                 "audio",
}

# ---------------------------------------------------------------------------
# Model complexity helpers (param count from model name)
# ---------------------------------------------------------------------------

_DATASET_PREFERRED_MAX_PARAMS: dict[str, Optional[int]] = {
    "tiny":   2_000,
    "small":  10_000,
    "medium": 30_000,
    "large":  None,
}


def _parse_model_params(model_name: str) -> Optional[int]:
    m = re.search(r"(\d+)[kK]", model_name)
    if m:
        return int(m.group(1)) * 1000
    candidates = [int(n) for n in re.findall(r"\d+", model_name) if int(n) >= 100]
    return max(candidates) if candidates else None


def _complexity_tier(param_count: int) -> str:
    tiers = [
        ("micro",  0,       1_000),
        ("tiny",   1_001,   4_000),
        ("small",  4_001,   10_000),
        ("medium", 10_001,  30_000),
        ("large",  30_001,  None),
    ]
    for name, lo, hi in tiers:
        if hi is None and param_count > lo:
            return name
        if hi is not None and lo <= param_count <= hi:
            return name
    return "micro"


# ---------------------------------------------------------------------------
# Example finder
# ---------------------------------------------------------------------------

def _find_modelzoo_examples_path() -> Optional[str]:
    """Locate the tinyml-modelzoo/examples directory.

    Search order:
      1. MMCLI_MODELZOO_PATH env var  (repo root)
      2. Sibling of tinyml-modelmaker inside the same monorepo checkout
      3. Common standalone paths
    """
    subpath = "examples"

    env_root = os.environ.get("MMCLI_MODELZOO_PATH")
    if env_root:
        candidate = os.path.join(env_root, subpath)
        if os.path.isdir(candidate):
            return candidate

    # Try to find via the installed tinyml_modelmaker package
    try:
        import tinyml_modelmaker
        pkg_dir = os.path.dirname(tinyml_modelmaker.__file__)
        # tinyml-modelzoo lives alongside tinyml-modelmaker in the monorepo
        candidate = os.path.normpath(
            os.path.join(pkg_dir, "..", "..", "..", "tinyml-modelzoo", "examples")
        )
        if os.path.isdir(candidate):
            return candidate
    except ImportError:
        pass

    # Fallback locations
    defaults = [
        os.path.expanduser("~/tinyml-tensorlab/tinyml-modelzoo"),
        os.path.expanduser("~/tinyml-modelzoo"),
        os.path.expanduser("~/projects/tinyml-modelzoo"),
        "/opt/tinyml-modelzoo",
    ]
    for root in defaults:
        candidate = os.path.join(root, subpath)
        if os.path.isdir(candidate):
            return candidate

    return None


def _parse_config_file(config_path: str) -> Optional[dict]:
    try:
        with open(config_path) as fh:
            return yaml.safe_load(fh)
    except Exception:
        return None


def _config_to_metadata(cfg: dict, example_dir: str) -> Optional[dict[str, Any]]:
    if not cfg or not isinstance(cfg, dict):
        return None
    try:
        common = cfg.get("common", {}) or {}
        training = cfg.get("training", {}) or {}
        feat_ext = cfg.get("data_processing_feature_extraction", {}) or {}

        task_type = common.get("task_type")
        target_module = common.get("target_module") or TASK_TYPE_TO_MODULE.get(task_type)

        raw_vars = feat_ext.get("variables")
        if isinstance(raw_vars, list):
            variables = len(raw_vars)
        elif isinstance(raw_vars, int):
            variables = raw_vars
        else:
            variables = None

        return {
            "task_type": task_type,
            "target_device": common.get("target_device"),
            "target_module": target_module,
            "variables": variables,
            "model_name": training.get("model_name"),
            "feature_extraction_name": feat_ext.get("feature_extraction_name"),
            "example_dir": example_dir,
        }
    except Exception:
        return None


def _list_examples(examples_root: str) -> list[dict[str, Any]]:
    """Read all examples, emitting one entry per config file found."""
    import glob
    out = []
    for item in os.listdir(examples_root):
        path = os.path.join(examples_root, item)
        if not os.path.isdir(path):
            continue

        primary = os.path.join(path, "config.yaml")
        if os.path.isfile(primary):
            meta = _config_to_metadata(_parse_config_file(primary), path)
            if meta:
                out.append(meta)
        else:
            for alt in sorted(glob.glob(os.path.join(path, "config_*.yaml"))):
                meta = _config_to_metadata(_parse_config_file(alt), path)
                if meta:
                    out.append(meta)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_example(
    example: dict,
    task_type: str,
    target_device: str,
    target_module: str,
    variables: Optional[int],
    dataset_size_bucket: Optional[str],
) -> tuple[float, dict[str, bool]]:
    score = 0
    bd = {k: False for k in (
        "task_type_match", "device_match", "module_match",
        "variables_match", "dataset_size_match",
    )}

    if example.get("task_type") == task_type:
        score += 1; bd["task_type_match"] = True
    if example.get("target_device") == target_device:
        score += 1; bd["device_match"] = True
    if example.get("target_module") == target_module:
        score += 1; bd["module_match"] = True
    if variables is not None and example.get("variables") == variables:
        score += 2; bd["variables_match"] = True
    if dataset_size_bucket:
        max_p = _DATASET_PREFERRED_MAX_PARAMS.get(dataset_size_bucket)
        param_count = _parse_model_params(example.get("model_name") or "")
        if max_p is None:
            score += 1; bd["dataset_size_match"] = True
        elif param_count is not None and param_count <= max_p:
            score += 1; bd["dataset_size_match"] = True

    return score, bd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_recommendations(
    task_type: str,
    target_device: str,
    target_module: str,
    variables: Optional[int] = None,
    dataset_size_bucket: Optional[str] = None,
    modelzoo_path: Optional[str] = None,
) -> dict[str, Any]:
    examples_root = modelzoo_path or _find_modelzoo_examples_path()
    if not examples_root:
        return {
            "success": False,
            "error": (
                "Could not find tinyml-modelzoo/examples. "
                "Set MMCLI_MODELZOO_PATH to the modelzoo repo root."
            ),
            "ranked": [],
        }

    examples = _list_examples(examples_root)
    if not examples:
        return {
            "success": False,
            "error": f"No valid examples in {examples_root}",
            "ranked": [],
        }

    scored = []
    for ex in examples:
        score, bd = _score_example(
            ex, task_type, target_device, target_module,
            variables, dataset_size_bucket,
        )
        param_count = _parse_model_params(ex.get("model_name") or "")
        scored.append({
            "model_name": ex.get("model_name"),
            "task_type": ex.get("task_type"),
            "target_device": ex.get("target_device"),
            "target_module": ex.get("target_module"),
            "variables": ex.get("variables"),
            "feature_extraction_name": ex.get("feature_extraction_name"),
            "param_count": param_count,
            "complexity_tier": _complexity_tier(param_count) if param_count else "unknown",
            "score": score,
            "match_breakdown": bd,
            "example_dir": ex.get("example_dir"),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    complexity_note = None
    if dataset_size_bucket:
        max_p = _DATASET_PREFERRED_MAX_PARAMS.get(dataset_size_bucket)
        if max_p:
            complexity_note = (
                f"Dataset '{dataset_size_bucket}': prefer models with ≤ {max_p:,} params "
                "(simpler models reduce overfitting risk on smaller datasets)."
            )
        else:
            complexity_note = (
                f"Dataset '{dataset_size_bucket}': no model complexity restriction."
            )

    return {
        "success": True,
        "error": None,
        "task_type": task_type,
        "recommended_model": scored[0]["model_name"] if scored else None,
        "recommended_fe_preset": scored[0]["feature_extraction_name"] if scored else None,
        "match_score": scored[0]["score"] if scored else 0,
        "match_breakdown": scored[0]["match_breakdown"] if scored else {},
        "dataset_complexity_note": complexity_note,
        "ranked": scored,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_recommendations(result: dict, top_n: int = 5) -> None:
    if not result["success"]:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        return

    ranked = result["ranked"]
    best = result["recommended_model"]
    fe = result["recommended_fe_preset"]
    task_type = result.get("task_type", "")
    task_guidance = _task_fe_recommendations(task_type)

    print(f"\nRecommended model:   {best or '(none)'}")
    if fe:
        print(f"Feature extraction:  {fe}")
    if result.get("dataset_complexity_note"):
        print(f"Size guidance:       {result['dataset_complexity_note']}")

    # Task-specific FE preset guidance from schema.yaml
    if task_guidance.get("recommended_presets"):
        presets = task_guidance["recommended_presets"]
        print(f"\nTask FE presets for '{task_type}':")
        for p in presets:
            print(f"  {p}")
        if task_guidance.get("preset_reason"):
            print(f"  — {task_guidance['preset_reason']}")
    if task_guidance.get("required_data_proc"):
        print(f"  Required transforms: {', '.join(task_guidance['required_data_proc'])}")
    if task_guidance.get("notes"):
        print(f"  Note: {task_guidance['notes']}")

    print(f"\nTop {min(top_n, len(ranked))} matches (score 0–6; variables match = +2):\n")
    name_w = max((len(r["model_name"] or "") for r in ranked[:top_n]), default=10) + 2
    name_w = max(name_w, 14)
    print(f"  {'Model':<{name_w}}  {'Score':>5}  {'Params':>8}  {'Tier':<7}  Match criteria")
    print(f"  {'─' * name_w}  {'─' * 5}  {'─' * 8}  {'─' * 7}  {'─' * 40}")
    for r in ranked[:top_n]:
        bd = r["match_breakdown"]
        criteria = []
        if bd.get("task_type_match"):  criteria.append("task")
        if bd.get("device_match"):     criteria.append("device")
        if bd.get("module_match"):     criteria.append("module")
        if bd.get("variables_match"):  criteria.append("variables(×2)")
        if bd.get("dataset_size_match"): criteria.append("size")
        params = f"{r['param_count']:,}" if r["param_count"] else "  ?"
        print(f"  {(r['model_name'] or '?'):<{name_w}}  {r['score']:>5}  {params:>8}  "
              f"{r['complexity_tier']:<7}  {', '.join(criteria) or '—'}")

    print()
    print("Use the recommended model with:")
    print(f"  mmcli train -m <module> -t <task_type> -d <device> "
          f"-n {best or 'MODEL_NAME'} -i <project>")
    if fe:
        print(f"  mmcli train ... --feature-extraction {fe}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_recommend(args) -> None:
    modelzoo_path = getattr(args, "modelzoo_path", None)
    if modelzoo_path:
        # If user gave the examples/ dir directly, use it; else append examples/
        if not os.path.basename(modelzoo_path) == "examples":
            candidate = os.path.join(modelzoo_path, "examples")
            if os.path.isdir(candidate):
                modelzoo_path = candidate

    task_type = args.task
    target_module = getattr(args, "module", None) or TASK_TYPE_TO_MODULE.get(task_type)
    if not target_module:
        print(
            f"ERROR: Cannot infer module from task '{task_type}'. "
            "Pass -m/--module explicitly.",
            file=sys.stderr,
        )
        sys.exit(2)

    result = get_recommendations(
        task_type=task_type,
        target_device=args.device,
        target_module=target_module,
        variables=getattr(args, "variables", None),
        dataset_size_bucket=getattr(args, "dataset_size_bucket", None),
        modelzoo_path=modelzoo_path,
    )

    # Format output based on arguments
    if args.format == "json":
        output_text = format_json(result)
    elif args.format == "csv":
        output_text = format_csv(result)
    elif args.format == "yaml":
        output_text = format_yaml(result)
    else:
        # Default to text format
        print_recommendations(result, top_n=getattr(args, "top", 5))
        if not result["success"]:
            sys.exit(1)
        return

    # Write to file or stdout
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_text)
    else:
        print(output_text)

    if not result["success"]:
        sys.exit(1)
