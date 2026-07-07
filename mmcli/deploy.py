"""
Device deployment utilities for mmcli.

Wraps the agent-skills device_deployment workflow as CLI-friendly functions.
Supports 5 subcommands:

- check-sdk — verify SDK installation for a device family
- artifacts — locate ModelMaker run outputs needed for CCS project
- create    — create CCS project from SDK template + compiled artifacts
- build     — headless CCS build
- flash     — flash compiled .out via dslite

CC1312 / CC1314 are SimpleLink LP devices; CCS type strings follow the
cc13xx lowercase convention established by the cc1352 entry in the reference.
"""

import glob
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

_DEFAULT_TINYML_BASE = os.path.expanduser("~/tinyml-tensorlab")


# ── Device family + SDK metadata ──────────────────────────────────────────────

DEVICE_FAMILY: Dict[str, str] = {
    # C2000 (F28x / F29x)
    "F280013": "c2000", "F280015": "c2000", "F28003": "c2000", "F28004": "c2000",
    "F2837":   "c2000", "F28P55":  "c2000", "F28P65": "c2000", "F29H85": "c2000",
    "F29P58":  "c2000", "F29P32":  "c2000",
    # MSPM0
    "MSPM0G3507": "mspm0", "MSPM0G3519": "mspm0", "MSPM0G5187": "mspm0",
    # MSPM33
    "MSPM33C32": "mspm33", "MSPM33C34": "mspm33",
    # AM13
    "AM13E2": "am13",
    # AM26x
    "AM263": "am26x", "AM263P": "am26x", "AM261": "am26x",
    # SimpleLink (CC27xx / CC13xx / CC35x)
    "CC2755":  "simplelink",
    "CC1312":  "simplelink", "CC1314":  "simplelink",
    "CC1352":  "simplelink", "CC1354":  "simplelink",
    "CC35X1":  "simplelink",
}

# Maps canonical device ID → CCS device_type string (used as folder name in SDK templates)
DEVICE_CCS_TYPE: Dict[str, str] = {
    "F28P55":     "f28p55x",
    "F28P65":     "f28p65x",
    "F28004":     "f28004x",
    "F2837":      "f2837x",
    "MSPM0G3507": "mspm0g3507",
    "MSPM0G5187": "mspm0g5187",
    "AM263":      "am263",
    "CC2755":     "cc2755",
    "CC1312":     "cc1312",
    "CC1314":     "cc1314",
    "CC1352":     "cc1352",
    "CC1354":     "cc1354",
}

SDK_INFO: Dict[str, Dict] = {
    "c2000": {
        "name":          "C2000Ware",
        "install_globs": [
            os.path.expanduser("~/ti/c2000ware_*"),
            os.path.expanduser("~/ti/C2000Ware_*"),
            "/opt/ti/c2000ware_*",
            "/opt/ti/C2000Ware_*",
            os.path.expanduser("~/ti/ccs*/c2000/C2000Ware_*"),
        ],
        "ai_examples_subpath": "libraries/ai/examples",
        "download_url": "https://www.ti.com/tool/C2000WARE",
    },
    "mspm0": {
        "name":          "MSPM0 SDK",
        "install_globs": [
            os.path.expanduser("~/ti/mspm0_sdk_*"),
            "/opt/ti/mspm0_sdk_*",
        ],
        "ai_examples_subpath": "examples",
        "download_url": "https://www.ti.com/tool/MSPM0-SDK",
    },
    "mspm33": {
        "name":          "MSPM33 SDK",
        "install_globs": [
            os.path.expanduser("~/ti/mspm33_sdk_*"),
            "/opt/ti/mspm33_sdk_*",
        ],
        "ai_examples_subpath": "examples",
        "download_url": "https://www.ti.com/tool/download/MSPM33-SDK",
    },
    "am13": {
        "name":          "MCU+ SDK (AM13x)",
        "install_globs": [
            os.path.expanduser("~/ti/mcu_plus_sdk_am263x_*"),
            os.path.expanduser("~/ti/mcu_plus_sdk_*"),
            "/opt/ti/mcu_plus_sdk_*",
        ],
        "ai_examples_subpath": "examples",
        "download_url": "https://www.ti.com/tool/MCU-PLUS-SDK-AM263X",
    },
    "am26x": {
        "name":          "MCU+ SDK (AM26x)",
        "install_globs": [
            os.path.expanduser("~/ti/mcu_plus_sdk_am263x_*"),
            os.path.expanduser("~/ti/mcu_plus_sdk_*"),
            "/opt/ti/mcu_plus_sdk_*",
        ],
        "ai_examples_subpath": "examples",
        "download_url": "https://www.ti.com/tool/MCU-PLUS-SDK-AM263X",
    },
    "simplelink": {
        "name":          "SimpleLink Low Power SDK",
        "install_globs": [
            os.path.expanduser("~/ti/simplelink_cc13xx_cc26xx_sdk_*"),
            os.path.expanduser("~/ti/simplelink_lowpower_f3_sdk_*"),
            "/opt/ti/simplelink_*",
        ],
        "ai_examples_subpath": "examples",
        "download_url": "https://www.ti.com/tool/SIMPLELINK-LOWPOWER-F3-SDK",
    },
}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _device_family(device_id: str) -> Optional[str]:
    return DEVICE_FAMILY.get(device_id) or DEVICE_FAMILY.get(device_id.upper())


def _find_sdk_root(family: str) -> Optional[str]:
    matches = []
    for pattern in SDK_INFO.get(family, {}).get("install_globs", []):
        matches += glob.glob(pattern)
    return sorted(matches)[-1] if matches else None


def _sdk_ai_examples_path(family: str, sdk_root: str) -> str:
    subpath = SDK_INFO.get(family, {}).get("ai_examples_subpath", "examples")
    return os.path.join(sdk_root, subpath)


def _runs_path(tinyml_base: Optional[str]) -> str:
    base = os.path.expanduser(tinyml_base) if tinyml_base else _DEFAULT_TINYML_BASE
    return os.path.join(base, "tinyml-modelmaker", "data", "projects")


def _golden_dir(tinyml_base, task_type, run_id, model_id, quantization) -> str:
    sub = "quantization" if quantization else "base"
    run_dir = os.path.join(_runs_path(tinyml_base), task_type, "run", run_id, model_id)
    return os.path.join(run_dir, "training", sub, "golden_vectors")


def _get_file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _validate_copied_artifacts(src_dir: str, dst_dir: str) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    tolerance = 0.5  # seconds

    if not os.path.isdir(src_dir):
        return False, [f"Source directory does not exist: {src_dir}"]
    if not os.path.isdir(dst_dir):
        return False, [f"Destination directory does not exist: {dst_dir}"]

    for root, _, files in os.walk(dst_dir):
        for fname in files:
            dst_file = os.path.join(root, fname)
            rel = os.path.relpath(dst_file, dst_dir)
            src_file = os.path.join(src_dir, rel)
            src_mtime = _get_file_mtime(src_file)
            dst_mtime = _get_file_mtime(dst_file)
            if src_mtime == 0.0:
                errors.append(f"Source file missing: {src_file} (but exists in dst: {dst_file})")
            elif abs(src_mtime - dst_mtime) > tolerance:
                errors.append(
                    f"Timestamp mismatch for {rel}: "
                    f"src={src_mtime:.1f} dst={dst_mtime:.1f} "
                    f"(diff={abs(src_mtime - dst_mtime):.1f}s)"
                )
    return len(errors) == 0, errors


# ── check_sdk ─────────────────────────────────────────────────────────────────

def check_sdk(device: str, sdk_path: Optional[str] = None) -> Dict[str, Any]:
    family = _device_family(device)
    if not family:
        return {
            "found": False, "family": None, "sdk_name": None,
            "sdk_root": None, "ai_examples_path": None,
            "errors": [f"Unknown device '{device}'. Cannot determine required SDK."],
        }

    info = SDK_INFO[family]
    sdk_name = info["name"]

    if sdk_path:
        sdk_root = os.path.expanduser(sdk_path)
        if not os.path.isdir(sdk_root):
            return {
                "found": False, "family": family, "sdk_name": sdk_name,
                "sdk_root": sdk_root, "ai_examples_path": None,
                "errors": [f"Provided sdk_path does not exist: {sdk_root}"],
            }
    else:
        sdk_root = _find_sdk_root(family)

    if not sdk_root:
        return {
            "found": False, "family": family, "sdk_name": sdk_name,
            "sdk_root": None, "ai_examples_path": None,
            "errors": [
                f"{sdk_name} not found. Searched: {info.get('install_globs', [])}",
                f"Download from: {info['download_url']}",
            ],
        }

    ai_path = _sdk_ai_examples_path(family, sdk_root)
    ai_exists = os.path.isdir(ai_path)
    return {
        "found": True, "family": family, "sdk_name": sdk_name,
        "sdk_root": sdk_root,
        "ai_examples_path": ai_path if ai_exists else None,
        "ai_examples_exists": ai_exists,
        "errors": [] if ai_exists else [
            f"SDK found at {sdk_root} but AI examples not found: {ai_path}. "
            "Ensure the SDK version includes AI library examples."
        ],
    }


# ── find_artifacts ─────────────────────────────────────────────────────────────

def find_artifacts(
    task_type: str,
    run_id: str,
    model_id: str,
    quantization: bool,
    tinyml_base: Optional[str] = None,
) -> Dict[str, Any]:
    run_dir = os.path.join(_runs_path(tinyml_base), task_type, "run", run_id, model_id)
    art_dir = os.path.join(run_dir, "compilation", "artifacts")
    gold_dir = _golden_dir(tinyml_base, task_type, run_id, model_id, quantization)

    required = {
        "mod.a":               os.path.join(art_dir,  "mod.a"),
        "tvmgen_default.h":    os.path.join(art_dir,  "tvmgen_default.h"),
        "test_vector.c":       os.path.join(gold_dir, "test_vector.c"),
        "user_input_config.h": os.path.join(gold_dir, "user_input_config.h"),
    }
    file_status = {n: {"path": p, "exists": os.path.exists(p)} for n, p in required.items()}
    missing = [n for n, s in file_status.items() if not s["exists"]]

    extra: List[str] = []
    if os.path.isdir(art_dir):
        extra = sorted(os.listdir(art_dir))

    return {
        "success":        len(missing) == 0,
        "run_dir":        run_dir,
        "artifacts_dir":  art_dir,
        "golden_dir":     gold_dir,
        "required_files": file_status,
        "extra_artifacts": extra,
        "missing": missing,
        "errors": [f"Missing: {n} (expected: {file_status[n]['path']})" for n in missing],
        "hint": (
            "Check that training and compilation completed successfully. "
            f"run_id is the timestamp folder under {_runs_path(tinyml_base)}/{task_type}/run/"
        ) if missing else "",
    }


# ── CCS project creator ────────────────────────────────────────────────────────

class _Deployer:
    def __init__(self, run_id, task_type, quantization, model_id, tinyml_base=None):
        self.run_id = run_id
        self.task_type = task_type
        self.quantization = quantization
        self.model_id = model_id
        runs = _runs_path(tinyml_base)
        self.artifacts_dir = os.path.join(runs, task_type, "run", run_id, model_id, "compilation", "artifacts")
        self.golden_dir = _golden_dir(tinyml_base, task_type, run_id, model_id, quantization)

    def create(self, project_name, device_type, templates_path):
        task_norm = self.task_type.lower().replace(" ", "_")
        task_dir = os.path.join(templates_path, task_norm)
        device_src = os.path.join(task_dir, device_type)

        if not os.path.exists(device_src):
            raise FileNotFoundError(
                f"Template not found: {device_src}\n"
                f"Searched SDK AI examples at: {templates_path}\n"
                "Verify the SDK is installed and the templates path points to its AI examples folder."
            )

        new_project = os.path.join(templates_path, project_name)
        if os.path.exists(new_project):
            raise FileExistsError(f"Project already exists: {new_project}")

        device_dst = os.path.join(new_project, device_type)
        shutil.copytree(device_src, device_dst)

        app_src = os.path.join(task_dir, "application_main.c")
        if os.path.exists(app_src):
            shutil.copy2(app_src, os.path.join(new_project, "application_main.c"))

        ccs_dir = os.path.join(device_dst, "CCS")
        old_spec = os.path.join(ccs_dir, f"{device_type}_{task_norm}.projectspec")
        new_spec = os.path.join(ccs_dir, f"{device_type}_{project_name}.projectspec")

        if os.path.exists(old_spec):
            self._update_projectspec(old_spec, new_spec, project_name, task_norm)
            os.remove(old_spec)
        else:
            raise FileNotFoundError(f"Projectspec not found: {old_spec}")

        art_ok = self._copy_artifacts(device_dst)
        gld_ok = self._copy_golden_files(device_dst)

        val = {"timestamp_validation_passed": True, "validation_errors": []}
        if art_ok:
            ok, errs = _validate_copied_artifacts(self.artifacts_dir, os.path.join(device_dst, "artifacts"))
            if not ok:
                val["timestamp_validation_passed"] = False
                val["validation_errors"].extend([f"Artifacts: {e}" for e in errs])
        if gld_ok:
            ok, errs = _validate_copied_artifacts(self.golden_dir, device_dst)
            if not ok:
                val["timestamp_validation_passed"] = False
                val["validation_errors"].extend([f"Golden vectors: {e}" for e in errs])

        return new_project, val

    def _update_projectspec(self, old, new, project_name, task_norm):
        tree = ET.parse(old)
        root = tree.getroot()
        for elem in root.iter():
            if elem.text and isinstance(elem.text, str):
                elem.text = elem.text.replace(task_norm, project_name)
            for k, v in elem.attrib.items():
                if isinstance(v, str):
                    elem.set(k, v.replace(task_norm, project_name))
        tree.write(new, encoding="utf-8", xml_declaration=False)

    def _copy_artifacts(self, project_path) -> bool:
        src = self.artifacts_dir
        dst = os.path.join(project_path, "artifacts")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        if os.path.exists(src):
            shutil.copytree(src, dst)
            return True
        return False

    def _copy_golden_files(self, project_root) -> bool:
        if not os.path.isdir(self.golden_dir):
            return False
        copied = False
        for fname in ("test_vector.c", "user_input_config.h"):
            src_file = os.path.join(self.golden_dir, fname)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, os.path.join(project_root, fname))
                copied = True
        return copied


def create_project(
    project_name: str,
    device: str,
    device_type: str,
    run_id: str,
    task_type: str,
    model_id: str,
    quantization: bool,
    tinyml_base: Optional[str] = None,
    sdk_path: Optional[str] = None,
    ccs_templates_path: Optional[str] = None,
) -> Dict[str, Any]:
    if ccs_templates_path:
        templates_path = os.path.expanduser(ccs_templates_path)
        sdk_meta = {"found": True, "family": None, "sdk_name": "custom", "sdk_root": None}
    else:
        sdk_meta = check_sdk(device, sdk_path)
        if not sdk_meta["found"]:
            return {"success": False, "project_path": None,
                    "family": sdk_meta.get("family"), "sdk_name": sdk_meta.get("sdk_name"),
                    "errors": sdk_meta["errors"]}
        if not sdk_meta.get("ai_examples_path"):
            return {"success": False, "project_path": None,
                    "family": sdk_meta["family"], "sdk_name": sdk_meta["sdk_name"],
                    "errors": sdk_meta["errors"]}
        templates_path = sdk_meta["ai_examples_path"]

    try:
        deployer = _Deployer(run_id, task_type, quantization, model_id, tinyml_base)
        project_path, val = deployer.create(project_name, device_type, templates_path)
        result: Dict[str, Any] = {
            "success": True,
            "project_path": project_path,
            "family": sdk_meta.get("family"),
            "sdk_name": sdk_meta.get("sdk_name"),
            "sdk_root": sdk_meta.get("sdk_root"),
            "errors": [],
            "timestamp_validation_passed": val["timestamp_validation_passed"],
            "validation_warnings": val["validation_errors"],
            "next_step": f"Build the project: mmcli deploy build --project-path '{project_path}' --ccs-path <CCS_ROOT>",
        }
        if not val["timestamp_validation_passed"]:
            result["errors"].extend(val["validation_errors"])
            result["critical_warning"] = (
                "TIMESTAMP VALIDATION FAILED: Copied artifacts may be from template or stale cache. "
                "Verify manually before building."
            )
        return result
    except FileNotFoundError as e:
        family = sdk_meta.get("family")
        sdk_info = SDK_INFO.get(family, {}) if family else {}
        errors = [str(e)]
        if sdk_info.get("download_url"):
            errors.append(f"Ensure {sdk_info['name']} is fully installed: {sdk_info['download_url']}")
        return {"success": False, "project_path": None, "family": family, "errors": errors}
    except FileExistsError as e:
        return {"success": False, "project_path": None, "errors": [str(e)]}
    except Exception as e:
        return {"success": False, "project_path": None, "errors": [f"Unexpected error: {e}"]}


# ── build_project ──────────────────────────────────────────────────────────────

def build_project(
    project_path: str,
    ccs_install_path: str,
    workspace_path: Optional[str] = None,
    build_type: str = "full",
) -> Dict[str, Any]:
    project_path = os.path.abspath(os.path.expanduser(project_path))
    ccs_path = os.path.expanduser(ccs_install_path)

    candidates = [
        os.path.join(ccs_path, "eclipse", "ccstudio"),
        os.path.join(ccs_path, "eclipse", "ccstudio.exe"),
        os.path.join(ccs_path, "eclipse", "eclipse"),
    ]
    launcher = next((c for c in candidates if os.path.isfile(c)), None)
    if not launcher:
        return {"success": False, "stdout": "", "stderr": "",
                "errors": [f"CCS launcher not found. Tried: {candidates}. "
                           "Verify ccs_install_path points to the CCS installation root."]}

    ws = workspace_path or f"/tmp/ccs_ws_{os.getpid()}"
    cmd = [
        launcher, "--launcher.suppressErrors", "-noSplash",
        "-data", ws,
        "-application", "com.ti.ccstudio.apps.projectBuild",
        "-ccs.projects", project_path,
        "-ccs.buildType", build_type,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        success = result.returncode == 0

        out_file = None
        pname = os.path.basename(project_path)
        for cfg in ("Debug", "Release"):
            candidate = os.path.join(project_path, cfg, f"{pname}.out")
            if os.path.isfile(candidate):
                out_file = candidate
                break

        return {
            "success": success,
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "out_file": out_file,
            "errors": [] if success else [
                f"Build failed (exit {result.returncode}). "
                "Check stderr. Common causes: missing include paths, linker errors, "
                "or missing CCS device support for the target."
            ],
            "next_step": f"Flash: mmcli deploy flash --project-path '{project_path}' --ccs-path '{ccs_install_path}'" if out_file else "",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "", "errors": ["Build timed out (10 min)."]}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": "", "errors": [str(e)]}


# ── flash_project ──────────────────────────────────────────────────────────────

def flash_project(
    project_path: str,
    ccs_install_path: str,
    project_name: Optional[str] = None,
    ccxml_path: Optional[str] = None,
    out_file: Optional[str] = None,
) -> Dict[str, Any]:
    project_path = os.path.abspath(os.path.expanduser(project_path))
    ccs_path = os.path.expanduser(ccs_install_path)
    pname = project_name or os.path.basename(project_path)

    if out_file:
        binary = os.path.expanduser(out_file)
    else:
        binary = None
        for cfg in ("Debug", "Release"):
            candidate = os.path.join(project_path, cfg, f"{pname}.out")
            if os.path.isfile(candidate):
                binary = candidate
                break
        if not binary:
            return {"success": False, "stdout": "", "stderr": "",
                    "errors": [f"Binary not found: {project_path}/Debug/{pname}.out. Build first."]}

    dslite_candidates = [
        os.path.join(ccs_path, "ccs_base", "DebugServer", "bin", "dslite.sh"),
        os.path.join(ccs_path, "ccs_base", "DebugServer", "bin", "dslite.bat"),
    ]
    dslite = next((d for d in dslite_candidates if os.path.isfile(d)), None)
    if not dslite:
        return {"success": False, "stdout": "", "stderr": "",
                "errors": [f"dslite not found. Tried: {dslite_candidates}. Verify ccs_install_path."]}

    if ccxml_path:
        ccxml = os.path.expanduser(ccxml_path)
    else:
        search_dirs = [project_path, os.path.join(project_path, "CCS")]
        ccxml_files = []
        for d in search_dirs:
            if os.path.isdir(d):
                ccxml_files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".ccxml")]
        if not ccxml_files:
            return {"success": False, "stdout": "", "stderr": "",
                    "errors": ["No .ccxml target config found. Provide --ccxml explicitly."]}
        lp = [f for f in ccxml_files if "LaunchPad" in os.path.basename(f) or
              "launchpad" in os.path.basename(f).lower()]
        ccxml = lp[0] if lp else ccxml_files[0]

    if not os.path.isfile(ccxml):
        return {"success": False, "stdout": "", "stderr": "",
                "errors": [f"ccxml file not found: {ccxml}"]}

    cmd = ["bash", dslite, "--mode", "flash", "--ccxml", ccxml, binary]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        success = result.returncode == 0
        return {
            "success": success,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "out_file": binary,
            "ccxml": ccxml,
            "errors": [] if success else [
                f"Flash failed (exit {result.returncode}). "
                "Check device connection, .ccxml path, and USB drivers."
            ],
            "next_step": (
                "Flash succeeded. In CCS debug perspective: Resume (F8), "
                "check test_result == 1 in Watch window to verify inference."
            ) if success else "",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "", "errors": ["Flash timed out. Check device connection."]}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": "", "errors": [str(e)]}


# ── CLI entry points ───────────────────────────────────────────────────────────

def run_deploy_check_sdk(args) -> None:
    result = check_sdk(args.device, getattr(args, "sdk_path", None))
    if result["found"]:
        print(f"SDK:       {result['sdk_name']}")
        print(f"Family:    {result['family']}")
        print(f"Root:      {result['sdk_root']}")
        ai = result.get("ai_examples_path")
        print(f"AI examples: {ai or '(not found)'}")
        if result.get("errors"):
            print("Warnings:")
            for e in result["errors"]:
                print(f"  {e}")
        ccs_type = DEVICE_CCS_TYPE.get(args.device)
        if ccs_type:
            print(f"\nCCS device_type for {args.device}: {ccs_type}")
        print("\nSDK installation OK." if not result["errors"] else "\nSDK found but with warnings.")
    else:
        print(f"ERROR: SDK not found for device '{args.device}'.", file=sys.stderr)
        for e in result["errors"]:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)


def run_deploy_artifacts(args) -> None:
    result = find_artifacts(
        task_type=args.task,
        run_id=args.run_id,
        model_id=args.model_id,
        quantization=not getattr(args, "no_quantization", False),
        tinyml_base=getattr(args, "tinyml_base", None),
    )
    if result["success"]:
        print("All required artifacts found:")
        for name, info in result["required_files"].items():
            print(f"  {name}: {info['path']}")
    else:
        print("Missing artifacts:", file=sys.stderr)
        for e in result["errors"]:
            print(f"  {e}", file=sys.stderr)
        if result["hint"]:
            print(f"\nHint: {result['hint']}", file=sys.stderr)
        sys.exit(1)


def run_deploy_create(args) -> None:
    device = args.device
    device_type = getattr(args, "device_type", None) or DEVICE_CCS_TYPE.get(device)
    if not device_type:
        print(
            f"ERROR: No CCS device_type known for '{device}'. "
            "Pass --device-type explicitly (e.g., f28p55x).",
            file=sys.stderr,
        )
        sys.exit(2)

    result = create_project(
        project_name=args.project_name,
        device=device,
        device_type=device_type,
        run_id=args.run_id,
        task_type=args.task,
        model_id=args.model_id,
        quantization=not getattr(args, "no_quantization", False),
        tinyml_base=getattr(args, "tinyml_base", None),
        sdk_path=getattr(args, "sdk_path", None),
        ccs_templates_path=getattr(args, "ccs_templates_path", None),
    )
    if result["success"]:
        print(f"Project created: {result['project_path']}")
        if result.get("critical_warning"):
            print(f"\nWARNING: {result['critical_warning']}", file=sys.stderr)
        print(f"\nNext step: {result['next_step']}")
    else:
        print("ERROR: Failed to create CCS project.", file=sys.stderr)
        for e in result["errors"]:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)


def run_deploy_build(args) -> None:
    result = build_project(
        project_path=args.project_path,
        ccs_install_path=args.ccs_path,
        workspace_path=getattr(args, "workspace", None),
        build_type=getattr(args, "build_type", "full"),
    )
    if result["stdout"]:
        print(result["stdout"])
    if result["stderr"]:
        print(result["stderr"], file=sys.stderr)

    if result["success"]:
        print(f"\nBuild succeeded.")
        if result.get("out_file"):
            print(f"Output: {result['out_file']}")
        if result.get("next_step"):
            print(f"Next: {result['next_step']}")
    else:
        print("\nBuild FAILED.", file=sys.stderr)
        for e in result["errors"]:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)


def run_deploy_flash(args) -> None:
    result = flash_project(
        project_path=args.project_path,
        ccs_install_path=args.ccs_path,
        project_name=getattr(args, "project_name", None),
        ccxml_path=getattr(args, "ccxml", None),
        out_file=getattr(args, "out_file", None),
    )
    if result["stdout"]:
        print(result["stdout"])
    if result["stderr"]:
        print(result["stderr"], file=sys.stderr)

    if result["success"]:
        print(f"\nFlash succeeded: {result['out_file']}")
        if result.get("next_step"):
            print(result["next_step"])
    else:
        print("\nFlash FAILED.", file=sys.stderr)
        for e in result["errors"]:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)
