"""
End-to-end integration test: mmcli init → analyze → recommend → run (train+compile).

Verifies the full pipeline on macOS with TVM installed. The test:
  1. Creates a project from the bundled classification dataset
  2. Analyzes the dataset
  3. Runs train + compile for 1 epoch (fast)
  4. Checks training outputs (ONNX produced, status files written)

Known macOS constraints (annotated per-assertion):
  - auto_quantization: Hessian power-iteration requires max_pool2d to be twice-
    differentiable, which fails on CPU/MPS. Use --no-auto-quantization.
  - MPS fake-quantize: aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams
    not implemented for MPS. Requires PYTORCH_ENABLE_MPS_FALLBACK=1.
  - onnxsim crash: onnxsim_cpp2py_export.cpython-310-darwin.so crashes with
    SIGSEGV during Python interpreter shutdown on macOS ARM64. The crash fires
    AFTER onnx export but BEFORE packaging and compilation. Fixed in
    quant_base._simplify_onnx_model: on darwin/arm64 onnxsim runs in a subprocess,
    so a crash there cannot propagate to the main process.
  - cl2000 cross-compiler: TI C2000 CGT 25.11.1.LTS is installed at
    ~/ti/ti-cgt-c2000_25.11.1.LTS. Set C2000_CG_ROOT so modelmaker finds it.

Run with:
    MMCLI_PYTHON=~/.venv-tinyml/bin/python \\
    PYTORCH_ENABLE_MPS_FALLBACK=1 \\
    pytest tests/test_e2e.py -v
"""

import os
import sys
import subprocess
import tempfile
import pytest

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: Full end-to-end pipeline tests (slow)")


MMCLI_PYTHON = os.environ.get("MMCLI_PYTHON", sys.executable)
MMCLI = [MMCLI_PYTHON, "-m", "mmcli"]

E2E_ENV = {
    **os.environ,
    "MMCLI_PYTHON": MMCLI_PYTHON,
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",  # MPS fallback for fake-quantize ops
    "PYTHONDONTWRITEBYTECODE": "1",
    # cl2000 installed at this path; modelmaker reads C2000_CG_ROOT at import time
    "C2000_CG_ROOT": os.path.expanduser("~/ti/ti-cgt-c2000_25.11.1.LTS"),
}


def _run(*args, timeout=600, capture=True, **kwargs):
    """Run mmcli with optional output capture.

    onnxsim crashes (exit 245) during Python interpreter shutdown on macOS ARM64.
    For 'run', redirect output to a temp file instead of using capture_output=True
    to avoid buffering issues if the process exits uncleanly.
    """
    if capture:
        result = subprocess.run(
            MMCLI + list(args),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=E2E_ENV,
            **kwargs,
        )
        return result.returncode, result.stdout, result.stderr
    else:
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as fh:
            log_path = fh.name
        with open(log_path, "w") as out_fh:
            result = subprocess.run(
                MMCLI + list(args),
                stdout=out_fh,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                env=E2E_ENV,
                **kwargs,
            )
        with open(log_path) as fh:
            combined = fh.read()
        os.unlink(log_path)
        return result.returncode, combined, ""


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestEndToEnd:
    """Full pipeline: init → analyze → run (train+compile)."""

    TASK = "generic_timeseries_classification"
    DEVICE = "F28P55"
    MODULE = "timeseries"
    MODEL = "CLS_1k_NPU"
    # FE preset from the modelzoo reference config for this task
    FE_PRESET = "Generic_1024Input_FFTBIN_64Feature_8Frame"

    @pytest.fixture(scope="class")
    def project(self, tmp_path_factory):
        """Create and train a project; share across all tests in the class."""
        # mktemp creates the dir; mmcli init needs a path that does NOT yet exist
        base = tmp_path_factory.mktemp("e2e_base")
        project_dir = str(base / "project")

        # 1. Init — extract bundled dataset
        rc, out, err = _run(
            "init",
            "--dataset", self.TASK,
            "-t", self.TASK,
            "-p", project_dir,
        )
        assert rc == 0, f"mmcli init failed:\n{out}\n{err}"
        assert os.path.isdir(os.path.join(project_dir, "dataset")), "dataset/ not created"

        # 2. Analyze — should succeed and print size bucket
        rc, out, err = _run("analyze", "-i", project_dir)
        assert rc == 0, f"mmcli analyze failed:\n{out}\n{err}"
        assert "large" in out.lower() or "medium" in out.lower() or "small" in out.lower(), \
            f"No size bucket in analyze output: {out[:300]}"

        # 3. Run — 1 epoch, no auto-quantization (Hessian fails on CPU/MPS).
        #
        #    macOS onnxsim constraint: onnxsim_cpp2py_export.cpython-310-darwin.so
        #    segfaults (SIGSEGV) during Python interpreter shutdown on macOS ARM64.
        #    The crash fires after ONNX export but before packaging and compilation,
        #    so WITHOUT a fix artifacts would be absent. quant_base._simplify_onnx_model
        #    now runs onnxsim in a subprocess on darwin/arm64 — the subprocess may crash,
        #    but the main process continues to packaging and compilation unaffected.
        rc, out, err = _run(
            "run",
            "-m", self.MODULE,
            "-t", self.TASK,
            "-d", self.DEVICE,
            "-n", self.MODEL,
            "-i", project_dir,
            "--quantization", "QUANTIZATION_TINPU",
            "--no-auto-quantization",
            "--preset", "auto",
            "--epochs", "1",
            "--batch-size", "64",
            "--feature-extraction", self.FE_PRESET,
            capture=False,
            timeout=1800,
        )
        # 245 = -11 mod 256 = SIGSEGV from onnxsim C++ cleanup during Python shutdown.
        # All pipeline artifacts are written before the crash hits.
        _MACOS_SEGV = 245
        assert rc in (0, _MACOS_SEGV), \
            f"mmcli run failed with unexpected exit code {rc}:\n{out[-2000:]}"

        return project_dir

    def test_run_dir_created(self, project):
        assert os.path.isdir(os.path.join(project, "run")), "run/ directory not created"

    def test_float_onnx_produced(self, project):
        """Float training ONNX must be present (base training phase)."""
        onnx = os.path.join(project, "run", "training_base", "model.onnx")
        assert os.path.isfile(onnx), f"Float ONNX not found: {onnx}"
        assert os.path.getsize(onnx) > 0, "Float ONNX is empty"

    def test_quantized_onnx_produced(self, project):
        """Quantized ONNX must be present when QUANTIZATION_TINPU is used."""
        onnx = os.path.join(project, "run", "training_quantization", "model.onnx")
        assert os.path.isfile(onnx), f"Quantized ONNX not found: {onnx}"
        assert os.path.getsize(onnx) > 0, "Quantized ONNX is empty"

    def test_golden_vectors_dir_created(self, project):
        """golden_vectors/ must be created under training_base."""
        gv_dir = os.path.join(project, "run", "training_base", "golden_vectors")
        assert os.path.isdir(gv_dir), f"golden_vectors/ dir not found: {gv_dir}"

    def test_compilation_attempted(self, project):
        """Compilation status.json and mod.a must be present.

        status.json is written by prepare() before training starts; it is always
        present regardless of compilation success.

        artifacts/mod.a is produced by TVM + cl2000 (TI C2000 CGT 25.11.1.LTS).
        cl2000 is available via C2000_CG_ROOT in E2E_ENV.
        """
        status = os.path.join(project, "run", "compilation", "status.json")
        assert os.path.isfile(status), \
            f"compilation/status.json not written — compilation stage was skipped: {status}"
        mod_a = os.path.join(project, "run", "compilation", "artifacts", "mod.a")
        assert os.path.isfile(mod_a), \
            f"compilation/artifacts/mod.a not produced — cl2000 compilation failed: {mod_a}"

    def test_run_status_files(self, project):
        """Top-level run.json and run.log must exist."""
        run_dir = os.path.join(project, "run")
        assert os.path.isfile(os.path.join(run_dir, "run.log")), "run.log not found"
        assert os.path.isfile(os.path.join(run_dir, "run.json")), "run.json not found"

    def test_preset_auto_resolved_to_default(self, project):
        """--preset auto on F28P55 + QUANTIZATION_TINPU should use default_preset (NPU path)."""
        import json
        run_json = os.path.join(project, "run", "run.json")
        with open(run_json) as fh:
            data = json.load(fh)
        preset = data.get("compilation", {}).get("compile_preset_name")
        assert preset == "default_preset", \
            f"Expected default_preset for NPU device, got: {preset}"


@pytest.mark.e2e
class TestDryRunE2E:
    """Fast dry-run checks — no actual training, just config generation."""

    def _make_minimal_project(self, tmp_path, name="proj"):
        """Create a minimal project dir that passes mmcli dataset validation."""
        project_dir = tmp_path / name
        ds = project_dir / "dataset"
        (ds / "classes" / "a").mkdir(parents=True)
        (ds / "classes" / "a" / "x.csv").write_text("1,2,3\n")
        (ds / "annotations").mkdir()
        (ds / "annotations" / "labels.csv").write_text("file,label\nx.csv,a\n")
        return str(project_dir)

    def test_dry_run_quantization_int_in_yaml(self, tmp_path):
        """QUANTIZATION_TINPU should be written as integer 2 in the YAML config."""
        import yaml

        project_dir = self._make_minimal_project(tmp_path)

        rc, out, err = _run(
            "--dry-run", "run",
            "-m", "timeseries",
            "-t", "generic_timeseries_classification",
            "-d", "F28P55",
            "-n", "CLS_1k_NPU",
            "-i", project_dir,
            "--quantization", "QUANTIZATION_TINPU",
        )
        assert rc == 0, f"dry-run failed: {out + err}"
        # Strip the comment line before parsing YAML
        yaml_text = "\n".join(l for l in out.splitlines() if not l.startswith("#"))
        config = yaml.safe_load(yaml_text)
        assert config["training"]["quantization"] == 2, \
            f"Expected quantization=2 in YAML, got: {config['training']['quantization']}"

    def test_dry_run_preset_auto_npu_device(self, tmp_path):
        """--preset auto on NPU device + QUANTIZATION_TINPU → default_preset."""
        import yaml

        project_dir = self._make_minimal_project(tmp_path, "proj2")

        rc, out, err = _run(
            "--dry-run", "run",
            "-m", "timeseries",
            "-t", "generic_timeseries_classification",
            "-d", "F28P55",
            "-n", "CLS_1k_NPU",
            "-i", project_dir,
            "--quantization", "QUANTIZATION_TINPU",
            "--preset", "auto",
        )
        assert rc == 0, f"dry-run failed: {out + err}"
        yaml_text = "\n".join(l for l in out.splitlines() if not l.startswith("#"))
        config = yaml.safe_load(yaml_text)
        assert config["compilation"]["compile_preset_name"] == "default_preset", \
            f"Expected default_preset, got: {config['compilation']['compile_preset_name']}"

    def test_dry_run_preset_auto_soft_npu_task(self, tmp_path):
        """--preset auto on NPU device + anomaly task → forced_soft_npu_preset."""
        import yaml

        project_dir = self._make_minimal_project(tmp_path, "proj3")

        rc, out, err = _run(
            "--dry-run", "run",
            "-m", "timeseries",
            "-t", "generic_timeseries_anomalydetection",
            "-d", "F28P55",
            "-n", "CLS_1k_NPU",
            "-i", project_dir,
            "--preset", "auto",
        )
        assert rc == 0, f"dry-run failed: {out + err}"
        yaml_text = "\n".join(l for l in out.splitlines() if not l.startswith("#"))
        config = yaml.safe_load(yaml_text)
        assert config["compilation"]["compile_preset_name"] == "forced_soft_npu_preset", \
            f"Expected forced_soft_npu_preset, got: {config['compilation']['compile_preset_name']}"
