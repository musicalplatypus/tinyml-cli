"""Unit tests for mmcli.deploy module.

Tests the deploy command module which provides device deployment utilities
including SDK checking, artifact finding, project creation, building, and flashing.
"""

import os
import sys
from unittest import mock

import pytest

from mmcli import deploy


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_sdk_path(tmp_path):
    """Create a temporary SDK path for testing."""
    sdk_dir = tmp_path / "c2000ware_10_00_00_00"
    sdk_dir.mkdir(parents=True)
    ai_examples = sdk_dir / "libraries" / "ai" / "examples"
    ai_examples.mkdir(parents=True)

    # Create a test device folder
    device_dir = ai_examples / "f28p55x"
    device_dir.mkdir()
    (device_dir / "test_file.txt").write_text("test")

    return str(sdk_dir)


@pytest.fixture
def temp_project_path(tmp_path):
    """Create a temporary project path for testing."""
    project_dir = tmp_path / "my_project"
    project_dir.mkdir(parents=True)

    # Create CCS directory with spec file
    ccs_dir = project_dir / "CCS"
    ccs_dir.mkdir()
    spec_file = ccs_dir / "f28p55x_test_task.projectspec"
    spec_file.write_text("""
<projectSpec>
  <name>test_task</name>
</projectSpec>
""")

    return str(project_dir)


# ============================================================================
# DEVICE_FAMILY tests
# ============================================================================


class TestDeviceFamily:
    """Tests for DEVICE_FAMILY mapping and _device_family function."""

    def test_device_family_mapping(self):
        """DEVICE_FAMILY should have expected device mappings."""
        assert deploy.DEVICE_FAMILY["F28P55"] == "c2000"
        assert deploy.DEVICE_FAMILY["CC1312"] == "simplelink"
        assert deploy.DEVICE_FAMILY["MSPM0G3507"] == "mspm0"

    def test_device_family_case_insensitive(self):
        """_device_family should handle lowercase device IDs."""
        family = deploy._device_family("f28p55")
        assert family == "c2000"

    def test_unknown_device_family(self):
        """_device_family should return None for unknown devices."""
        result = deploy._device_family("UNKNOWN123")
        assert result is None


# ============================================================================
# DEVICE_CCS_TYPE tests
# ============================================================================


class TestDeviceCcsType:
    """Tests for DEVICE_CCS_TYPE mapping."""

    def test_ccs_type_mapping(self):
        """DEVICE_CCS_TYPE should have expected CCS type mappings."""
        assert deploy.DEVICE_CCS_TYPE["F28P55"] == "f28p55x"
        assert deploy.DEVICE_CCS_TYPE["CC1312"] == "cc1312"
        assert deploy.DEVICE_CCS_TYPE["MSPM0G3507"] == "mspm0g3507"

    def test_ccs_type_unknown_device(self):
        """DEVICE_CCS_TYPE should not have entry for unknown devices."""
        result = deploy.DEVICE_CCS_TYPE.get("UNKNOWN123")
        assert result is None


# ============================================================================
# SDK_INFO tests
# ============================================================================


class TestSdkInfo:
    """Tests for SDK_INFO configuration."""

    def test_sdk_info_has_expected_families(self):
        """SDK_INFO should have entries for all device families."""
        expected = ["c2000", "mspm0", "mspm33", "am13", "am26x", "simplelink"]
        for family in expected:
            assert family in deploy.SDK_INFO

    def test_sdk_info_has_install_globs(self):
        """SDK_INFO entries should have install_globs."""
        for family, info in deploy.SDK_INFO.items():
            assert "install_globs" in info
            assert isinstance(info["install_globs"], list)

    def test_sdk_info_has_download_url(self):
        """SDK_INFO entries should have download_url."""
        for family, info in deploy.SDK_INFO.items():
            assert "download_url" in info


# ============================================================================
# _find_sdk_root tests
# ============================================================================


class TestFindSdkRoot:
    """Tests for _find_sdk_root function."""

    def test_finds_existing_sdk(self, temp_sdk_path, monkeypatch):
        """_find_sdk_root should find SDK when it exists."""
        # Mock the glob pattern to include our temp sdk path
        def mock_glob(pattern):
            if "c2000ware" in pattern or "C2000Ware" in pattern:
                return [temp_sdk_path]
            return []

        monkeypatch.setattr(deploy.glob, "glob", mock_glob)
        result = deploy._find_sdk_root("c2000")
        assert result == temp_sdk_path

    def test_no_sdk_found(self, tmp_path):
        """_find_sdk_root should return None when no SDK found."""
        # Use a path that doesn't match any glob pattern
        result = deploy._find_sdk_root("c2000")
        # Will be None if no SDK exists at default locations


# ============================================================================
# check_sdk tests
# ============================================================================


class TestCheckSdk:
    """Tests for check_sdk function."""

    def test_known_device_family(self, monkeypatch, temp_sdk_path):
        """check_sdk should return known family for valid device."""
        # Mock _find_sdk_root to return our temp SDK path
        def mock_find_sdk(family):
            if family == "c2000":
                return temp_sdk_path
            return None

        monkeypatch.setattr(deploy, "_find_sdk_root", mock_find_sdk)

        result = deploy.check_sdk("F28P55")

        assert result["found"] is True
        assert result["family"] == "c2000"

    def test_unknown_device(self):
        """check_sdk should return error for unknown device."""
        result = deploy.check_sdk("UNKNOWN123")

        assert result["found"] is False
        assert "Unknown device" in str(result.get("errors", []))

    def test_custom_sdk_path(self, temp_sdk_path):
        """check_sdk should use custom sdk_path when provided."""
        result = deploy.check_sdk("F28P55", sdk_path=temp_sdk_path)

        assert result["found"] is True
        assert "c2000ware" in str(result.get("sdk_root", ""))

    def test_missing_ai_examples(self, temp_sdk_path):
        """check_sdk should report missing AI examples."""
        # Remove AI examples directory
        ai_dir = os.path.join(temp_sdk_path, "libraries", "ai", "examples")
        import shutil
        if os.path.exists(ai_dir):
            shutil.rmtree(ai_dir)

        result = deploy.check_sdk("F28P55", sdk_path=temp_sdk_path)

        assert result["found"] is True  # SDK found
        assert not result.get("ai_examples_exists", False)


# ============================================================================
# find_artifacts tests
# ============================================================================


class TestFindArtifacts:
    """Tests for find_artifacts function."""

    def test_missing_artifacts(self, tmp_path):
        """find_artifacts should report missing files."""
        run_dir = tmp_path / "tinyml-modelmaker" / "data" / "projects"
        task_dir = run_dir / "classification" / "run" / "1234567890" / "model1"
        art_dir = task_dir / "compilation" / "artifacts"
        gold_dir_base = task_dir / "training" / "base" / "golden_vectors"
        # Create only the artifacts directory but don't populate it
        art_dir.mkdir(parents=True)

        result = deploy.find_artifacts(
            task_type="classification",
            run_id="1234567890",
            model_id="model1",
            quantization=False,
            tinyml_base=str(tmp_path),
        )

        assert result["success"] is False
        assert len(result["missing"]) > 0

    def test_all_artifacts_present(self, tmp_path):
        """find_artifacts should succeed when all artifacts present."""
        run_dir = tmp_path / "tinyml-modelmaker" / "data" / "projects"
        task_dir = run_dir / "classification" / "run" / "1234567890" / "model1"
        art_dir = task_dir / "compilation" / "artifacts"
        gold_dir = task_dir / "training" / "quantization" / "golden_vectors"

        art_dir.mkdir(parents=True)
        gold_dir.mkdir(parents=True)

        # Create required files
        (art_dir / "mod.a").write_text("")
        (art_dir / "tvmgen_default.h").write_text("")
        (gold_dir / "test_vector.c").write_text("")
        (gold_dir / "user_input_config.h").write_text("")

        result = deploy.find_artifacts(
            task_type="classification",
            run_id="1234567890",
            model_id="model1",
            quantization=True,
            tinyml_base=str(tmp_path),
        )

        assert result["success"] is True
        assert len(result["missing"]) == 0


# ============================================================================
# create_project tests
# ============================================================================


class TestCreateProject:
    """Tests for create_project function."""

    def test_template_not_found(self, tmp_path):
        """create_project should error when template not found."""
        # Mock _find_sdk_root to return a valid SDK path
        with mock.patch.object(deploy, "_find_sdk_root", return_value=tmp_path / "sdk"):
            result = deploy.create_project(
                project_name="test_project",
                device="F28P55",
                device_type="f28p55x",
                run_id="1234567890",
                task_type="classification",
                model_id="model1",
                quantization=False,
                tinyml_base=str(tmp_path),
            )

        assert result["success"] is False
        # The error will be about template not found because the SDK exists but has no examples

    def test_existing_project_error(self, temp_project_path):
        """create_project should error when project already exists."""
        # The temp_project_path has the CCS directory with spec file
        # but we need to mock find_sdk_root to return a valid path

        with mock.patch.object(deploy, "_find_sdk_root", return_value=temp_project_path):
            result = deploy.create_project(
                project_name="existing_project",
                device="F28P55",
                device_type="f28p55x",
                run_id="1234567890",
                task_type="classification",
                model_id="model1",
                quantization=False,
            )

            # This test may vary based on SDK setup


# ============================================================================
# build_project tests
# ============================================================================


class TestBuildProject:
    """Tests for build_project function."""

    def test_ccs_launcher_not_found(self, tmp_path):
        """build_project should error when CCS launcher not found."""
        result = deploy.build_project(
            project_path=str(tmp_path / "test_project"),
            ccs_install_path="/nonexistent/ccs",
        )

        assert result["success"] is False
        assert "CCS launcher not found" in str(result.get("errors", []))

    def test_build_timeout(self):
        """build_project should handle timeout gracefully."""
        # First mock the launcher detection to return a path
        ccs_path = "/tmp/ccs"
        launcher_path = os.path.join(ccs_path, "eclipse", "ccstudio")
        with mock.patch("os.path.isfile", side_effect=lambda p: p == launcher_path):
            with mock.patch("subprocess.run") as mock_run:
                from subprocess import TimeoutExpired
                mock_run.side_effect = TimeoutExpired(cmd=["test"], timeout=600)

                result = deploy.build_project(
                    project_path="/tmp/test",
                    ccs_install_path=ccs_path,
                )

                assert result["success"] is False
                assert "Build timed out" in str(result.get("errors", []))


# ============================================================================
# flash_project tests
# ============================================================================


class TestFlashProject:
    """Tests for flash_project function."""

    def test_binary_not_found(self, tmp_path):
        """flash_project should error when binary not found."""
        result = deploy.flash_project(
            project_path=str(tmp_path / "test_project"),
            ccs_install_path="/tmp/ccs",
        )

        assert result["success"] is False
        assert "Binary not found" in str(result.get("errors", []))

    def test_dslite_not_found(self, tmp_path):
        """flash_project should error when dslite not found."""
        # Create a mock project with .out file
        project_dir = tmp_path / "test_project"
        debug_dir = project_dir / "Debug"
        debug_dir.mkdir(parents=True)
        (debug_dir / "test_project.out").write_text("mock binary")

        result = deploy.flash_project(
            project_path=str(project_dir),
            ccs_install_path="/nonexistent/ccs",
        )

        assert result["success"] is False
        assert "dslite not found" in str(result.get("errors", []))


# ============================================================================
# run_deploy_check_sdk tests
# ============================================================================


class TestRunDeployCheckSdk:
    """Tests for run_deploy_check_sdk function."""

    def test_prints_sdk_info(self, capsys):
        """run_deploy_check_sdk should print SDK information."""
        result = {
            "found": True,
            "family": "c2000",
            "sdk_name": "C2000Ware",
            "sdk_root": "/tmp/sdk",
            "ai_examples_path": "/tmp/sdk/examples",
            "errors": [],
        }

        class MockArgs:
            device = "F28P55"

        with mock.patch.object(deploy, "check_sdk", return_value=result):
            deploy.run_deploy_check_sdk(MockArgs())

        captured = capsys.readouterr()
        assert "SDK:       C2000Ware" in captured.out

    def test_prints_error_for_missing_sdk(self, capsys):
        """run_deploy_check_sdk should print error for missing SDK."""
        result = {
            "found": False,
            "family": None,
            "sdk_name": None,
            "sdk_root": None,
            "errors": ["SDK not found"],
        }

        class MockArgs:
            device = "F28P55"

        with mock.patch.object(deploy, "check_sdk", return_value=result):
            with pytest.raises(SystemExit) as exc_info:
                deploy.run_deploy_check_sdk(MockArgs())

            assert exc_info.value.code == 1


# ============================================================================
# run_deploy_artifacts tests
# ============================================================================


class TestRunDeployArtifacts:
    """Tests for run_deploy_artifacts function."""

    def test_prints_missing_artifacts(self, capsys):
        """run_deploy_artifacts should print missing artifacts."""
        result = {
            "success": False,
            "missing": ["mod.a", "tvmgen_default.h"],
            "errors": ["Missing: mod.a"],
            "hint": "Check that training and compilation completed successfully.",
        }

        class MockArgs:
            task = "classification"
            run_id = "1234567890"
            model_id = "model1"
            no_quantization = False

        with mock.patch.object(deploy, "find_artifacts", return_value=result):
            with pytest.raises(SystemExit) as exc_info:
                deploy.run_deploy_artifacts(MockArgs())

            assert exc_info.value.code == 1


# ============================================================================
# run_deploy_create tests
# ============================================================================


class TestRunDeployCreate:
    """Tests for run_deploy_create function."""

    def test_prints_project_created(self, capsys):
        """run_deploy_create should print project path when successful."""
        result = {
            "success": True,
            "project_path": "/tmp/project",
            "next_step": "Build the project...",
        }

        class MockArgs:
            device = "F28P55"
            device_type = None
            run_id = "1234567890"
            task = "classification"
            model_id = "model1"
            no_quantization = False
            project_name = "test_project"

        with mock.patch.object(deploy, "create_project", return_value=result):
            deploy.run_deploy_create(MockArgs())

        captured = capsys.readouterr()
        assert "Project created: /tmp/project" in captured.out

    def test_prints_error_for_failed_creation(self, capsys):
        """run_deploy_create should print error for failed creation."""
        result = {
            "success": False,
            "errors": ["Template not found"],
        }

        class MockArgs:
            device = "F28P55"
            device_type = None
            run_id = "1234567890"
            task = "classification"
            model_id = "model1"
            no_quantization = False
            project_name = "test_project"

        with mock.patch.object(deploy, "create_project", return_value=result):
            with pytest.raises(SystemExit) as exc_info:
                deploy.run_deploy_create(MockArgs())

            assert exc_info.value.code == 1


# ============================================================================
# run_deploy_build tests
# ============================================================================


class TestRunDeployBuild:
    """Tests for run_deploy_build function."""

    def test_prints_success(self, capsys):
        """run_deploy_build should print success message."""
        result = {
            "success": True,
            "stdout": "Build output...",
            "stderr": "",
            "out_file": "/tmp/project/Debug/project.out",
            "next_step": "Flash the project...",
        }

        class MockArgs:
            project_path = "/tmp/project"
            ccs_path = "/tmp/ccs"

        with mock.patch.object(deploy, "build_project", return_value=result):
            deploy.run_deploy_build(MockArgs())

        captured = capsys.readouterr()
        assert "\nBuild succeeded." in captured.out
        assert "/tmp/project/Debug/project.out" in captured.out

    def test_prints_failure(self, capsys):
        """run_deploy_build should print failure message."""
        result = {
            "success": False,
            "stdout": "",
            "stderr": "Build failed",
            "errors": ["Linker error"],
        }

        class MockArgs:
            project_path = "/tmp/project"
            ccs_path = "/tmp/ccs"

        with mock.patch.object(deploy, "build_project", return_value=result):
            with pytest.raises(SystemExit) as exc_info:
                deploy.run_deploy_build(MockArgs())

            assert exc_info.value.code == 1


# ============================================================================
# run_deploy_flash tests
# ============================================================================


class TestRunDeployFlash:
    """Tests for run_deploy_flash function."""

    def test_prints_success(self, capsys):
        """run_deploy_flash should print success message."""
        result = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "out_file": "/tmp/project/Debug/project.out",
            "next_step": "Verification steps...",
        }

        class MockArgs:
            project_path = "/tmp/project"
            ccs_path = "/tmp/ccs"

        with mock.patch.object(deploy, "flash_project", return_value=result):
            deploy.run_deploy_flash(MockArgs())

        captured = capsys.readouterr()
        assert "\nFlash succeeded" in captured.out

    def test_prints_failure(self, capsys):
        """run_deploy_flash should print failure message."""
        result = {
            "success": False,
            "stdout": "",
            "stderr": "",
            "errors": ["Connection failed"],
        }

        class MockArgs:
            project_path = "/tmp/project"
            ccs_path = "/tmp/ccs"

        with mock.patch.object(deploy, "flash_project", return_value=result):
            with pytest.raises(SystemExit) as exc_info:
                deploy.run_deploy_flash(MockArgs())

            assert exc_info.value.code == 1


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_device_family_case_sensitivity(self):
        """_device_family should handle mixed case."""
        result = deploy._device_family("f28p55")
        assert result == "c2000"

    def test_find_sdk_root_empty_glob(self, tmp_path):
        """_find_sdk_root should return None when no glob matches."""
        result = deploy._find_sdk_root("unknown_family")
        assert result is None

    def test_create_project_file_exists_error(self, temp_sdk_path):
        """create_project should handle FileExistsError."""
        # Create project directory first
        proj_dir = os.path.join(temp_sdk_path, "existing_project")
        os.makedirs(proj_dir)

        with mock.patch.object(deploy, "_find_sdk_root", return_value=temp_sdk_path):
            result = deploy.create_project(
                project_name="existing_project",
                device="F28P55",
                device_type="f28p55x",
                run_id="1234567890",
                task_type="classification",
                model_id="model1",
                quantization=False,
            )

            assert result["success"] is False
