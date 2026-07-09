"""Tests for F28E12 device and TASK_TYPES_AUDIO constant — device/task coverage."""
import subprocess
import sys
import pytest
from mmcli.cli import TARGET_DEVICES


class TestF28E12Device:
    def test_f28e12_in_target_devices(self):
        assert "F28E12" in TARGET_DEVICES

    def test_f28e12_between_f2837_and_f28p55(self):
        idx = TARGET_DEVICES.index("F28E12")
        assert "F2837" in TARGET_DEVICES
        assert "F28P55" in TARGET_DEVICES
        assert TARGET_DEVICES.index("F2837") < idx < TARGET_DEVICES.index("F28P55")

    def test_train_help_lists_f28e12(self):
        result = subprocess.run(
            [sys.executable, "-m", "mmcli", "train", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "F28E12" in result.stdout

    def test_deploy_help_lists_f28e12(self):
        # deploy is a sub-dispatcher; device choices appear in subcommand help
        result = subprocess.run(
            [sys.executable, "-m", "mmcli", "deploy", "check-sdk", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "F28E12" in result.stdout

    def test_f28e12_accepted_by_train(self):
        result = subprocess.run(
            [sys.executable, "-m", "mmcli", "train", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_f28e12_not_rejected_by_compile(self):
        result = subprocess.run(
            [sys.executable, "-m", "mmcli", "compile", "-d", "F28E12", "--help"],
            capture_output=True, text=True,
        )
        assert "invalid choice" not in result.stderr


class TestTaskTypesAudio:
    def test_task_types_audio_exists(self):
        from mmcli.cli import TASK_TYPES_AUDIO
        assert TASK_TYPES_AUDIO is not None

    def test_audio_classification_in_task_types_audio(self):
        from mmcli.cli import TASK_TYPES_AUDIO
        assert "audio_classification" in TASK_TYPES_AUDIO

    def test_train_help_mentions_audio_classification(self):
        result = subprocess.run(
            [sys.executable, "-m", "mmcli", "train", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "audio_classification" in result.stdout

    def test_run_help_mentions_audio_classification(self):
        result = subprocess.run(
            [sys.executable, "-m", "mmcli", "run", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "audio_classification" in result.stdout

    def test_task_types_audio_is_list(self):
        from mmcli.cli import TASK_TYPES_AUDIO
        assert isinstance(TASK_TYPES_AUDIO, list)
        assert len(TASK_TYPES_AUDIO) > 0

    def test_nas_supported_tasks_unchanged(self):
        from mmcli.cli import NAS_SUPPORTED_TASKS
        # NAS_SUPPORTED_TASKS must NOT include audio tasks
        assert "audio_classification" not in NAS_SUPPORTED_TASKS
