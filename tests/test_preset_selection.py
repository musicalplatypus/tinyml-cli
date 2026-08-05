"""Tests for channel-aware feature-extraction preset selection.

Background: `mmcli train` without --feature-extraction failed for nearly every
timeseries task because the preset in play expected a different channel count
than the dataset had. See .planning/FINDINGS-training-matrix.md F-5 and
.planning/SPEC-channel-aware-preset-selection.md.

These exercise the pure logic (detection + selection), which needs no training
environment. The values encoded here are the REAL ones measured from the shipped
datasets and the upstream catalog, not invented fixtures.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmcli import preset_selection as ps  # noqa: E402


# Real preset shapes, copied from FEATURE_EXTRACTION_PRESET_DESCRIPTIONS.
RAW_1CH = {
    "name": "Generic_256Input_RAW_256Feature_1Frame",
    "variables": 1,
    "feat_ext_transform": ["RAW_FE", "CONCAT"],
    "frame_size": 256, "feature_size_per_frame": 256, "num_frame_concat": 1,
}
FFT_1CH = {
    "name": "Generic_256Input_FFT_128Feature_1Frame",
    "variables": 1,
    "feat_ext_transform": ["FFT_FE", "CONCAT"],
    "frame_size": 256, "feature_size_per_frame": 128, "num_frame_concat": 1,
}
# The trap: declares variables=1 but has NO transforms and no frame structure.
# Selecting it produces no features, the tensor stays 2-D, and training dies with
# "Not enough dimensions present" — verified against the real CLI.
CUSTOM_DEFAULT = {
    "name": "Custom_Default",
    "variables": 1,
    "feat_ext_transform": [],
    "frame_size": None, "feature_size_per_frame": None, "num_frame_concat": None,
}
ABS_11CH = {
    "name": "Generic_8Input_ABS_8Feature_1Frame",
    "variables": 11,
    "feat_ext_transform": ["ABS", "LOG_DB", "CONCAT"],
    "frame_size": 8, "feature_size_per_frame": 8, "num_frame_concat": 1,
}


class TestColumnCounting:
    def test_headerless_single_column(self, tmp_path):
        """The shipped classification dataset: bare numbers, no header."""
        f = tmp_path / "saw.csv"
        f.write_text("1417\n1497\n1580\n")
        assert ps.count_columns_csv(str(f)) == 1

    def test_header_is_not_counted_as_data(self, tmp_path):
        """The shipped regression dataset has an `x,y` header."""
        f = tmp_path / "file_10.csv"
        f.write_text("x,y\n1.52,1.36\n2.01,1.99\n")
        assert ps.count_columns_csv(str(f)) == 2

    def test_time_columns_are_dropped(self, tmp_path):
        """tinyverse drops columns whose first value contains 'time'
        (timeseries_dataset.py:708), so detection must too or it over-counts."""
        f = tmp_path / "t.csv"
        f.write_text("time,ax,ay\n0.0,1.0,2.0\n0.1,1.1,2.1\n")
        assert ps.count_columns_csv(str(f)) == 2

    def test_empty_file_yields_none(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("")
        assert ps.count_columns_csv(str(f)) is None


class TestChannelDetection:
    def _mk(self, tmp_path, name, text):
        d = tmp_path / "dataset" / "classes" / name
        d.mkdir(parents=True)
        (d / "a.csv").write_text(text)
        return tmp_path / "dataset"

    def test_classification_counts_every_column(self, tmp_path):
        ds = self._mk(tmp_path, "sine", "1417\n1497\n")
        assert ps.detect_channels(str(ds), "generic_timeseries_classification") == 1

    def test_regression_subtracts_the_target_column(self, tmp_path):
        """GenericTSDatasetReg takes the LAST column as a continuous target
        (timeseries_dataset.py:1172/1188), so x,y is 1 input channel, not 2.

        This is the case that made the sweep pick an 11-channel preset and fail
        with 'index 2 is out of bounds for axis 0 with size 2'.
        """
        ds = self._mk(tmp_path, "files", "x,y\n1.5,1.3\n2.0,1.9\n")
        assert ps.detect_channels(str(ds), "generic_timeseries_regression") == 1

    def test_disagreement_between_files_is_reported_not_averaged(self, tmp_path):
        base = tmp_path / "dataset" / "classes"
        (base / "a").mkdir(parents=True)
        (base / "b").mkdir(parents=True)
        (base / "a" / "x.csv").write_text("1\n2\n")
        (base / "b" / "y.csv").write_text("1,2,3\n4,5,6\n")
        assert ps.detect_channels(str(tmp_path / "dataset"),
                                  "generic_timeseries_classification") is None

    def test_annotations_dir_is_ignored(self, tmp_path):
        """annotations/ holds split lists, not sample data."""
        ds = tmp_path / "dataset"
        (ds / "annotations").mkdir(parents=True)
        (ds / "annotations" / "instances_train_list.txt").write_text("a\nb\n")
        (ds / "classes" / "sine").mkdir(parents=True)
        (ds / "classes" / "sine" / "a.csv").write_text("5\n6\n")
        assert ps.detect_channels(str(ds), "generic_timeseries_classification") == 1


class TestPresetChoice:
    def test_picks_a_matching_usable_preset(self):
        got = ps.choose_preset([RAW_1CH, ABS_11CH], 1, "t")
        assert got == RAW_1CH["name"]

    def test_prefers_raw_passthrough_over_fft(self):
        """RAW makes no assumption about the signal's frequency content."""
        got = ps.choose_preset([FFT_1CH, RAW_1CH], 1, "t")
        assert got == RAW_1CH["name"]

    def test_rejects_custom_default_despite_matching_channels(self):
        """The core regression: Custom_Default declares variables=1 but does no
        feature extraction, so it must never be selected."""
        with pytest.raises(ps.PresetSelectionError) as exc:
            ps.choose_preset([CUSTOM_DEFAULT], 1, "generic_timeseries_regression")
        assert "Custom_Default" in str(exc.value)
        assert "unusable" in str(exc.value).lower()

    def test_channel_mismatch_is_fatal_not_a_fallback(self):
        """Guessing would recreate the failure this module exists to prevent."""
        with pytest.raises(ps.PresetSelectionError) as exc:
            ps.choose_preset([ABS_11CH], 1, "t")
        assert "11" in str(exc.value)

    def test_task_with_no_presets_says_so_plainly(self):
        """Anomaly detection and forecasting genuinely have zero presets."""
        with pytest.raises(ps.PresetSelectionError) as exc:
            ps.choose_preset([], 1, "generic_timeseries_anomalydetection")
        assert "no feature-extraction presets" in str(exc.value)


class TestQueryFailuresAreNotReportedAsEmptyCatalog:
    """A broken interpreter must not be reported as 'this task has no presets'.

    The first version returned [] on any subprocess failure, so an ImportError in
    the MMCLI_PYTHON environment surfaced to the user as a catalog gap — blaming
    the wrong thing entirely.
    """

    def test_nonzero_exit_raises_query_error(self, tmp_path):
        fake = tmp_path / "boom.py"
        fake.write_text("import sys; sys.stderr.write('ImportError: nope\\n'); sys.exit(1)\n")
        with pytest.raises(ps.PresetQueryError) as exc:
            ps.query_presets(sys.executable, "t") if False else ps.query_presets(
                str(fake), "t")
        assert "failed" in str(exc.value).lower() or "could not run" in str(exc.value).lower()

    def test_missing_interpreter_raises_query_error(self):
        with pytest.raises(ps.PresetQueryError):
            ps.query_presets("/nonexistent/python-does-not-exist", "t")

    def test_selection_declines_rather_than_blaming_the_dataset(self, tmp_path, capsys):
        ds = tmp_path / "dataset" / "classes" / "sine"
        ds.mkdir(parents=True)
        (ds / "a.csv").write_text("1\n2\n")
        got = ps.select_for_project(str(tmp_path), "generic_timeseries_classification",
                                    "/nonexistent/python-does-not-exist")
        assert got is None, "a failed query must not select a preset"
        err = capsys.readouterr().err
        assert "skipping automatic feature-extraction preset selection" in err
        assert "no feature-extraction presets available" not in err
