"""Tests for mmcli.report — TrainingLogParser and HTMLReportGenerator."""

import os
import tempfile

import pytest

from mmcli.report import (
    TrainingLogParser,
    HTMLReportGenerator,
    create_report_handler,
    _pca_images_to_html,
    _forecasting_var_table_to_html,
)


# ---------------------------------------------------------------------------
# Sample log lines (match real tinyml_modelmaker output format)
# ---------------------------------------------------------------------------

FLOAT_EPOCH_0 = (
    "   INFO: root.utils.MetricLogger.FloatTrain: "
    "Epoch: [0]  [0/3]  eta: 0:00:05  loss: 2.1234  acc1: 45.0000  lr: 0.001  samples/s: 128"
)
FLOAT_EPOCH_1 = (
    "   INFO: root.utils.MetricLogger.FloatTrain: "
    "Epoch: [1]  [0/3]  eta: 0:00:04  loss: 0.8765  acc1: 82.0000  lr: 0.001  samples/s: 135"
)
FLOAT_TEST_ACC_0 = "   INFO: root.train_utils.evaluate.FloatTrain: Test:  Acc@1 55.000"
FLOAT_TEST_F1_0 = "   INFO: root.train_utils.evaluate.FloatTrain: Test:  F1-Score 0.520"
FLOAT_TEST_AUC_0 = "   INFO: root.train_utils.evaluate.FloatTrain: AU-ROC Score: 0.610"

FLOAT_TEST_ACC_1 = "   INFO: root.train_utils.evaluate.FloatTrain: Test:  Acc@1 85.000"
FLOAT_TEST_F1_1 = "   INFO: root.train_utils.evaluate.FloatTrain: Test:  F1-Score 0.840"
FLOAT_TEST_AUC_1 = "   INFO: root.train_utils.evaluate.FloatTrain: AU-ROC Score: 0.910"

QUANT_EPOCH_0 = (
    "   INFO: root.utils.MetricLogger.QuantTrain: "
    "Epoch: [0]  [0/3]  eta: 0:00:06  loss: 1.5000  acc1: 60.0000  lr: 0.0005  samples/s: 100"
)
QUANT_TEST_ACC_0 = "   INFO: root.train_utils.evaluate.QuantTrain: Test:  Acc@1 70.000"
QUANT_TEST_F1_0 = "   INFO: root.train_utils.evaluate.QuantTrain: Test:  F1-Score 0.680"
QUANT_TEST_AUC_0 = "   INFO: root.train_utils.evaluate.QuantTrain: AU-ROC Score: 0.720"

BEST_EPOCH_LINE = "   INFO: root.main.FloatTrain.BestEpoch: Best Epoch: 1"
BEST_ACC_LINE = "   INFO: root.main.FloatTrain.BestEpoch: Acc@1 85.000"
BEST_F1_LINE = "   INFO: root.main.FloatTrain.BestEpoch: F1-Score 0.840"
BEST_AUC_LINE = "   INFO: root.main.FloatTrain.BestEpoch: AUC ROC Score 0.910"

CONF_MATRIX_LINES = [
    "   INFO: root.main.FloatTrain.BestEpoch: Confusion Matrix:",
    "+---+---+---+---+",
    "| | Predicted as: A | Predicted as: B | Predicted as: C |",
    "+---+---+---+---+",
    "| Ground Truth: A |  40 |   3 |   2 |",
    "| Ground Truth: B |   2 |  36 |   2 |",
    "| Ground Truth: C |   1 |   2 |  41 |",
    "+---+---+---+---+",
    "   INFO: root.main.FloatTrain: Next log line after matrix",  # non-table line ends matrix
]


# ---------------------------------------------------------------------------
# Helper: feed a sequence of lines through a parser
# ---------------------------------------------------------------------------

def _feed_lines(parser: TrainingLogParser, lines: list[str]) -> None:
    for line in lines:
        parser.feed_line(line)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrainingLogParser:
    """Tests for TrainingLogParser."""

    def test_parse_training_epoch(self):
        """Epoch line should populate _pending_eval with epoch, loss, acc."""
        p = TrainingLogParser()
        p.feed_line(FLOAT_EPOCH_0)
        assert p._pending_eval["epoch"] == 0
        assert p._pending_eval["train_loss"] == pytest.approx(2.1234)
        assert p._pending_eval["train_acc"] == pytest.approx(45.0)

    def test_parse_eval_metrics(self):
        """Full epoch cycle (train + eval) should append to float_epochs."""
        p = TrainingLogParser()
        _feed_lines(p, [
            FLOAT_EPOCH_0,
            FLOAT_TEST_ACC_0,
            FLOAT_TEST_F1_0,
            FLOAT_TEST_AUC_0,  # AUC triggers _flush_pending_eval
        ])
        assert len(p.float_epochs) == 1
        e = p.float_epochs[0]
        assert e["epoch"] == 0
        assert e["train_loss"] == pytest.approx(2.1234)
        assert e["val_acc"] == pytest.approx(55.0)
        assert e["val_f1"] == pytest.approx(0.52)
        assert e["val_auc"] == pytest.approx(0.61)

    def test_phase_detection(self):
        """FloatTrain and QuantTrain lines go to separate epoch lists."""
        p = TrainingLogParser()
        # Float phase
        _feed_lines(p, [
            FLOAT_EPOCH_0, FLOAT_TEST_ACC_0, FLOAT_TEST_F1_0, FLOAT_TEST_AUC_0,
        ])
        # Quant phase
        _feed_lines(p, [
            QUANT_EPOCH_0, QUANT_TEST_ACC_0, QUANT_TEST_F1_0, QUANT_TEST_AUC_0,
        ])
        assert len(p.float_epochs) == 1
        assert len(p.quant_epochs) == 1
        assert p.quant_epochs[0]["val_acc"] == pytest.approx(70.0)

    def test_confusion_matrix_parsing(self):
        """Full confusion matrix block should be parsed into structured dict."""
        p = TrainingLogParser()
        _feed_lines(p, CONF_MATRIX_LINES)
        cm = p.float_conf_matrix
        assert cm is not None
        assert cm["headers"] == ["A", "B", "C"]
        assert len(cm["rows"]) == 3
        assert cm["rows"][0]["label"] == "A"
        assert cm["rows"][0]["values"] == [40, 3, 2]
        assert cm["rows"][1]["label"] == "B"
        assert cm["rows"][1]["values"] == [2, 36, 2]
        assert cm["rows"][2]["label"] == "C"
        assert cm["rows"][2]["values"] == [1, 2, 41]

    def test_best_epoch(self):
        """BestEpoch lines should populate best_float dict."""
        p = TrainingLogParser()
        _feed_lines(p, [BEST_EPOCH_LINE, BEST_ACC_LINE, BEST_F1_LINE, BEST_AUC_LINE])
        assert p.best_float["epoch"] == 1
        assert p.best_float["acc"] == pytest.approx(85.0)
        assert p.best_float["f1"] == pytest.approx(0.84)
        assert p.best_float["auc"] == pytest.approx(0.91)

    def test_two_float_epochs(self):
        """Two complete float epoch cycles should produce two entries."""
        p = TrainingLogParser()
        _feed_lines(p, [
            FLOAT_EPOCH_0, FLOAT_TEST_ACC_0, FLOAT_TEST_F1_0, FLOAT_TEST_AUC_0,
            FLOAT_EPOCH_1, FLOAT_TEST_ACC_1, FLOAT_TEST_F1_1, FLOAT_TEST_AUC_1,
        ])
        assert len(p.float_epochs) == 2
        assert p.float_epochs[0]["val_acc"] == pytest.approx(55.0)
        assert p.float_epochs[1]["val_acc"] == pytest.approx(85.0)


class TestHTMLReportGenerator:
    """Tests for HTMLReportGenerator."""

    def test_html_output_contains_key_elements(self):
        """Generated HTML should contain chart, cards, and structure."""
        p = TrainingLogParser()
        _feed_lines(p, [
            FLOAT_EPOCH_0, FLOAT_TEST_ACC_0, FLOAT_TEST_F1_0, FLOAT_TEST_AUC_0,
            FLOAT_EPOCH_1, FLOAT_TEST_ACC_1, FLOAT_TEST_F1_1, FLOAT_TEST_AUC_1,
        ])
        _feed_lines(p, [BEST_EPOCH_LINE, BEST_ACC_LINE, BEST_F1_LINE])
        _feed_lines(p, CONF_MATRIX_LINES)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "report.html")
            gen = HTMLReportGenerator(out)
            gen.generate(p, is_complete=True)

            html = open(out).read()

        # Chart canvas
        assert 'id="metricsChart"' in html
        # Status badge (complete)
        assert "Complete" in html
        assert "complete" in html
        # No auto-refresh when complete
        assert 'http-equiv="refresh"' not in html
        # Confusion matrix table
        assert "Confusion Matrix" in html
        assert "Ground Truth" not in html  # labels are stripped in parsing
        # Summary cards
        assert "Float Best Acc" in html
        assert "85.0%" in html

    def test_training_status_shows_auto_refresh(self):
        """During training, HTML should include auto-refresh meta tag."""
        p = TrainingLogParser()
        _feed_lines(p, [
            FLOAT_EPOCH_0, FLOAT_TEST_ACC_0, FLOAT_TEST_F1_0, FLOAT_TEST_AUC_0,
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "report.html")
            gen = HTMLReportGenerator(out)
            gen.generate(p, is_complete=False)

            html = open(out).read()

        assert 'http-equiv="refresh"' in html
        assert "Training" in html


class TestCreateReportHandler:
    """Tests for the create_report_handler() convenience function."""

    def test_creates_html_file(self):
        """feed_line + finalize should produce a valid HTML report file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "report.html")
            feed_line, finalize = create_report_handler(out)

            # Initial empty report should exist
            assert os.path.isfile(out)

            # Feed some lines
            for line in [
                FLOAT_EPOCH_0, FLOAT_TEST_ACC_0, FLOAT_TEST_F1_0, FLOAT_TEST_AUC_0,
            ]:
                feed_line(line)

            finalize()

            html = open(out).read()
            assert "Complete" in html
            assert 'id="metricsChart"' in html
            assert "55.0%" in html


class TestPCAImages:
    """Tests for PCA visualization image embedding."""

    def _create_tiny_png(self, path: str) -> None:
        """Create a minimal valid 1x1 PNG file."""
        import struct, zlib
        # Minimal 1x1 red PNG
        raw = b'\x00\xff\x00\x00'  # filter byte + RGB
        compressed = zlib.compress(raw)
        def chunk(ctype, data):
            c = ctype + data
            return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
        ihdr = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
        png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', compressed) + chunk(b'IEND', b'')
        with open(path, 'wb') as f:
            f.write(png)

    def test_pca_images_embedded(self):
        """PCA PNGs in report directory should be base64-embedded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_tiny_png(os.path.join(tmpdir, 'pca_on_feature_extracted_train_data.png'))
            self._create_tiny_png(os.path.join(tmpdir, 'pca_on_feature_extracted_validation_data.png'))
            html = _pca_images_to_html(tmpdir)
            assert 'PCA Feature Visualization' in html
            assert 'data:image/png;base64,' in html
            assert 'Train Data' in html
            assert 'Validation Data' in html

    def test_no_pca_images(self):
        """No PCA section when PNGs are absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            html = _pca_images_to_html(tmpdir)
            assert html == ''


# ---------------------------------------------------------------------------
# Regression log lines
# ---------------------------------------------------------------------------

REG_EPOCH_0 = (
    "   INFO: root.utils.MetricLogger.FloatTrain: "
    "Epoch: [0]  [0/3]  eta: 0:00:05  loss: 5.1234  lr: 0.001  samples/s: 128"
)
REG_EPOCH_1 = (
    "   INFO: root.utils.MetricLogger.FloatTrain: "
    "Epoch: [1]  [0/3]  eta: 0:00:04  loss: 1.2345  lr: 0.001  samples/s: 135"
)
REG_TEST_MSE_0 = "   INFO: root.train_utils.evaluate.FloatTrain: Test:  MSE 5306.965"
REG_TEST_R2_0 = "   INFO: root.train_utils.evaluate.FloatTrain: Test:  R2-Score -21386.584"
REG_TEST_MSE_1 = "   INFO: root.train_utils.evaluate.FloatTrain: Test:  MSE 15.475"
REG_TEST_R2_1 = "   INFO: root.train_utils.evaluate.FloatTrain: Test:  R2-Score 0.994"
REG_BEST_EPOCH = "   INFO: root.main.FloatTrain.BestEpoch: Best Epoch: 1"
REG_BEST_MSE = "   INFO: root.main.FloatTrain.BestEpoch: MSE 15.475"
REG_BEST_R2 = "   INFO: root.main.FloatTrain.BestEpoch: R2-Score 0.994"


class TestRegressionParser:
    """Tests for regression metric parsing."""

    def test_regression_epoch_metrics(self):
        """Regression epoch lines (loss only) should flush on R2-Score."""
        p = TrainingLogParser()
        _feed_lines(p, [REG_EPOCH_0, REG_TEST_MSE_0, REG_TEST_R2_0])
        assert len(p.float_epochs) == 1
        e = p.float_epochs[0]
        assert e['epoch'] == 0
        assert e['train_loss'] == pytest.approx(5.1234)
        assert e['val_mse'] == pytest.approx(5306.965)
        assert e['val_r2'] == pytest.approx(-21386.584)
        assert 'train_acc' not in e  # no accuracy for regression

    def test_regression_task_type(self):
        """Task type should auto-detect as 'regression'."""
        p = TrainingLogParser()
        _feed_lines(p, [REG_EPOCH_0, REG_TEST_MSE_0, REG_TEST_R2_0])
        assert p.task_type == 'regression'

    def test_regression_best_epoch(self):
        """Best epoch MSE and R2-Score should be captured."""
        p = TrainingLogParser()
        _feed_lines(p, [REG_BEST_EPOCH, REG_BEST_MSE, REG_BEST_R2])
        assert p.best_float['epoch'] == 1
        assert p.best_float['mse'] == pytest.approx(15.475)
        assert p.best_float['r2'] == pytest.approx(0.994)

    def test_regression_html_r2_axis(self):
        """Regression report should use R²-Score on Y-axis."""
        p = TrainingLogParser()
        _feed_lines(p, [
            REG_EPOCH_0, REG_TEST_MSE_0, REG_TEST_R2_0,
            REG_EPOCH_1, REG_TEST_MSE_1, REG_TEST_R2_1,
        ])
        _feed_lines(p, [REG_BEST_EPOCH, REG_BEST_MSE, REG_BEST_R2])
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "report.html")
            gen = HTMLReportGenerator(out)
            gen.generate(p, is_complete=True)
            html = open(out).read()
        # Y-axis label is R²-Score, not Accuracy
        assert "R²-Score" in html or "R\\u00b2-Score" in html
        assert "Float Best R" in html  # summary card
        assert "Float Best MSE" in html
        assert "Complete" in html


# ---------------------------------------------------------------------------
# Forecasting log lines
# ---------------------------------------------------------------------------

FORECAST_EPOCH_0 = (
    "   INFO: root.utils.MetricLogger.FloatTrain: "
    "Epoch: [0] Total time: 0:00:03"
)
FORECAST_EPOCH_0_TEST = (
    "   INFO: root.utils.MetricLogger.FloatTrain: "
    "Test:   [  0/314]  eta: 0:00:45  loss: 1.8408 (1.8408)  smape: 9.6656 (9.6656)"
)
FORECAST_SMAPE_0 = (
    "   INFO: root.train_utils.evaluate.FloatTrain: "
    "Current SMAPE across all target variables and across all predicted timesteps: 8.54%"
)
FORECAST_EPOCH_1 = (
    "   INFO: root.utils.MetricLogger.FloatTrain: "
    "Epoch: [1] Total time: 0:00:03"
)
# Use loss from the epoch-total line (which our regex captures from batch lines)
FORECAST_EPOCH_0_BATCH = (
    "   INFO: root.utils.MetricLogger.FloatTrain: "
    "Epoch: [0]  [0/314]  eta: 0:00:45  loss: 1.8408  smape: 9.6656"
)
FORECAST_EPOCH_1_BATCH = (
    "   INFO: root.utils.MetricLogger.FloatTrain: "
    "Epoch: [1]  [0/314]  eta: 0:00:45  loss: 0.5123  smape: 3.1234"
)
FORECAST_SMAPE_1 = (
    "   INFO: root.train_utils.evaluate.FloatTrain: "
    "Current SMAPE across all target variables and across all predicted timesteps: 2.15%"
)
FORECAST_BEST_EPOCH = "   INFO: root.main.FloatTrain.BestEpoch: Best epoch:10"
FORECAST_BEST_SMAPE = "   INFO: root.main.FloatTrain.BestEpoch: Overall SMAPE across all variables: 0.36%"
FORECAST_BEST_VAR_SMAPE = (
    "   INFO: root.main.FloatTrain.BestEpoch: "
    "      SMAPE of indoorTemperature across all predicted timesteps: 0.36%"
)
FORECAST_BEST_VAR_R2 = (
    "   INFO: root.main.FloatTrain.BestEpoch: "
    "      R² of indoorTemperature across all predicted timesteps: 0.9967"
)
FORECAST_TEST_SMAPE = (
    "   INFO: root.main.test_data : "
    "  SMAPE of indoorTemperature across all predicted timesteps: 0.95%"
)
FORECAST_TEST_R2 = (
    "   INFO: root.main.test_data : "
    "  R² of indoorTemperature across all predicted timesteps: 0.9833"
)


class TestForecastingParser:
    """Tests for forecasting metric parsing."""

    def test_forecasting_epoch_metrics(self):
        """Forecasting epoch should flush on SMAPE."""
        p = TrainingLogParser()
        _feed_lines(p, [FORECAST_EPOCH_0_BATCH, FORECAST_SMAPE_0])
        assert len(p.float_epochs) == 1
        e = p.float_epochs[0]
        assert e['epoch'] == 0
        assert e['train_loss'] == pytest.approx(1.8408)
        assert e['val_smape'] == pytest.approx(8.54)
        assert 'train_acc' not in e

    def test_forecasting_task_type(self):
        """Task type should auto-detect as 'forecasting'."""
        p = TrainingLogParser()
        _feed_lines(p, [FORECAST_EPOCH_0_BATCH, FORECAST_SMAPE_0])
        assert p.task_type == 'forecasting'

    def test_forecasting_best_epoch(self):
        """Best epoch overall SMAPE should be captured."""
        p = TrainingLogParser()
        _feed_lines(p, [FORECAST_BEST_EPOCH, FORECAST_BEST_SMAPE])
        assert p.best_float['epoch'] == 10
        assert p.best_float['smape'] == pytest.approx(0.36)

    def test_forecasting_per_var_best(self):
        """Per-variable SMAPE/R² from BestEpoch should be captured."""
        p = TrainingLogParser()
        _feed_lines(p, [
            FORECAST_BEST_EPOCH, FORECAST_BEST_SMAPE,
            FORECAST_BEST_VAR_SMAPE, FORECAST_BEST_VAR_R2,
        ])
        assert len(p.best_float.get('per_var_smape', [])) == 1
        assert p.best_float['per_var_smape'][0]['var'] == 'indoorTemperature'
        assert p.best_float['per_var_smape'][0]['smape'] == pytest.approx(0.36)
        assert len(p.best_float.get('per_var_r2', [])) == 1
        assert p.best_float['per_var_r2'][0]['r2'] == pytest.approx(0.9967)

    def test_forecasting_test_data(self):
        """Per-variable test_data SMAPE/R² should be captured."""
        p = TrainingLogParser()
        _feed_lines(p, [FORECAST_TEST_SMAPE, FORECAST_TEST_R2])
        assert len(p.test_data_metrics) == 1
        entry = p.test_data_metrics[0]
        assert entry['var'] == 'indoorTemperature'
        assert entry['smape'] == pytest.approx(0.95)
        assert entry['r2'] == pytest.approx(0.9833)

    def test_forecasting_html_smape_axis(self):
        """Forecasting report should use SMAPE on Y-axis and show per-var table."""
        p = TrainingLogParser()
        _feed_lines(p, [
            FORECAST_EPOCH_0_BATCH, FORECAST_SMAPE_0,
            FORECAST_EPOCH_1_BATCH, FORECAST_SMAPE_1,
        ])
        _feed_lines(p, [
            FORECAST_BEST_EPOCH, FORECAST_BEST_SMAPE,
            FORECAST_BEST_VAR_SMAPE, FORECAST_BEST_VAR_R2,
            FORECAST_TEST_SMAPE, FORECAST_TEST_R2,
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "report.html")
            gen = HTMLReportGenerator(out)
            gen.generate(p, is_complete=True)
            html = open(out).read()
        assert "SMAPE" in html
        assert "Float Best SMAPE" in html
        assert "Per-Variable Forecasting Metrics" in html
        assert "indoorTemperature" in html
        assert "Complete" in html

    def test_forecasting_var_table_helper(self):
        """The per-variable table helper should render properly."""
        best = {
            'epoch': 10,
            'smape': 0.36,
            'per_var_smape': [{'var': 'temp', 'smape': 0.36}],
            'per_var_r2': [{'var': 'temp', 'r2': 0.9967}],
        }
        test_metrics = [{'var': 'temp', 'smape': 0.95, 'r2': 0.9833}]
        html = _forecasting_var_table_to_html(best, test_metrics)
        assert 'temp' in html
        assert '0.36' in html
        assert '0.9967' in html
        assert '0.95' in html
        assert '0.9833' in html


# ---------------------------------------------------------------------------
# NAS search report tests
# ---------------------------------------------------------------------------

# NAS log line samples
NAS_DEVICE = "   INFO: root.modelopt.nas   : NAS device: mps (Apple Metal)"
NAS_PARAM = "   INFO: root.modelopt.nas.search: param size = 0.013075MB"
NAS_STEP_0 = (
    "   INFO: root.modelopt.nas.train: "
    "Epoch: [0]  [000/128]  eta: 00:12:34  lr: 0.025000  samples/s: 45.200  loss: 1.23  acc1: 33.33  time: 0.1234  max_mem: 0.0"
)
NAS_TRAIN_ACC_0 = "   INFO: root.modelopt.nas.search: Train:  Acc@1 52.345678"
NAS_TEST_ACC_0 = "   INFO: root.modelopt.nas.search: Test:  Acc@1 48.123456"
NAS_TRAIN_ACC_1 = "   INFO: root.modelopt.nas.search: Train:  Acc@1 64.567890"
NAS_TEST_ACC_1 = "   INFO: root.modelopt.nas.search: Test:  Acc@1 58.234567"
NAS_BEST = "   INFO: root.modelopt.nas.search: New best genotype at epoch 1 (Acc@1 58.234567)"


class TestNASParser:
    """Tests for NAS architecture search parsing."""

    def test_nas_device_detection(self):
        p = TrainingLogParser()
        p.feed_line(NAS_DEVICE)
        assert p.is_nas is True
        assert p.nas_device == 'mps'

    def test_nas_param_size(self):
        p = TrainingLogParser()
        p.feed_line(NAS_PARAM)
        assert p.nas_param_size == pytest.approx(0.013075)

    def test_nas_search_epochs(self):
        p = TrainingLogParser()
        p.feed_line(NAS_TRAIN_ACC_0)
        p.feed_line(NAS_TEST_ACC_0)
        p.feed_line(NAS_TRAIN_ACC_1)
        p.feed_line(NAS_TEST_ACC_1)
        assert len(p.nas_epochs) == 2
        assert p.nas_epochs[0]['train_acc'] == pytest.approx(52.345678)
        assert p.nas_epochs[0]['test_acc'] == pytest.approx(48.123456)
        assert p.nas_epochs[1]['train_acc'] == pytest.approx(64.567890)
        assert p.nas_epochs[1]['test_acc'] == pytest.approx(58.234567)

    def test_nas_best_genotype(self):
        p = TrainingLogParser()
        p.feed_line(NAS_BEST)
        assert p.nas_best['epoch'] == 1
        assert p.nas_best['acc'] == pytest.approx(58.234567)

    def test_nas_step_parsing(self):
        p = TrainingLogParser()
        p.feed_line(NAS_STEP_0)
        assert p._nas_last_step['epoch'] == 0
        assert p._nas_last_step['step'] == 0
        assert p._nas_last_step['total_steps'] == 128
        assert p._nas_last_step['loss'] == pytest.approx(1.23)
        assert p._nas_last_step['acc'] == pytest.approx(33.33)
        # Step should also be accumulated in nas_steps
        assert len(p.nas_steps) == 1
        assert p.nas_steps[0]['epoch'] == 0
        assert p.nas_steps[0]['loss'] == pytest.approx(1.23)
        assert p.nas_steps[0]['acc'] == pytest.approx(33.33)

    def test_nas_step_chart_in_report(self):
        """NAS step data should produce a step chart with mode switcher."""
        p = TrainingLogParser()
        # Feed step data + epoch data (chart needs both)
        for line in [NAS_STEP_0, NAS_TRAIN_ACC_0, NAS_TEST_ACC_0, NAS_BEST]:
            p.feed_line(line)
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            path = f.name
        try:
            gen = HTMLReportGenerator(path)
            gen.generate(p, is_complete=True)
            html = open(path).read()
            assert 'nasStepChart' in html
            assert 'NAS Step Metrics' in html
            assert 'nasStepSwitchMode' in html
            assert 'Smoothed' in html
            assert 'Aggregated' in html
        finally:
            os.unlink(path)

    def test_nas_no_interference_with_training(self):
        """NAS lines should not affect float_epochs or task_type."""
        p = TrainingLogParser()
        for line in [NAS_DEVICE, NAS_PARAM, NAS_TRAIN_ACC_0, NAS_TEST_ACC_0]:
            p.feed_line(line)
        assert p.float_epochs == []
        assert p.task_type is None
        # Now feed normal training
        p.feed_line(FLOAT_EPOCH_0)
        p.feed_line(FLOAT_TEST_ACC_0)
        p.feed_line(FLOAT_TEST_F1_0)
        p.feed_line(FLOAT_TEST_AUC_0)
        assert len(p.float_epochs) == 1
        assert p.task_type == 'classification'

    def test_nas_report_html_contains_search_chart(self):
        p = TrainingLogParser()
        for line in [NAS_DEVICE, NAS_PARAM, NAS_TRAIN_ACC_0, NAS_TEST_ACC_0, NAS_BEST]:
            p.feed_line(line)
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            path = f.name
        try:
            gen = HTMLReportGenerator(path)
            gen.generate(p, is_complete=False)
            html = open(path).read()
            assert 'nasChart' in html
            assert 'NAS Architecture Search' in html
            assert 'NAS Search' in html
            assert 'MPS' in html
        finally:
            os.unlink(path)

    def test_nas_report_html_has_both_charts(self):
        """When NAS + training data present, both charts appear."""
        p = TrainingLogParser()
        for line in [NAS_DEVICE, NAS_TRAIN_ACC_0, NAS_TEST_ACC_0]:
            p.feed_line(line)
        # Normal training
        p.feed_line(FLOAT_EPOCH_0)
        p.feed_line(FLOAT_TEST_ACC_0)
        p.feed_line(FLOAT_TEST_F1_0)
        p.feed_line(FLOAT_TEST_AUC_0)
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            path = f.name
        try:
            gen = HTMLReportGenerator(path)
            gen.generate(p)
            html = open(path).read()
            assert 'nasChart' in html
            assert 'metricsChart' in html
            assert 'NAS search epochs' in html
        finally:
            os.unlink(path)

    def test_nas_subtitle_includes_search_count(self):
        p = TrainingLogParser()
        for line in [NAS_TRAIN_ACC_0, NAS_TEST_ACC_0, NAS_TRAIN_ACC_1, NAS_TEST_ACC_1]:
            p.feed_line(line)
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            path = f.name
        try:
            gen = HTMLReportGenerator(path)
            gen.generate(p)
            html = open(path).read()
            assert '2 NAS search epochs' in html
        finally:
            os.unlink(path)
