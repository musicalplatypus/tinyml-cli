"""
mmcli report — training report generator.

Parses tinyml_modelmaker training stdout line-by-line and generates
a self-contained HTML report with live-updating loss/accuracy charts
and a heatmap confusion matrix.
"""

import base64
import glob
import os
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Log line parser
# ---------------------------------------------------------------------------

# Log format: "   INFO: root.utils.MetricLogger.FloatTrain: Epoch: [0] ..."
# Also handles: "   INFO: root.main.test_data : ..." (space before colon)
_RE_LOG_PREFIX = re.compile(
    r'^\s*\w+:\s+'          # level (INFO, DEBUG, etc.)
    r'([\w.]+)\s*:\s+'      # logger name (captured, allowing space before colon)
    r'(.*)',                # message (captured)
    re.DOTALL,
)

# Training epoch line: "Epoch: [5]  [0/12]  ... loss: 0.1234 ... acc1: 95.00 ..."
_RE_EPOCH_TRAIN = re.compile(
    r'Epoch:\s*\[(\d+)\].*?loss:\s*([\d.]+).*?acc1:\s*([\d.]+)'
)

# Training epoch line WITHOUT acc1 (regression/forecasting):
# "Epoch: [5]  [0/12]  ... loss: 0.1234 ..." or
# "Epoch: [0] Total time: 0:00:03"
_RE_EPOCH_TRAIN_LOSS_ONLY = re.compile(
    r'Epoch:\s*\[(\d+)\].*?loss:\s*([\d.]+)'
)

# Evaluation accuracy: "Test:  Acc@1 95.000" or "Test: EMA Acc@1 95.000"
_RE_TEST_ACC = re.compile(r'Test:\s*(?:EMA\s+)?Acc@1\s+([\d.]+)')

# Evaluation F1: "Test:  F1-Score 0.840" or "Test: EMA F1-Score 0.840"
_RE_TEST_F1 = re.compile(r'Test:\s*(?:EMA\s+)?F1-Score\s+([\d.]+)')

# AU-ROC score: "AU-ROC Score: 0.950"
_RE_AUC = re.compile(r'AU-ROC Score:\s*([\d.]+)')

# Best epoch accuracy: "Acc@1 95.000"
_RE_BEST_ACC = re.compile(r'^Acc@1\s+([\d.]+)')

# Best epoch F1: "F1-Score 0.840"
_RE_BEST_F1 = re.compile(r'^F1-Score\s+([\d.]+)')

# Best epoch AUC: "AUC ROC Score 0.910"
_RE_BEST_AUC = re.compile(r'^AUC ROC Score\s+([\d.]+)')

# Best epoch: "Best Epoch: 15" or "Best epoch:15" (forecasting variant)
_RE_BEST_EPOCH = re.compile(r'Best [Ee]poch:?\s*(\d+)')

# ---------------------------------------------------------------------------
# Regression-specific patterns
# ---------------------------------------------------------------------------

# Validation MSE: "Test:  MSE 5306.965"
_RE_TEST_MSE = re.compile(r'Test:\s*MSE\s+([-+eE\d.]+)')

# Validation R2-Score: "Test:  R2-Score -21386.584"
_RE_TEST_R2 = re.compile(r'Test:\s*R2-Score\s+([-+eE\d.]+)')

# Best epoch MSE: "MSE 15.475"
_RE_BEST_MSE = re.compile(r'^MSE\s+([-+eE\d.]+)')

# Best epoch R2-Score: "R2-Score 0.994"
_RE_BEST_R2 = re.compile(r'^R2-Score\s+([-+eE\d.]+)')

# ---------------------------------------------------------------------------
# Forecasting-specific patterns
# ---------------------------------------------------------------------------

# Per-epoch overall SMAPE: "Current SMAPE across all target variables and across all predicted timesteps: 8.54%"
_RE_CURRENT_SMAPE = re.compile(
    r'Current SMAPE across all target variables.*?:\s*([-+eE\d.]+)%'
)

# Best epoch overall SMAPE: "Overall SMAPE across all variables: 0.36%"
_RE_BEST_OVERALL_SMAPE = re.compile(
    r'Overall SMAPE across all variables:\s*([-+eE\d.]+)%'
)

# Per-variable SMAPE: "SMAPE of indoorTemperature across all predicted timesteps: 0.36%"
_RE_PER_VAR_SMAPE = re.compile(
    r'SMAPE of (\S+) across all predicted timesteps:\s*([-+eE\d.]+)%'
)

# Per-variable R²: "R² of indoorTemperature across all predicted timesteps: 0.9967"
_RE_PER_VAR_R2 = re.compile(
    r'R[²2] of (\S+) across all predicted timesteps:\s*([-+eE\d.]+)'
)

# ---------------------------------------------------------------------------
# Common patterns (all task types)
# ---------------------------------------------------------------------------

# Confusion matrix start: "Confusion Matrix:"
_RE_CONF_MATRIX_START = re.compile(r'Confusion Matrix:')

# Confusion matrix row (tabulate grid): "| Ground Truth: class_a |  50 |   2 |"
_RE_CONF_ROW = re.compile(
    r'\|\s*(?:Ground Truth:\s*)?(.+?)\s*\|'     # row label
    r'((?:\s*[\d.]+\s*\|)+)'                     # values separated by |
)

# Confusion matrix header row: "| | Predicted as: A | Predicted as: B |"
_RE_CONF_HEADER = re.compile(
    r'\|((?:\s*Predicted as:\s*.+?\s*\|)+)'
)

# File-level classification summary log path
_RE_FLCS_PATH = re.compile(
    r'Generated (?:F|f)ile.level classification summary.*?:\s*(.+\.log)'
)

# Dataset path detection (from modelmaker loading)
_RE_DATASET_PATH = re.compile(
    r'input_data_path["\']?\s*[:=]\s*["\']?([^"\',}]+)'
)


# ---------------------------------------------------------------------------
# NAS search-specific patterns
# ---------------------------------------------------------------------------

# NAS per-epoch summary: "Train:  Acc@1 68.123456" (from nas.search logger)
_RE_NAS_TRAIN_ACC = re.compile(r'Train:\s+Acc@1\s+([\d.]+)')

# NAS per-epoch summary: "Test:  Acc@1 62.345678" (from nas.search logger)
_RE_NAS_TEST_ACC = re.compile(r'Test:\s+Acc@1\s+([\d.]+)')

# NAS best genotype: "New best genotype at epoch 3 (Acc@1 65.432100)"
_RE_NAS_BEST_GENOTYPE = re.compile(
    r'New best genotype at epoch\s+(\d+)\s+\(Acc@1\s+([\d.]+)\)'
)

# NAS per-step progress (from nas.train or nas.infer):
# "Epoch: [0]  [000/128]  eta: 00:12:34  lr: 0.025  samples/s: 45.2  loss: 1.23  acc1: 33.33  ..."
_RE_NAS_STEP = re.compile(
    r'Epoch:\s*\[(\d+)\]\s+\[(\d+)/(\d+)\]\s+eta:\s*(\S+).*?loss:\s*([\d.]+).*?acc1:\s*([\d.]+)'
)

# NAS device info: "NAS device: mps (Apple Metal)"
_RE_NAS_DEVICE = re.compile(r'NAS device:\s*(\S+)')

# NAS param size: "param size = 0.013075MB"
_RE_NAS_PARAM_SIZE = re.compile(r'param size\s*=\s*([\d.]+)MB')

# NAS genotype: captures the ops from "genotype = Genotype_CNN(normal=[('skip_connect', 0), ...])"
_RE_NAS_GENOTYPE = re.compile(r'genotype\s*=\s*(Genotype_CNN\(.*\))')

# Model summary from torchinfo: "Total params: 1,983"
_RE_TOTAL_PARAMS = re.compile(r'Total params:\s*([\d,]+)')

# Estimated model size from torchinfo: "Estimated Total Size (MB): 0.08"
_RE_MODEL_SIZE = re.compile(r'Estimated Total Size \(MB\):\s*([\d.]+)')

# NAS budget/epochs: "nas_budget=20" or "nas_epochs=20" in args dump
_RE_NAS_BUDGET = re.compile(r'nas_budget[=:]\s*(\d+)')


class TrainingLogParser:
    """Parse tinyml_modelmaker training log lines incrementally."""

    def __init__(self):
        self.float_epochs = []   # list of dicts (keys vary by task type)
        self.quant_epochs = []
        self.best_float = {}     # {epoch, acc, f1, auc} or {epoch, mse, r2} or {epoch, smape}
        self.best_quant = {}
        self.float_conf_matrix = None   # {headers: [...], rows: [{label, values: [...]}]}
        self.quant_conf_matrix = None
        self._current_phase = 'FloatTrain'
        self._parsing_conf_matrix = False
        self._conf_headers = []
        self._conf_rows = []
        self._pending_eval = {}  # accumulate eval metrics before epoch append
        self._last_train_epoch = -1
        self.flcs_log_path = None  # file-level classification summary log
        self.dataset_path = None   # dataset directory for hyperlinks
        self.task_type = None      # 'classification', 'regression', 'forecasting' (auto-detected)
        self.test_data_metrics = []  # forecasting: [{var, smape, r2}] from test_data lines
        # NAS search state
        self.nas_epochs = []         # list of {epoch, train_acc, test_acc}
        self.nas_best = {}           # {epoch, acc}
        self.is_nas = False          # auto-detected from NAS log lines
        self.nas_device = None       # 'mps', 'cuda', 'cpu'
        self.nas_param_size = None   # search model size in MB
        self._nas_pending = {}       # accumulate NAS epoch data
        self._nas_last_step = {}     # last per-step data for progress display
        self.nas_genotype = None     # latest genotype string
        self.nas_best_genotype = None  # genotype at best epoch
        self.nas_budget = None       # total NAS search budget (epochs)
        self.nas_model_params = None  # total params in discovered model
        self.nas_model_size = None   # estimated model size (MB)

    def feed_line(self, line: str) -> bool:
        """
        Feed a single log line. Returns True if new data was extracted
        (caller should regenerate the report).
        """
        line = line.rstrip()
        if not line:
            return False

        # Detect phase from logger name
        if 'QuantTrain' in line:
            self._current_phase = 'QuantTrain'
        elif 'FloatTrain' in line:
            self._current_phase = 'FloatTrain'

        # Strip log prefix to get the message
        m = _RE_LOG_PREFIX.match(line)
        if m:
            logger_name = m.group(1)
            message = m.group(2)
        else:
            # Line without standard log prefix (e.g. continuation of confusion matrix)
            logger_name = ''
            message = line

        # ---- NAS search lines (from root.modelopt.nas.* loggers) ----
        if 'modelopt.nas' in logger_name:
            self.is_nas = True
            return self._parse_nas_line(logger_name, message)

        is_best = 'BestEpoch' in logger_name

        # Confusion matrix parsing (multi-line block)
        if self._parsing_conf_matrix:
            matrix_done = self._parse_conf_line(message)
            if matrix_done and self._parsing_conf_matrix is False:
                # Matrix ended on a non-table line — fall through to process
                # the current line for other patterns (e.g. FLCS path)
                pass
            else:
                return matrix_done

        if _RE_CONF_MATRIX_START.search(message):
            self._parsing_conf_matrix = True
            self._conf_headers = []
            self._conf_rows = []
            # The rest of the line after "Confusion Matrix:" might have data
            rest = _RE_CONF_MATRIX_START.split(message, 1)[-1].strip()
            if rest:
                self._parse_conf_line(rest)
            return False

        # Training epoch metrics (classification: has acc1)
        tm = _RE_EPOCH_TRAIN.search(message)
        if tm:
            epoch = int(tm.group(1))
            loss = float(tm.group(2))
            acc = float(tm.group(3))
            self._last_train_epoch = epoch
            self._pending_eval = {'epoch': epoch, 'train_loss': loss, 'train_acc': acc}
            if self.task_type is None:
                self.task_type = 'classification'
            return False

        # Training epoch metrics (regression/forecasting: loss only, no acc1)
        if not tm:
            tm2 = _RE_EPOCH_TRAIN_LOSS_ONLY.search(message)
            if tm2 and 'Epoch:' in message:
                epoch = int(tm2.group(1))
                loss = float(tm2.group(2))
                self._last_train_epoch = epoch
                self._pending_eval = {'epoch': epoch, 'train_loss': loss}
                return False

        # ---- Classification evaluation metrics ----

        # Evaluation accuracy
        ta = _RE_TEST_ACC.search(message)
        if ta and not is_best:
            acc = float(ta.group(1))
            self._pending_eval['val_acc'] = acc
            return False

        # Evaluation F1
        tf = _RE_TEST_F1.search(message)
        if tf and not is_best:
            f1 = float(tf.group(1))
            self._pending_eval['val_f1'] = f1
            return False

        # AU-ROC (triggers flush for classification)
        au = _RE_AUC.search(message)
        if au and not is_best:
            auc = float(au.group(1))
            self._pending_eval['val_auc'] = auc
            self._flush_pending_eval()
            return True

        # ---- Regression evaluation metrics ----

        # Validation MSE
        mse_m = _RE_TEST_MSE.search(message)
        if mse_m and not is_best:
            self._pending_eval['val_mse'] = float(mse_m.group(1))
            if self.task_type is None:
                self.task_type = 'regression'
            return False

        # Validation R2-Score (triggers flush for regression)
        r2_m = _RE_TEST_R2.search(message)
        if r2_m and not is_best:
            self._pending_eval['val_r2'] = float(r2_m.group(1))
            if self.task_type is None:
                self.task_type = 'regression'
            self._flush_pending_eval()
            return True

        # ---- Forecasting evaluation metrics ----

        # Per-epoch overall SMAPE (triggers flush for forecasting)
        smape_m = _RE_CURRENT_SMAPE.search(message)
        if smape_m and not is_best:
            self._pending_eval['val_smape'] = float(smape_m.group(1))
            if self.task_type is None:
                self.task_type = 'forecasting'
            self._flush_pending_eval()
            return True

        # ---- Best epoch info (all task types) ----

        be = _RE_BEST_EPOCH.search(message)
        if be:
            best = self._get_best_dict()
            best['epoch'] = int(be.group(1))
            return False

        # Classification best-epoch metrics
        ba = _RE_BEST_ACC.match(message.strip())
        if ba and is_best:
            best = self._get_best_dict()
            best['acc'] = float(ba.group(1))
            return True

        bf = _RE_BEST_F1.match(message.strip())
        if bf and is_best:
            best = self._get_best_dict()
            best['f1'] = float(bf.group(1))
            return False

        bau = _RE_BEST_AUC.match(message.strip())
        if bau and is_best:
            best = self._get_best_dict()
            best['auc'] = float(bau.group(1))
            return True

        # Regression best-epoch metrics
        bmse = _RE_BEST_MSE.match(message.strip())
        if bmse and is_best:
            best = self._get_best_dict()
            best['mse'] = float(bmse.group(1))
            return False

        br2 = _RE_BEST_R2.match(message.strip())
        if br2 and is_best:
            best = self._get_best_dict()
            best['r2'] = float(br2.group(1))
            return True

        # Forecasting best-epoch metrics
        bsmape = _RE_BEST_OVERALL_SMAPE.search(message)
        if bsmape and is_best:
            best = self._get_best_dict()
            best['smape'] = float(bsmape.group(1))
            return True

        # Per-variable SMAPE (best epoch or test_data)
        pvs = _RE_PER_VAR_SMAPE.search(message)
        if pvs:
            var_name = pvs.group(1)
            val = float(pvs.group(2))
            is_test_data = 'test_data' in logger_name
            if is_test_data:
                # Find or create entry in test_data_metrics
                entry = next((e for e in self.test_data_metrics if e['var'] == var_name), None)
                if not entry:
                    entry = {'var': var_name}
                    self.test_data_metrics.append(entry)
                entry['smape'] = val
            elif is_best:
                best = self._get_best_dict()
                if 'per_var_smape' not in best:
                    best['per_var_smape'] = []
                best['per_var_smape'].append({'var': var_name, 'smape': val})
            return False

        # Per-variable R² (best epoch or test_data)
        pvr = _RE_PER_VAR_R2.search(message)
        if pvr:
            var_name = pvr.group(1)
            val = float(pvr.group(2))
            is_test_data = 'test_data' in logger_name
            if is_test_data:
                entry = next((e for e in self.test_data_metrics if e['var'] == var_name), None)
                if not entry:
                    entry = {'var': var_name}
                    self.test_data_metrics.append(entry)
                entry['r2'] = val
            elif is_best:
                best = self._get_best_dict()
                if 'per_var_r2' not in best:
                    best['per_var_r2'] = []
                best['per_var_r2'].append({'var': var_name, 'r2': val})
            return True

        # File-level classification summary log path
        fp = _RE_FLCS_PATH.search(line)
        if fp:
            self.flcs_log_path = fp.group(1).strip()
            return False

        # Dataset path
        dp = _RE_DATASET_PATH.search(line)
        if dp:
            self.dataset_path = dp.group(1).strip()
            return False

        # Torchinfo model summary (captured when NAS is active)
        if self.is_nas:
            tp = _RE_TOTAL_PARAMS.search(message)
            if tp:
                self.nas_model_params = int(tp.group(1).replace(',', ''))
                return False
            ms = _RE_MODEL_SIZE.search(message)
            if ms:
                self.nas_model_size = float(ms.group(1))
                return True

        # NAS budget from args dump (comes from root.main, not NAS logger)
        bm = _RE_NAS_BUDGET.search(line)
        if bm:
            self.nas_budget = int(bm.group(1))
            return False

        return False

    def _flush_pending_eval(self):
        """Append accumulated eval metrics to the current phase's epoch list."""
        if not self._pending_eval:
            return
        epochs = self.float_epochs if self._current_phase == 'FloatTrain' else self.quant_epochs
        entry = dict(self._pending_eval)  # preserve all keys from the accumulator
        entry.setdefault('epoch', len(epochs))
        entry.setdefault('train_loss', 0)
        epochs.append(entry)
        self._pending_eval = {}

    def _get_best_dict(self):
        return self.best_float if self._current_phase == 'FloatTrain' else self.best_quant

    def _parse_conf_line(self, line: str) -> bool:
        """Parse a line that's part of a confusion matrix block. Returns True when matrix is complete."""
        line = line.strip()

        # Grid separator lines: +---+---+---+
        if re.match(r'^[+\-=]+$', line):
            return False

        # Empty or non-table line signals end of matrix
        if not line or (not line.startswith('|') and not line.startswith('+')):
            self._parsing_conf_matrix = False
            matrix = {'headers': self._conf_headers, 'rows': self._conf_rows}
            if self._current_phase == 'QuantTrain':
                self.quant_conf_matrix = matrix
            else:
                self.float_conf_matrix = matrix
            return True

        # Header row
        hm = _RE_CONF_HEADER.search(line)
        if hm:
            headers_str = hm.group(1)
            self._conf_headers = [
                h.replace('Predicted as:', '').strip()
                for h in re.findall(r'Predicted as:\s*([^|]+)', headers_str)
            ]
            return False

        # Data row
        rm = _RE_CONF_ROW.match(line)
        if rm:
            label = rm.group(1).replace('Ground Truth:', '').strip()
            values_str = rm.group(2)
            values = [float(v.strip()) for v in values_str.strip('|').split('|') if v.strip()]
            self._conf_rows.append({'label': label, 'values': values})
            return False

        return False

    # ------------------------------------------------------------------
    # NAS search parsing
    # ------------------------------------------------------------------

    def _parse_nas_line(self, logger_name: str, message: str) -> bool:
        """Parse a line from root.modelopt.nas.* loggers. Returns True if new data."""

        # NAS device info
        dm = _RE_NAS_DEVICE.search(message)
        if dm:
            self.nas_device = dm.group(1)
            return False

        # NAS param size
        pm = _RE_NAS_PARAM_SIZE.search(message)
        if pm:
            self.nas_param_size = float(pm.group(1))
            return False

        # Per-step progress (for live ETA display)
        sm = _RE_NAS_STEP.search(message)
        if sm:
            self._nas_last_step = {
                'epoch': int(sm.group(1)),
                'step': int(sm.group(2)),
                'total_steps': int(sm.group(3)),
                'eta': sm.group(4),
                'loss': float(sm.group(5)),
                'acc': float(sm.group(6)),
            }
            return False

        # Per-epoch train accuracy
        ta = _RE_NAS_TRAIN_ACC.search(message)
        if ta:
            self._nas_pending['train_acc'] = float(ta.group(1))
            return False

        # Per-epoch test accuracy (triggers flush)
        te = _RE_NAS_TEST_ACC.search(message)
        if te:
            self._nas_pending['test_acc'] = float(te.group(1))
            self._flush_nas_epoch()
            return True

        # Best genotype
        bg = _RE_NAS_BEST_GENOTYPE.search(message)
        if bg:
            self.nas_best = {
                'epoch': int(bg.group(1)),
                'acc': float(bg.group(2)),
            }
            # Snapshot the latest genotype as the best
            if self.nas_genotype:
                self.nas_best_genotype = self.nas_genotype
            return True

        # Genotype dump (logged each epoch)
        gm = _RE_NAS_GENOTYPE.search(message)
        if gm:
            self.nas_genotype = gm.group(1)
            return False

        return False

    def _flush_nas_epoch(self):
        """Append accumulated NAS epoch data to nas_epochs."""
        if not self._nas_pending:
            return
        entry = dict(self._nas_pending)
        entry.setdefault('epoch', len(self.nas_epochs))
        self.nas_epochs.append(entry)
        self._nas_pending = {}


# ---------------------------------------------------------------------------
# HTML report generator
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
{auto_refresh}
<title>mmcli Training Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #242837;
    --border: #2e3348;
    --text: #e4e6f0;
    --text-dim: #8b8fa3;
    --accent: #6c63ff;
    --accent2: #00d4aa;
    --accent3: #ff6b8a;
    --float-color: #6c63ff;
    --quant-color: #00d4aa;
    --success: #00d4aa;
    --danger: #ff6b8a;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Inter',sans-serif; background:var(--bg); color:var(--text); padding:24px; min-height:100vh; }}
  .container {{ max-width:1200px; margin:0 auto; }}
  h1 {{ font-size:28px; font-weight:700; margin-bottom:4px; background:linear-gradient(135deg,var(--accent),var(--accent2)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
  .subtitle {{ color:var(--text-dim); font-size:14px; margin-bottom:32px; }}
  .cards {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:16px; margin-bottom:32px; }}
  .card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:20px; }}
  .card .label {{ font-size:12px; text-transform:uppercase; letter-spacing:1px; color:var(--text-dim); margin-bottom:8px; }}
  .card .value {{ font-size:24px; font-weight:600; }}
  .card .value.accent {{ color:var(--accent); }}
  .card .value.accent2 {{ color:var(--accent2); }}
  .card .value.accent3 {{ color:var(--accent3); }}
  .chart-section {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:24px; margin-bottom:24px; }}
  .chart-section h2 {{ font-size:18px; font-weight:600; margin-bottom:16px; color:var(--text); }}
  .chart-container {{ position:relative; height:320px; }}
  .conf-section {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:24px; margin-bottom:24px; }}
  .conf-section h2 {{ font-size:18px; font-weight:600; margin-bottom:16px; }}
  .conf-phase {{ margin-bottom:24px; }}
  .conf-phase h3 {{ font-size:15px; font-weight:500; margin-bottom:12px; color:var(--text-dim); }}
  table.conf {{ border-collapse:collapse; width:auto; font-size:13px; }}
  table.conf th, table.conf td {{ padding:10px 14px; text-align:center; border:1px solid var(--border); }}
  table.conf th {{ background:var(--surface2); font-weight:500; color:var(--text-dim); }}
  table.conf td.label-cell {{ text-align:left; font-weight:500; background:var(--surface2); color:var(--text-dim); white-space:nowrap; }}
  .flcs-section {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:24px; margin-bottom:24px; }}
  .flcs-section h2 {{ font-size:18px; font-weight:600; margin-bottom:16px; }}
  table.flcs {{ border-collapse:collapse; width:100%; font-size:13px; }}
  table.flcs th {{ background:var(--surface2); font-weight:500; color:var(--text-dim); padding:10px 12px; text-align:left; border:1px solid var(--border); white-space:nowrap; }}
  table.flcs td {{ padding:8px 12px; border:1px solid var(--border); }}
  table.flcs tr:hover {{ background:var(--surface2); }}
  table.flcs a {{ color:var(--accent); text-decoration:none; }}
  table.flcs a:hover {{ text-decoration:underline; }}
  table.flcs .correct {{ color:var(--success); font-weight:500; }}
  table.flcs .incorrect {{ color:var(--danger); font-weight:500; }}
  .pca-section {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:24px; margin-bottom:24px; }}
  .pca-section h2 {{ font-size:18px; font-weight:600; margin-bottom:16px; }}
  .pca-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap:20px; }}
  .pca-grid figure {{ margin:0; text-align:center; }}
  .pca-grid img {{ max-width:100%; height:auto; border-radius:8px; border:1px solid var(--border); }}
  .pca-grid figcaption {{ color:var(--text-dim); font-size:13px; margin-top:8px; }}
  .status {{ display:inline-block; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:500; }}
  .status.training {{ background:rgba(108,99,255,0.15); color:var(--accent); }}
  .status.complete {{ background:rgba(0,212,170,0.15); color:var(--accent2); }}
  .footer {{ text-align:center; color:var(--text-dim); font-size:12px; margin-top:40px; padding-top:20px; border-top:1px solid var(--border); }}
</style>
</head>
<body>
<div class="container">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
    <h1>mmcli Training Report</h1>
    <span class="status {status_class}">{status_label}</span>
  </div>
  <div class="subtitle">{subtitle}</div>

  <div class="cards">
    {summary_cards}
  </div>

  {nas_chart_html}

  <div class="chart-section">
    <h2>Training Metrics</h2>
    <div class="chart-container"><canvas id="metricsChart"></canvas></div>
  </div>

  {conf_matrix_html}

  {flcs_html}

  {pca_html}

  <div class="footer">Generated by mmcli · Musical Platypus Vibes © 2026</div>
</div>

<script>
const chartOpts = {{
  responsive:true, maintainAspectRatio:false,
  plugins:{{ legend:{{ labels:{{ color:'#8b8fa3',font:{{family:'Inter'}} }} }} }},
  scales:{{
    x:{{ title:{{display:true,text:'Epoch',color:'#8b8fa3',font:{{family:'Inter'}}}}, grid:{{color:'#2e3348'}}, ticks:{{color:'#8b8fa3'}} }},
    y:{{ grid:{{color:'#2e3348'}}, ticks:{{color:'#8b8fa3'}} }}
  }}
}};

// Combined metrics chart (dual Y-axes)
new Chart(document.getElementById('metricsChart'), {{
  type:'line',
  data:{{
    labels:{chart_labels},
    datasets:[
      {chart_datasets}
    ]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    interaction:{{ mode:'index', intersect:false }},
    plugins:{{ legend:{{ labels:{{ color:'#8b8fa3',font:{{family:'Inter'}} }} }} }},
    scales:{{
      x:{{ title:{{display:true,text:'Epoch',color:'#8b8fa3',font:{{family:'Inter'}}}}, grid:{{color:'#2e3348'}}, ticks:{{color:'#8b8fa3'}} }},
      y:{{ type:'linear', position:'left', title:{{display:true,text:'{y_axis_label}',color:'#8b8fa3',font:{{family:'Inter'}}}}, grid:{{color:'#2e3348'}}, ticks:{{color:'#8b8fa3'}} }},
      y1:{{ type:'linear', position:'right', title:{{display:true,text:'Loss',color:'#8b8fa3',font:{{family:'Inter'}}}}, grid:{{drawOnChartArea:false}}, ticks:{{color:'#8b8fa3'}} }}
    }}
  }}
}});

{nas_chart_js}
</script>
</body>
</html>"""


def _make_dataset_js(label: str, data: list, color: str, dashed: bool = False, yaxis: str = 'y') -> str:
    """Build a Chart.js dataset object string."""
    dash = "borderDash:[5,5]," if dashed else ""
    return (
        f"{{label:'{label}',data:{data},borderColor:'{color}',"
        f"backgroundColor:'{color}22',tension:0.3,pointRadius:2,"
        f"borderWidth:2,{dash}yAxisID:'{yaxis}',fill:false}}"
    )


def _extract_genotype_ops(genotype_str: str) -> list:
    """Extract unique operation names from a Genotype_CNN string.

    E.g. from "Genotype_CNN(normal=[('skip_connect', 0), ('conv_bn_relu_3x1', 0), ...]"
    returns ['skip_connect', 'conv_bn_relu_3x1', 'max_pool_3x1', ...]
    """
    # Match both single and double quoted operation names
    ops = re.findall(r'["\']([a-z][a-z_\d]*)["\']', genotype_str)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for op in ops:
        if op not in seen:
            seen.add(op)
            unique.append(op)
    return unique


def _conf_matrix_to_html(matrix: dict, title: str) -> str:
    """Render a confusion matrix dict as a styled HTML table with heatmap."""
    if not matrix or not matrix.get('rows'):
        return ''

    headers = matrix['headers']
    rows = matrix['rows']

    # Find max value for color scaling
    all_vals = [v for r in rows for v in r['values']]
    max_val = max(all_vals) if all_vals else 1

    html = f'<div class="conf-phase"><h3>{title}</h3>\n'
    html += '<table class="conf"><thead><tr><th></th>'
    for h in headers:
        html += f'<th>{h}</th>'
    html += '</tr></thead><tbody>\n'

    for i, row in enumerate(rows):
        html += f'<tr><td class="label-cell">{row["label"]}</td>'
        for j, val in enumerate(row['values']):
            intensity = val / max_val if max_val > 0 else 0
            if i == j:  # diagonal = correct predictions
                bg = f'rgba(0,212,170,{intensity * 0.6 + 0.05})'
                c = '#fff' if intensity > 0.3 else 'var(--text-dim)'
            else:  # off-diagonal = errors
                bg = f'rgba(255,107,138,{intensity * 0.6})'
                c = '#fff' if intensity > 0.3 else 'var(--text-dim)'
            html += f'<td style="background:{bg};color:{c}">{int(val)}</td>'
        html += '</tr>\n'

    html += '</tbody></table></div>\n'
    return html


def _parse_flcs_log(log_path: str) -> list:
    """
    Parse a file_level_classification_summary.log file.
    Returns list of dicts with keys from the table header.
    The log uses tabulate 'pretty' format.
    """
    if not log_path or not os.path.isfile(log_path):
        return []

    with open(log_path, 'r') as f:
        content = f.read()

    # Find table lines (lines starting with | or +)
    lines = content.split('\n')
    table_lines = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        # Strip log prefix if present
        if ':' in stripped and ('INFO' in stripped or 'DEBUG' in stripped):
            # Extract message part after last known logger prefix
            parts = stripped.split(': ', 2)
            if len(parts) >= 3:
                stripped = parts[-1].strip()

        if stripped.startswith('+') or stripped.startswith('|'):
            table_lines.append(stripped)
            in_table = True
        elif in_table and not stripped:
            break  # end of table

    if not table_lines:
        return []

    # Parse header and data rows
    headers = []
    rows = []
    for line in table_lines:
        if line.startswith('+'):
            continue  # separator
        if line.startswith('|'):
            cells = [c.strip() for c in line.split('|')[1:-1]]
            if not headers:
                headers = cells
            else:
                if cells and cells[0] and not all(c == '' for c in cells):
                    row = {}
                    for i, h in enumerate(headers):
                        row[h] = cells[i] if i < len(cells) else ''
                    rows.append(row)

    return rows


def _flcs_to_html(rows: list, dataset_path: str = None) -> str:
    """
    Render file-level classification summary as a prettified HTML table.
    Hyperlinks filenames to their location in the dataset directory.
    """
    if not rows:
        return ''

    headers = list(rows[0].keys())
    # Skip index column if present (numeric first column)
    if headers and headers[0] == '':
        headers = headers[1:]

    html = '<div class="flcs-section"><h2>File-Level Classification Summary</h2>\n'
    html += '<div style="overflow-x:auto;">\n'
    html += '<table class="flcs"><thead><tr>'
    for h in headers:
        html += f'<th>{h}</th>'
    html += '</tr></thead><tbody>\n'

    for row in rows:
        html += '<tr>'
        for h in headers:
            val = row.get(h, '')
            if h == 'File' and val:
                # Hyperlink the filename to the dataset
                if dataset_path:
                    true_class = row.get('True Class', '')
                    # Dataset files are typically in <dataset_path>/<class_name>/
                    file_path = os.path.join(dataset_path, true_class, val)
                    html += f'<td><a href="file:///{file_path}" title="{file_path}">{val}</a></td>'
                else:
                    html += f'<td>{val}</td>'
            elif h.startswith('Predicted as'):
                # Highlight: green if this is the correct class column, red if mispredicted
                true_class = row.get('True Class', '')
                pred_class = h.replace('Predicted as ', '')
                count = int(val) if val.isdigit() else 0
                if pred_class == true_class and count > 0:
                    html += f'<td class="correct">{val}</td>'
                elif count > 0:
                    html += f'<td class="incorrect">{val}</td>'
                else:
                    html += f'<td>{val}</td>'
            else:
                html += f'<td>{val}</td>'
        html += '</tr>\n'

    html += '</tbody></table></div></div>\n'
    return html


def _find_pca_images(report_dir: str) -> list:
    """
    Search for PCA visualization PNGs near the report directory.
    Returns list of (title, abs_path) tuples for found images.
    """
    pca_files = [
        ('Train Data', 'pca_on_feature_extracted_train_data.png'),
        ('Validation Data', 'pca_on_feature_extracted_validation_data.png'),
    ]
    found = []
    # Search report dir and up to 3 levels of parent directories
    search_dirs = [report_dir]
    parent = report_dir
    for _ in range(3):
        parent = os.path.dirname(parent)
        if parent and parent != report_dir:
            search_dirs.append(parent)

    for title, filename in pca_files:
        for search_dir in search_dirs:
            # Direct match
            path = os.path.join(search_dir, filename)
            if os.path.isfile(path):
                found.append((title, path))
                break
            # Recursive glob within this dir
            matches = glob.glob(os.path.join(search_dir, '**', filename), recursive=True)
            if matches:
                found.append((title, matches[0]))
                break
    return found


def _pca_images_to_html(report_dir: str) -> str:
    """Render PCA visualization images as base64-embedded HTML."""
    images = _find_pca_images(report_dir)
    if not images:
        return ''

    html = '<div class="pca-section"><h2>PCA Feature Visualization</h2>\n'
    html += '<div class="pca-grid">\n'
    for title, path in images:
        try:
            with open(path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('ascii')
            html += (
                f'<figure>'
                f'<img src="data:image/png;base64,{b64}" alt="{title}">'
                f'<figcaption>{title}</figcaption>'
                f'</figure>\n'
            )
        except OSError:
            continue
    html += '</div></div>\n'
    return html


def _forecasting_var_table_to_html(best_dict: dict, test_data_metrics: list) -> str:
    """Render per-variable forecasting metrics as a styled HTML table."""
    # Merge best-epoch per-var data with test_data per-var data
    vars_data = {}  # var_name -> {best_smape, best_r2, test_smape, test_r2}
    for entry in best_dict.get('per_var_smape', []):
        vars_data.setdefault(entry['var'], {})['best_smape'] = entry['smape']
    for entry in best_dict.get('per_var_r2', []):
        vars_data.setdefault(entry['var'], {})['best_r2'] = entry['r2']
    for entry in test_data_metrics:
        v = entry['var']
        if 'smape' in entry:
            vars_data.setdefault(v, {})['test_smape'] = entry['smape']
        if 'r2' in entry:
            vars_data.setdefault(v, {})['test_r2'] = entry['r2']

    if not vars_data:
        return ''

    html = '<div class="flcs-section"><h2>Per-Variable Forecasting Metrics</h2>\n'
    html += '<div style="overflow-x:auto;">\n'
    html += '<table class="flcs"><thead><tr>'
    html += '<th>Variable</th>'
    has_best = any('best_smape' in v for v in vars_data.values())
    has_test = any('test_smape' in v for v in vars_data.values())
    if has_best:
        html += '<th>Best Epoch SMAPE (%)</th><th>Best Epoch R²</th>'
    if has_test:
        html += '<th>Test SMAPE (%)</th><th>Test R²</th>'
    html += '</tr></thead><tbody>\n'

    for var_name, data in vars_data.items():
        html += f'<tr><td>{var_name}</td>'
        if has_best:
            bs = f"{data['best_smape']:.2f}" if 'best_smape' in data else '—'
            br = f"{data['best_r2']:.4f}" if 'best_r2' in data else '—'
            html += f'<td>{bs}</td><td>{br}</td>'
        if has_test:
            ts = f"{data['test_smape']:.2f}" if 'test_smape' in data else '—'
            tr2 = f"{data['test_r2']:.4f}" if 'test_r2' in data else '—'
            html += f'<td>{ts}</td><td>{tr2}</td>'
        html += '</tr>\n'

    html += '</tbody></table></div></div>\n'
    return html


class HTMLReportGenerator:
    """Generate the HTML training report from parsed data."""

    def __init__(self, output_path: str):
        self.output_path = output_path

    def generate(self, parser: TrainingLogParser, is_complete: bool = False) -> None:
        """Generate/overwrite the HTML report file from current parser state."""
        float_epochs = parser.float_epochs
        quant_epochs = parser.quant_epochs
        has_quant = len(quant_epochs) > 0

        # Status
        status_class = 'complete' if is_complete else 'training'
        status_label = 'Complete' if is_complete else 'Training…'

        # Auto-refresh during training (every 5 seconds)
        auto_refresh = '' if is_complete else '<meta http-equiv="refresh" content="5">'

        # Subtitle
        total = len(float_epochs) + len(quant_epochs)
        parts = []
        if parser.nas_epochs:
            parts.append(f'{len(parser.nas_epochs)} NAS search epochs')
        if float_epochs:
            parts.append(f'{len(float_epochs)} float epochs')
        if quant_epochs:
            parts.append(f'{len(quant_epochs)} quant epochs')
        subtitle = ' · '.join(parts) if parts else (
            'NAS architecture search starting…' if parser.is_nas else 'Waiting for training data…'
        )

        # Render
        task = parser.task_type or 'classification'

        # Summary cards (adaptive by task type)
        cards = []

        # NAS search cards (always first if NAS was used)
        if parser.is_nas:
            # Progress info
            n_nas = len(parser.nas_epochs)
            if n_nas == 0 and not is_complete:
                cards.append(('NAS Search', 'Searching…', 'accent3'))
            elif parser.nas_budget:
                cards.append(('NAS Search', f'{n_nas}/{parser.nas_budget} epochs', 'accent3'))
            else:
                cards.append(('NAS Search', f'{n_nas} epochs', 'accent3'))

            if parser.nas_best.get('acc') is not None:
                cards.append(('NAS Best Acc', f"{parser.nas_best['acc']:.1f}%", 'accent3'))
                cards.append(('Best @ Epoch', str(parser.nas_best['epoch']), ''))

            # Live ETA from last step
            if parser._nas_last_step.get('eta') and not parser.float_epochs:
                cards.append(('Search ETA', parser._nas_last_step['eta'], ''))

            if parser.nas_device:
                cards.append(('NAS Device', parser.nas_device.upper(), ''))
            if parser.nas_param_size is not None:
                cards.append(('Search Model', f'{parser.nas_param_size:.3f} MB', ''))

            # Discovered model info (appears after training starts)
            if parser.nas_model_params is not None:
                cards.append(('Final Params', f'{parser.nas_model_params:,}', 'accent'))
            if parser.nas_model_size is not None:
                cards.append(('Final Size', f'{parser.nas_model_size} MB', 'accent'))

        if task == 'classification':
            if float_epochs:
                best_acc = max(e.get('val_acc', 0) for e in float_epochs)
                latest_loss = float_epochs[-1]['train_loss']
                cards.append(('Float Best Acc', f'{best_acc:.1f}%', 'accent'))
                cards.append(('Float Epochs', str(len(float_epochs)), ''))
                cards.append(('Float Latest Loss', f'{latest_loss:.4f}', ''))
            if quant_epochs:
                best_acc_q = max(e.get('val_acc', 0) for e in quant_epochs)
                cards.append(('Quant Best Acc', f'{best_acc_q:.1f}%', 'accent2'))
                cards.append(('Quant Epochs', str(len(quant_epochs)), ''))
        elif task == 'regression':
            if float_epochs:
                best_r2 = max(e.get('val_r2', -1e9) for e in float_epochs)
                best_mse = min(e.get('val_mse', 1e9) for e in float_epochs)
                latest_loss = float_epochs[-1]['train_loss']
                cards.append(('Float Best R²', f'{best_r2:.4f}', 'accent'))
                cards.append(('Float Best MSE', f'{best_mse:.3f}', ''))
                cards.append(('Float Epochs', str(len(float_epochs)), ''))
                cards.append(('Float Latest Loss', f'{latest_loss:.4f}', ''))
            if quant_epochs:
                best_r2_q = max(e.get('val_r2', -1e9) for e in quant_epochs)
                best_mse_q = min(e.get('val_mse', 1e9) for e in quant_epochs)
                cards.append(('Quant Best R²', f'{best_r2_q:.4f}', 'accent2'))
                cards.append(('Quant Best MSE', f'{best_mse_q:.3f}', ''))
                cards.append(('Quant Epochs', str(len(quant_epochs)), ''))
        elif task == 'forecasting':
            if float_epochs:
                best_smape = min(e.get('val_smape', 1e9) for e in float_epochs)
                latest_loss = float_epochs[-1]['train_loss']
                cards.append(('Float Best SMAPE', f'{best_smape:.2f}%', 'accent'))
                cards.append(('Float Epochs', str(len(float_epochs)), ''))
                cards.append(('Float Latest Loss', f'{latest_loss:.4f}', ''))
            if quant_epochs:
                best_smape_q = min(e.get('val_smape', 1e9) for e in quant_epochs)
                cards.append(('Quant Best SMAPE', f'{best_smape_q:.2f}%', 'accent2'))
                cards.append(('Quant Epochs', str(len(quant_epochs)), ''))

        # Best epoch cards (all task types)
        for phase_label, best_dict in [('Float', parser.best_float), ('Quant', parser.best_quant)]:
            accent = 'accent' if phase_label == 'Float' else 'accent2'
            if best_dict.get('epoch') is not None:
                cards.append((f'Best {phase_label} Epoch', str(best_dict['epoch']), accent))
                # Classification
                if 'f1' in best_dict:
                    cards.append((f'{phase_label} F1-Score', f"{best_dict['f1']:.3f}", ''))
                if 'auc' in best_dict:
                    cards.append((f'{phase_label} AUC ROC', f"{best_dict['auc']:.3f}", ''))
                # Regression
                if 'mse' in best_dict:
                    cards.append((f'{phase_label} Best MSE', f"{best_dict['mse']:.3f}", ''))
                if 'r2' in best_dict:
                    cards.append((f'{phase_label} Best R²', f"{best_dict['r2']:.4f}", accent))
                # Forecasting
                if 'smape' in best_dict:
                    cards.append((f'{phase_label} Best SMAPE', f"{best_dict['smape']:.2f}%", accent))

        cards_html = ''
        for label, value, cls in cards:
            cls_attr = f' {cls}' if cls else ''
            cards_html += (
                f'<div class="card"><div class="label">{label}</div>'
                f'<div class="value{cls_attr}">{value}</div></div>\n'
            )

        # Chart data — adaptive by task type
        chart_labels = [e['epoch'] for e in float_epochs]
        float_loss = [e['train_loss'] for e in float_epochs]

        if task == 'classification':
            y_axis_label = 'Accuracy (%)'
            float_primary = [e.get('val_acc', 0) for e in float_epochs]
            float_secondary = [e.get('train_acc', 0) for e in float_epochs]
            primary_label = 'Val Accuracy'
            secondary_label = 'Train Accuracy'
        elif task == 'regression':
            y_axis_label = 'R²-Score'
            float_primary = [e.get('val_r2', 0) for e in float_epochs]
            float_secondary = [e.get('val_mse', 0) for e in float_epochs]
            primary_label = 'Val R²-Score'
            secondary_label = 'Val MSE'
        else:  # forecasting
            y_axis_label = 'SMAPE (%)'
            float_primary = [e.get('val_smape', 0) for e in float_epochs]
            float_secondary = []  # no secondary for forecasting
            primary_label = 'Val SMAPE'
            secondary_label = None

        chart_datasets = []
        chart_datasets.append(_make_dataset_js(f'Float {primary_label}', float_primary, '#6c63ff', yaxis='y'))
        if float_secondary:
            chart_datasets.append(_make_dataset_js(f'Float {secondary_label}', float_secondary, '#6c63ff', dashed=True, yaxis='y' if task == 'classification' else 'y1'))
        if task == 'classification':
            chart_datasets.append(_make_dataset_js('Float Loss', float_loss, '#ff6b8a', yaxis='y1'))
        else:
            chart_datasets.append(_make_dataset_js('Float Loss', float_loss, '#ff6b8a', yaxis='y1'))

        if quant_epochs:
            q_offset = len(float_epochs)
            q_labels = [q_offset + e['epoch'] for e in quant_epochs]
            chart_labels = chart_labels + q_labels
            pad_float = ['null'] * len(quant_epochs)
            pad_quant = ['null'] * len(float_epochs)

            if task == 'classification':
                quant_primary = pad_quant + [e.get('val_acc', 0) for e in quant_epochs]
                quant_secondary = pad_quant + [e.get('train_acc', 0) for e in quant_epochs]
            elif task == 'regression':
                quant_primary = pad_quant + [e.get('val_r2', 0) for e in quant_epochs]
                quant_secondary = pad_quant + [e.get('val_mse', 0) for e in quant_epochs]
            else:
                quant_primary = pad_quant + [e.get('val_smape', 0) for e in quant_epochs]
                quant_secondary = []
            quant_loss = pad_quant + [e['train_loss'] for e in quant_epochs]

            # Pad float datasets
            float_primary_padded = float_primary + pad_float
            float_loss_padded = float_loss + pad_float
            chart_datasets[0] = _make_dataset_js(f'Float {primary_label}', float_primary_padded, '#6c63ff', yaxis='y')
            if float_secondary:
                float_secondary_padded = float_secondary + pad_float
                chart_datasets[1] = _make_dataset_js(f'Float {secondary_label}', float_secondary_padded, '#6c63ff', dashed=True, yaxis='y' if task == 'classification' else 'y1')
            # Update loss dataset index
            loss_idx = 2 if float_secondary else 1
            chart_datasets[loss_idx] = _make_dataset_js('Float Loss', float_loss_padded, '#ff6b8a', yaxis='y1')

            chart_datasets.append(_make_dataset_js(f'Quant {primary_label}', quant_primary, '#00d4aa', yaxis='y'))
            if quant_secondary:
                chart_datasets.append(_make_dataset_js(f'Quant {secondary_label}', quant_secondary, '#00d4aa', dashed=True, yaxis='y' if task == 'classification' else 'y1'))
            chart_datasets.append(_make_dataset_js('Quant Loss', quant_loss, '#ffa726', yaxis='y1'))

        # Confusion matrix HTML (classification only)
        conf_html = ''
        if parser.float_conf_matrix or parser.quant_conf_matrix:
            conf_html = '<div class="conf-section"><h2>Confusion Matrix</h2>\n'
            if parser.float_conf_matrix:
                conf_html += _conf_matrix_to_html(parser.float_conf_matrix, 'Float Training — Best Epoch')
            if parser.quant_conf_matrix:
                conf_html += _conf_matrix_to_html(parser.quant_conf_matrix, 'Quantized Training — Best Epoch')
            conf_html += '</div>'

        # File-level classification summary
        flcs_html = ''
        if is_complete and parser.flcs_log_path:
            flcs_rows = _parse_flcs_log(parser.flcs_log_path)
            if flcs_rows:
                flcs_html = _flcs_to_html(flcs_rows, parser.dataset_path)

        # Forecasting per-variable table
        if is_complete and task == 'forecasting':
            var_table = _forecasting_var_table_to_html(
                parser.best_float, parser.test_data_metrics)
            if var_table:
                flcs_html += var_table

        # PCA visualization images (only in final report)
        pca_html = ''
        if is_complete:
            report_dir = os.path.dirname(os.path.abspath(self.output_path))
            pca_html = _pca_images_to_html(report_dir)

        # NAS search chart
        nas_chart_html = ''
        nas_chart_js = ''
        if parser.nas_epochs:
            nas_chart_html = (
                '<div class="chart-section">'
                '<h2>NAS Architecture Search</h2>'
                '<div class="chart-container"><canvas id="nasChart"></canvas></div>'
            )
            # Add discovered architecture info below the chart
            if parser.nas_best_genotype:
                ops = _extract_genotype_ops(parser.nas_best_genotype)
                if ops:
                    nas_chart_html += (
                        '<div style="margin-top:16px;padding:16px;background:var(--surface2);'
                        'border-radius:8px;border:1px solid var(--border);">'
                        '<h3 style="font-size:14px;font-weight:600;color:var(--accent3);'
                        'margin-bottom:8px;">Discovered Architecture</h3>'
                        '<div style="display:flex;flex-wrap:wrap;gap:6px;">'
                    )
                    for op in ops:
                        nas_chart_html += (
                            f'<span style="padding:4px 10px;border-radius:6px;'
                            f'font-size:12px;background:rgba(255,107,138,0.12);'
                            f'color:var(--accent3);border:1px solid rgba(255,107,138,0.25);">'
                            f'{op}</span>'
                        )
                    nas_chart_html += '</div></div>'
            nas_chart_html += '</div>'
            nas_labels = [e.get('epoch', i) for i, e in enumerate(parser.nas_epochs)]
            nas_train = [e.get('train_acc', 0) for e in parser.nas_epochs]
            nas_test = [e.get('test_acc', 0) for e in parser.nas_epochs]
            nas_ds = []
            nas_ds.append(_make_dataset_js('Train Accuracy', nas_train, '#ffa726', yaxis='y'))
            nas_ds.append(_make_dataset_js('Test Accuracy', nas_test, '#ff6b8a', yaxis='y'))
            nas_chart_js = (
                "new Chart(document.getElementById('nasChart'), {{"
                "  type:'line',"
                "  data:{{ labels:{labels}, datasets:[{datasets}] }},"
                "  options:{{"
                "    responsive:true, maintainAspectRatio:false,"
                "    interaction:{{ mode:'index', intersect:false }},"
                "    plugins:{{ legend:{{ labels:{{ color:'#8b8fa3',font:{{family:'Inter'}} }} }} }},"
                "    scales:{{"
                "      x:{{ title:{{display:true,text:'Search Epoch',color:'#8b8fa3',font:{{family:'Inter'}}}}, grid:{{color:'#2e3348'}}, ticks:{{color:'#8b8fa3'}} }},"
                "      y:{{ type:'linear', position:'left', title:{{display:true,text:'Accuracy (%)',color:'#8b8fa3',font:{{family:'Inter'}}}}, grid:{{color:'#2e3348'}}, ticks:{{color:'#8b8fa3'}} }}"
                "    }}"
                "  }}"
                "}});"
            ).format(
                labels=nas_labels,
                datasets=','.join(nas_ds),
            )

        elif parser.is_nas and not is_complete:
            # NAS enabled but no epoch data yet — show placeholder chart section
            nas_chart_html = (
                '<div class="chart-section">'
                '<h2>NAS Architecture Search</h2>'
                '<div style="display:flex;align-items:center;justify-content:center;'
                'height:200px;color:var(--text-dim);font-size:14px;">'
                'Searching for optimal architecture…</div>'
                '</div>'
            )

        # Render
        html = _HTML_TEMPLATE.format(
            auto_refresh=auto_refresh,
            status_class=status_class,
            status_label=status_label,
            subtitle=subtitle,
            summary_cards=cards_html,
            chart_labels=chart_labels,
            chart_datasets=',\n      '.join(chart_datasets) if chart_datasets else '',
            conf_matrix_html=conf_html,
            flcs_html=flcs_html,
            pca_html=pca_html,
            y_axis_label=y_axis_label,
            nas_chart_html=nas_chart_html,
            nas_chart_js=nas_chart_js,
        )

        os.makedirs(os.path.dirname(self.output_path) or '.', exist_ok=True)
        with open(self.output_path, 'w') as f:
            f.write(html)


def create_report_handler(report_path: str, nas_enabled: bool = False):
    """
    Create a line handler for use with subprocess stdout capture.

    Returns (feed_line_fn, finalize_fn):
      - feed_line_fn(line: str) -> None: call for each stdout/stderr line
      - finalize_fn() -> None: call when training is complete
    """
    parser = TrainingLogParser()
    if nas_enabled:
        parser.is_nas = True
    generator = HTMLReportGenerator(report_path)
    # Generate initial report (shows NAS placeholders if nas_enabled)
    generator.generate(parser, is_complete=False)

    def feed_line(line: str) -> None:
        changed = parser.feed_line(line)
        if changed:
            generator.generate(parser, is_complete=False)

    def finalize() -> None:
        generator.generate(parser, is_complete=True)

    return feed_line, finalize

