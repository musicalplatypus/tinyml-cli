# mmcli

[tinyml-modelmaker](https://github.com/musicalplatypus/tinyml-tensorlab) 的命令行界面工具。

`mmcli` 是一个**独立的 macOS 原生二进制文件**，可完全通过命令行驱动 tinyml-modelmaker
训练和编译流水线——无需 YAML 配置文件（但也可以选择使用 YAML 文件作为基础配置）。

工具内置 **9 个示例数据集**，覆盖所有任务类型，只需一条命令即可创建项目并开始训练。

## 工作原理

`mmcli` 是一个轻量级二进制文件（约 10 MB），**不**捆绑
`tinyml_modelmaker`、PyTorch 或 TVM。它的工作方式是：

1. 将您的命令行参数转换为 `tinyml_modelmaker` 所需的配置字典
2. 写入临时 YAML 文件
3. 以子进程方式调用 `$MMCLI_PYTHON run_tinyml_modelmaker.py <tempfile>`

这意味着 `tinyml_modelmaker` 在您现有的 Python 3.10 环境中运行，包含所有
原生依赖（包括 macOS 训练所需的 MPS/Metal）。

> **关于编译：** TVM 编译器后端（`ti_mcu_nnc`）目前仅提供 Linux 和 Windows
> 的 wheel 包。`mmcli compile` 和 `mmcli run` 子命令可在 Linux/Windows 上运行。
> 在 macOS 上，您可以使用 `mmcli train` 通过 Metal/MPS 进行训练，
> 然后将生成的 `model.onnx` 传输到 Linux 机器上进行编译。

---

## 安装设置

### 1. 安装 tinyml_modelmaker（Python 3.10 环境）

```bash
# 创建并激活 Python 3.10 虚拟环境
python3.10 -m venv ~/.venv-tinyml
source ~/.venv-tinyml/bin/activate

# 从发布标签安装 tinyml_modelmaker
pip install "tinyml_modelmaker @ git+https://github.com/musicalplatypus/tinyml-tensorlab.git@PlatypusCLI_0.9.0_Release#subdirectory=tinyml-modelmaker"
```

### 2. 设置环境变量

添加到您的 `~/.zshrc`（或 `~/.bash_profile`）：

```bash
export MMCLI_PYTHON="$HOME/.venv-tinyml/bin/python"
```

### 3. 获取二进制文件

**方案 A — 使用预编译二进制：**
```bash
cp dist/mmcli /usr/local/bin/mmcli   # 或 PATH 中的任何位置
```

**方案 B — 自行编译**（需要任意 Python + 已激活虚拟环境中的 PyInstaller）：
```bash
git clone --branch PlatypusCLI_0.9.0_Release https://github.com/musicalplatypus/tinyml-cli.git
cd tinyml-cli
source ~/.venv-ai/bin/activate        # 任何安装了 PyInstaller 的虚拟环境
pip install pyinstaller -q
pip install -e .
bash build_macos.sh                   # → dist/mmcli（约 10 MB）
```

---

## 子命令

### `mmcli init` — 从示例数据集创建项目

创建一个预填充数据集的新项目目录，可直接用于训练。

```
mmcli init -t TASK --dataset DATASET_NAME -p PROJECT_DIR [-m MODULE]
```

| 参数 | 缩写 | 说明 |
|------|------|------|
| `--task` | `-t` | 任务类型 **（必填）** |
| `--dataset` | | 示例数据集名称 **（必填）** |
| `--project` | `-p` | 新项目目录路径 **（必填，目录不能已存在）** |
| `--module` | `-m` | AI 模块（如省略则从数据集自动检测） |

**示例：**
```bash
# 创建一个电弧故障分类项目
mmcli init -t arc_fault --dataset arc_fault_classification -p ./my_arc_project

# 然后使用该项目进行训练
mmcli train -m timeseries -t arc_fault -d F28P55 -n CLS_1k_NPU -i ./my_arc_project
```

> **提示：** 运行 `mmcli info -m timeseries -t <task>` 可查看指定任务可用的数据集。

---

### `mmcli train` — 仅训练

训练模型并导出 `model.onnx`，跳过编译步骤。

```
mmcli train -m MODULE -t TASK -d DEVICE -n MODEL -i PROJECT_DIR [选项]
mmcli train -m MODULE -t TASK -d DEVICE --nas SIZE -i PROJECT_DIR [选项]
```

| 参数 | 缩写 | 说明 |
|------|------|------|
| `--module` | `-m` | `timeseries` 或 `vision` |
| `--task` | `-t` | 任务类型（见下方列表） |
| `--device` | `-d` | 目标设备（例如 `F28P55`） |
| `--model` | `-n` | 模型目录中的模型名称（使用 `--nas` 时可选） |
| `--project` | `-i` | 包含 `dataset/` 的项目目录路径 |
| `--config` | `-c` | 可选的基础 YAML 文件（命令行参数会覆盖其中的值） |
| `--feature-extraction` | | 特征提取预设名称 |
| `--epochs` | | 训练轮数 |
| `--batch-size` | | 批次大小 |
| `--lr` | | 学习率 |
| `--gpus` | | GPU 数量（0 = CPU/MPS，macOS 默认值） |
| `--quantization` | | `NO_QUANTIZATION` 或 `QUANTIZATION_TINPU` |
| `--run-name` | | 输出文件夹名称（支持 `{date-time}`、`{model_name}` 占位符） |
| `--output` | | 根输出目录 |
| `--compile-model` | | `0`（默认）或 `1` 启用 torch.compile（推荐 CUDA） |
| `--native-amp` | | 启用原生混合精度（推荐 CUDA，不适用于 MPS） |
| `--report` | | 生成实时更新的 HTML 训练报告（图表 + 混淆矩阵） |
| `--nas` | | NAS 模型大小预设：`s`、`m`、`l`、`xl`（仅分类任务） |
| `--nas-epochs` | | NAS 搜索轮数（默认：10） |
| `--nas-optimize` | | NAS 资源优化目标：`Memory`（默认）或 `Compute` |

**示例：**
```bash
mmcli train \
  -m timeseries \
  -t generic_timeseries_classification \
  -d F28P55 \
  -n CLS_1k_NPU \
  -i ./data/my_project \
  --epochs 30 \
  --batch-size 256
```

---

### `mmcli compile` — 仅编译

编译预训练的 ONNX 文件，不需要训练数据。

```
mmcli compile -m MODULE -t TASK -d DEVICE -n MODEL -o ONNX_FILE [选项]
```

| 参数 | 缩写 | 说明 |
|------|------|------|
| `--onnx` | `-o` | 已有 ONNX 模型文件路径 **（必填）** |
| `--preset` | | 编译预设（默认：`default_preset`） |
| *（上述通用参数）* | | |

**示例：**
```bash
mmcli compile \
  -m timeseries \
  -t generic_timeseries_classification \
  -d F28P55 \
  -n CLS_1k_NPU \
  -o ./data/projects/my_run/model.onnx
```

---

### `mmcli run` — 完整流水线

先训练后编译。接受 `train` 和 `compile` 的所有参数。

```bash
mmcli run \
  -m timeseries \
  -t generic_timeseries_classification \
  -d F28P55 \
  -n CLS_1k_NPU \
  -i ./data/my_dataset \
  --quantization QUANTIZATION_TINPU
```

---

## 示例数据集

`mmcli` 内置 9 个从 TI 服务器下载的示例数据集。使用 `mmcli init` 将其提取到新项目中。

| 数据集名称 | 任务类型 | 大小 | 说明 |
|-----------|---------|------|------|
| `generic_timeseries_classification` | 分类 | 2.5 MB | 合成波形（锯齿波、正弦波、方波） |
| `generic_timeseries_regression` | 回归 | 885 KB | 合成时序回归数据 |
| `generic_timeseries_anomalydetection` | 异常检测 | 4.0 MB | 幅度/频率异常 |
| `generic_timeseries_forecasting` | 预测 | 69 KB | 模拟恒温器温度 |
| `arc_fault_classification` | 电弧故障 | 13 MB | 直流电弧故障电流分类（DSI 传感器） |
| `ecg_classification` | 心电分类 | 4.4 MB | ECG 二分类心跳（正常 vs 异常） |
| `fan_blade_fault` | 电机故障 | 54 MB | 风扇叶片振动数据（三轴加速度计） |
| `pir_detection` | PIR 检测 | 1.5 MB | PIR 运动检测分类（人类 vs 非人类） |
| `mnist_image_classification` | 图像分类 | 45 MB | MNIST 手写数字（28×28 图像） |

如需使用外部数据集目录替代内置目录，请设置：
```bash
export MMCLI_DATASETS=/path/to/your/datasets
```


## 实用选项

### `mmcli info` — 查询模型注册表

显示支持的任务类型、模型、设备、特征提取预设和可用的示例数据集。

```
mmcli info -m MODULE [-t TASK] [-d DEVICE]
```

| 参数 | 缩写 | 说明 |
|------|------|------|
| `--module` | `-m` | `timeseries` 或 `vision` **（必填）** |
| `--task` | `-t` | 要显示详情的任务类型。省略则列出所有任务类型。 |
| `--device` | `-d` | 用于筛选模型的目标设备。 |

**示例：**
```bash
mmcli info -m timeseries                        # 列出任务类型
mmcli info -m timeseries -t arc_fault           # arc_fault 的详情
mmcli info -m timeseries -t arc_fault -d F28P55 # F28P55 可用的模型
```

---

### 试运行 — 查看生成的 YAML 但不执行

```bash
mmcli --dry-run train \
  -m timeseries -t generic_timeseries_classification \
  -d F28P55 -n CLS_1k_NPU -i ./data
```

### 覆盖基础 YAML 配置

```bash
mmcli train --config examples/hello_world/config.yaml \
            --epochs 50 --device F29H85
```

### 详细输出（显示子进程命令 + 调试日志）

```bash
mmcli --verbose train ...
```

### 训练报告

生成包含实时更新准确率/损失图表和热力图混淆矩阵的独立 HTML 报告：

```bash
mmcli train \
  -m timeseries \
  -t generic_timeseries_classification \
  -d F28P55 \
  -n CLS_1k_NPU \
  -i ./data/my_project \
  --report
```

报告将写入 `<project_dir>/run/report.html`，训练过程中每 5 秒自动刷新一次。
训练完成后，自动刷新功能将被移除，最终报告将包含混淆矩阵和文件级分类摘要（如可用）。

---

## 神经架构搜索（NAS）

您可以让 NAS 自动发现最优架构，而不是从模型目录中选择模型（`-n`）。NAS 仅支持
**分类任务**（时序和视觉）。

当设置 `--nas` 时，`--model/-n` 变为可选——会自动生成类似 `NAS_m` 的合成名称。

### NAS 中的特征提取

> **重要：** 对于时序任务，使用 `--nas` 时**必须**指定与数据集匹配的
> `--feature-extraction` 预设。
>
> 使用目录模型（`-n`）时，特征提取配置由模型描述自动提供。NAS 没有目录条目，
> 因此流水线不知道应用哪种特征提取。如果不设置此参数，训练将因
> *"Not enough dimensions present"* 而失败，因为原始传感器数据未被转换为
> 分类流水线所需的特征表示。

查看任务可用的特征提取预设：

```bash
mmcli info -m timeseries -t generic_timeseries_classification
```

通用时序分类的常用预设包括：
- `Generic_256Input_FFTBIN_16Feature_8Frame`
- `Generic_1024Input_FFTBIN_32Feature_32Frame`

### NAS 参数

| 参数 | 说明 |
|------|------|
| `--nas SIZE` | 启用 NAS 并设置模型大小预设：`s`（小）、`m`（中）、`l`（大）、`xl`（超大）。控制搜索空间复杂度和生成的模型大小。 |
| `--nas-epochs N` | NAS 搜索轮数（默认：10）。值越高探索的架构越多，但耗时更长。 |
| `--nas-optimize MODE` | 资源优化目标：`Memory`（更少参数，默认）或 `Compute`（更少 MAC 运算，更低延迟）。 |
| `--feature-extraction` | 特征提取预设 **（时序 NAS 必填）**。使用 `mmcli info` 查看可用预设。 |

### NAS 示例

```bash
# 基础 NAS — 中等大小模型 + 特征提取
mmcli train \
  -m timeseries \
  -t generic_timeseries_classification \
  -d F28P55 \
  -i ./data/my_project \
  --nas m \
  --feature-extraction Generic_256Input_FFTBIN_16Feature_8Frame \
  --epochs 50

# NAS — 指定搜索预算和计算优化
mmcli train \
  -m timeseries \
  -t motor_fault \
  -d F28P55 \
  -i ./data/motor_project \
  --nas l \
  --nas-epochs 20 \
  --nas-optimize Compute \
  --feature-extraction Generic_256Input_FFTBIN_16Feature_8Frame

# 视觉分类 NAS（不需要 --feature-extraction）
mmcli train \
  -m vision \
  -t image_classification \
  -d F29H85 \
  -i ./data/image_project \
  --nas s

# 试运行查看 NAS 配置
mmcli --dry-run train \
  -m timeseries -t generic_timeseries_classification \
  -d F28P55 -i ./data/my_project --nas m \
  --feature-extraction Generic_256Input_FFTBIN_16Feature_8Frame
```

### NAS 支持的任务

`generic_timeseries_classification` · `arc_fault` · `ecg_classification`
`motor_fault` · `blower_imbalance` · `pir_detection` · `image_classification`

---

## 可用任务类型

**时序：**
`generic_timeseries_classification` · `generic_timeseries_regression`
`generic_timeseries_anomalydetection` · `generic_timeseries_forecasting`
`arc_fault` · `motor_fault` · `blower_imbalance` · `pir_detection`

**视觉：**
`image_classification`

## 可用目标设备

`F280013` `F280015` `F28003` `F28004` `F2837` `F28P55` `F28P65` `F29H85`
`MSPM0G3507` `MSPM0G5187` `CC2755` `AM263`

## 示例模型名称（时序）

**分类：** `CLS_100_NPU` `CLS_500_NPU` `CLS_1k_NPU` `CLS_2k_NPU`
`CLS_4k_NPU` `CLS_6k_NPU` `CLS_8k_NPU` `CLS_13k_NPU` `CLS_20k_NPU`
`CLS_55k_NPU` `CLS_ResAdd_3k` `CLS_ResCat_3k`

**回归：** `REGR_1k` `REGR_2k` `REGR_3k` `REGR_4k` `REGR_10k` `REGR_13k`
`REGR_500_NPU` `REGR_2k_NPU` `REGR_6k_NPU` `REGR_8k_NPU` `REGR_20k_NPU`

**异常检测：** `AD_1k` `AD_4k` `AD_16k` `AD_17k` `AD_Linear`
`AD_500_NPU` `AD_2k_NPU` `AD_6k_NPU` `AD_8k_NPU` `AD_10k_NPU` `AD_20k_NPU`

**预测：** `FCST_3k` `FCST_13k` `FCST_LSTM8` `FCST_LSTM10`
`FCST_500_NPU` `FCST_1k_NPU` `FCST_2k_NPU` `FCST_4k_NPU` `FCST_6k_NPU`
`FCST_8k_NPU` `FCST_10k_NPU` `FCST_20k_NPU`

**特定应用：** `ArcFault_model_200_t` `ArcFault_model_300_t`
`ArcFault_model_700_t` `ArcFault_model_1400_t`
`MotorFault_model_1_t` `MotorFault_model_2_t` `MotorFault_model_3_t`
`FanImbalance_model_1_t` `FanImbalance_model_2_t` `FanImbalance_model_3_t`
`ECG_55k_NPU` `PIRDetection_model_1_t`

> **提示：** 运行 `mmcli info -m timeseries -t <task>` 可查看特定任务的可用模型。

---

## 编译二进制文件

```bash
bash build_macos.sh              # arm64（Apple Silicon，默认）
ARCH=x86_64 bash build_macos.sh  # Intel Mac
ARCH=universal2 bash build_macos.sh  # 通用二进制（两者皆可）
# 输出：dist/mmcli
```

将 `dist/mmcli` 复制到 `PATH` 中的任何位置。**运行**二进制文件不需要 Python 环境
——只需 `MMCLI_PYTHON` 指向安装了 `tinyml_modelmaker` 的 Python 3.10。

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MMCLI_PYTHON` | PATH 中的 `python` 或 `python3` | 安装了 `tinyml_modelmaker` 的 Python 解释器 |
| `MMCLI_MODELMAKER` | 自动检测 | tinyml-modelmaker 源代码目录路径（仅在自动检测失败时需要） |
| `MMCLI_DATASETS` | 内置 `example_datasets/` | 覆盖包含示例数据集 zip 文件的目录 |
