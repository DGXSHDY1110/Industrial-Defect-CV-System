# Cline Task: Day2 - Run YOLOv8n Training for Industrial-Defect-CV-System

## 0. 当前任务背景

项目路径：

```bash
/root/autodl-tmp/Industrial-Defect-CV-System
```

Conda 环境路径：

```bash
/root/miniconda3/envs/defect-cv
```

Day1 已经完成：

* MVTec 3 类原始数据校验
* 数据预处理
* mask 转 YOLO bbox
* YOLO 格式 labels 生成
* 处理后的数据集已经放在项目目录的 `data/processed/` 中

本次 Day2 任务目标：

1. 跑通 `yolov8n` 训练。
2. 保存训练日志、PR 曲线、confusion matrix 到项目目录合适位置。
3. 记录 `mAP50`、`Precision`、`Recall`。
4. 整理 10 张成功案例 + 5 张失败案例。

重要要求：

* 不要重新下载数据集。
* 不要把处理好的数据移动到 `data/processed/mvtec_yolo/`。
* 必须使用现有的 `data/processed/`。
* 必须在 `/root/miniconda3/envs/defect-cv` 环境中执行。
* 所有输出必须保存在项目目录内，不能散落到默认 `runs/` 目录外。
* 如果使用 Ultralytics 默认生成的 `runs/`，必须自动复制或移动到项目规范目录。

---

## 1. 先检查项目状态

进入项目：

```bash
cd /root/autodl-tmp/Industrial-Defect-CV-System
```

激活 conda 环境：

```bash
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate /root/miniconda3/envs/defect-cv
```

检查 Python：

```bash
which python
python -V
```

期望：

```bash
/root/miniconda3/envs/defect-cv/bin/python
```

检查 CUDA：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

如果 `torch` 未安装或 CUDA 不可用，不要盲目重装整个环境，先记录问题，再根据当前 CUDA 版本安装匹配的 PyTorch。

---

## 2. 安装或补齐 Day2 依赖

检查依赖：

```bash
python - <<'PY'
mods = ["ultralytics", "cv2", "yaml", "pandas", "matplotlib", "numpy"]
for m in mods:
    try:
        __import__(m)
        print(f"[OK] {m}")
    except Exception as e:
        print(f"[MISSING] {m}: {e}")
PY
```

如果缺少依赖，在当前 conda 环境中安装：

```bash
pip install ultralytics opencv-python pyyaml pandas matplotlib tqdm scikit-learn
```

如果项目已有 `requirements.txt`，可以补充但不要破坏已有内容。

建议追加依赖：

```txt
ultralytics
opencv-python
pyyaml
pandas
matplotlib
tqdm
scikit-learn
```

---

## 3. 检查 YOLO 数据集结构

检查：

```bash
tree -L 4 data/processed | head -100
```

期望至少存在类似结构之一：

```text
data/processed/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

或者：

```text
data/processed/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

如果 `dataset.yaml` 不存在，需要创建：

```bash
data/processed/dataset.yaml
```

推荐内容：

```yaml
path: /root/autodl-tmp/Industrial-Defect-CV-System/data/processed
train: images/train
val: images/val
test: images/test

names:
  0: defect
```

如果现有数据集是多类别，比如：

```yaml
names:
  0: bottle_defect
  1: capsule_defect
  2: metal_nut_defect
```

则保留现有类别，不要强行改成单类。

要求：

* 训练脚本必须自动读取 `data/processed/dataset.yaml`。
* 如果 `dataset.yaml` 中路径错误，修正为项目内绝对路径。
* 不要改变图片和 label 的实际存放位置。

---

## 4. 新增或修改 YOLOv8n 训练脚本

请创建或完善：

```bash
scripts/train_yolo.py
```

功能要求：

1. 使用 Ultralytics YOLO。
2. 默认模型：`yolov8n.pt`。
3. 默认数据集配置：`data/processed/dataset.yaml`。
4. 默认输出目录：`outputs/checkpoints/yolo/yolov8n_day2`。
5. 保存训练日志。
6. 训练结束后自动验证 best.pt。
7. 自动提取并保存：

   * mAP50
   * Precision
   * Recall
   * mAP50-95
8. 自动复制关键图表：

   * `PR_curve.png`
   * `confusion_matrix.png`
   * `results.png`
   * `F1_curve.png`
   * `P_curve.png`
   * `R_curve.png`
   * `results.csv`
9. 生成：

   * `outputs/reports/day2_yolo/metrics.json`
   * `outputs/reports/day2_yolo/metrics.md`

训练默认参数：

```yaml
model: yolov8n.pt
imgsz: 768
epochs: 50
batch: 8
patience: 15
optimizer: AdamW
lr0: 0.001
weight_decay: 0.0005
cos_lr: true
seed: 42
device: 0
workers: 4
```

如果显存不足，自动降级：

```yaml
imgsz: 640
batch: 4
```

训练命令应支持：

```bash
python scripts/train_yolo.py \
  --data data/processed/dataset.yaml \
  --model yolov8n.pt \
  --imgsz 768 \
  --epochs 50 \
  --batch 8 \
  --device 0
```

脚本中建议使用的输出路径：

```text
outputs/checkpoints/yolo/yolov8n_day2/
outputs/logs/yolo/yolov8n_day2/
outputs/reports/day2_yolo/
```

---

## 5. 先跑 smoke test

正式训练前，先跑 1 epoch，避免浪费时间。

命令：

```bash
python scripts/train_yolo.py \
  --data data/processed/dataset.yaml \
  --model yolov8n.pt \
  --imgsz 640 \
  --epochs 1 \
  --batch 4 \
  --device 0 \
  --run_name smoke_test
```

smoke test 验收：

```text
1. 能正常加载 dataset.yaml
2. 能正常读取 images 和 labels
3. 能正常开始训练
4. 能生成 weights
5. 能生成 results.csv
```

如果 smoke test 失败，优先排查：

1. `dataset.yaml` 路径是否正确。
2. 图片路径和 labels 路径是否匹配。
3. label 格式是否为 YOLO 格式：

   ```text
   class_id x_center y_center width height
   ```
4. 坐标是否归一化到 0-1。
5. 是否存在空 label 文件。注意：正常图允许空 label，但路径必须存在。
6. 类别 id 是否超过 `names` 定义范围。

---

## 6. 正式训练 YOLOv8n

smoke test 通过后，执行正式训练：

```bash
python scripts/train_yolo.py \
  --data data/processed/dataset.yaml \
  --model yolov8n.pt \
  --imgsz 768 \
  --epochs 50 \
  --batch 8 \
  --device 0 \
  --run_name yolov8n_mvtec_3cls_day2
```

如果显存不足，改用：

```bash
python scripts/train_yolo.py \
  --data data/processed/dataset.yaml \
  --model yolov8n.pt \
  --imgsz 640 \
  --epochs 50 \
  --batch 4 \
  --device 0 \
  --run_name yolov8n_mvtec_3cls_day2
```

训练完成后，期望产生：

```text
outputs/checkpoints/yolo/yolov8n_day2/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.csv
├── results.png
├── PR_curve.png
├── confusion_matrix.png
├── F1_curve.png
├── P_curve.png
└── R_curve.png
```

同时复制关键文件到：

```text
outputs/reports/day2_yolo/
├── metrics.json
├── metrics.md
├── results.csv
├── results.png
├── PR_curve.png
├── confusion_matrix.png
├── F1_curve.png
├── P_curve.png
└── R_curve.png
```

---

## 7. 指标记录格式

请生成：

```bash
outputs/reports/day2_yolo/metrics.json
```

格式如下：

```json
{
  "model": "yolov8n",
  "dataset": "MVTec AD subset",
  "categories": ["bottle", "capsule", "metal_nut"],
  "image_size": 768,
  "epochs": 50,
  "batch_size": 8,
  "metrics": {
    "precision": 0.0,
    "recall": 0.0,
    "map50": 0.0,
    "map50_95": 0.0
  },
  "artifacts": {
    "best_weights": "outputs/checkpoints/yolo/yolov8n_day2/weights/best.pt",
    "pr_curve": "outputs/reports/day2_yolo/PR_curve.png",
    "confusion_matrix": "outputs/reports/day2_yolo/confusion_matrix.png",
    "results_csv": "outputs/reports/day2_yolo/results.csv"
  }
}
```

请同时生成：

```bash
outputs/reports/day2_yolo/metrics.md
```

内容模板：

```md
# Day2 YOLOv8n Training Report

## Model

- Model: YOLOv8n
- Dataset: MVTec AD subset
- Image size: 768
- Epochs: 50
- Batch size: 8

## Metrics

| Metric | Value |
|---|---:|
| Precision | TBD |
| Recall | TBD |
| mAP50 | TBD |
| mAP50-95 | TBD |

## Artifacts

- PR Curve: `outputs/reports/day2_yolo/PR_curve.png`
- Confusion Matrix: `outputs/reports/day2_yolo/confusion_matrix.png`
- Results CSV: `outputs/reports/day2_yolo/results.csv`
- Best Weights: `outputs/checkpoints/yolo/yolov8n_day2/weights/best.pt`
```

要求：不要留 `TBD`，要填真实训练结果。

---

## 8. 整理 10 张成功案例 + 5 张失败案例

请创建脚本：

```bash
scripts/analyze_yolo_cases.py
```

功能：

1. 加载：

   ```text
   outputs/checkpoints/yolo/yolov8n_day2/weights/best.pt
   ```
2. 在 val 或 test 集上推理。
3. 读取对应 YOLO label。
4. 计算预测框与 GT 框 IoU。
5. 按规则整理成功和失败案例。
6. 输出可视化图片和 CSV 说明。

输出目录：

```text
outputs/reports/day2_yolo/cases/
├── success/
│   ├── success_001.jpg
│   ├── success_002.jpg
│   └── ...
├── failure/
│   ├── failure_001.jpg
│   ├── failure_002.jpg
│   └── ...
└── case_summary.csv
```

可视化要求：

* 原图上画 GT bbox 和 Pred bbox。
* GT bbox 标注为 `GT`。
* Pred bbox 标注为 `Pred: conf=xx`。
* 图片左上角写：

  * case type
  * max IoU
  * confidence
  * failure reason

成功案例定义：

```text
有 GT defect，且至少一个预测框与 GT 的 IoU >= 0.5，confidence >= 0.25
```

失败案例定义，按优先级：

```text
1. False Negative：有 GT，但没有预测框
2. Low IoU：有预测框，但 max IoU < 0.5
3. Low Confidence：IoU 合格但 confidence < 0.25
4. False Positive：无 GT，但有预测框
```

如果真实失败案例不足 5 张：

* 不要伪造失败案例。
* 选择 IoU 最低或 confidence 最低的 borderline cases。
* 在 `case_summary.csv` 中标记为：

  ```text
  borderline_case
  ```

CSV 字段：

```csv
case_id,image_path,case_type,reason,gt_count,pred_count,max_iou,max_conf,output_image
```

运行命令：

```bash
python scripts/analyze_yolo_cases.py \
  --weights outputs/checkpoints/yolo/yolov8n_day2/weights/best.pt \
  --data data/processed/dataset.yaml \
  --split val \
  --conf 0.25 \
  --iou 0.5 \
  --num_success 10 \
  --num_failure 5 \
  --out_dir outputs/reports/day2_yolo/cases
```

---

## 9. 更新 README 的 Day2 结果区域

请在 `README.md` 中新增或更新以下部分：

````md
## YOLOv8n Detection Baseline

| Model | Dataset | Image Size | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|
| YOLOv8n | MVTec AD subset | 768 | value | value | value | value |

### Training Artifacts

- PR Curve: `outputs/reports/day2_yolo/PR_curve.png`
- Confusion Matrix: `outputs/reports/day2_yolo/confusion_matrix.png`
- Training Log: `outputs/reports/day2_yolo/results.csv`
- Best Weights: `outputs/checkpoints/yolo/yolov8n_day2/weights/best.pt`

### Qualitative Results

Success cases:

```text
outputs/reports/day2_yolo/cases/success/
````

Failure cases:

```text
outputs/reports/day2_yolo/cases/failure/
```

````

要求：

- 表格必须填入真实数值。
- 不要写虚假指标。
- 如果训练结果很差，也照实记录，并在备注中写原因，例如数据少、bbox 来自 mask 转换、缺陷尺度小等。

---

## 10. 最终验收清单

Day2 完成后，请确保以下文件存在：

```text
scripts/train_yolo.py
scripts/analyze_yolo_cases.py

outputs/checkpoints/yolo/yolov8n_day2/weights/best.pt
outputs/checkpoints/yolo/yolov8n_day2/weights/last.pt

outputs/reports/day2_yolo/metrics.json
outputs/reports/day2_yolo/metrics.md
outputs/reports/day2_yolo/results.csv
outputs/reports/day2_yolo/PR_curve.png
outputs/reports/day2_yolo/confusion_matrix.png

outputs/reports/day2_yolo/cases/success/
outputs/reports/day2_yolo/cases/failure/
outputs/reports/day2_yolo/cases/case_summary.csv
````

检查命令：

```bash
find outputs/reports/day2_yolo -maxdepth 3 -type f | sort
```

成功案例数量检查：

```bash
ls outputs/reports/day2_yolo/cases/success | wc -l
ls outputs/reports/day2_yolo/cases/failure | wc -l
```

指标检查：

```bash
cat outputs/reports/day2_yolo/metrics.md
```

---

## 11. Git 提交建议

完成后执行：

```bash
git status
git add scripts/train_yolo.py scripts/analyze_yolo_cases.py README.md outputs/reports/day2_yolo
git commit -m "feat: add YOLOv8n training baseline and case analysis"
```

注意：

* 不要提交大模型权重到 GitHub，除非仓库已经配置 Git LFS。
* `best.pt` 和 `last.pt` 建议加入 `.gitignore`。
* 可以提交轻量报告、曲线图、案例图和 metrics 文件。
* 如果需要保留权重，后续用 GitHub Release 或网盘链接。

建议 `.gitignore` 包含：

```gitignore
outputs/checkpoints/
*.pt
*.onnx
*.engine
data/raw/
```

但保留：

```text
outputs/reports/day2_yolo/
```

用于面试展示。

---

## 12. Cline 执行完成后请输出总结

请在任务完成后输出：

```md
# Day2 Summary

## Environment
- Python:
- Torch:
- CUDA:
- GPU:

## Dataset
- Dataset yaml:
- Train images:
- Val images:
- Test images:
- Classes:

## Training
- Model:
- Epochs:
- Image size:
- Batch size:
- Best weights:

## Metrics
- Precision:
- Recall:
- mAP50:
- mAP50-95:

## Artifacts
- PR curve:
- Confusion matrix:
- Results CSV:
- Metrics report:
- Success cases:
- Failure cases:

## Issues Encountered
- issue 1:
- issue 2:

## Next Step
Proceed to Day3: ResNet classification baseline + Grad-CAM explainability.
```
