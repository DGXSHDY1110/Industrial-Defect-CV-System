# Cline 任务：实现 MVTec AD 子集到 YOLOv8 检测格式的数据转换

## 0. 当前背景

项目名称：`Industrial-Defect-CV-System`

重要目录约束：请严格遵守初始项目结构，处理好的 YOLO 数据集必须直接放在：

```text
data/processed/
```

不要再额外创建：

```text
data/processed/processed/
```

也就是说，最终应是：

```text
data/processed/images/
data/processed/labels/
data/processed/dataset.yaml
```

而不是：

```text
data/processed/processed/images/
data/processed/processed/labels/
data/processed/processed/dataset.yaml
```

目标：你需要在当前项目中实现两个脚本：

```text
scripts/prepare_mvtec.py
scripts/mask_to_yolo_bbox.py
```

并将已经下载并校验通过的 MVTec AD 三类数据：

```text
bottle
capsule
metal_nut
```

从原始 MVTec AD 目录转换为 YOLOv8 可直接训练的数据集格式。

当前原始数据目录已经通过校验，路径为：

```text
/root/autodl-tmp/Industrial-Defect-CV-System/data/raw/mvtec_ad
```

原始目录中应包含：

```text
data/raw/mvtec_ad/
├── bottle/
├── capsule/
└── metal_nut/
```

每个类别内部是标准 MVTec AD 结构：

```text
category/
├── train/
│   └── good/
├── test/
│   ├── good/
│   ├── broken_large/
│   ├── broken_small/
│   └── ...
└── ground_truth/
    ├── broken_large/
    ├── broken_small/
    └── ...
```

注意：不同类别的缺陷子目录名称不完全相同，代码不能写死缺陷类型名称。

---

## 1. 最终目标

实现后，运行：

```bash
python scripts/prepare_mvtec.py \
  --raw_dir data/raw/mvtec_ad \
  --out_dir data/processed \
  --categories bottle capsule metal_nut \
  --val_ratio 0.2 \
  --seed 42 \
  --min_area 10 \
  --single_class
```

应生成 YOLOv8 标准数据集：

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
├── visualizations/
│   ├── train/
│   ├── val/
│   └── test/
├── dataset.yaml
├── split_summary.json
└── conversion_summary.json
```

其中：

```text
images/train/*.png
images/val/*.png
images/test/*.png

labels/train/*.txt
labels/val/*.txt
labels/test/*.txt
```

每张图片必须对应一个同名 `.txt` label 文件。

正常图片的 label 文件可以为空文件。

缺陷图片的 label 文件写入 YOLO bbox：

```text
class_id x_center y_center width height
```

所有坐标必须归一化到 `[0, 1]`。

---

## 2. 类别策略

本 MVP 阶段使用 `single_class` 策略。

也就是说，不区分 bottle / capsule / metal_nut，也不区分具体 defect type，所有异常统一标记为：

```text
0 defect
```

生成的 `dataset.yaml` 应为：

```yaml
path: data/processed
train: images/train
val: images/val
test: images/test

names:
  0: defect
```

原因：MVTec AD 原始任务是 anomaly detection，不是标准 detection。当前项目目标是在 5 天内做出可演示的 YOLOv8 工业缺陷检测 MVP，所以第一版先做 binary defect detection：只回答“缺陷在哪里”。

后续可以再扩展为多类别：

```text
bottle_broken_large
capsule_crack
metal_nut_scratch
...
```

但本任务不要做多类别，避免复杂化。

---

## 3. 数据划分规则

MVTec AD 原始结构中：

```text
train/good/
```

只有正常图。

```text
test/good/
```

是测试正常图。

```text
test/<defect_type>/
```

是测试缺陷图。

```text
ground_truth/<defect_type>/
```

包含对应缺陷 mask。

因为原始训练集没有缺陷图，所以为了训练 YOLO，需要从原始 `test/<defect_type>` 缺陷图中切分出一部分作为 train / val / test。

请按以下规则处理：

### 3.1 正常图

- `train/good` 中的正常图：
  - 按 `val_ratio` 从中随机划分出一部分到 `val`
  - 剩余放入 `train`
- `test/good` 中的正常图：
  - 全部放入 `test`

### 3.2 缺陷图

对每个类别、每个 defect type：

- 读取 `test/<defect_type>/*.png`
- 找到对应 mask：`ground_truth/<defect_type>/*_mask.png`
- 按比例切分：
  - 约 70% → train
  - 约 20% → val
  - 约 10% → test

可以使用简单策略：

```python
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
```

但需要保证每个 defect type 至少有样本进入 test；如果样本数太少，可以采用稳健逻辑，避免某个 split 为空。

### 3.3 文件命名

为避免不同类别、不同 defect type 下文件名冲突，复制到 processed 目录时统一重命名：

```text
{category}__{split_source}__{defect_type}__{original_stem}.png
```

示例：

```text
bottle__test__broken_large__000.png
capsule__test__crack__012.png
metal_nut__train__good__045.png
```

对应 label：

```text
bottle__test__broken_large__000.txt
capsule__test__crack__012.txt
metal_nut__train__good__045.txt
```

---

## 4. mask_to_yolo_bbox.py 要求

你需要实现独立脚本：

```text
scripts/mask_to_yolo_bbox.py
```

它负责将 MVTec 的 mask 图转换成 YOLO bbox label。

### 4.1 CLI 参数

支持如下参数：

```bash
python scripts/mask_to_yolo_bbox.py \
  --image_path path/to/image.png \
  --mask_path path/to/mask.png \
  --label_path path/to/output.txt \
  --vis_path path/to/output_vis.png \
  --class_id 0 \
  --min_area 10
```

参数说明：

```text
--image_path    原始图像路径，必填
--mask_path     mask 路径，缺陷图必填；正常图可不传
--label_path    输出 YOLO label txt 路径，必填
--vis_path      可视化输出路径，可选
--class_id      默认 0
--min_area      过滤过小连通域，默认 10
```

### 4.2 函数设计

请至少实现以下函数，方便 prepare_mvtec.py 复用：

```python
def mask_to_bboxes(mask_path: str | Path, min_area: int = 10) -> list[tuple[int, int, int, int]]:
    """
    输入 mask 路径，输出 bbox 列表。
    bbox 格式为 (x_min, y_min, x_max, y_max)，像素坐标。
    需要支持一个 mask 中存在多个缺陷连通域。
    """

def bbox_to_yolo(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    将像素 bbox 转为 YOLO 格式：
    x_center, y_center, width, height
    均归一化到 [0, 1]。
    """

def write_yolo_label(
    label_path: str | Path,
    bboxes: list[tuple[int, int, int, int]],
    image_width: int,
    image_height: int,
    class_id: int = 0,
) -> None:
    """
    写入 YOLO label 文件。
    如果 bboxes 为空，也要创建空 txt 文件。
    """

def draw_bboxes(
    image_path: str | Path,
    bboxes: list[tuple[int, int, int, int]],
    output_path: str | Path,
) -> None:
    """
    保存 bbox 可视化结果，用于人工检查转换是否正确。
    """
```

### 4.3 mask 处理细节

mask 读取方式：

- 用 OpenCV 或 PIL 读取灰度图。
- 所有像素值 > 0 的区域视为 defect。
- 使用 connected components 或 contour 找出多个缺陷区域。
- 过滤面积小于 `min_area` 的区域。
- bbox 坐标必须 clamp 到图像边界内。
- 若 mask 文件不存在，必须抛出清晰错误。
- 若 mask 为空，仍然生成空 label 文件，并在 summary 中记录。

建议使用 OpenCV：

```python
cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
cv2.threshold(...)
cv2.findContours(...)
cv2.boundingRect(...)
```

也可以用：

```python
cv2.connectedComponentsWithStats(...)
```

优先推荐 connected components，因为可以更直接按面积过滤。

### 4.4 YOLO 坐标检查

每个 YOLO bbox 必须满足：

```text
0 <= x_center <= 1
0 <= y_center <= 1
0 < width <= 1
0 < height <= 1
```

如果不满足，应该跳过该 bbox 并打印 warning。

---

## 5. prepare_mvtec.py 要求

你需要实现主转换脚本：

```text
scripts/prepare_mvtec.py
```

它负责完整处理 3 类 MVTec AD 数据，并生成 YOLOv8 数据集。

### 5.1 CLI 参数

支持：

```bash
python scripts/prepare_mvtec.py \
  --raw_dir data/raw/mvtec_ad \
  --out_dir data/processed \
  --categories bottle capsule metal_nut \
  --val_ratio 0.2 \
  --seed 42 \
  --min_area 10 \
  --single_class \
  --vis_samples 50
```

参数说明：

```text
--raw_dir       MVTec AD 原始目录
--out_dir       YOLO 格式输出目录
--categories    类别列表，默认 bottle capsule metal_nut
--val_ratio     正常 train/good 中划分到 val 的比例
--seed          随机种子
--min_area      mask 连通域最小面积
--single_class  是否统一为 defect 类，当前必须支持
--vis_samples   每个 split 最多生成多少张可视化结果
--overwrite     如传入则清空 out_dir 后重新生成
```

建议 `--overwrite` 默认为 False。如果 out_dir 已存在且非空，并且没有传 `--overwrite`，应提示用户加 `--overwrite` 或换目录，避免误删数据。

### 5.2 输出目录创建

脚本应创建：

```text
out_dir/images/train
out_dir/images/val
out_dir/images/test
out_dir/labels/train
out_dir/labels/val
out_dir/labels/test
out_dir/visualizations/train
out_dir/visualizations/val
out_dir/visualizations/test
```

### 5.3 数据处理流程

伪代码：

```python
def main():
    parse_args()
    set_seed(args.seed)
    validate_raw_dataset(args.raw_dir, args.categories)
    create_output_dirs(args.out_dir)

    all_records = []

    for category in categories:
        records = collect_mvtec_records(category_dir)

        normal_train_records = records from train/good
        normal_test_records = records from test/good
        defect_records = records from test/<defect_type> with masks

        split normal_train into train/val by args.val_ratio
        put normal_test into test

        split defect_records into train/val/test by 0.7/0.2/0.1

        for each record:
            copy image to out_dir/images/{split}/new_name.png
            if normal:
                create empty label txt
            else:
                convert mask to bbox label using mask_to_yolo_bbox functions
            optionally draw visualization
            append conversion info to summary

    write dataset.yaml
    write split_summary.json
    write conversion_summary.json
    run sanity checks
    print final report
```

### 5.4 Record 数据结构

建议用 dataclass：

```python
@dataclass
class MVTecRecord:
    category: str
    defect_type: str
    image_path: Path
    mask_path: Path | None
    source_split: str
    target_split: str | None = None
    is_defect: bool = False
```

### 5.5 mask 文件匹配规则

MVTec AD 常见命名：

```text
test/<defect_type>/000.png
ground_truth/<defect_type>/000_mask.png
```

匹配逻辑：

```python
mask_path = category_dir / "ground_truth" / defect_type / f"{image_stem}_mask.png"
```

但请写得稳健一些：

1. 首先尝试 `{stem}_mask.png`
2. 如果不存在，再尝试 glob：`{stem}*`
3. 如果还不存在，记录错误并跳过该缺陷图，最后 summary 里列出 missing masks

### 5.6 dataset.yaml

生成：

```yaml
path: /root/autodl-tmp/Industrial-Defect-CV-System/data/processed
train: images/train
val: images/val
test: images/test

names:
  0: defect
```

注意：

- `path` 建议写绝对路径，避免 YOLO 在不同工作目录下找不到数据。
- 同时可以在 README 中使用相对路径说明。

### 5.7 summary 文件

生成 `split_summary.json`，示例结构：

```json
{
  "categories": ["bottle", "capsule", "metal_nut"],
  "splits": {
    "train": {
      "images": 0,
      "normal": 0,
      "defect": 0,
      "labels": 0
    },
    "val": {
      "images": 0,
      "normal": 0,
      "defect": 0,
      "labels": 0
    },
    "test": {
      "images": 0,
      "normal": 0,
      "defect": 0,
      "labels": 0
    }
  },
  "by_category": {
    "bottle": {
      "train": 0,
      "val": 0,
      "test": 0
    }
  }
}
```

生成 `conversion_summary.json`，示例结构：

```json
{
  "total_images": 0,
  "total_labels": 0,
  "total_bboxes": 0,
  "empty_label_files": 0,
  "missing_masks": [],
  "invalid_bboxes": [],
  "records": [
    {
      "category": "bottle",
      "defect_type": "broken_large",
      "source_image": "...",
      "source_mask": "...",
      "target_image": "...",
      "target_label": "...",
      "target_split": "train",
      "is_defect": true,
      "num_bboxes": 1
    }
  ]
}
```

---

## 6. 必须实现的数据集校验

转换完成后，脚本必须自动执行 sanity check。

### 6.1 图片与 label 数量一致

检查：

```text
len(images/train) == len(labels/train)
len(images/val) == len(labels/val)
len(images/test) == len(labels/test)
```

### 6.2 每张图片都有同名 label

例如：

```text
images/train/abc.png
labels/train/abc.txt
```

必须存在。

### 6.3 label 格式合法

检查每一行：

```text
class_id x_center y_center width height
```

要求：

```text
class_id == 0
0 <= x_center <= 1
0 <= y_center <= 1
0 < width <= 1
0 < height <= 1
```

### 6.4 dataset.yaml 可被 YOLO 使用

脚本结尾打印：

```bash
yolo detect train data=data/processed/dataset.yaml model=yolov8n.pt imgsz=768 epochs=50 batch=8
```

---

## 7. 可视化要求

转换完成后，请在：

```text
data/processed/visualizations/
```

下保存部分 bbox 可视化图。

要求：

- 每个 split 最多保存 `--vis_samples` 张。
- 缺陷图画红色 bbox。
- 正常图可以保存原图并标注 `normal`。
- 文件名与原图一致。

用于人工检查：

```bash
ls data/processed/visualizations/train | head
```

---

## 8. 代码质量要求

请遵守：

1. 使用 `pathlib.Path`，不要大量拼接字符串路径。
2. 所有脚本支持从项目根目录运行。
3. 加入清晰的日志输出。
4. 对缺失目录、缺失 mask、空 mask、非法 bbox 给出清晰 warning。
5. 所有随机划分必须由 seed 控制，保证可复现。
6. 脚本重复运行时不要静默覆盖，除非传入 `--overwrite`。
7. 函数尽量可测试，不要把所有逻辑堆在 main 里。
8. 中文注释可以有，但核心函数 docstring 建议用英文，方便 GitHub 展示。
9. 不要把原始 MVTec 数据复制进 GitHub，只保留 `.gitkeep`。
10. 不要提交大文件、模型权重、原始图片。

---

## 9. 需要同时创建或更新的文件

请创建：

```text
scripts/prepare_mvtec.py
scripts/mask_to_yolo_bbox.py
configs/dataset/yolo_mvtec.yaml
```

请确保 `.gitignore` 至少包含：

```gitignore
data/raw/
data/processed/
outputs/
runs/
*.pt
*.pth
*.onnx
__pycache__/
.ipynb_checkpoints/
```

如果目录不存在，请创建：

```text
scripts/
configs/dataset/
data/processed/
```

---

## 10. 运行命令

请在项目根目录运行：

```bash
cd /root/autodl-tmp/Industrial-Defect-CV-System
```

如果第一次运行，建议：

```bash
python scripts/prepare_mvtec.py \
  --raw_dir data/raw/mvtec_ad \
  --out_dir data/processed \
  --categories bottle capsule metal_nut \
  --val_ratio 0.2 \
  --seed 42 \
  --min_area 10 \
  --single_class \
  --vis_samples 50 \
  --overwrite
```

然后检查：

```bash
tree -L 3 data/processed
```

检查 summary：

```bash
cat data/processed/split_summary.json
cat data/processed/conversion_summary.json
```

检查 label：

```bash
find data/processed/labels -name "*.txt" | head
```

检查可视化：

```bash
find data/processed/visualizations -type f | head
```

---

## 11. YOLOv8 训练命令

转换完成后，应能直接运行：

```bash
yolo detect train \
  data=data/processed/dataset.yaml \
  model=yolov8n.pt \
  imgsz=768 \
  epochs=50 \
  batch=8 \
  workers=4 \
  project=outputs/checkpoints/yolo \
  name=processedv8n
```

也可以用 Python 脚本后续封装训练，但本任务只要求数据转换脚本可用。

---

## 12. 验收标准

任务完成后必须满足：

```text
[ ] scripts/prepare_mvtec.py 存在并可运行
[ ] scripts/mask_to_yolo_bbox.py 存在并可单独运行
[ ] data/processed/images/train 存在
[ ] data/processed/images/val 存在
[ ] data/processed/images/test 存在
[ ] data/processed/labels/train 存在
[ ] data/processed/labels/val 存在
[ ] data/processed/labels/test 存在
[ ] 每张图片都有同名 label
[ ] 正常图 label 为空文件
[ ] 缺陷图 label 至少包含一个 bbox，除非 mask 为空
[ ] 所有 YOLO bbox 坐标均在合法范围内
[ ] dataset.yaml 可直接用于 YOLOv8 训练
[ ] split_summary.json 存在
[ ] conversion_summary.json 存在
[ ] visualizations 目录下有 bbox 可视化图片
```

---

## 13. 最终输出给用户的内容

完成后，请在终端输出类似：

```text
[MVTec → YOLO Conversion Done]

Raw dir:
  data/raw/mvtec_ad

Output dir:
  data/processed

Categories:
  bottle, capsule, metal_nut

Split summary:
  train: xxx images, xxx defect, xxx normal
  val  : xxx images, xxx defect, xxx normal
  test : xxx images, xxx defect, xxx normal

Labels:
  total label files: xxx
  total bboxes     : xxx
  empty labels     : xxx

Missing masks:
  0

Dataset yaml:
  data/processed/dataset.yaml

Next command:
  yolo detect train data=data/processed/dataset.yaml model=yolov8n.pt imgsz=768 epochs=50 batch=8
```

---

## 14. 重要提醒

当前任务只做数据转换，不要开始写 YOLO 训练脚本。

本次任务完成后，下一步才是：

```text
scripts/train_yolo.py
scripts/eval_yolo.py
outputs/checkpoints/yolo
```

请确保本次转换脚本稳定、可重复、可校验，因为后续所有训练和 Demo 都依赖这个数据格式。
