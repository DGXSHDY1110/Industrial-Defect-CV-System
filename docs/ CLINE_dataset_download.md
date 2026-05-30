# Cline Task: 下载并整理 MVTec AD 三类数据集到项目正确目录

## 任务背景

当前项目仓库 `Industrial-Defect-CV-System` 已经完成目录结构搭建。现在需要下载 MVTec AD 数据集中以下 3 个类别：

```text
bottle
capsule
metal_nut
```

这 3 类数据将作为项目 MVP 的原始数据，用于后续：

```text
MVTec mask → YOLO bbox 转换
YOLOv8 缺陷检测训练
ResNet 分类 baseline
Grad-CAM 可解释性
ONNX 导出与推理延迟统计
Streamlit Demo
```

请只完成本任务中要求的数据下载、解压、目录整理、校验和 `.gitignore` 更新。不要开始训练模型，不要改动训练逻辑。

---

## 一、目标目录结构

请将数据最终整理为以下结构：

```text
Industrial-Defect-CV-System/
└── data/
    └── raw/
        ├── downloads/
        │   ├── bottle.tar.xz
        │   ├── capsule.tar.xz
        │   └── metal_nut.tar.xz
        └── mvtec_ad/
            ├── bottle/
            │   ├── train/
            │   ├── test/
            │   └── ground_truth/
            ├── capsule/
            │   ├── train/
            │   ├── test/
            │   └── ground_truth/
            └── metal_nut/
                ├── train/
                ├── test/
                └── ground_truth/
```

最终训练和数据转换脚本默认从这里读取：

```text
data/raw/mvtec_ad/
```

---

## 二、下载链接

请使用以下 3 个链接下载数据：

```text
bottle:
https://www.mydrive.ch/shares/150452/132a93367fb17cdf968dfb5c4013f6e7/download/420937370-1629958698/bottle.tar.xz

capsule:
https://www.mydrive.ch/shares/150454/e0ce6dd74eb150f46c0d98131b3703f2/download/420937454-1629958872/capsule.tar.xz

metal_nut:
https://www.mydrive.ch/shares/150459/c68856a21dca589b0f8ff6d4ee0f18f4/download/420937637-1629959294/metal_nut.tar.xz
```

下载时请支持：

* 自动创建目录
* 断点续传
* 自动重试
* 下载失败时给出清晰错误信息
* 解压前检查文件是否存在
* 解压后检查目录是否符合预期

---

## 三、请创建下载脚本

请新增脚本：

```text
scripts/download_mvtec_subset.py
```

脚本功能要求：

1. 从项目根目录运行。
2. 自动创建以下目录：

```text
data/raw/downloads/
data/raw/mvtec_ad/
```

3. 下载 3 个 `.tar.xz` 文件到：

```text
data/raw/downloads/
```

4. 解压到：

```text
data/raw/mvtec_ad/
```

5. 解压后最终必须存在：

```text
data/raw/mvtec_ad/bottle/
data/raw/mvtec_ad/capsule/
data/raw/mvtec_ad/metal_nut/
```

6. 每个类别目录下至少应包含：

```text
train/
test/
ground_truth/
```

7. 如果目标类别目录已存在，默认跳过解压，避免重复覆盖。
8. 支持参数 `--force`，用于强制重新下载和重新解压。
9. 支持参数 `--skip-download`，用于只执行解压和校验。
10. 下载完成后输出每个类别的文件数量统计。

建议运行方式：

```bash
python scripts/download_mvtec_subset.py
```

强制重新下载：

```bash
python scripts/download_mvtec_subset.py --force
```

只校验已有文件：

```bash
python scripts/download_mvtec_subset.py --skip-download
```

---

## 四、下载实现建议

优先使用 Python 标准库实现，避免额外依赖。

建议使用：

```python
urllib.request
tarfile
pathlib
argparse
shutil
time
```

如果你认为 `requests` 更稳，也可以使用 `requests`，但需要把依赖写入 `requirements.txt`。

下载要求：

* 显示当前下载类别
* 显示保存路径
* 支持重试，例如最多 3 次
* 遇到网络错误时不要静默失败
* 文件已经存在时可以跳过下载
* `--force` 时删除旧文件重新下载

解压要求：

* 使用 `tarfile.open(..., "r:xz")`
* 先解压到临时目录，例如：

```text
data/raw/_tmp_extract/
```

* 再把类别目录移动到：

```text
data/raw/mvtec_ad/
```

注意：有些 tar 包解压后可能直接得到：

```text
bottle/
```

也可能得到：

```text
./bottle/
```

脚本需要兼容这两种情况。

---

## 五、请更新 `.gitignore`

请确认 `.gitignore` 中包含以下内容，避免把大型数据集提交到 GitHub：

```gitignore
# Dataset files
data/raw/downloads/
data/raw/mvtec_ad/
data/raw/_tmp_extract/
data/processed/
outputs/

# Large model files
*.pt
*.pth
*.onnx
*.engine

# Python cache
__pycache__/
*.pyc
.ipynb_checkpoints/

# OS files
.DS_Store
```

注意：

* `data/raw/.gitkeep` 可以保留。
* 数据集本体不要提交到 GitHub。
* 后续 README 中只写下载方式和目录结构。

---

## 六、请新增数据说明文件

请新增：

```text
data/raw/README.md
```

内容包括：

````markdown
# Raw Dataset Directory

This directory stores raw datasets for the Industrial-Defect-CV-System project.

Expected MVTec AD subset structure:

```text
data/raw/mvtec_ad/
├── bottle/
├── capsule/
└── metal_nut/
````

The dataset files are not committed to GitHub.

To download the required subset, run:

```bash
python scripts/download_mvtec_subset.py
```

To force re-download:

```bash
python scripts/download_mvtec_subset.py --force
```

````

---

## 七、请新增数据集配置文件

请新增或更新：

```text
configs/dataset/mvtec_subset.yaml
````

内容如下：

```yaml
dataset_name: mvtec_ad_subset
raw_root: data/raw/mvtec_ad
processed_root: data/processed/mvtec_yolo

categories:
  - bottle
  - capsule
  - metal_nut

expected_structure:
  train_dir: train
  test_dir: test
  ground_truth_dir: ground_truth

task:
  detection: true
  classification: true
  explainability: true

notes:
  - MVTec AD is originally designed for anomaly detection.
  - This project converts pixel-level anomaly masks into YOLO-style bounding boxes for supervised defect detection demo.
  - The dataset itself should not be committed to GitHub.
```

---

## 八、请新增一个快速校验脚本

请新增：

```text
scripts/check_mvtec_dataset.py
```

功能：

1. 检查以下目录是否存在：

```text
data/raw/mvtec_ad/bottle
data/raw/mvtec_ad/capsule
data/raw/mvtec_ad/metal_nut
```

2. 检查每个类别是否有：

```text
train/
test/
ground_truth/
```

3. 统计每个类别下：

```text
train 图片数量
test 图片数量
ground_truth mask 数量
```

4. 输出类似：

```text
[MVTec AD Subset Check]

Category: bottle
  train images: xxx
  test images: xxx
  masks: xxx
  status: OK

Category: capsule
  train images: xxx
  test images: xxx
  masks: xxx
  status: OK

Category: metal_nut
  train images: xxx
  test images: xxx
  masks: xxx
  status: OK

Overall status: OK
```

运行方式：

```bash
python scripts/check_mvtec_dataset.py
```

如果缺少目录或文件，请明确输出错误原因。

---

## 九、请更新 README.md 的数据准备部分

请在项目根目录 `README.md` 中增加或更新以下章节：

````markdown
## Dataset Preparation

This project uses a 3-category subset of MVTec AD for the MVP demo:

- bottle
- capsule
- metal_nut

The dataset is stored locally under:

```text
data/raw/mvtec_ad/
````

Download and extract the subset:

```bash
python scripts/download_mvtec_subset.py
```

Verify the dataset structure:

```bash
python scripts/check_mvtec_dataset.py
```

Expected structure:

```text
data/raw/mvtec_ad/
├── bottle/
│   ├── train/
│   ├── test/
│   └── ground_truth/
├── capsule/
│   ├── train/
│   ├── test/
│   └── ground_truth/
└── metal_nut/
    ├── train/
    ├── test/
    └── ground_truth/
```

Note:

MVTec AD is originally an anomaly detection dataset. In this project, the pixel-level masks are converted into YOLO-style bounding boxes for a supervised industrial defect detection demo.

````

---

## 十、验收标准

完成后请确保以下命令可以成功运行：

```bash
python scripts/download_mvtec_subset.py
python scripts/check_mvtec_dataset.py
````

最终必须满足：

```text
✅ data/raw/downloads/ 下存在 bottle.tar.xz、capsule.tar.xz、metal_nut.tar.xz
✅ data/raw/mvtec_ad/bottle/ 存在 train、test、ground_truth
✅ data/raw/mvtec_ad/capsule/ 存在 train、test、ground_truth
✅ data/raw/mvtec_ad/metal_nut/ 存在 train、test、ground_truth
✅ scripts/check_mvtec_dataset.py 输出 Overall status: OK
✅ .gitignore 已忽略大型数据和模型文件
✅ README.md 已包含数据准备说明
✅ 没有把数据集文件加入 git tracked files
```

请最后运行：

```bash
git status
```

确认大型数据没有被 Git 追踪。

如果发现数据被追踪，请执行：

```bash
git rm -r --cached data/raw/downloads data/raw/mvtec_ad data/processed outputs
```

然后重新检查：

```bash
git status
```

---

## 十一、完成后请汇报

完成任务后，请在回复中给出：

1. 新增或修改的文件列表
2. 三个数据集是否下载成功
3. 每个类别的 train/test/ground_truth 文件数量
4. `python scripts/check_mvtec_dataset.py` 的完整输出
5. `git status` 的关键结果
6. 是否有任何下载失败、网络超时或目录结构异常
