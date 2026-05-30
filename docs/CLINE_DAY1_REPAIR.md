# Cline Day1 Repair Task: Complete Missing Project Skeleton, Tests, Docs, App, and Initial Git Commit

## Current Status

The project root already exists:

```bash
/root/autodl-tmp/Industrial-Defect-CV-System
```

The conda environment already exists and is active:

```bash
defect-cv
```

Python path is correct:

```bash
/root/miniconda3/envs/defect-cv/bin/python
```

However, Day1 validation did not pass because:

1. `pytest -q` returned `no tests ran`
2. Git has no initial commit yet
3. Several required files and directories are missing

Your task is to repair the project skeleton without recreating the whole project destructively.

Do not delete existing files unless necessary.

---

## 1. Enter Project and Activate Environment

```bash
cd /root/autodl-tmp/Industrial-Defect-CV-System
conda activate defect-cv
```

Verify:

```bash
pwd
which python
python --version
```

Expected Python path:

```bash
/root/miniconda3/envs/defect-cv/bin/python
```

---

## 2. Create Missing Directories

Create the missing required directories:

```bash
mkdir -p docker
mkdir -p docs
mkdir -p notebooks
mkdir -p app/pages
mkdir -p app/assets
mkdir -p outputs/checkpoints
mkdir -p outputs/logs
mkdir -p outputs/reports
mkdir -p outputs/predictions
mkdir -p outputs/exported
mkdir -p tests
```

Create `.gitkeep` files:

```bash
touch outputs/checkpoints/.gitkeep
touch outputs/logs/.gitkeep
touch outputs/reports/.gitkeep
touch outputs/predictions/.gitkeep
touch outputs/exported/.gitkeep
touch app/assets/.gitkeep
```

---

## 3. Create Missing Top-Level Files

Create these files if missing:

```bash
touch README.md
touch LICENSE
touch docker/Dockerfile
touch docker/docker-compose.yml
```

### LICENSE

Use MIT License placeholder:

```text
MIT License

Copyright (c) 2026 Delman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the Software
without restriction, subject to the conditions in the full MIT License text.
```

---

## 4. Create README.md

Overwrite `README.md` with a clean Day1 version:

````markdown
# Industrial-Defect-CV-System

Industrial-Defect-CV-System is an end-to-end industrial surface defect detection MVP.

It demonstrates a complete engineering workflow:

- MVTec AD dataset conversion
- YOLOv8 defect detection
- ResNet classification baseline
- Grad-CAM explainability
- ONNX export
- latency benchmark
- Streamlit visualization

This repository is designed as a public and anonymized engineering reproduction of industrial CV experience.

---

## MVP Scope

Day1 focuses on:

- GitHub-ready project skeleton
- isolated conda environment
- configuration files
- placeholder scripts
- basic tests
- Streamlit placeholder app

No training is implemented on Day1.

---

## Architecture

```mermaid
flowchart TD
    A[Raw MVTec AD Dataset] --> B[Mask to YOLO BBox Converter]
    B --> C[Processed YOLO Dataset]
    C --> D[YOLOv8 Detector Training]
    C --> E[ResNet Classification Baseline]

    D --> F[PyTorch Detector]
    F --> G[ONNX Export]
    G --> H[ONNXRuntime Inference]

    E --> I[Grad-CAM Explainability]

    H --> J[Streamlit Demo]
    I --> J

    J --> K[Detection Visualization]
    J --> L[Latency Benchmark]
    J --> M[Inspection Report Export]
````

---

## Project Structure

```text
configs/
data/
docs/
scripts/
src/defect_cv/
app/
outputs/
tests/
```

---

## Environment Setup

```bash
conda create -n defect-cv python=3.10 -y
conda activate defect-cv
pip install -r requirements.txt
```

Verify:

```bash
which python
pytest -q
```

---

## Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

---

## Day1 Status

* [x] Project skeleton
* [x] Conda environment
* [x] Git initialization
* [x] Placeholder scripts
* [x] Basic tests
* [ ] Dataset preparation
* [ ] YOLOv8 training
* [ ] ResNet baseline
* [ ] Grad-CAM
* [ ] ONNX export
* [ ] latency benchmark

````

---

## 5. Create Docs

Create `docs/architecture.md`:

```markdown
# Architecture

This project is organized into five layers:

1. Data layer: MVTec AD loading and mask-to-bbox conversion
2. Model layer: YOLOv8 detector and ResNet classification baseline
3. Explainability layer: Grad-CAM visualization
4. Deployment layer: ONNX export and ONNXRuntime inference
5. Demo layer: Streamlit interface and report export
````

Create `docs/dataset_conversion.md`:

```markdown
# Dataset Conversion

This project plans to use MVTec AD masks and convert them into YOLO-style bounding boxes.

This is for a supervised detection demo, not for claiming official anomaly detection benchmark performance.
```

Create `docs/benchmark.md`:

```markdown
# Benchmark

| Model | Runtime | Device | Image Size | Precision | p50 Latency | p95 Latency | FPS | Model Size |
|---|---|---|---|---|---:|---:|---:|---:|
| YOLOv8n | PyTorch | CUDA | 768 | FP32 | TBD | TBD | TBD | TBD |
| YOLOv8n | ONNXRuntime | CUDA | 768 | FP32 | TBD | TBD | TBD | TBD |
```

Create `docs/deployment.md`:

````markdown
# Deployment

Planned deployment path:

```text
PyTorch checkpoint -> ONNX export -> ONNXRuntime inference -> latency benchmark -> Streamlit demo
````

````

Create `docs/interview_notes.md`:

```markdown
# Interview Notes

This project is not a toy YOLO demo.

It is designed to demonstrate an industrial CV engineering workflow:

- small defect detection
- weak texture handling
- reflective surface robustness
- class imbalance handling
- explainability
- deployment latency
- visual reporting
````

Create or update `docs/day1_environment_check.md` with current validation output.

---

## 6. Create Minimal Streamlit App

Create `app/streamlit_app.py`:

```python
import streamlit as st

st.set_page_config(
    page_title="Industrial Defect CV System",
    page_icon="🏭",
    layout="wide",
)

st.title("Industrial-Defect-CV-System")

st.markdown(
    """
    This is a Day1 placeholder demo for an industrial surface defect detection MVP.

    Planned modules:

    - YOLOv8 defect detection
    - ResNet classification baseline
    - Grad-CAM explainability
    - ONNXRuntime inference
    - latency benchmark
    - inspection report export

    Day1 only validates the project skeleton and environment.
    """
)

st.info("Model training and inference will be implemented in later MVP stages.")
```

Create placeholder pages:

```bash
cat > app/pages/1_Detection.py <<'PY'
import streamlit as st

st.title("Detection")
st.info("YOLOv8 detection page placeholder.")
PY

cat > app/pages/2_GradCAM.py <<'PY'
import streamlit as st

st.title("Grad-CAM")
st.info("Grad-CAM explainability page placeholder.")
PY

cat > app/pages/3_Benchmark.py <<'PY'
import streamlit as st

st.title("Benchmark")
st.info("Latency benchmark page placeholder.")
PY
```

---

## 7. Create Minimal Tests

Create `tests/test_dataset.py`:

```python
from pathlib import Path


def test_mvtec_subset_config_exists():
    assert Path("configs/dataset/mvtec_subset.yaml").exists()
```

Create `tests/test_mask_to_bbox.py`:

```python
def test_mask_to_bbox_placeholder():
    assert True
```

Create `tests/test_onnx_infer.py`:

```python
def test_onnx_infer_placeholder():
    assert True
```

Create `tests/test_latency_profiler.py`:

```python
def test_latency_profiler_placeholder():
    assert True
```

Run:

```bash
pytest -q
```

Expected:

```text
4 passed
```

---

## 8. Check Required Files Exist

Run:

```bash
ls -lah
ls -lah docs
ls -lah app
ls -lah tests
ls -lah outputs
```

Confirm these exist:

```text
README.md
LICENSE
.gitignore
requirements.txt
environment.yml
pyproject.toml
.env.example
docker/
configs/
data/
docs/
scripts/
src/
app/
outputs/
tests/
```

---

## 9. Run Final Validation

From project root:

```bash
pwd
which python
python --version
pytest -q
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"
```

Start Streamlit briefly:

```bash
streamlit run app/streamlit_app.py --server.headless true --server.port 8501
```

If it starts successfully, stop it with Ctrl+C.

---

## 10. Initial Git Commit

Run:

```bash
git status
git add .
git commit -m "chore: initialize industrial defect CV project skeleton"
```

If Git identity is missing, set local identity only:

```bash
git config user.name "Delman"
git config user.email "delman@example.com"
git commit -m "chore: initialize industrial defect CV project skeleton"
```

Finally run:

```bash
git log --oneline -1
git status
```

Expected final status:

```text
nothing to commit, working tree clean
```

---

## 11. Final Report

After finishing, report:

1. project path
2. Python path
3. pytest result
4. Streamlit result
5. Git commit hash
6. remaining issues, if any
