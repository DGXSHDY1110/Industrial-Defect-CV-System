请严格清理并修正项目结构，不要再创建任何 day2 相关脚本。

项目路径：
/root/autodl-tmp/Industrial-Defect-CV-System

目标：
scripts/ 目录必须只保留以下文件：
- prepare_mvtec.py
- mask_to_yolo_bbox.py
- train_yolo.py
- train_resnet.py
- eval_yolo.py
- eval_resnet.py
- export_onnx.py
- benchmark_latency.py
- generate_report.py

要求：

1. 不要创建 run_day2_yolo.sh。
2. 不要创建 collect_day2_yolo_metrics.py。
3. 不要创建 analyze_yolo_cases.py。
4. 不要出现 day2_yolo、yolov8n_day2 这种命名。
5. 所有 YOLOv8n 训练逻辑合并到 scripts/train_yolo.py。
6. 所有 YOLOv8n 评估、metrics.json、metrics.md、PR_curve、confusion_matrix 复制逻辑合并到 scripts/eval_yolo.py。
7. 10 张成功案例 + 5 张失败案例整理逻辑合并到 scripts/generate_report.py。
8. 输出目录统一使用：
   - outputs/checkpoints/yolo/yolov8n_mvtec/
   - outputs/logs/yolo/yolov8n_mvtec.log
   - outputs/reports/yolo/
   - outputs/reports/yolo/cases/success/
   - outputs/reports/yolo/cases/failure/

不要运行正式训练。
最多只允许运行：
python -m py_compile scripts/train_yolo.py
python -m py_compile scripts/eval_yolo.py
python -m py_compile scripts/generate_report.py

完成后只输出修改了哪些文件，不要跑 50 epoch 训练。