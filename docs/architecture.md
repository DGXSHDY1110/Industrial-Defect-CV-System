# Architecture

This project is organized into five layers:

1. Data layer: MVTec AD loading and mask-to-bbox conversion
2. Model layer: YOLOv8 detector and ResNet classification baseline
3. Explainability layer: Grad-CAM visualization
4. Deployment layer: ONNX export and ONNXRuntime inference
5. Demo layer: Streamlit interface and report export