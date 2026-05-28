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
