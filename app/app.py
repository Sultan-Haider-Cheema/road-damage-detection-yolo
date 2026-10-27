from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
MODEL_PATH = Path("road_damage_model.pt")

EXPECTED_CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Other Corruption",
    4: "Pothole",
}


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.6rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1.05rem;
            color: #6b7280;
            margin-bottom: 1.5rem;
        }

        .info-box {
            padding: 1rem;
            border-radius: 0.7rem;
            background-color: rgba(128, 128, 128, 0.10);
            margin-bottom: 1rem;
        }

        .footer {
            text-align: center;
            color: #6b7280;
            padding-top: 2rem;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_path: str) -> YOLO:
    """
    Load the YOLO model once and keep it cached between Streamlit reruns.
    """
    return YOLO(model_path)


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def get_class_name(
    class_id: int,
    model_names: Optional[dict],
) -> str:
    """
    Return the class name provided by the model.
    Fall back to the expected project class names.
    """
    if isinstance(model_names, dict):
        return str(
            model_names.get(
                class_id,
                EXPECTED_CLASS_NAMES.get(class_id, f"Class {class_id}"),
            )
        )

    if isinstance(model_names, list) and class_id < len(model_names):
        return str(model_names[class_id])

    return EXPECTED_CLASS_NAMES.get(class_id, f"Class {class_id}")


def create_detection_table(result) -> pd.DataFrame:
    """
    Convert YOLO detections into a pandas DataFrame.
    """
    rows = []

    if result.boxes is None or len(result.boxes) == 0:
        return pd.DataFrame(
            columns=[
                "Detection",
                "Class",
                "Confidence",
                "X1",
                "Y1",
                "X2",
                "Y2",
            ]
        )

    model_names = result.names

    for index, box in enumerate(result.boxes, start=1):
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        rows.append(
            {
                "Detection": index,
                "Class": get_class_name(class_id, model_names),
                "Confidence": round(confidence, 4),
                "X1": round(x1, 1),
                "Y1": round(y1, 1),
                "X2": round(x2, 1),
                "Y2": round(y2, 1),
            }
        )

    return pd.DataFrame(rows)


def create_class_summary(detections: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of detections for each predicted class.
    """
    if detections.empty:
        return pd.DataFrame(columns=["Damage Type", "Count"])

    summary = (
        detections.groupby("Class")
        .size()
        .reset_index(name="Count")
        .rename(columns={"Class": "Damage Type"})
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )

    return summary


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🛣️ Automatic Road Damage Detection</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Upload a road image to detect cracks, potholes, and other road damage
        using a trained YOLO object-detection model.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.header("Detection Settings")

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help=(
            "Predictions below this confidence score will not be displayed."
        ),
    )

    iou_threshold = st.slider(
        "IoU threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help=(
            "Controls how strongly overlapping predictions are suppressed."
        ),
    )

    image_size = st.selectbox(
        "Inference image size",
        options=[320, 480, 640, 800],
        index=2,
        help="Use 640 for consistency with model training.",
    )

    st.divider()

    st.subheader("Project Classes")

    for class_id, class_name in EXPECTED_CLASS_NAMES.items():
        st.write(f"**{class_id}:** {class_name}")

    st.divider()

    st.caption(
        "The model can later be replaced by overwriting "
        "`road_damage_model.pt`."
    )


# ---------------------------------------------------------
# CHECK MODEL FILE
# ---------------------------------------------------------
if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: `{MODEL_PATH}`\n\n"
        "Place your trained `best.pt` file in the project folder "
        "and rename it to `road_damage_model.pt`."
    )
    st.stop()


try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error("The model could not be loaded.")
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------
with st.expander("Model information"):
    st.write(f"**Model file:** `{MODEL_PATH.name}`")

    model_names = getattr(model, "names", None)

    if model_names:
        st.write("**Classes stored in the model:**")
        st.json(model_names)
    else:
        st.warning("Class names could not be read from the model.")


# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"],
    help="Accepted formats: JPG, JPEG, and PNG.",
)


if uploaded_file is None:
    st.markdown(
        """
        <div class="info-box">
            Upload an image to begin road-damage detection.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "For a useful demonstration, choose an image with visible road "
        "cracks, potholes, or surface damage."
    )

else:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")
    except Exception as error:
        st.error("The uploaded file could not be opened as an image.")
        st.exception(error)
        st.stop()

    st.subheader("Input image")

    st.image(
        original_image,
        caption="Uploaded road image",
        use_container_width=True,
    )

    detect_button = st.button(
        "Run damage detection",
        type="primary",
        use_container_width=True,
    )

    if detect_button:
        with st.spinner("Analyzing the road image..."):
            try:
                results = model.predict(
                    source=original_image,
                    conf=confidence_threshold,
                    iou=iou_threshold,
                    imgsz=image_size,
                    device="cpu",
                    verbose=False,
                )
            except Exception as error:
                st.error("An error occurred during model inference.")
                st.exception(error)
                st.stop()

        if not results:
            st.warning("The model did not return any result.")
            st.stop()

        result = results[0]

        # result.plot() returns a BGR NumPy array.
        annotated_bgr = result.plot(
            labels=True,
            conf=True,
            boxes=True,
        )

        # Convert BGR to RGB for correct Streamlit display.
        annotated_rgb = annotated_bgr[:, :, ::-1]

        detections = create_detection_table(result)
        class_summary = create_class_summary(detections)

        st.success("Detection completed.")

        st.subheader("Detection result")

        image_col1, image_col2 = st.columns(2)

        with image_col1:
            st.image(
                original_image,
                caption="Original image",
                use_container_width=True,
            )

        with image_col2:
            st.image(
                annotated_rgb,
                caption="Detected road damage",
                use_container_width=True,
            )

        st.divider()

        total_detections = len(detections)

        if total_detections == 0:
            st.warning(
                "No road damage was detected at the selected "
                "confidence threshold."
            )

        else:
            unique_classes = detections["Class"].nunique()
            average_confidence = detections["Confidence"].mean()

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            metric_col1.metric(
                "Total detections",
                total_detections,
            )

            metric_col2.metric(
                "Detected damage types",
                unique_classes,
            )

            metric_col3.metric(
                "Average confidence",
                f"{average_confidence:.2%}",
            )

            st.subheader("Damage summary")
            st.dataframe(
                class_summary,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Detailed detections")
            st.dataframe(
                detections,
                use_container_width=True,
                hide_index=True,
            )

            csv_data = detections.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download detection results as CSV",
                data=csv_data,
                file_name="road_damage_detections.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ---------------------------------------------------------
# PROJECT DESCRIPTION
# ---------------------------------------------------------
st.divider()

with st.expander("About this project"):
    st.write(
        """
        This application uses a YOLO object-detection model trained on
        the RDD2022 road-damage dataset. The system identifies damaged
        road regions and draws bounding boxes around detected objects.

        The project compares YOLOv8n, YOLOv8s, and YOLOv5s in terms of
        detection accuracy, inference speed, and model size.
        """
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Road Damage Detection Project · Built with Streamlit and Ultralytics YOLO
    </div>
    """,
    unsafe_allow_html=True,
)