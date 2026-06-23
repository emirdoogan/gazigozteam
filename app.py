from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from time import perf_counter
import traceback
from urllib.request import Request, urlopen

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from raw_preprocessing import crop_raw_image

# --- Constants ---
MODEL_CANDIDATES = [
    "complex_unet_edi_best.keras",
    "complex_unet_edi_last.keras",
    "complex_unet_enhancedv5.keras",
]
REMOTE_MODEL_DIR = Path("/tmp/edi_oct_models")
SIZE       = 496
N_CLASSES  = 13
CLASS_NAMES = {
    0: "Background",
    1: "Vitreus",
    2: "RNFL",
    3: "GCL",
    4: "IPL",
    5: "INL",
    6: "OPL",
    7: "ONL",
    8: "ELM",
    9: "EZ",
    10: "IZ",
    11: "RPE",
    12: "Choriocapillaris",
}
PALETTE = {
    0:  (  0,   0,   0),
    1:  (255,   0,   0),
    2:  (  0, 255,   0),
    3:  (  0,   0, 255),
    4:  (255, 255,   0),
    5:  (255,   0, 255),
    6:  (  0, 255, 255),
    7:  (128,   0,   128),
    8:  (128, 128,   0),
    9:  (  0, 128, 128),
    10:(128, 128, 128),
    11:(255, 165,   0),
    12:(180, 120, 255),
}

# #region debug-point A:reporter
def report_debug_event(hypothesis_id, location, message, data=None, run_id="pre-fix"):
    env_path = Path(".dbg/streamlit-upload-crash.env")
    server_url = "http://127.0.0.1:7777/event"
    session_id = "streamlit-upload-crash"

    try:
        env_text = env_path.read_text()
        for line in env_text.splitlines():
            if line.startswith("DEBUG_SERVER_URL="):
                server_url = line.split("=", 1)[1].strip()
            elif line.startswith("DEBUG_SESSION_ID="):
                session_id = line.split("=", 1)[1].strip()
    except Exception:
        pass

    payload = {
        "sessionId": session_id,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "msg": f"[DEBUG] {message}",
        "data": data or {},
    }

    try:
        request = Request(
            server_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=0.25):
            pass
    except Exception:
        pass
# #endregion

# --- Cache model load as a resource ---
def get_runtime_config(name, default=None):
    if name in os.environ:
        return os.environ[name]
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def download_model_if_needed():
    model_url = get_runtime_config("MODEL_URL")
    model_filename = get_runtime_config("MODEL_FILENAME", "complex_unet_edi_best.keras")
    if not model_url:
        return None

    REMOTE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    destination = REMOTE_MODEL_DIR / model_filename
    if destination.exists():
        return str(destination)

    with st.spinner("Downloading segmentation model for this app instance..."):
        with urlopen(model_url) as response, destination.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)
    return str(destination)


def resolve_model_path():
    for candidate in MODEL_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return str(path)
    return download_model_if_needed()


@st.cache_resource
def load_segmentation_model(path):
    import tensorflow as tf
    # Delay TensorFlow/Keras imports until the user actually uploads an image.
    from tensorflow.keras.models import load_model
    from simple_unet import GroupNormalization

    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

    return load_model(
        path,
        compile=False,
        custom_objects={
            "GroupNormalization": GroupNormalization,
        },
    )


def init_session_state():
    if "measurement_history" not in st.session_state:
        st.session_state.measurement_history = []
    if "roi_range" not in st.session_state:
        st.session_state.roi_range = None
    if "roi_image_name" not in st.session_state:
        st.session_state.roi_image_name = None


def build_summary_row(image_name, scale_px, roi_width_px, roi_width_um, layer_df, composite_df):
    row = {
        "Image": image_name,
        "MeasuredAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ScalePixelsPer200um": int(scale_px),
        "ROIWidthPx": int(roi_width_px),
        "ROIWidthUm": round(float(roi_width_um), 2),
    }

    for _, item in layer_df.iterrows():
        layer_name = str(item["Layer"])
        row[f"{layer_name} MeanThickness (µm)"] = round(float(item["MeanThickness (µm)"]), 2)
        row[f"{layer_name} MeanReflectivity"] = round(float(item["MeanReflectivity"]), 4)

    for _, item in composite_df.iterrows():
        layer_name = str(item["Layer"])
        row[f"{layer_name} MeanThickness (µm)"] = round(float(item["MeanThickness (µm)"]), 2)
        row[f"{layer_name} MeanReflectivity"] = round(float(item["MeanReflectivity"]), 4)

    return row


def compute_layer_measurements(mask, raw_norm, class_names, n_classes, um_per_px, col_range=None):
    h, w = mask.shape
    rows = np.arange(h)
    records = []

    if col_range is None:
        columns = range(w)
        roi_bool = np.ones_like(mask, dtype=bool)
    else:
        col_start, col_end = col_range
        columns = range(col_start, col_end)
        roi_bool = np.zeros_like(mask, dtype=bool)
        roi_bool[:, col_start:col_end] = True

    for cls in range(1, n_classes):
        layer = mask == cls
        if not layer.any():
            continue

        thickness_px = []
        for c in columns:
            ys = rows[layer[:, c]]
            if ys.size:
                thickness_px.append(ys[-1] - ys[0] + 1)

        thickness_px = np.array(thickness_px, float)
        thickness_um = thickness_px * um_per_px
        refl_vals = raw_norm[layer & roi_bool]

        records.append({
            "Layer": class_names[cls],
            "ColumnsMeasured": len(thickness_px),
            "MeanThickness (µm)": np.mean(thickness_um) if len(thickness_px) > 0 else 0,
            "MedianThickness (µm)": np.median(thickness_um) if len(thickness_px) > 0 else 0,
            "SDThickness (µm)": np.std(thickness_um) if len(thickness_px) > 0 else 0,
            "MeanReflectivity": np.mean(refl_vals) if len(refl_vals) > 0 else 0,
            "MedianReflectivity": np.median(refl_vals) if len(refl_vals) > 0 else 0,
            "SDReflectivity": np.std(refl_vals) if len(refl_vals) > 0 else 0,
        })

    return pd.DataFrame.from_records(records)


def get_val(df, layer, col):
    try:
        return float(df.loc[df["Layer"] == layer, col].iloc[0])
    except Exception:
        return 0.0


def compute_composite_measurements(df):
    ganglion_layers = ["RNFL", "GCL", "IPL"]
    g_mean_t = sum(get_val(df, layer, "MeanThickness (µm)") for layer in ganglion_layers)
    g_mr_num = sum(
        get_val(df, layer, "MeanReflectivity") * get_val(df, layer, "MeanThickness (µm)")
        for layer in ganglion_layers
    )
    g_mean_r = g_mr_num / g_mean_t if g_mean_t > 0 else 0
    g_med_t = sum(get_val(df, layer, "MedianThickness (µm)") for layer in ganglion_layers)
    ganglion_df = df.loc[df["Layer"].isin(ganglion_layers)]
    g_med_r = ganglion_df["MedianReflectivity"].median() if not ganglion_df.empty else 0

    composite = [
        {
            "Layer": "Ganglion Layer",
            "MeanThickness (µm)": g_mean_t,
            "MedianThickness (µm)": g_med_t,
            "MeanReflectivity": g_mean_r,
            "MedianReflectivity": g_med_r,
        },
        {
            "Layer": "Outer Nuclear Layer (ONL)",
            "MeanThickness (µm)": get_val(df, "ONL", "MeanThickness (µm)"),
            "MedianThickness (µm)": get_val(df, "ONL", "MedianThickness (µm)"),
            "MeanReflectivity": get_val(df, "ONL", "MeanReflectivity"),
            "MedianReflectivity": get_val(df, "ONL", "MedianReflectivity"),
        },
        {
            "Layer": "Ellipsoid Zone (EZ)",
            "MeanThickness (µm)": get_val(df, "EZ", "MeanThickness (µm)"),
            "MedianThickness (µm)": get_val(df, "EZ", "MedianThickness (µm)"),
            "MeanReflectivity": get_val(df, "EZ", "MeanReflectivity"),
            "MedianReflectivity": get_val(df, "EZ", "MedianReflectivity"),
        },
        {
            "Layer": "Interdigitation Zone (IZ)",
            "MeanThickness (µm)": get_val(df, "IZ", "MeanThickness (µm)"),
            "MedianThickness (µm)": get_val(df, "IZ", "MedianThickness (µm)"),
            "MeanReflectivity": get_val(df, "IZ", "MeanReflectivity"),
            "MedianReflectivity": get_val(df, "IZ", "MedianReflectivity"),
        },
        {
            "Layer": "RPE + Choriocapillaris",
            "MeanThickness (µm)": (
                get_val(df, "RPE", "MeanThickness (µm)")
                + get_val(df, "Choriocapillaris", "MeanThickness (µm)")
            ),
            "MedianThickness (µm)": (
                get_val(df, "RPE", "MedianThickness (µm)")
                + get_val(df, "Choriocapillaris", "MedianThickness (µm)")
            ),
            "MeanReflectivity": np.mean([
                get_val(df, "RPE", "MeanReflectivity"),
                get_val(df, "Choriocapillaris", "MeanReflectivity"),
            ]),
            "MedianReflectivity": np.mean([
                get_val(df, "RPE", "MedianReflectivity"),
                get_val(df, "Choriocapillaris", "MedianReflectivity"),
            ]),
        },
    ]
    return pd.DataFrame(composite)


def measurement_download_bytes(df):
    return df.to_csv(index=False).encode()


def run_segmentation_inference(model, batch):
    # A direct forward pass is more reliable than model.predict() in this app's
    # Streamlit runtime and avoids the extra Keras prediction loop.
    pred = model(batch, training=False)
    if hasattr(pred, "numpy"):
        pred = pred.numpy()
    return pred

# --- App UI ---
st.set_page_config(
    page_title="EDI OCT Measurement Studio",
    page_icon=":bar_chart:",
    layout="wide",
)
init_session_state()

st.title("EDI OCT Measurement Studio")
st.caption("Upload an OCT image, review the segmentation, define an ROI, and keep each measurement as a reusable session row.")

with st.sidebar:
    st.header("Input Settings")
    st.caption("Model loads when you upload an image.")
    scale_px = st.number_input(
        "Scale bar length in pixels (200 µm)",
        min_value=1,
        value=50,
        help="Count how many pixels correspond to the 200 µm scale bar in this OCT image.",
    )
    compute_full = st.checkbox("Also compute full-image tables", value=False)
    st.divider()
    st.markdown("**Workflow**")
    st.markdown(
        "1. Upload one EDI OCT image.\n"
        "2. Adjust the ROI on the predicted mask.\n"
        "3. Review measurements and add the case to the session history."
    )

hero_col1, hero_col2, hero_col3 = st.columns(3)
hero_col1.metric("Next Step", "Upload EDI Image")
hero_col2.metric("Scale Entry", "Enter 200 µm bar")
hero_col3.metric("Saved Rows", str(len(st.session_state.measurement_history)))

uploaded = st.file_uploader("Upload OCT image", type=["png", "jpg", "jpeg"])
if not uploaded:
    if st.session_state.measurement_history:
        st.subheader("Session Measurement History")
        history_df = pd.DataFrame(st.session_state.measurement_history)
        st.dataframe(history_df, use_container_width=True)
        st.download_button(
            label="Download session history as CSV",
            data=measurement_download_bytes(history_df),
            file_name="edi_measurement_history.csv",
            mime="text/csv",
        )
    st.stop()

# --- Read image from upload ---
status = st.status("Processing uploaded EDI image...", expanded=False)
perf = {}

# #region debug-point A:upload-received
report_debug_event(
    "A",
    "app.py:upload",
    "Upload received",
    {
        "file_name": uploaded.name,
        "file_type": uploaded.type,
        "file_size": getattr(uploaded, "size", None),
    },
)
# #endregion

try:
    start = perf_counter()
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    perf["decode"] = perf_counter() - start
    if img_bgr is None:
        status.update(label="Image decoding failed.", state="error")
        st.error("Couldn't decode image!")
        # #region debug-point A:decode-none
        report_debug_event(
            "A",
            "app.py:decode",
            "Image decode returned None",
            {"buffer_length": int(file_bytes.size)},
        )
        # #endregion
        st.stop()

    # #region debug-point B:decode-success
    report_debug_event(
        "B",
        "app.py:decode",
        "Image decoded successfully",
        {
            "decode_seconds": round(perf["decode"], 4),
            "decoded_shape": list(img_bgr.shape),
            "decoded_dtype": str(img_bgr.dtype),
        },
    )
    # #endregion

    status.update(label="Preparing OCT image...")
    start = perf_counter()
    if img_bgr.ndim == 3 and img_bgr.shape[2] > 1:
        raw_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        raw_gray = img_bgr.copy()

    cropped = crop_raw_image(raw_gray)
    resized = cv2.resize(cropped, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    raw_norm = resized.astype(np.float32) / 255.0
    raw_3d = raw_norm[..., np.newaxis]  # (H,W,1)
    perf["preprocess"] = perf_counter() - start

    # #region debug-point B:preprocess-success
    report_debug_event(
        "B",
        "app.py:preprocess",
        "Preprocess completed",
        {
            "preprocess_seconds": round(perf["preprocess"], 4),
            "gray_shape": list(raw_gray.shape),
            "cropped_shape": list(cropped.shape),
            "resized_shape": list(resized.shape),
            "raw_min": float(raw_norm.min()),
            "raw_max": float(raw_norm.max()),
            "raw_mean": float(raw_norm.mean()),
        },
    )
    # #endregion
except Exception as exc:
    # #region debug-point B:preprocess-error
    report_debug_event(
        "B",
        "app.py:preprocess",
        "Decode or preprocess failed",
        {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        },
    )
    # #endregion
    raise

status.update(label="Loading segmentation model...")
model_path = resolve_model_path()
if model_path is None:
    status.update(label="Model file is missing.", state="error")
    st.error(
        "No model file was found. Expected one of: "
        + ", ".join(MODEL_CANDIDATES)
    )
    st.stop()

try:
    # #region debug-point A:model-path
    report_debug_event(
        "A",
        "app.py:model",
        "Resolved model path",
        {
            "model_path": model_path,
            "model_exists": Path(model_path).exists(),
            "model_size_mb": round(Path(model_path).stat().st_size / (1024 * 1024), 2),
        },
    )
    # #endregion

    start = perf_counter()
    model = load_segmentation_model(model_path)
    perf["load_model"] = perf_counter() - start

    # #region debug-point A:model-loaded
    report_debug_event(
        "A",
        "app.py:model",
        "Model loaded successfully",
        {
            "load_model_seconds": round(perf["load_model"], 4),
            "input_shape": [dim if dim is not None else -1 for dim in model.input_shape],
            "output_shape": [dim if dim is not None else -1 for dim in model.output_shape],
            "parameter_count": int(model.count_params()),
        },
    )
    # #endregion

    status.update(label="Running segmentation inference...")
    start = perf_counter()
    X_batch = raw_3d[np.newaxis, ...].astype(np.float32)

    # #region debug-point C:predict-start
    report_debug_event(
        "C",
        "app.py:predict",
        "Starting model inference",
        {
            "batch_shape": list(X_batch.shape),
            "batch_dtype": str(X_batch.dtype),
            "batch_min": float(X_batch.min()),
            "batch_max": float(X_batch.max()),
        },
    )
    # #endregion

    pred = run_segmentation_inference(model, X_batch)[0]
    mask = np.argmax(pred, axis=-1).astype(np.uint8)
    perf["predict"] = perf_counter() - start
    status.update(label="EDI image processed successfully.", state="complete")

    # #region debug-point C:predict-success
    report_debug_event(
        "C",
        "app.py:predict",
        "Inference completed",
        {
            "predict_seconds": round(perf["predict"], 4),
            "prediction_shape": list(pred.shape),
            "prediction_dtype": str(pred.dtype),
            "mask_shape": list(mask.shape),
            "mask_unique_classes": [int(v) for v in np.unique(mask)],
        },
    )
    # #endregion
except Exception as exc:
    # #region debug-point C:predict-error
    report_debug_event(
        "C",
        "app.py:predict",
        "Model load or inference failed",
        {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        },
    )
    # #endregion
    raise

color_mask = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
for cls, col in PALETTE.items():
    color_mask[mask == cls] = col

h, w = mask.shape

# Default ROI: center 60% of the image width
mid = w // 2
half = int(0.3 * w)
min_width = max(20, int(0.1 * w))
default_roi = (mid - half, mid + half)

if st.session_state.roi_image_name != uploaded.name or st.session_state.roi_range is None:
    st.session_state.roi_range = default_roi
    st.session_state.roi_image_name = uploaded.name

# Calculate microns per pixel from scale input
um_per_px = 200.0 / scale_px

preview_tab, measurement_tab, history_tab = st.tabs([
    "Preview",
    "Measurements",
    "Session History",
])

with preview_tab:
    st.subheader("Image Review")
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(raw_gray, caption="Original Upload", clamp=True, use_container_width=True)
    with img_col2:
        st.image(raw_norm, caption="Cropped & Resized", clamp=True, use_container_width=True)

    st.subheader("ROI Selection")
    st.caption("Drag the range handles or use the quick actions to position the measurement region.")

    current_start, current_end = st.session_state.roi_range
    current_width = max(min_width, current_end - current_start)

    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    with action_col1:
        if st.button("Center 60%", use_container_width=True):
            st.session_state.roi_range = default_roi
    with action_col2:
        if st.button("Full Width", use_container_width=True):
            st.session_state.roi_range = (0, w)
    with action_col3:
        if st.button("Shift Left", use_container_width=True):
            new_start = max(0, current_start - 20)
            st.session_state.roi_range = (new_start, new_start + current_width)
    with action_col4:
        if st.button("Shift Right", use_container_width=True):
            new_end = min(w, current_end + 20)
            st.session_state.roi_range = (new_end - current_width, new_end)

    col_start, col_end = st.slider(
        "ROI Column Range",
        min_value=0,
        max_value=w,
        value=st.session_state.roi_range,
        step=1,
        key="roi_range",
        help="Select the horizontal image span used for thickness and reflectivity measurements.",
    )

    if col_end - col_start < min_width:
        adjusted_end = min(w, col_start + min_width)
        adjusted_start = max(0, adjusted_end - min_width)
        st.session_state.roi_range = (adjusted_start, adjusted_end)
        col_start, col_end = st.session_state.roi_range

    roi_width_px = col_end - col_start
    roi_width_um = roi_width_px * um_per_px
    overlay = color_mask.copy()
    cv2.rectangle(overlay, (col_start, 0), (col_end - 1, h - 1), (255, 255, 255), 2)

    roi_stat1, roi_stat2, roi_stat3 = st.columns(3)
    roi_stat1.metric("Start Column", col_start)
    roi_stat2.metric("End Column", col_end)
    roi_stat3.metric("ROI Width", f"{roi_width_px}px")

    if roi_width_px < min_width:
        st.warning(
            f"ROI width must be at least {min_width} pixels. Current width: {roi_width_px} pixels."
        )
        valid_roi = False
    else:
        valid_roi = True

    st.subheader("Segmentation")
    seg_col1, seg_col2 = st.columns(2)
    with seg_col1:
        st.image(color_mask, caption="Color-coded Mask", use_container_width=True)
    with seg_col2:
        st.image(
            overlay,
            caption=f"ROI overlay: {roi_width_px}px (~{roi_width_um:.0f} µm)",
            use_container_width=True,
        )

info_col1, info_col2, info_col3 = st.columns(3)
info_col1.metric("File", uploaded.name)
info_col2.metric("Scale", f"{scale_px} px / 200 µm")
info_col3.metric("ROI Width", f"{roi_width_px}px")

with st.expander("Processing details", expanded=False):
    st.caption(f"Model file: {Path(model_path).name}")
    st.write(
        {
            "Decode (s)": round(perf["decode"], 3),
            "Preprocess (s)": round(perf["preprocess"], 3),
            "Model load (s)": round(perf["load_model"], 3),
            "Predict (s)": round(perf["predict"], 3),
        }
    )

fmt = {
    "MeanThickness (µm)": "{:.2f}",
    "MedianThickness (µm)": "{:.2f}",
    "SDThickness (µm)": "{:.2f}",
    "MeanReflectivity": "{:.4f}",
    "MedianReflectivity": "{:.4f}",
    "SDReflectivity": "{:.4f}",
}

if valid_roi:
    try:
        df = compute_layer_measurements(
            mask,
            raw_norm,
            CLASS_NAMES,
            N_CLASSES,
            um_per_px,
            col_range=(col_start, col_end),
        )
        comp_df = compute_composite_measurements(df)

        # #region debug-point D:measurements-success
        report_debug_event(
            "D",
            "app.py:measurements",
            "ROI measurements computed",
            {
                "roi_range": [int(col_start), int(col_end)],
                "roi_width_px": int(roi_width_px),
                "layer_rows": int(len(df)),
                "composite_rows": int(len(comp_df)),
                "layer_names": df["Layer"].tolist(),
            },
        )
        # #endregion
    except Exception as exc:
        # #region debug-point D:measurements-error
        report_debug_event(
            "D",
            "app.py:measurements",
            "Measurement computation failed",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        # #endregion
        raise

    with measurement_tab:
        st.subheader("ROI Layer Measurements")
        top_col1, top_col2, top_col3 = st.columns(3)
        top_col1.metric("Measured Layers", str(len(df)))
        top_col2.metric("Columns in ROI", str(roi_width_px))
        top_col3.metric("ROI Width (µm)", f"{roi_width_um:.1f}")

        st.dataframe(df.style.format(fmt), use_container_width=True)

        st.subheader("Composite & Photoreceptor Measurements")
        st.dataframe(
            comp_df.style.format({
                "MeanThickness (µm)": "{:.2f}",
                "MedianThickness (µm)": "{:.2f}",
                "MeanReflectivity": "{:.4f}",
                "MedianReflectivity": "{:.4f}",
            }),
            use_container_width=True,
        )

        action_col1, action_col2, action_col3 = st.columns([1.3, 1.3, 2.4])
        with action_col1:
            st.download_button(
                label="Download ROI basic CSV",
                data=measurement_download_bytes(df),
                file_name="layer_measurements_roi.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with action_col2:
            st.download_button(
                label="Download ROI composite CSV",
                data=measurement_download_bytes(comp_df),
                file_name="composite_layer_measurements_roi.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with action_col3:
            if st.button("Add Current Image To Session History", use_container_width=True):
                summary_row = build_summary_row(
                    uploaded.name,
                    scale_px,
                    roi_width_px,
                    roi_width_um,
                    df,
                    comp_df,
                )
                st.session_state.measurement_history.append(summary_row)
                st.success(f"Added `{uploaded.name}` to session history.")

        if compute_full:
            df_full = compute_layer_measurements(
                mask,
                raw_norm,
                CLASS_NAMES,
                N_CLASSES,
                um_per_px,
                col_range=None,
            )
            comp_df_full = compute_composite_measurements(df_full)

            st.subheader("Full Image Measurements")
            st.dataframe(df_full.style.format(fmt), use_container_width=True)

            st.subheader("Full Image Composite Measurements")
            st.dataframe(
                comp_df_full.style.format({
                    "MeanThickness (µm)": "{:.2f}",
                    "MedianThickness (µm)": "{:.2f}",
                    "MeanReflectivity": "{:.4f}",
                    "MedianReflectivity": "{:.4f}",
                }),
                use_container_width=True,
            )

            full_col1, full_col2 = st.columns(2)
            with full_col1:
                st.download_button(
                    label="Download full-image basic CSV",
                    data=measurement_download_bytes(df_full),
                    file_name="layer_measurements_full.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with full_col2:
                st.download_button(
                    label="Download full-image composite CSV",
                    data=measurement_download_bytes(comp_df_full),
                    file_name="composite_layer_measurements_full.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

with history_tab:
    st.subheader("Session Measurement History")
    if st.session_state.measurement_history:
        history_df = pd.DataFrame(st.session_state.measurement_history)
        st.dataframe(history_df, use_container_width=True)

        history_col1, history_col2 = st.columns(2)
        with history_col1:
            st.download_button(
                label="Download session history as CSV",
                data=measurement_download_bytes(history_df),
                file_name="edi_measurement_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with history_col2:
            if st.button("Clear Session History", use_container_width=True):
                st.session_state.measurement_history = []
                st.success("Session history cleared.")
    else:
        st.info("No saved measurements yet. Measure an image and add it to the session history.")

# Model Architecture Visualization removed as requested
