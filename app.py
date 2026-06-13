from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from film_splitter.processing import (
    CropBox,
    detect_frames,
    draw_boxes_on_preview,
    estimate_global_skew_degrees,
    export_zip_file,
    make_preview,
    open_uploaded_image,
    rotate_image_for_detection,
)


st.set_page_config(page_title="Film Scan Frame Splitter", layout="wide")

OUTPUT_DIR = Path("outputs")
LARGE_UPLOAD_MB = 300


def boxes_to_dataframe(boxes: list[CropBox]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "enabled": box.enabled,
                "order": box.order,
                "x": box.x,
                "y": box.y,
                "width": box.width,
                "height": box.height,
            }
            for box in boxes
        ]
    )


def dataframe_to_boxes(df: pd.DataFrame, image_width: int, image_height: int) -> list[CropBox]:
    boxes: list[CropBox] = []
    for idx, row in df.fillna(0).iterrows():
        width = int(row.get("width", 0))
        height = int(row.get("height", 0))
        if width <= 0 or height <= 0:
            continue
        order_value = int(row.get("order", idx + 1))
        boxes.append(
            CropBox(
                x=int(row.get("x", 0)),
                y=int(row.get("y", 0)),
                width=width,
                height=height,
                order=order_value,
                enabled=bool(row.get("enabled", True)),
            ).clamp(image_width, image_height)
        )
    boxes.sort(key=lambda box: box.order)
    return boxes


def reset_detection_state() -> None:
    for key in (
        "boxes_df",
        "detected_orientation",
        "working_image",
        "preview",
        "preview_scale",
        "skew_angle",
        "crop_editor",
        "export_path",
        "export_message",
        "export_signature",
    ):
        st.session_state.pop(key, None)


st.title("Film Scan Frame Splitter")

with st.sidebar:
    st.header("Detection")
    detection_method = st.selectbox(
        "Detection method",
        ["Black border pitch", "Regular 3:2 grid", "Frame bands", "Auto", "Contours"],
        index=0,
    )
    orientation_label = st.selectbox("Strip direction", ["Auto", "Vertical", "Horizontal"], index=0)
    lane_label = st.selectbox("Parallel lanes", ["Auto", 1, 2, 3, 4], index=0)
    lane_count = 0 if lane_label == "Auto" else int(lane_label)
    aspect_mode = st.selectbox("Frame aspect", ["Landscape 3:2", "Portrait 2:3"], index=0)
    expected_frames_per_lane = st.number_input(
        "Frames per lane (0 = variable gaps)",
        min_value=0,
        max_value=80,
        value=0,
        step=1,
    )
    black_threshold = st.slider("Black border threshold", 20, 140, 70, 5)
    base_sensitivity = st.slider("Frame/base sensitivity", 6, 80, 18, 2)
    preview_max_side = st.slider("Preview max side", 900, 2600, 2600, 100)
    min_area = st.slider("Minimum frame area (%)", 0.1, 8.0, 0.8, 0.1)
    padding = st.slider("Crop padding (%)", -8.0, 12.0, -1.0, 0.5)
    auto_deskew = st.checkbox("Auto-deskew before detection", value=False)

    st.header("Export")
    export_label = st.radio(
        "Format",
        ["Original quality / lossless TIFF", "High-quality JPEG", "PNG"],
        index=0,
    )
    export_format = {
        "Original quality / lossless TIFF": "TIFF",
        "High-quality JPEG": "JPEG",
        "PNG": "PNG",
    }[export_label]
    trim_borders = st.checkbox("Trim solid borders", value=False)
    invert_negative = st.checkbox("Invert color or B&W negative", value=False)
    normalize_negative = st.checkbox("Basic negative color correction", value=False, disabled=not invert_negative)

uploaded = st.file_uploader(
    "Upload one uncut film scan",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    accept_multiple_files=False,
    on_change=reset_detection_state,
)

if uploaded is None:
    st.info("Upload a JPG, PNG, or TIFF film-strip scan to begin.")
    st.stop()

upload_size_mb = getattr(uploaded, "size", 0) / (1024 * 1024)
if upload_size_mb >= LARGE_UPLOAD_MB:
    st.warning(
        f"This file is about {upload_size_mb:.0f} MB. Large TIFF export can take a while, "
        "so the app will save the ZIP to the local outputs folder when you click Generate ZIP."
    )

original = open_uploaded_image(uploaded)
image_for_detection = original
skew_angle = 0.0

if auto_deskew:
    quick_preview, _ = make_preview(original, max_side=preview_max_side)
    skew_angle = estimate_global_skew_degrees(quick_preview)
    image_for_detection = rotate_image_for_detection(original, skew_angle)

preview, preview_scale = make_preview(image_for_detection, max_side=preview_max_side)

run_detection = st.button("Detect Frames", type="primary")
if run_detection:
    for key in ("crop_editor", "export_path", "export_message", "export_signature"):
        st.session_state.pop(key, None)

if run_detection or "boxes_df" not in st.session_state:
    orientation_override = {
        "Auto": None,
        "Vertical": "vertical",
        "Horizontal": "horizontal",
    }[orientation_label]
    boxes, orientation = detect_frames(
        preview,
        preview_scale,
        min_frame_area_percent=min_area,
        padding_percent=padding,
        detection_method=detection_method,
        orientation_override=orientation_override,
        base_sensitivity=base_sensitivity,
        lane_count=lane_count,
        aspect_mode=aspect_mode,
        expected_frames_per_lane=expected_frames_per_lane,
        black_threshold=black_threshold,
    )
    boxes = [box.clamp(*image_for_detection.size) for box in boxes]
    st.session_state.boxes_df = boxes_to_dataframe(boxes)
    st.session_state.detected_orientation = orientation
    st.session_state.working_image = image_for_detection
    st.session_state.preview = preview
    st.session_state.preview_scale = preview_scale
    st.session_state.skew_angle = skew_angle

working_image = st.session_state.get("working_image", image_for_detection)
preview = st.session_state.get("preview", preview)
preview_scale = st.session_state.get("preview_scale", preview_scale)

left, right = st.columns([1.25, 1])

with right:
    st.subheader("Crop Boxes")
    if st.session_state.get("detected_orientation"):
        st.caption(
            f"Detected {st.session_state.detected_orientation}; "
            f"working size {working_image.size[0]} x {working_image.size[1]} px"
        )
    if auto_deskew and abs(st.session_state.get("skew_angle", 0.0)) >= 0.05:
        st.caption(f"Deskew applied for detection/export: {st.session_state.skew_angle:.2f} degrees")

    edited_df = st.data_editor(
        st.session_state.boxes_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "enabled": st.column_config.CheckboxColumn("enabled"),
            "order": st.column_config.NumberColumn("order", min_value=1, step=1),
            "x": st.column_config.NumberColumn("x", min_value=0, step=1),
            "y": st.column_config.NumberColumn("y", min_value=0, step=1),
            "width": st.column_config.NumberColumn("width", min_value=1, step=1),
            "height": st.column_config.NumberColumn("height", min_value=1, step=1),
        },
        key="crop_editor",
    )
    st.session_state.boxes_df = edited_df

    boxes = dataframe_to_boxes(edited_df, *working_image.size)
    enabled_count = sum(1 for box in boxes if box.enabled)
    export_signature = (
        tuple((box.x, box.y, box.width, box.height, box.order, box.enabled) for box in boxes),
        export_format,
        trim_borders,
        invert_negative,
        normalize_negative,
    )
    st.caption(f"{enabled_count} frame(s) enabled for export")

    if enabled_count:
        if st.button("Generate ZIP", type="primary", use_container_width=True):
            with st.spinner("Cropping full-resolution frames and writing ZIP..."):
                output_path = OUTPUT_DIR / "film_frames.zip"
                export_zip_file(
                    working_image,
                    boxes,
                    output_path,
                    fmt=export_format,
                    trim_borders=trim_borders,
                    invert=invert_negative,
                    normalize_negative=normalize_negative,
                )
                st.session_state.export_path = str(output_path.resolve())
                size_mb = output_path.stat().st_size / (1024 * 1024)
                st.session_state.export_message = f"Saved ZIP ({size_mb:.1f} MB)"
                st.session_state.export_signature = export_signature

    export_path_value = st.session_state.get("export_path")
    if export_path_value and st.session_state.get("export_signature") == export_signature:
        export_path = Path(export_path_value)
        if export_path.exists():
            st.success(f"{st.session_state.get('export_message', 'Saved ZIP')}: {export_path}")
            with export_path.open("rb") as zip_file:
                st.download_button(
                    "Download ZIP",
                    data=zip_file,
                    file_name=export_path.name,
                    mime="application/zip",
                    use_container_width=True,
                )

with left:
    st.subheader("Preview")
    boxes = dataframe_to_boxes(st.session_state.boxes_df, *working_image.size)
    overlay = draw_boxes_on_preview(preview, boxes, preview_scale)
    st.image(overlay, use_container_width=True)
