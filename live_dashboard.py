
import csv
import json
import os
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import av
import cv2
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

from src.track_vehicles import (
    CONFIDENCE,
    IMAGE_SIZE,
    LINE_HEIGHT_RATIO,
    LINE_TOLERANCE,
)

st.set_page_config(
    page_title="Road Vision | Live Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "yolo26n.pt"
TRACKER_PATH = ROOT / "bytetrack_live.yaml"
DATA_DIR = ROOT / "data" / "live_session"
EVENTS_CSV = DATA_DIR / "live_events.csv"
SUMMARY_JSON = DATA_DIR / "live_summary.json"

VEHICLE_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
VEHICLE_COLORS = {
    "Car": (60, 180, 75),
    "Motorcycle": (255, 165, 0),
    "Bus": (190, 90, 220),
    "Truck": (60, 130, 240),
}

LIVE_CONFIDENCE = min(float(CONFIDENCE), 0.10)
LIVE_IMAGE_SIZE = max(int(IMAGE_SIZE), 640)
LIVE_TRACK_BUFFER = 90

SETTINGS_LOCK = threading.Lock()
SETTINGS = {"line_ratio": LINE_HEIGHT_RATIO}
STORE_LOCK = threading.RLock()

EVENT_COLUMNS = [
    "Time", "Frame", "Track ID", "Vehicle", "Direction", "Confidence"
]


def resolve_model_path():
    """Use the local model or ask Ultralytics to download the model asset."""
    if MODEL_PATH.exists():
        return MODEL_PATH

    try:
        downloaded = attempt_download_asset("yolo26n.pt")
        downloaded = Path(downloaded)

        if downloaded.exists():
            return downloaded

    except Exception as exc:
        raise RuntimeError(
            "Could not download yolo26n.pt. Check network access or provide "
            "the model file through your deployment's file storage."
        ) from exc

    raise FileNotFoundError(
        "yolo26n.pt was not found locally and could not be downloaded."
    )


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1440px;
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,.18);
    }
    .section-note {
        color: var(--text-color);
        opacity: .72;
        font-size: .9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def empty_summary(device="CPU", active=False):
    return {
        "total": 0,
        "top_to_bottom": 0,
        "bottom_to_top": 0,
        "classes": {name: 0 for name in VEHICLE_NAMES.values()},
        "fps": 0.0,
        "device": device,
        "active": active,
        "events": [],
        "updated_at": None,
    }


def write_summary_file(summary):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = SUMMARY_JSON.with_suffix(".tmp")
    temp_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temp_path.replace(SUMMARY_JSON)


def ensure_storage():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not EVENTS_CSV.exists():
        with EVENTS_CSV.open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            csv.DictWriter(file, fieldnames=EVENT_COLUMNS).writeheader()

    if not SUMMARY_JSON.exists():
        write_summary_file(empty_summary())


def ensure_tracker_config():
    TRACKER_PATH.write_text(
        "tracker_type: bytetrack\n"
        "track_high_thresh: 0.20\n"
        "track_low_thresh: 0.05\n"
        "new_track_thresh: 0.20\n"
        f"track_buffer: {LIVE_TRACK_BUFFER}\n"
        "match_thresh: 0.80\n"
        "fuse_score: True\n",
        encoding="utf-8",
    )


def read_events():
    ensure_storage()

    try:
        df = pd.read_csv(EVENTS_CSV)

        for column in EVENT_COLUMNS:
            if column not in df.columns:
                df[column] = ""

        return df[EVENT_COLUMNS].fillna("")

    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame(columns=EVENT_COLUMNS)


def summarize_events(df, fps=0.0, device="CPU", active=False):
    if df is None or df.empty:
        top = bottom = total = 0
        counts = {name: 0 for name in VEHICLE_NAMES.values()}
        records = []

    else:
        directions = df["Direction"].astype(str)

        top = int((directions == "Top to bottom").sum())
        bottom = int((directions == "Bottom to top").sum())
        total = top + bottom

        vehicle_counts = df["Vehicle"].astype(str).value_counts().to_dict()
        counts = {
            name: int(vehicle_counts.get(name, 0))
            for name in VEHICLE_NAMES.values()
        }

        records = df.to_dict("records")

    return {
        "total": total,
        "top_to_bottom": top,
        "bottom_to_top": bottom,
        "classes": counts,
        "fps": round(float(fps), 1),
        "device": device,
        "active": bool(active),
        "events": records,
    }


def persist_event(row):
    ensure_storage()

    with STORE_LOCK:
        with EVENTS_CSV.open(
            "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=EVENT_COLUMNS)
            writer.writerow({
                key: row.get(key, "")
                for key in EVENT_COLUMNS
            })
            file.flush()
            os.fsync(file.fileno())

        summary = summarize_events(read_events())
        summary["updated_at"] = datetime.now().isoformat(
            timespec="seconds"
        )
        write_summary_file(summary)


def clear_saved_events():
    ensure_storage()

    with STORE_LOCK:
        with EVENTS_CSV.open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            csv.DictWriter(file, fieldnames=EVENT_COLUMNS).writeheader()

        summary = empty_summary()
        summary["updated_at"] = datetime.now().isoformat(
            timespec="seconds"
        )
        write_summary_file(summary)


ensure_storage()


class RoadVisionProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.RLock()
        self.device = "0" if torch.cuda.is_available() else "cpu"

        ensure_tracker_config()
        self.model = YOLO(str(resolve_model_path()))

        self.previous_side = {}
        self.counted_events = set()

        saved = read_events()
        restored = summarize_events(saved, device=self.device)

        self.direction_counts = Counter({
            "Top to bottom": restored["top_to_bottom"],
            "Bottom to top": restored["bottom_to_top"],
        })
        self.class_counts = Counter(restored["classes"])
        self.event_rows = saved.to_dict("records")

        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        self.active = True

    def get_line_ratio(self):
        with SETTINGS_LOCK:
            return float(SETTINGS["line_ratio"])

    def get_stats(self):
        with self.lock:
            return summarize_events(
                pd.DataFrame(self.event_rows, columns=EVENT_COLUMNS),
                fps=self.fps,
                device="CUDA GPU" if self.device == "0" else "CPU",
                active=self.active,
            )

    def reset_stats(self):
        with self.lock:
            self.previous_side.clear()
            self.counted_events.clear()

            self.direction_counts = Counter({
                "Top to bottom": 0,
                "Bottom to top": 0,
            })
            self.class_counts = Counter({
                name: 0 for name in VEHICLE_NAMES.values()
            })
            self.event_rows.clear()
            self.frame_count = 0
            self.fps = 0.0
            self.last_fps_time = time.time()

        clear_saved_events()

    @staticmethod
    def side_of_line(y, line_y, tolerance):
        if y < line_y - tolerance:
            return -1
        if y > line_y + tolerance:
            return 1
        return 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        height, width = image.shape[:2]

        line_y = int(height * self.get_line_ratio())
        tolerance = max(
            int(LINE_TOLERANCE),
            int(height * 0.008),
        )

        cv2.line(
            image, (0, line_y), (width, line_y), (0, 255, 255), 3
        )
        cv2.putText(
            image,
            "COUNTING LINE",
            (16, max(28, line_y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        try:
            results = self.model.track(
                image,
                persist=True,
                tracker=str(TRACKER_PATH),
                classes=list(VEHICLE_NAMES.keys()),
                conf=LIVE_CONFIDENCE,
                imgsz=LIVE_IMAGE_SIZE,
                device=self.device,
                verbose=False,
            )

        except Exception as exc:
            cv2.putText(
                image,
                f"Detection error: {str(exc)[:70]}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )
            return av.VideoFrame.from_ndarray(
                image, format="bgr24"
            )

        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()

            track_ids = (
                boxes.id.cpu().numpy().astype(int)
                if boxes.id is not None
                else [-1] * len(xyxy)
            )

            for box, class_id, track_id, confidence in zip(
                xyxy, classes, track_ids, confidences
            ):
                if class_id not in VEHICLE_NAMES:
                    continue

                x1, y1, x2, y2 = map(int, box)
                name = VEHICLE_NAMES[class_id]
                color = VEHICLE_COLORS[name]
                anchor_x, anchor_y = (x1 + x2) // 2, y2

                cv2.rectangle(
                    image, (x1, y1), (x2, y2), color, 2
                )
                cv2.circle(
                    image, (anchor_x, anchor_y), 5, color, -1
                )

                label = (
                    f"{name} | ID {track_id} | {confidence:.2f}"
                    if track_id >= 0
                    else f"{name} | {confidence:.2f}"
                )

                cv2.putText(
                    image,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    color,
                    2,
                )

                if track_id < 0:
                    continue

                side = self.side_of_line(
                    anchor_y, line_y, tolerance
                )

                if side == 0:
                    continue

                event_row = None

                with self.lock:
                    old_side = self.previous_side.get(track_id)

                    if old_side is not None and old_side != side:
                        direction = (
                            "Top to bottom"
                            if old_side == -1 and side == 1
                            else "Bottom to top"
                            if old_side == 1 and side == -1
                            else None
                        )

                        key = (int(track_id), direction)

                        if (
                            direction
                            and key not in self.counted_events
                        ):
                            self.counted_events.add(key)
                            self.direction_counts[direction] += 1
                            self.class_counts[name] += 1

                            event_row = {
                                "Time": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                                "Frame": self.frame_count,
                                "Track ID": int(track_id),
                                "Vehicle": name,
                                "Direction": direction,
                                "Confidence": round(
                                    float(confidence), 3
                                ),
                            }
                            self.event_rows.append(event_row)

                    self.previous_side[track_id] = side

                if event_row is not None:
                    try:
                        persist_event(event_row)
                    except Exception as exc:
                        print(f"Could not save live event: {exc}")

        now = time.time()

        with self.lock:
            self.frame_count += 1
            elapsed = now - self.last_fps_time

            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.last_fps_time = now

            total = sum(self.direction_counts.values())
            top = self.direction_counts["Top to bottom"]
            bottom = self.direction_counts["Bottom to top"]
            fps = self.fps

        cv2.rectangle(
            image, (10, 10), (335, 112), (25, 25, 25), -1
        )
        cv2.putText(
            image, f"Crossings: {total}", (22, 38),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.putText(
            image, f"Down: {top}  Up: {bottom}", (22, 67),
            cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2,
        )
        cv2.putText(
            image, f"FPS: {fps:.1f} | conf: {LIVE_CONFIDENCE:.2f}",
            (22, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (200, 220, 200), 1,
        )

        return av.VideoFrame.from_ndarray(
            image, format="bgr24"
        )


# ----------------------------- Dashboard UI -----------------------------

st.title("Road Vision Agent")
st.caption(
    "Live traffic monitoring · Vehicle tracking · "
    "Two-way crossing analytics"
)

left, right = st.columns([3, 1])

with left:
    st.markdown(
        '<p class="section-note">Events are saved to this server’s '
        'local storage. Hosted storage may not survive restarts.</p>',
        unsafe_allow_html=True,
    )

with right:
    st.caption(
        "GPU enabled" if torch.cuda.is_available()
        else "CPU processing"
    )

with st.container(border=True):
    control_col, info_col = st.columns([2, 3])

    with control_col:
        default_percent = int(
            st.session_state.get(
                "line_ratio", LINE_HEIGHT_RATIO
            ) * 100
        )

        line_percent = st.slider(
            "Counting line position",
            min_value=10,
            max_value=90,
            value=default_percent,
            step=1,
            help="Move the line to change where crossings are counted.",
            key="line_position_slider",
        )

        st.session_state["line_ratio"] = line_percent / 100.0

        with SETTINGS_LOCK:
            SETTINGS["line_ratio"] = line_percent / 100.0

    with info_col:
        st.markdown("**Counting rules**")
        st.caption(
            "Both directions are counted separately. Each track is "
            "counted once per direction during a tracker session."
        )
        st.caption(
            f"Live confidence threshold: {LIVE_CONFIDENCE:.2f}. "
            "A low threshold can increase false positives."
        )

camera_col, analytics_col = st.columns(
    [1.7, 1], gap="large"
)

with camera_col:
    st.subheader("Live camera")

    webrtc_ctx = webrtc_streamer(
        key="road-vision-live-persistent-v3",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=RoadVisionProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    st.caption(
        "Allow camera access when prompted. Keep this tab open "
        "while monitoring."
    )

with analytics_col:
    st.subheader("Live analytics")
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    report_placeholder = st.empty()

    @st.fragment(run_every="1s")
    def render_live_analytics():
        processor = webrtc_ctx.video_processor

        if processor is not None:
            stats = processor.get_stats()
            events = pd.DataFrame(
                stats["events"], columns=EVENT_COLUMNS
            )
            status = "Camera running"
        else:
            events = read_events()
            stats = summarize_events(
                events, device="Saved session"
            )
            status = "Camera stopped · showing saved results"

        with metrics_placeholder.container():
            st.caption(status)

            m1, m2 = st.columns(2)
            m1.metric("Total crossings", stats["total"])
            m2.metric(
                "Processing FPS",
                stats["fps"] if processor is not None else "—",
            )

            m3, m4 = st.columns(2)
            m3.metric("Top → bottom", stats["top_to_bottom"])
            m4.metric("Bottom → top", stats["bottom_to_top"])

            st.caption(
                f"Processing device: "
                f"{stats.get('device', 'Saved session')}"
            )

        chart_data = pd.DataFrame([
            {
                "Vehicle type": name,
                "Crossings": int(
                    stats["classes"].get(name, 0)
                ),
            }
            for name in ["Car", "Motorcycle", "Bus", "Truck"]
        ])

        with chart_placeholder.container():
            st.markdown("**Vehicle distribution**")

            if int(chart_data["Crossings"].sum()) > 0:
                fig = px.bar(
                    chart_data,
                    x="Vehicle type",
                    y="Crossings",
                    text="Crossings",
                )
                fig.update_layout(
                    height=270,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                    xaxis_title=None,
                    yaxis_title="Crossings",
                )
                fig.update_traces(textposition="outside")
                st.plotly_chart(
                    fig, use_container_width=True
                )
            else:
                st.caption(
                    "Vehicle distribution will appear after crossings."
                )

        with report_placeholder.container():
            st.markdown("**Crossing event log**")

            if not events.empty:
                st.dataframe(
                    events.tail(20).iloc[::-1],
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "Download live CSV",
                    data=events.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name="road_vision_live_events.csv",
                    mime="text/csv",
                    key="download_live_events_persistent",
                )
            else:
                st.caption("No crossings recorded yet.")

    render_live_analytics()

st.divider()

reset_col, reset_info = st.columns([1, 3])

with reset_col:
    if st.button(
        "Reset saved live records", type="secondary"
    ):
        processor = webrtc_ctx.video_processor

        if processor is not None:
            processor.reset_stats()
        else:
            clear_saved_events()

        st.success("Saved live records cleared.")
        st.rerun()

with reset_info:
    st.caption(
        "Use this to start a fresh report. Files on hosted services "
        "may not persist across restarts or redeployments."
    )

st.divider()
st.caption(
    "Road Vision Agent · Counts depend on visibility, tracking "
    "stability, camera angle, frame rate, and counting-line placement."
)
