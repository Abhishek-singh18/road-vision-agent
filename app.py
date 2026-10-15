
import io
import json

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# =========================================================
# Configuration
# =========================================================

st.set_page_config(
    page_title="Road Vision | Video Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_API_URL = "http://127.0.0.1:8000"
MAX_UPLOAD_MB = 100

VEHICLE_NAMES = ["Car", "Motorcycle", "Bus", "Truck"]

VEHICLE_ALIASES = {
    "car": "Car",
    "cars": "Car",
    "2": "Car",
    "motorcycle": "Motorcycle",
    "motorcycles": "Motorcycle",
    "motorbike": "Motorcycle",
    "motorbikes": "Motorcycle",
    "3": "Motorcycle",
    "bus": "Bus",
    "buses": "Bus",
    "5": "Bus",
    "truck": "Truck",
    "trucks": "Truck",
    "7": "Truck",
}


# =========================================================
# Styling
# =========================================================

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
        border: 1px solid rgba(128, 128, 128, .18);
    }

    .section-note {
        opacity: .72;
        font-size: .9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Helper functions
# =========================================================

def normalize_label(value):
    """Normalize text for reliable comparisons."""
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace("→", "to")
    )


def normalize_vehicle_name(value):
    """Convert common vehicle labels or COCO class IDs to display names."""
    if value is None:
        return None

    return VEHICLE_ALIASES.get(str(value).strip().lower())


def first_present(mapping, keys, default=None):
    """Return the first non-None field in a dictionary."""
    if not isinstance(mapping, dict):
        return default

    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value

    return default


def to_int_or_none(value):
    """Safely convert a value to an integer."""
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def parse_api_summary(payload):
    """Read summary fields returned by the FastAPI backend."""
    if not isinstance(payload, dict):
        return {}

    nested = payload.get("summary")
    if not isinstance(nested, dict):
        nested = payload.get("results")

    summary = nested if isinstance(nested, dict) else payload

    raw_classes = first_present(
        summary,
        [
            "vehicle_counts",
            "counts_by_class",
            "class_counts",
            "counts",
        ],
        default={},
    )

    classes = {}

    if isinstance(raw_classes, dict):
        for key, value in raw_classes.items():
            vehicle = normalize_vehicle_name(key)
            count = to_int_or_none(value)

            if vehicle is not None and count is not None:
                classes[vehicle] = count

    return {
        "total": to_int_or_none(
            first_present(
                summary,
                [
                    "total_counted",
                    "total_crossings",
                    "total_crossed",
                    "total_vehicles",
                    "total_count",
                    "total",
                ],
            )
        ),
        "classes": classes,
        "top_to_bottom": to_int_or_none(
            first_present(
                summary,
                [
                    "top_to_bottom_count",
                    "top_to_bottom",
                    "top_to_bottom_total",
                    "down_count",
                ],
            )
        ),
        "bottom_to_top": to_int_or_none(
            first_present(
                summary,
                [
                    "bottom_to_top_count",
                    "bottom_to_top",
                    "bottom_to_top_total",
                    "up_count",
                ],
            )
        ),
    }


def find_column(dataframe, candidates):
    """Find a dataframe column without depending on letter case or spaces."""
    if dataframe is None or dataframe.empty:
        return None

    lookup = {
        normalize_label(column): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        normalized = normalize_label(candidate)

        if normalized in lookup:
            return lookup[normalized]

    return None


def direction_value(directions, desired):
    """
    Match direction labels including:
    top_to_bottom, top-to-bottom, top to bottom, and top→bottom.
    """
    target = normalize_label(desired)
    total = 0

    for key, value in directions.items():
        if normalize_label(key) == target:
            count = to_int_or_none(value)

            if count is not None:
                total += count

    return total


def summarize_csv(dataframe):
    """
    Summarize the crossing-event CSV.

    The backend CSV contains one row per counted crossing event.
    Therefore, rows must not be deduplicated by track ID: the same
    track may legitimately cross in both directions.
    """
    if dataframe is None or dataframe.empty:
        return {
            "total": 0,
            "classes": {},
            "directions": {},
            "vehicle_column": None,
            "direction_column": None,
            "is_event_log": False,
        }

    vehicle_column = find_column(
        dataframe,
        [
            "vehicle_type",
            "vehicle",
            "class_name",
            "vehicle_class",
            "class",
            "object_type",
            "label",
        ],
    )

    direction_column = find_column(
        dataframe,
        [
            "direction",
            "crossing_direction",
            "movement",
            "travel_direction",
        ],
    )

    # For this project, the CSV is an event log with one row per crossing.
    # Count rows directly when the event fields are available.
    is_event_log = (
        vehicle_column is not None
        and direction_column is not None
    )

    classes = {}

    if vehicle_column is not None:
        normalized_vehicles = dataframe[vehicle_column].map(
            normalize_vehicle_name
        )

        classes = {
            str(name): int(count)
            for name, count in normalized_vehicles.value_counts().items()
            if name is not None
        }

    directions = {}

    if direction_column is not None:
        direction_series = (
            dataframe[direction_column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

        directions = {
            str(name): int(count)
            for name, count in direction_series.value_counts().items()
        }

    # One row equals one crossing event in this project's CSV format.
    if is_event_log:
        total = int(len(dataframe))
    elif classes:
        total = int(sum(classes.values()))
    else:
        total = int(len(dataframe))

    return {
        "total": total,
        "classes": classes,
        "directions": directions,
        "vehicle_column": vehicle_column,
        "direction_column": direction_column,
        "is_event_log": is_event_log,
    }


def request_file(url, timeout=120):
    """Download a result file from FastAPI."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def safe_job_id(payload):
    """Return the job ID if the API response contains one."""
    if not isinstance(payload, dict):
        return None

    job_id = payload.get("job_id")

    if job_id is None:
        return None

    return str(job_id)


# =========================================================
# Page header and API settings
# =========================================================

st.title("Road Vision Agent")
st.caption(
    "Video analysis · Vehicle counting · Traffic reports"
)

with st.container(border=True):
    api_col, help_col = st.columns([2, 3])

    with api_col:
        api_url = st.text_input(
            "FastAPI server URL",
            value=st.session_state.get(
                "road_vision_api_url",
                DEFAULT_API_URL,
            ),
            help="Address where src.api is running.",
        ).strip().rstrip("/")

        st.session_state["road_vision_api_url"] = api_url

    with help_col:
        st.markdown("**Supported video formats**")
        st.caption(
            f"MP4, AVI and MOV · Maximum upload size: {MAX_UPLOAD_MB} MB"
        )


# =========================================================
# API health check
# =========================================================

health_col, status_col = st.columns([1, 3])

with health_col:
    check_api = st.button("Check API connection")

if check_api:
    try:
        response = requests.get(
            f"{api_url}/health",
            timeout=5,
        )
        response.raise_for_status()

        st.session_state["road_vision_api_health"] = (
            "connected",
            "FastAPI is reachable.",
        )

    except requests.RequestException as exc:
        st.session_state["road_vision_api_health"] = (
            "error",
            str(exc),
        )

health = st.session_state.get("road_vision_api_health")

with status_col:
    if health:
        if health[0] == "connected":
            st.success(health[1])
        else:
            st.error(f"Could not connect to FastAPI: {health[1]}")


st.divider()


# =========================================================
# Upload and analyze
# =========================================================

st.subheader("Analyze a traffic video")

uploaded_file = st.file_uploader(
    "Choose a road video",
    type=["mp4", "avi", "mov"],
    help=(
        "For best results, use a stable traffic video with "
        "a clearly visible crossing line."
    ),
)

if uploaded_file is not None:
    upload_signature = (
        uploaded_file.name,
        uploaded_file.size,
    )

    previous_signature = st.session_state.get(
        "road_vision_upload_signature"
    )

    if previous_signature != upload_signature:
        st.session_state["road_vision_upload_signature"] = (
            upload_signature
        )
        st.session_state.pop("road_vision_result", None)

    if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(
            f"Your video exceeds {MAX_UPLOAD_MB} MB. "
            "Choose a smaller video."
        )

    else:
        preview_col, detail_col = st.columns(
            [1.3, 1],
            gap="large",
        )

        with preview_col:
            st.markdown("**Selected video**")
            st.video(uploaded_file)

        with detail_col:
            st.markdown("**File details**")
            st.write(f"**Name:** {uploaded_file.name}")
            st.write(
                f"**Size:** {uploaded_file.size / (1024 * 1024):.2f} MB"
            )
            st.caption(
                "YOLO detects vehicles and ByteTrack follows their "
                "IDs. Analysis time depends on video length, "
                "resolution and hardware."
            )

        analyze = st.button(
            "Run traffic analysis",
            type="primary",
            disabled=(
                uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024
            ),
        )

        if analyze:
            try:
                with st.spinner(
                    "Uploading video and running YOLO + ByteTrack..."
                ):
                    uploaded_file.seek(0)

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                            or "application/octet-stream",
                        )
                    }

                    response = requests.post(
                        f"{api_url}/analyze",
                        files=files,
                        timeout=(15, 1800),
                    )

                    response.raise_for_status()

                    try:
                        payload = response.json()
                    except ValueError:
                        payload = {"message": response.text}

                    job_id = safe_job_id(payload)

                    if not job_id:
                        st.error(
                            "The API response did not contain a job_id. "
                            "Check src/api.py."
                        )
                        st.code(
                            json.dumps(
                                payload,
                                indent=2,
                                default=str,
                            )
                        )

                    else:
                        st.session_state["road_vision_result"] = {
                            "job_id": job_id,
                            "api_url": api_url,
                            "payload": payload,
                            "upload_name": uploaded_file.name,
                        }

                        st.success("Analysis completed successfully.")

            except requests.Timeout:
                st.error(
                    "The request timed out. Check whether the "
                    "FastAPI server is still processing the video."
                )

            except requests.RequestException as exc:
                st.error(f"Could not complete the analysis: {exc}")
                st.caption(
                    "Confirm FastAPI is running and the server URL is correct."
                )


# =========================================================
# Results
# =========================================================

result = st.session_state.get("road_vision_result")

if result:
    job_id = result["job_id"]
    result_api_url = result.get("api_url", api_url)
    payload = result.get("payload", {})

    api_summary = parse_api_summary(payload)

    st.divider()
    st.subheader("Analysis results")
    st.caption(
        f"Source: {result.get('upload_name', 'Uploaded video')}"
    )

    # -----------------------------------------------------
    # Load CSV
    # -----------------------------------------------------

    csv_bytes = None
    csv_df = pd.DataFrame()
    csv_error = None

    try:
        csv_bytes = request_file(
            f"{result_api_url}/files/{job_id}/csv",
            timeout=60,
        )

        csv_df = pd.read_csv(io.BytesIO(csv_bytes))

    except Exception as exc:
        csv_error = str(exc)

    csv_summary = (
        summarize_csv(csv_df)
        if not csv_df.empty
        else None
    )

    # -----------------------------------------------------
    # Calculate consistent summary metrics
    # -----------------------------------------------------

    # Prefer CSV event data because it directly represents the
    # crossing records displayed in the report.
    if csv_summary and csv_summary["vehicle_column"]:
        classes = csv_summary["classes"]
    else:
        classes = api_summary.get("classes", {})

    if csv_summary and csv_summary["is_event_log"]:
        total = csv_summary["total"]
    elif api_summary.get("total") is not None:
        total = api_summary["total"]
    elif csv_summary:
        total = csv_summary["total"]
    else:
        total = sum(classes.values())

    directions = (
        csv_summary["directions"]
        if csv_summary
        else {}
    )

    # Initialize from API summary.
    top_count = api_summary.get("top_to_bottom")
    bottom_count = api_summary.get("bottom_to_top")

    # Correctly normalize underscores, hyphens, spaces and arrows.
    if csv_summary and csv_summary["direction_column"]:
        top_count = direction_value(
            directions,
            "top_to_bottom",
        )
        bottom_count = direction_value(
            directions,
            "bottom_to_top",
        )

    # -----------------------------------------------------
    # Summary cards
    # -----------------------------------------------------

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Total crossings", int(total))
    k2.metric("Cars", int(classes.get("Car", 0)))
    k3.metric("Motorcycles", int(classes.get("Motorcycle", 0)))
    k4.metric(
        "Trucks + buses",
        int(classes.get("Truck", 0) + classes.get("Bus", 0)),
    )

    d1, d2 = st.columns(2)

    d1.metric(
        "Top → bottom",
        top_count if top_count is not None else "Not provided",
    )

    d2.metric(
        "Bottom → top",
        bottom_count if bottom_count is not None else "Not provided",
    )

    # -----------------------------------------------------
    # Charts and annotated video
    # -----------------------------------------------------

    chart_col, output_col = st.columns(
        [1, 1],
        gap="large",
    )

    with chart_col:
        st.markdown("**Vehicle-type distribution**")

        chart_df = pd.DataFrame([
            {
                "Vehicle": name,
                "Count": int(classes.get(name, 0)),
            }
            for name in VEHICLE_NAMES
        ])

        if chart_df["Count"].sum() > 0:
            fig = px.bar(
                chart_df,
                x="Vehicle",
                y="Count",
                text="Count",
            )

            fig.update_layout(
                height=330,
                margin=dict(
                    l=10,
                    r=10,
                    t=15,
                    b=10,
                ),
                showlegend=False,
                xaxis_title=None,
                yaxis_title="Crossings",
            )

            fig.update_traces(textposition="outside")

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:
            st.info(
                "No vehicle-class breakdown was found "
                "in the CSV or API response."
            )

        if directions:
            st.markdown("**Crossings by direction**")

            direction_df = pd.DataFrame([
                {
                    "Direction": key,
                    "Count": int(value),
                }
                for key, value in directions.items()
            ])

            direction_fig = px.bar(
                direction_df,
                x="Direction",
                y="Count",
                text="Count",
            )

            direction_fig.update_layout(
                height=280,
                margin=dict(
                    l=10,
                    r=10,
                    t=15,
                    b=10,
                ),
                showlegend=False,
                xaxis_title=None,
                yaxis_title="Crossings",
            )

            direction_fig.update_traces(
                textposition="outside"
            )

            st.plotly_chart(
                direction_fig,
                use_container_width=True,
            )

    with output_col:
        st.markdown("**Annotated video**")

        try:
            video_bytes = request_file(
                f"{result_api_url}/files/{job_id}/video",
                timeout=180,
            )

            st.video(
                video_bytes,
                format="video/mp4",
            )

            st.download_button(
                "Download annotated video",
                data=video_bytes,
                file_name=f"road_vision_{job_id}.mp4",
                mime="video/mp4",
                key=f"download_video_{job_id}",
            )

        except Exception as exc:
            st.error(
                f"Could not load the annotated video: {exc}"
            )

        if csv_bytes is not None:
            st.download_button(
                "Download CSV report",
                data=csv_bytes,
                file_name=f"road_vision_{job_id}.csv",
                mime="text/csv",
                key=f"download_csv_{job_id}",
            )

        elif csv_error:
            st.warning(
                f"CSV report could not be loaded: {csv_error}"
            )

    # -----------------------------------------------------
    # CSV report table
    # -----------------------------------------------------

    if not csv_df.empty:
        st.divider()
        st.markdown("**Crossing report data**")

        st.dataframe(
            csv_df,
            use_container_width=True,
            hide_index=True,
        )

        # Data consistency check.
        if (
            csv_summary
            and csv_summary["direction_column"]
            and csv_summary["is_event_log"]
        ):
            directional_total = (
                int(top_count or 0)
                + int(bottom_count or 0)
            )

            if directional_total == int(total):
                st.success(
                    "Consistency check passed: directional counts "
                    "equal the total crossing-event count."
                )
            else:
                st.warning(
                    f"Consistency check: total crossings = {total}, "
                    f"directional sum = {directional_total}. "
                    "Check whether the CSV contains unknown or "
                    "unexpected direction labels."
                )

    # -----------------------------------------------------
    # Technical API response
    # -----------------------------------------------------

    with st.expander("Technical response from FastAPI"):
        st.json(payload)


# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    "Road Vision Agent · Phase 2 · Counts depend on detection "
    "quality, tracking stability, video quality and counting rules."
)
