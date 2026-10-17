
from pathlib import Path
import logging
import os
import re
import secrets
import subprocess
import uuid

import imageio_ffmpeg
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from src.track_vehicles import analyze_video


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Road Vision Agent API",
    description="Vehicle detection and traffic-crossing analysis",
    version="2.1.0",
)

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "input"
OUTPUT_DIR = ROOT / "data" / "output"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
JOB_ID_PATTERN = re.compile(r"[a-f0-9]{12}")


# ---------------------------------------------------------
# API KEY AUTHENTICATION
# ---------------------------------------------------------

def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> bool:
    """Require a valid API key for protected endpoints."""

    expected_key = os.getenv("ROAD_VISION_API_KEY")

    if not expected_key:
        logger.error("ROAD_VISION_API_KEY is not configured.")
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )

    if not x_api_key or not secrets.compare_digest(
        x_api_key,
        expected_key,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return True


# ---------------------------------------------------------
# PUBLIC ENDPOINTS
# ---------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Road Vision Agent API is running",
        "docs": "/docs",
        "status": "healthy",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# ---------------------------------------------------------
# VIDEO CONVERSION
# ---------------------------------------------------------

def convert_to_h264(input_path: Path, output_path: Path) -> None:
    """Convert the annotated video to browser-compatible H.264 MP4."""

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(output_path),
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )

    if completed.returncode != 0 or not output_path.is_file():
        # Keep detailed FFmpeg output in server logs, not API responses.
        logger.error(
            "FFmpeg conversion failed: %s",
            completed.stderr[-2000:],
        )
        raise RuntimeError("Video conversion failed.")


# ---------------------------------------------------------
# PROTECTED VIDEO ANALYSIS ENDPOINT
# ---------------------------------------------------------

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    _: bool = Depends(verify_api_key),
):
    original_filename = file.filename or "uploaded_video.mp4"
    extension = Path(original_filename).suffix.lower()

    if extension not in VIDEO_EXTENSIONS:
        await file.close()
        raise HTTPException(
            status_code=400,
            detail="Upload a video in MP4, AVI, or MOV format.",
        )

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    job_id = uuid.uuid4().hex[:12]

    input_path = INPUT_DIR / f"{job_id}{extension}"
    raw_video_path = OUTPUT_DIR / f"{job_id}_raw.mp4"
    video_path = OUTPUT_DIR / f"{job_id}_tracked.mp4"
    csv_path = OUTPUT_DIR / f"{job_id}_crossings.csv"

    size = 0

    # Save the uploaded file in chunks to enforce the size limit.
    try:
        with input_path.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                size += len(chunk)

                if size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the 100 MB upload limit.",
                    )

                destination.write(chunk)

    except HTTPException:
        input_path.unlink(missing_ok=True)
        raise

    except Exception as exc:
        logger.exception("Could not save uploaded video.")
        input_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="Could not save the uploaded video.",
        ) from exc

    finally:
        await file.close()

    # Analyze the video and convert the annotated result.
    try:
        result = await run_in_threadpool(
            analyze_video,
            input_path,
            raw_video_path,
            csv_path,
        )

        await run_in_threadpool(
            convert_to_h264,
            raw_video_path,
            video_path,
        )

        raw_video_path.unlink(missing_ok=True)

        return {
            "job_id": job_id,
            "status": "completed",
            "original_filename": original_filename,
            "size_bytes": size,
            "total_counted": result["total_counted"],
            "vehicle_counts": result["vehicle_counts"],
            "top_to_bottom_count": result["top_to_bottom_count"],
            "bottom_to_top_count": result["bottom_to_top_count"],
            "frames_processed": result["frames_processed"],
            "fps": result["fps"],
            "counting_line_y": result["line_y"],
            "device": result["device"],
            "annotated_video_url": f"/files/{job_id}/video",
            "crossing_report_url": f"/files/{job_id}/csv",
        }

    except Exception as exc:
        logger.exception("Video analysis or conversion failed.")

        # Remove incomplete output files after a failed job.
        raw_video_path.unlink(missing_ok=True)
        video_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="Video analysis or conversion failed. Check server logs.",
        ) from exc

    finally:
        # The original upload is no longer needed after processing.
        input_path.unlink(missing_ok=True)


# ---------------------------------------------------------
# PROTECTED RESULT DOWNLOAD ENDPOINT
# ---------------------------------------------------------

@app.get("/files/{job_id}/{kind}")
def download_result(
    job_id: str,
    kind: str,
    _: bool = Depends(verify_api_key),
):
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid job ID.",
        )

    if kind == "video":
        path = OUTPUT_DIR / f"{job_id}_tracked.mp4"
        media_type = "video/mp4"

    elif kind == "csv":
        path = OUTPUT_DIR / f"{job_id}_crossings.csv"
        media_type = "text/csv"

    else:
        raise HTTPException(
            status_code=400,
            detail="File type must be video or csv.",
        )

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Result file not found.",
        )

    return FileResponse(
        path=path,
        media_type=media_type,
        filename=path.name,
    )


# ---------------------------------------------------------
# LOCAL DEVELOPMENT ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
