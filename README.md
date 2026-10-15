# Road Vision Agent

Road Vision Agent is a computer-vision application for detecting, tracking, and counting vehicles in road videos. It provides an uploaded-video dashboard, a live-camera dashboard, and a FastAPI backend.

## Features

- Vehicle detection using Ultralytics YOLO.
- Multi-object tracking using ByteTrack.
- Virtual-line vehicle crossing counts.
- Vehicle categories: cars, motorcycles, buses, and trucks.
- Directional counts, annotated videos, and CSV reports.
- Uploaded-video processing through Streamlit.
- Live-camera processing through Streamlit-WebRTC.
- FastAPI endpoint for video-analysis requests.
- NVIDIA GPU acceleration when compatible CUDA-enabled PyTorch is available.

## Technology Stack

Python, Streamlit, Streamlit-WebRTC, FastAPI, Uvicorn, Ultralytics YOLO, ByteTrack, OpenCV, PyTorch, and Pandas.

## Project Structure

```text
road-vision-agent/
├── app.py
├── live_dashboard.py
├── live_dashboard.py.backup
├── live_dashboard_backup.py
├── live_dashboard_old.py
├── app.py.backup
├── requirements.txt
├── README.md
├── .gitignore
├── bytetrack_live.yaml
├── bytetrack_road.yaml
├── yolo26n.pt
├── data/
└── src/
    ├── api.py
    ├── api.py.backup
    ├── detect_vehicles.py
    ├── live_camera.py
    └── track_vehicles.py
```

The `.venv` virtual environment and generated files are local development resources and should not be committed to source control.

## Requirements

- Python and compatible project dependencies.
- A webcam and browser camera permission for live-camera use.
- A compatible NVIDIA GPU and CUDA-enabled PyTorch installation for NVIDIA GPU acceleration. CPU execution may be possible but slower.
- Internet access when dependencies or model weights need to be downloaded.

The development environment uses Python 3.13.9 and an NVIDIA RTX 3050 Laptop GPU.

## Installation

Open PowerShell in the project directory and activate the existing environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies when setting up or repairing an environment:

```powershell
python -m pip install -r requirements.txt
```

For a new machine, verify PyTorch and CUDA compatibility before installing a GPU-enabled PyTorch build. The existing environment's package versions may be platform-specific.

## Running the Application

Run each service in a separate terminal and keep the terminals open.

### 1. FastAPI backend

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Health endpoint:

http://127.0.0.1:8000/health

Interactive API documentation, if enabled:

http://127.0.0.1:8000/docs

### 2. Uploaded-video dashboard

```powershell
python -m streamlit run app.py --server.port 8505
```

Open:

http://localhost:8505

Select a supported video, check the API connection, and start analysis. Review the vehicle counts, annotated video, and available report.

The dashboard is configured to accept MP4, AVI, and MOV uploads with a maximum size of 100 MB.

### 3. Live-camera dashboard

```powershell
python -m streamlit run live_dashboard.py --server.port 8504
```

Open:

http://localhost:8504

Grant camera permission and follow the dashboard controls to start live processing. Camera access and remote WebRTC connectivity depend on browser and network configuration.

## API Usage

The backend accepts video uploads at:

`POST /analyze`

The multipart upload field is named `file`.

Example Python request:

```python
import requests

with open("road_video.mp4", "rb") as video:
    response = requests.post(
        "http://127.0.0.1:8000/analyze",
        files={"file": ("road_video.mp4", video, "video/mp4")},
        timeout=600,
    )

response.raise_for_status()
print(response.json())
```

The API returns analysis information and result URLs. The project provides these result routes:

- `/files/{job_id}/video`
- `/files/{job_id}/csv`

Use the job ID returned by the API.

## Outputs

Depending on the workflow, outputs can include annotated videos, CSV crossing-event reports, vehicle-type totals, directional totals, and processing statistics. The live dashboard can also persist session records.

Generated output locations and filenames depend on the application configuration and analysis job.

## GPU Verification

Run this command in the active virtual environment:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

YOLO inference is the main GPU-accelerated workload when CUDA is available. Other operations, such as video decoding, rendering, tracking, and file writing, may use the CPU.

## Troubleshooting

### API connection fails

1. Start the FastAPI backend.
2. Open `http://127.0.0.1:8000/health`.
3. Check that the dashboard points to the correct API URL.

### Dashboard does not start

Activate the virtual environment and inspect the required packages:

```powershell
python -m pip show streamlit streamlit-webrtc
```

### GPU is unavailable

Run the GPU verification command above. Check driver and PyTorch compatibility before changing the installed GPU packages.

### Live camera does not start

Check browser camera permissions. Remote use may require HTTPS and working WebRTC connectivity.

### Port is already in use

Stop the previous process or choose a different port, then open the corresponding URL.

## Limitations

- Detection and counting depend on video quality, camera angle, lighting, occlusion, confidence thresholds, and tracking behavior.
- A result from one test video does not establish general detection accuracy.
- Concurrent multi-camera and multi-user performance needs workload testing.
- CUDA availability does not mean every processing step runs on the GPU.
- Do not publish private footage, secrets, or sensitive data in a public repository.

## Development Status

The uploaded-video dashboard, live-camera dashboard, and FastAPI backend have been reported working in the development environment. Re-test after changing code or dependencies.

## License and Model Notes

Review the applicable licenses for the project code, dependencies, and model weights before redistribution or deployment.
