<div align="center">

# 👁️ AI Video Auditor

**Real-time crowd analysis powered by YOLOv8 + Google Gemini**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00A8E0?style=flat)](https://ultralytics.com/yolov8)
[![Gemini](https://img.shields.io/badge/Gemini-Google%20AI-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat)](LICENSE)

*Detect people in video streams, analyse crowd behaviour, and surface safety insights — all in your browser.*

[Demo](#demo) · [Quick Start](#quick-start) · [Features](#features) · [Configuration](#configuration) · [Architecture](#architecture) · [Contributing](#contributing)

</div>

---

## Overview

**AI Video Auditor** is a Streamlit web application that combines:

- **YOLOv8** (via Ultralytics) for fast, accurate person detection on every frame
- **Google Gemini** multimodal LLM for contextual scene analysis — crowd behaviour, density, and risk-level assessment
- A smart **anomaly engine** that skips quiet scenes and triggers analysis only on meaningful crowd changes — preserving your API quota

It runs entirely in your browser with no cloud infrastructure required beyond a single Gemini API key.

---

## Demo

| Idle State | Live Analysis |
|:---:|:---:|
| ![Idle state screenshot](assets/screenshot-idle.png) | ![Live analysis screenshot](assets/screenshot-live.png) |

> *Screenshots are illustrative. Replace with your own once the app is running.*

---

## Features

### 🎯 Core Capabilities

- **Real-time person detection** — YOLOv8n runs on every frame from webcam or uploaded video
- **AI scene analysis** — Gemini describes crowd behaviour and rates safety risk (Low / Medium / High)
- **Anomaly-driven triggering** — API calls fire on population spikes, not on a fixed clock
- **Silence suppression** — Stable scenes (< 15 % change) are silently skipped to conserve quota

### 📊 Analytics Dashboard

- Live KPI row: current count, session peak, session average, total API calls
- Scrolling area chart of crowd size over time
- Timestamped analysis log with colour-coded trigger badges (SPIKE · INTERVAL · EVENT)
- One-click JSON export of the full audit log

### ⚙️ Tunable Parameters

| Parameter | Range | Effect |
|---|---|---|
| Report Interval | 30 – 300 s | Minimum time between Gemini calls |
| Anomaly Sensitivity | 1 – 5 | Maps to a 0.60 – 0.20 spike threshold |
| Gemini Model | 3 options | Trade speed vs. daily quota |

### 🔒 Quota Safety

- Visual quota progress bar in the sidebar (green → amber → red)
- Per-model RPD limits enforced at runtime
- Exponential back-off with retry on HTTP 429 responses

---

## Quick Start

### 1 — Prerequisites

| Tool | Version |
|---|---|
| Python | ≥ 3.10 |
| pip | ≥ 23 |
| Webcam *(optional)* | — |

### 2 — Clone & install

```bash
git clone https://github.com/yourusername/ai-video-auditor.git
cd ai-video-auditor

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3 — Set your API key

```bash
# Option A — .env file (recommended)
echo "GEMINI_API_KEY=your_key_here" > .env

# Option B — environment variable
export GEMINI_API_KEY=your_key_here   # macOS / Linux
set GEMINI_API_KEY=your_key_here      # Windows CMD
```

> 🔑 Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey).

### 4 — Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
ai-video-auditor/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # API keys — never commit this!
├── .gitignore              # Excludes keys, models, media files
│
├── assets/                 # Optional: demo screenshots, logo
│   ├── screenshot-idle.png
│   └── screenshot-live.png
│
└── README.md
```

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Your Google Gemini API key |

### Gemini Model Options

| Model ID | Free Tier RPD | Best For |
|---|---|---|
| `gemini-2.5-flash-lite-preview-06-17` | 1 000 | Daily monitoring, long sessions |
| `gemini-2.5-flash` | 250 | High-quality short sessions |
| `gemini-1.5-flash` | 1 500 | Maximum free-tier calls |

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Streamlit Frontend                   │
│  Sidebar controls · KPI row · Video feed · Log cards │
└───────────────────┬──────────────────────────────────┘
                    │ frame + count
          ┌─────────▼─────────┐
          │  YOLOv8n (YOLO)   │  ← @st.cache_resource
          │  Person detection  │
          └─────────┬─────────┘
                    │ annotated frame + bbox list
          ┌─────────▼──────────────┐
          │  Anomaly Engine        │
          │  spike / silence check │
          └─────────┬──────────────┘
                    │ trigger decision
          ┌─────────▼─────────┐
          │  Google Gemini    │  ← multimodal: frame + prompt
          │  Scene analysis   │
          └─────────┬─────────┘
                    │ text analysis
          ┌─────────▼─────────┐
          │  Session State    │  timeline_data, audit_history
          └───────────────────┘
```

### Detection → Analysis Flow

1. **Capture** — OpenCV reads a frame from webcam or uploaded file
2. **Detect** — YOLOv8n runs inference (class 0 = person only) in < 30 ms on CPU
3. **Evaluate** — Anomaly engine compares count against rolling mean
4. **Analyse** *(conditional)* — If triggered, JPEG-encode the frame and send with a structured prompt to Gemini
5. **Display** — Annotated frame, updated KPIs, chart, and log card rendered in browser

---

## `requirements.txt`

```txt
streamlit>=1.35.0
ultralytics>=8.2.0
opencv-python-headless>=4.9.0
google-generativeai>=0.7.0
python-dotenv>=1.0.0
pandas>=2.0.0
```

> Use `opencv-python-headless` on servers (no GUI dependency). Switch to `opencv-python` if you need `cv2.imshow()` locally.

---

## Tips for Saving API Quota

- Choose **gemini-2.5-flash-lite** (1 000 RPD) for long monitoring sessions
- Raise the **Report Interval** slider to 120 – 300 s in stable environments
- Set **Sensitivity** to 2 or 3 — very high sensitivity triggers on minor fluctuations
- Use uploaded video instead of webcam during development to replay the same footage

---

## Roadmap

- [ ] Multi-camera support (camera index selector)
- [ ] Heatmap overlay for crowd density visualisation
- [ ] Email / webhook alerts on High-risk detections
- [ ] Historical session comparison dashboard
- [ ] Docker image for one-command deployment

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push to your fork: `git push origin feature/your-feature`
5. Open a Pull Request

Please keep commits atomic and write clear PR descriptions.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

Made with ❤️ using [Streamlit](https://streamlit.io) · [Ultralytics YOLOv8](https://ultralytics.com/yolov8) · [Google Gemini](https://ai.google.dev)

</div>
