import streamlit as st
import cv2
import time
import json
import os
import math
import tempfile
import statistics
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from google import genai
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Video Auditor", page_icon="👁️", layout="wide")

st.markdown("""
<style>
    .audit-card {
        background: #1a1f2e;
        border: 1px solid #2d3a52;
        border-left: 4px solid #4f8ef7;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 14px;
        color: #e0e8f0;
    }
    .audit-card .audit-time {
        font-size: 11px;
        color: #7a9abf;
        font-family: monospace;
        margin-bottom: 6px;
    }
    .audit-card .audit-body {
        font-size: 13.5px;
        line-height: 1.65;
        color: #ccd9ea;
        white-space: pre-wrap;
    }
    .audit-card.anomaly  { border-left-color: #f74f4f; }
    .audit-card.skipped  { border-left-color: #f7c44f; opacity: 0.7; }
    .audit-card.retried  { border-left-color: #a855f7; }
    .badge {
        display: inline-block;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 10px;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .badge-normal   { background: #1e3a5f; color: #4f8ef7; }
    .badge-anomaly  { background: #3a1e1e; color: #f74f4f; }
    .badge-skipped  { background: #3a2e1e; color: #f7c44f; }
    .badge-retried  { background: #2e1e3a; color: #a855f7; }
    .section-header {
        font-size: 13px;
        font-weight: 600;
        color: #7a9abf;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 18px 0 8px 0;
        padding-bottom: 4px;
        border-bottom: 1px solid #2d3a52;
    }
    div[data-testid="stMetric"] {
        background: #1a1f2e;
        border: 1px solid #2d3a52;
        border-radius: 8px;
        padding: 12px 16px;
    }
    .quota-bar-wrap {
        background: #1a1f2e;
        border: 1px solid #2d3a52;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 6px;
        font-size: 12px;
        color: #7a9abf;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# RATE-LIMIT CONFIG — tweak these to match your tier
# ---------------------------------------------------------------------------
# Free-tier safe defaults for gemini-2.5-flash-lite (1000 RPD / 15 RPM)
# Change MAX_CALLS_PER_DAY to 250 if using gemini-2.5-flash (250 RPD)
GEMINI_MODEL_OPTIONS = {
    "gemini-2.5-flash-lite (Free ✓ 1000 RPD)": "gemini-2.5-flash-lite-preview-06-17",
    "gemini-2.5-flash   (Free  250 RPD)":       "gemini-2.5-flash",
    "gemini-1.5-flash   (Free  1500 RPD)":      "gemini-1.5-flash",
}
RPD_BY_MODEL = {
    "gemini-2.5-flash-lite-preview-06-17": 1000,
    "gemini-2.5-flash":  250,
    "gemini-1.5-flash": 1500,
}

# Anomaly detection thresholds
SPIKE_THRESHOLD   = 0.40   # relative jump > 40% of rolling mean → anomaly candidate
SILENCE_THRESHOLD = 0.15   # relative change < 15% → skip the API call (scene is quiet)

# Retry config
MAX_RETRIES   = 3
BASE_BACKOFF  = 15   # seconds (Gemini hint says ~12 s)

# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

def init_gemini(api_key):
    if not api_key:
        return None
    os.environ["GEMINI_API_KEY"] = api_key
    return genai.Client()

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
defaults = {
    "audit_history":  [],
    "timeline_data":  [],
    "run_auditor":    False,
    "session_start":  None,
    "peak_count":     0,
    "total_frames":   0,
    "api_calls_today": 0,         # rough per-session counter
    "retry_after":    0,          # epoch time when we can retry after a 429
    "last_mean":      None,       # rolling mean for spike detection
    "skipped_intervals": 0,       # quiet intervals that were suppressed
    "last_analyzed":  None,       # agg dict from most recent Gemini window
    "session_counts": [],         # all counts ever seen this session (for true avg)
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

load_dotenv()
default_api_key = os.environ.get("GEMINI_API_KEY", "")

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Auditor Settings")
api_key_input     = st.sidebar.text_input("Gemini API Key", value=default_api_key, type="password")
model_label       = st.sidebar.selectbox("Gemini Model", list(GEMINI_MODEL_OPTIONS.keys()))
selected_model    = GEMINI_MODEL_OPTIONS[model_label]
max_rpd           = RPD_BY_MODEL[selected_model]
video_source      = st.sidebar.selectbox("Video Source", ["Webcam (0)", "Upload Video File"])
reporting_interval = st.sidebar.slider(
    "Min Report Interval (s) — actual may be longer if scene is quiet",
    min_value=30, max_value=300, value=60,
    help="Increase this to save API calls. Event-driven spikes can still trigger earlier."
)
sensitivity = st.sidebar.slider(
    "Anomaly Sensitivity", min_value=1, max_value=5, value=3,
    help="Higher = triggers API call on smaller changes"
)
# Map sensitivity 1–5 to spike threshold 0.60–0.20
dynamic_spike_threshold = round(0.70 - (sensitivity * 0.10), 2)

st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Quota saver tips**")
st.sidebar.markdown(
    "- Use `flash-lite` for free tier (1000 RPD)\n"
    "- Raise the interval slider\n"
    "- Lower sensitivity avoids false triggers\n"
    "- Stable scenes are silently skipped"
)

def start_auditor():
    st.session_state.run_auditor   = True
    st.session_state.session_start = datetime.now().strftime("%H:%M:%S")

def stop_auditor():
    st.session_state.run_auditor = False

def clear_history():
    for k, v in defaults.items():
        st.session_state[k] = v

col1, col2 = st.sidebar.columns(2)
col1.button("▶️ Start", on_click=start_auditor, use_container_width=True)
col2.button("⏹️ Stop",  on_click=stop_auditor,  use_container_width=True)
st.sidebar.button("🗑️ Clear History", on_click=clear_history, use_container_width=True)

if st.session_state.audit_history:
    export_data = json.dumps(st.session_state.audit_history, indent=2)
    st.sidebar.download_button(
        "📥 Export Audit Log", data=export_data,
        file_name=f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json", use_container_width=True
    )

# Quota bar
used = st.session_state.api_calls_today
pct  = min(used / max_rpd * 100, 100)
bar_color = "#4f8ef7" if pct < 70 else ("#f7c44f" if pct < 90 else "#f74f4f")
st.sidebar.markdown(
    f"""<div class="quota-bar-wrap">
        API calls this session: <b>{used}</b> / {max_rpd} RPD estimate<br>
        <div style="background:#2d3a52;border-radius:4px;height:6px;margin-top:6px;">
          <div style="background:{bar_color};width:{pct:.1f}%;height:6px;border-radius:4px;"></div>
        </div>
    </div>""",
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# TELEMETRY AGGREGATION — compress N raw frames into a compact stats object
# ---------------------------------------------------------------------------
def aggregate_telemetry(event_logs: list) -> dict:
    """
    Turn a list of {timestamp, detected_count} into a tiny stats dict.
    This drastically reduces prompt token count.
    """
    if not event_logs:
        return {}
    counts = [e["detected_count"] for e in event_logs]
    return {
        "window_start":   event_logs[0]["timestamp"],
        "window_end":     event_logs[-1]["timestamp"],
        "samples":        len(counts),
        "min":            min(counts),
        "max":            max(counts),
        "mean":           round(statistics.mean(counts), 1),
        "std_dev":        round(statistics.stdev(counts), 2) if len(counts) > 1 else 0,
        "trend":          "rising" if counts[-1] > counts[0] else ("falling" if counts[-1] < counts[0] else "stable"),
        "peak_at":        event_logs[counts.index(max(counts))]["timestamp"],
    }

# ---------------------------------------------------------------------------
# CHANGE DETECTION — decide whether it's worth an API call
# ---------------------------------------------------------------------------
def should_call_api(agg: dict, last_mean: float | None, spike_threshold: float) -> tuple[bool, str]:
    """
    Returns (True, reason) if the API should be called, else (False, reason).
    Strategies:
      1. First call ever — always report.
      2. Spike detected — deviation > spike_threshold from rolling mean.
      3. Significant drop — same logic downward.
      4. Scene is quiet (< SILENCE_THRESHOLD change) — skip.
    """
    if last_mean is None:
        return True, "initial_baseline"

    current_mean = agg.get("mean", 0)
    if last_mean == 0:
        # Avoid division by zero; any non-zero reading is a spike
        return current_mean > 0, "zero_to_nonzero"

    relative_change = abs(current_mean - last_mean) / last_mean

    if relative_change >= spike_threshold:
        direction = "spike_up" if current_mean > last_mean else "spike_down"
        return True, direction

    if relative_change < SILENCE_THRESHOLD:
        return False, "quiet_scene_suppressed"

    return True, "routine_interval"

# ---------------------------------------------------------------------------
# GEMINI CALL WITH EXPONENTIAL BACKOFF + MODEL FALLBACK
# ---------------------------------------------------------------------------
FALLBACK_MODELS = ["gemini-1.5-flash", "gemini-1.5-flash-8b"]

def generate_gemini_audit(client, agg: dict, interval: int, primary_model: str) -> tuple[str, bool, str]:
    """
    Returns (report_text, has_anomaly, model_used).
    Tries primary_model first, then fallbacks, then graceful degradation.
    """
    prompt = (
        f"You are an AI Security Auditor. Analyse this {interval}s telemetry window:\n"
        f"{json.dumps(agg)}\n\n"
        "Reply with 2-4 sentences. "
        "Start with [ANOMALY] if std_dev > 2 or a spike/drop > 30% occurred, else [NORMAL]. "
        "State the trend, flag risks, and suggest action if needed."
    )

    models_to_try = [primary_model] + [m for m in FALLBACK_MODELS if m != primary_model]

    for attempt in range(MAX_RETRIES):
        for model in models_to_try:
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                text = response.text.strip()
                has_anomaly = text.startswith("[ANOMALY]")
                display = text.replace("[NORMAL]", "").replace("[ANOMALY]", "").strip()
                st.session_state.api_calls_today += 1
                return display, has_anomaly, model
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    # Extract retry delay hint from error if present
                    delay = BASE_BACKOFF * (2 ** attempt)
                    # Try to read Google's suggested retry delay
                    try:
                        import re
                        m = re.search(r"retryDelay.*?(\d+)s", err_str)
                        if m:
                            delay = max(int(m.group(1)) + 2, delay)
                    except Exception:
                        pass
                    st.session_state.retry_after = time.time() + delay
                    # Continue to next model in this attempt before giving up
                    continue
                else:
                    # Non-rate-limit error — don't retry same model
                    break
        # All models exhausted for this attempt — wait before next attempt
        if attempt < MAX_RETRIES - 1:
            wait = BASE_BACKOFF * (2 ** attempt)
            time.sleep(min(wait, 60))

    # Graceful degradation — local summary with no API call
    trend_emoji = {"rising": "📈", "falling": "📉", "stable": "➡️"}.get(agg.get("trend",""), "")
    local_text = (
        f"[Local summary — API quota reached] {trend_emoji} "
        f"Occupancy: avg {agg.get('mean','?')} people over {agg.get('samples','?')} samples "
        f"(min {agg.get('min','?')}, max {agg.get('max','?')}). "
        f"Trend: {agg.get('trend','unknown')}. "
        f"Std dev: {agg.get('std_dev','?')} — "
        + ("⚠️ High variance, manual review recommended." if (agg.get('std_dev') or 0) > 2 else "Variance normal.")
    )
    return local_text, (agg.get("std_dev", 0) or 0) > 2, "local_fallback"

# ---------------------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------------------
st.title("👁️ Multi-Stream Multimodal Video Auditor")
st.markdown(
    "Edge AI detection (YOLOv8, local) + Cloud LLM reasoning (Gemini). "
    "**Quota-aware:** skips silent scenes, retries with backoff, falls back to local summaries."
)

video_col, chart_col, report_col = st.columns([0.38, 0.28, 0.34])

with video_col:
    st.subheader("Live Edge AI Stream")
    video_placeholder   = st.empty()
    metrics_row         = st.empty()
    next_report_ph      = st.empty()   # countdown to next Gemini call

with chart_col:
    st.subheader("Live Analytics")
    chart_placeholder   = st.empty()
    stats_cols_ph       = st.empty()

with report_col:
    st.subheader("Audit Reports")
    status_placeholder  = st.empty()
    reports_container   = st.container()

# ---------------------------------------------------------------------------
# RENDER HELPERS
# ---------------------------------------------------------------------------
def render_charts(timeline_data):
    if not timeline_data:
        chart_placeholder.info("Charts appear once data is collected.")
        stats_cols_ph.empty()
        return

    df = pd.DataFrame(timeline_data)
    df["count"] = pd.to_numeric(df["count"])

    # Line chart — always replaces itself inside the placeholder
    chart_placeholder.line_chart(
        df.set_index("time")["count"],
        use_container_width=True, height=200, color="#4f8ef7"
    )

    # --- Stats: Last Analyzed window + Session Average ---
    # These are the only two numbers the user asked for.
    last = st.session_state.last_analyzed          # dict or None
    session_counts = st.session_state.session_counts

    last_mean_val  = f"{last['mean']:.1f}"  if last else "—"
    last_max_val   = str(last['max'])        if last else "—"
    last_trend     = last.get('trend', '')   if last else ""
    trend_icon     = {"rising": "📈", "falling": "📉", "stable": "➡️"}.get(last_trend, "")
    last_window    = f"{last['window_start']} → {last['window_end']}" if last else "waiting for first report…"

    session_avg    = f"{statistics.mean(session_counts):.1f}" if session_counts else "—"

    # Rendered as a single HTML block so st.empty() truly replaces it on each frame
    stats_cols_ph.markdown(
        f"""
        <div style="display:flex;gap:10px;margin-top:8px;">
          <div style="flex:1;background:#1a1f2e;border:1px solid #2d3a52;border-radius:8px;
                      padding:12px 14px;">
            <div style="font-size:11px;color:#7a9abf;margin-bottom:4px;font-family:monospace;">
              LAST ANALYZED &nbsp;{trend_icon}
            </div>
            <div style="font-size:11px;color:#4a5a6a;margin-bottom:6px;">{last_window}</div>
            <div style="display:flex;gap:16px;">
              <span>
                <div style="font-size:10px;color:#7a9abf;">Avg</div>
                <div style="font-size:22px;font-weight:700;color:#e0e8f0;">{last_mean_val}</div>
              </span>
              <span>
                <div style="font-size:10px;color:#7a9abf;">Peak</div>
                <div style="font-size:22px;font-weight:700;color:#e0e8f0;">{last_max_val}</div>
              </span>
            </div>
          </div>
          <div style="flex:1;background:#1a1f2e;border:1px solid #2d3a52;border-radius:8px;
                      padding:12px 14px;">
            <div style="font-size:11px;color:#7a9abf;margin-bottom:4px;">SESSION AVERAGE</div>
            <div style="font-size:11px;color:#4a5a6a;margin-bottom:6px;">
              across {len(session_counts)} frames
            </div>
            <div style="font-size:22px;font-weight:700;color:#e0e8f0;">{session_avg}</div>
            <div style="font-size:10px;color:#7a9abf;margin-top:2px;">people / frame</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_audit_history():
    if not st.session_state.audit_history:
        with reports_container:
            st.info("No audit reports yet. Start the auditor to begin.")
        return
    with reports_container:
        total    = len(st.session_state.audit_history)
        anomalies = sum(1 for r in st.session_state.audit_history if r.get("has_anomaly"))
        skipped  = st.session_state.skipped_intervals
        st.markdown(
            f"<div class='section-header'>"
            f"{total} reports · {anomalies} anomalies · {skipped} quiet intervals skipped"
            f"</div>",
            unsafe_allow_html=True
        )
        for report in reversed(st.session_state.audit_history):
            model_used  = report.get("model_used", "")
            is_local    = model_used == "local_fallback"
            is_anomaly  = report.get("has_anomaly", False)
            is_retried  = report.get("trigger") in ("spike_up", "spike_down")

            if is_local:
                card_cls, badge_cls, badge_lbl = "audit-card skipped", "badge-skipped", "⚡ Local"
            elif is_anomaly:
                card_cls, badge_cls, badge_lbl = "audit-card anomaly", "badge-anomaly", "⚠ Anomaly"
            elif is_retried:
                card_cls, badge_cls, badge_lbl = "audit-card retried", "badge-retried", "🔺 Spike"
            else:
                card_cls, badge_cls, badge_lbl = "audit-card", "badge-normal", "✓ Normal"

            model_tag = f"<span style='font-size:10px;color:#4a5a6a;margin-left:8px;'>via {model_used}</span>" if model_used else ""
            st.markdown(
                f"""<div class="{card_cls}">
                  <div class="audit-time">{report['timestamp']}{model_tag}</div>
                  <span class="badge {badge_cls}">{badge_lbl}</span>
                  <div class="audit-body">{report['text']}</div>
                </div>""",
                unsafe_allow_html=True
            )

# Initial render
render_charts(st.session_state.timeline_data)
render_audit_history()

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
if st.session_state.run_auditor:
    if not api_key_input:
        st.warning("⚠️ Please enter a Gemini API Key in the sidebar.")
        st.stop()

    gemini_client = init_gemini(api_key_input)
    yolo_model    = load_yolo_model()

    source = 0 if video_source == "Webcam (0)" else None
    if video_source == "Upload Video File":
        uploaded_file = st.sidebar.file_uploader("Upload an MP4", type=["mp4", "mov", "avi"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            source = tfile.name
        else:
            st.warning("⚠️ Please upload a video file.")
            st.stop()

    cap = cv2.VideoCapture(source)
    event_logs       = []
    last_report_time = time.time()
    status_placeholder.success("🔴 Live — collecting telemetry...")

    while cap.isOpened() and st.session_state.run_auditor:
        success, frame = cap.read()
        if not success:
            st.warning("Video stream ended or could not be read.")
            break

        results       = yolo_model.track(frame, classes=[0], persist=True, verbose=False)
        current_count = len(results[0].boxes) if results[0].boxes is not None else 0

        st.session_state.total_frames += 1
        st.session_state.peak_count    = max(st.session_state.peak_count, current_count)

        annotated_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
        video_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
        metrics_row.metric("🧍 People Detected", current_count)

        current_time = time.time()
        time_str     = time.strftime("%H:%M:%S", time.localtime(current_time))

        event_logs.append({"timestamp": time_str, "detected_count": current_count})

        st.session_state.timeline_data.append({"time": time_str, "count": current_count})
        if len(st.session_state.timeline_data) > 500:
            st.session_state.timeline_data = st.session_state.timeline_data[-500:]

        # Running list for session-wide average (kept separately from timeline_data)
        st.session_state.session_counts.append(current_count)
        if len(st.session_state.session_counts) > 5000:
            st.session_state.session_counts = st.session_state.session_counts[-5000:]

        render_charts(st.session_state.timeline_data)

        # Countdown display
        elapsed  = current_time - last_report_time
        remaining = max(0, reporting_interval - elapsed)
        next_report_ph.caption(
            f"⏱ Next report check in ~{remaining:.0f}s "
            f"| Calls today: {st.session_state.api_calls_today}"
        )

        # --- REPORT TRIGGER ---
        if elapsed >= reporting_interval:
            # Respect rate-limit backoff window
            if time.time() < st.session_state.retry_after:
                wait_left = st.session_state.retry_after - time.time()
                status_placeholder.warning(
                    f"⏳ Rate limit backoff — resuming in {wait_left:.0f}s (no data lost)"
                )
                last_report_time = current_time   # reset so we don't hammer
            else:
                agg = aggregate_telemetry(event_logs)
                if agg:
                    should_call, trigger_reason = should_call_api(
                        agg, st.session_state.last_mean, dynamic_spike_threshold
                    )

                    if not should_call:
                        # Quiet scene — record suppression, skip API
                        st.session_state.skipped_intervals += 1
                        status_placeholder.info(
                            f"🔇 Quiet scene at {time_str} — API call skipped "
                            f"(saved {st.session_state.skipped_intervals} calls so far)"
                        )
                    else:
                        report_ts = time_str
                        with report_col:
                            with st.spinner(f"Gemini analysing [{trigger_reason}]..."):
                                report_text, has_anomaly, model_used = generate_gemini_audit(
                                    gemini_client, agg, reporting_interval, selected_model
                                )

                        st.session_state.last_mean     = agg["mean"]
                        st.session_state.last_analyzed = agg   # ← powers the stats panel
                        st.session_state.audit_history.append({
                            "timestamp":  report_ts,
                            "text":       report_text,
                            "has_anomaly": has_anomaly,
                            "model_used": model_used,
                            "trigger":    trigger_reason,
                            "agg":        agg,
                        })
                        render_audit_history()

                event_logs       = []
                last_report_time = current_time

    cap.release()
    status_placeholder.warning("⏹️ Auditor stopped. History preserved.")
    render_audit_history()

else:
    if st.session_state.audit_history:
        status_placeholder.warning("⏹️ Auditor stopped. Showing saved history.")
    else:
        status_placeholder.info("Press ▶️ Start to begin auditing.")