"""
app.py — Vantage Streamlit Dashboard

Run with:  streamlit run app.py

Performance notes
─────────────────
• @st.fragment(run_every=100 ms) isolates the live feed — the rest of the
  page never re-renders during analysis.
• SVG arc gauge replaces Plotly — < 1 KB per frame.
• Frames downscaled to 480 px wide + JPEG-encoded before transfer.
• Engine + SessionManager persist in st.session_state across fragment ticks.
"""

import math
import os
import time
from datetime import timedelta

import numpy as np
import streamlit as st
from PIL import Image

from session_manager import SessionManager, generate_tips

# ─────────────────────────────────────────────────────────────────────────────
# Cloud vs Local detection
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Cloud containers mount the repo at /mount/src/<repo-name>.
IS_CLOUD = os.path.exists("/mount/src")

if IS_CLOUD:
    from frame_analyzer import FrameAnalyzer
else:
    import cv2
    from confidence_engine import ConfidenceEngine

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vantage",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens + global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main .block-container {
    background: #09090f !important;
    color: #d4d4e8;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background: #0f0f18 !important;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] * { color: #b0b0c8 !important; }
.block-container { padding-top: 2rem !important; max-width: 1280px; }

h1, h2, h3 { letter-spacing: -0.6px; color: #eeeef8; }
p { line-height: 1.65; }

.vt-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e1e32, transparent);
    margin: 18px 0;
}

.vt-stat {
    background: #10101c;
    border: 1px solid #1c1c2e;
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
}
.vt-stat-val {
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -1px;
    margin-bottom: 6px;
}
.vt-stat-lbl {
    font-size: 0.67rem;
    font-weight: 600;
    color: #3e3e58;
    text-transform: uppercase;
    letter-spacing: 1.3px;
}

.sig-wrap { margin-bottom: 14px; }
.sig-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 5px;
}
.sig-name {
    font-size: 0.72rem;
    font-weight: 600;
    color: #5a5a80;
    text-transform: uppercase;
    letter-spacing: 0.9px;
}
.sig-val { font-size: 0.82rem; font-weight: 700; }
.sig-track {
    background: #16162a;
    border-radius: 99px;
    height: 7px;
    overflow: hidden;
}
.sig-fill { height: 100%; border-radius: 99px; transition: width 0.25s ease; }

.vt-insight {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #12122a;
    border: 1px solid #26264a;
    border-radius: 10px;
    padding: 10px 14px;
    margin-top: 14px;
    font-size: 0.83rem;
    color: #8888c0;
    font-weight: 500;
}
.vt-insight-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #5050a0;
    flex-shrink: 0;
}

.vt-tip {
    background: #0d0d1a;
    border: 1px solid #1a1a2e;
    border-left: 3px solid #38388a;
    border-radius: 0 10px 10px 0;
    padding: 11px 16px;
    margin-bottom: 9px;
    font-size: 0.87rem;
    color: #8888b0;
    line-height: 1.65;
}

.vt-trend {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 99px;
    background: #12121e;
    border: 1px solid #1e1e30;
    letter-spacing: 0.2px;
}

.vt-section-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #3a3a58;
    margin-bottom: 12px;
}

.vt-pill {
    display: inline-block;
    background: #10101e;
    border: 1px solid #22223a;
    border-radius: 99px;
    padding: 5px 14px;
    font-size: 0.76rem;
    color: #5a5a88;
    font-weight: 500;
    margin: 3px;
}

.vt-calib-track {
    background: #14142a;
    border-radius: 99px;
    height: 5px;
    overflow: hidden;
    margin: 10px 0 6px;
}
.vt-calib-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #38388a, #6868c0);
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stStatusWidget"] { display: none; }

.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.87rem !important;
    letter-spacing: 0.2px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = dict(
    page="landing",
    engine=None,
    analyzer=None,
    last_cloud_result=None,
    session_mgr=None,
    live_running=False,
    last_summary=None,
    session_start=None,
    # Timestamp of the last engine.stop() call.  Used to enforce a minimum
    # delay before re-opening the camera — Windows DirectShow keeps the
    # device locked for ~500-1500 ms after cap.release(), so attempting to
    # reopen too quickly causes "camera already in use" errors.
    engine_stop_time=0.0,
)
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def goto(page: str):
    st.session_state.page = page


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers  (logic identical, HTML polished)
# ─────────────────────────────────────────────────────────────────────────────
def _score_color_hex(score: float) -> str:
    if score >= 85: return "#22d47e"
    if score >= 70: return "#1abc6a"
    if score >= 55: return "#f0b429"
    if score >= 40: return "#f06a30"
    return "#e84040"


def _signal_bar(label: str, score: float, muted: bool = False) -> str:
    color = "#28283e" if muted else _score_color_hex(score)
    pct   = 0 if muted else int(score)
    val   = "MUTED" if muted else f"{pct}%"
    val_color = "#38385a" if muted else color
    return (
        f'<div class="sig-wrap">'
        f'<div class="sig-header">'
        f'<span class="sig-name">{label}</span>'
        f'<span class="sig-val" style="color:{val_color}">{val}</span>'
        f'</div>'
        f'<div class="sig-track">'
        f'<div class="sig-fill" style="width:{pct}%;background:{color}"></div>'
        f'</div></div>'
    )


def _svg_gauge(score: float, state: str) -> str:
    """Lightweight SVG semicircular gauge."""
    color = _score_color_hex(score)
    cx, cy, r, sw = 130, 118, 96, 13
    sx, sy = cx - r, cy
    tx, ty = cx, cy - r
    ex, ey = cx + r, cy

    bg = f"M {sx} {sy} A {r} {r} 0 0 1 {tx} {ty} A {r} {r} 0 0 1 {ex} {ey}"

    sc = max(0.0, min(100.0, score))
    if sc < 0.5:
        fill = ""
    elif sc > 99.5:
        fill = (
            f'<path d="{bg}" fill="none" stroke="{color}" '
            f'stroke-width="{sw}" stroke-linecap="round"/>'
        )
    else:
        a  = math.pi * (1 - sc / 100)
        fx = cx + r * math.cos(a)
        fy = cy - r * math.sin(a)
        la = 1 if sc > 50 else 0
        fill = (
            f'<path d="M {sx} {sy} A {r} {r} 0 {la} 1 {fx:.1f} {fy:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round"/>'
        )

    return (
        f'<div style="text-align:center;padding:4px 0">'
        f'<svg viewBox="0 0 260 148" width="100%"'
        f' style="max-width:300px;display:block;margin:0 auto">'
        # Outer glow ring
        f'<path d="{bg}" fill="none" stroke="#18182a" '
        f'stroke-width="{sw + 5}" stroke-linecap="round"/>'
        # Background ring
        f'<path d="{bg}" fill="none" stroke="#1e1e30" '
        f'stroke-width="{sw}" stroke-linecap="round"/>'
        # Filled arc
        f'{fill}'
        # Score number
        f'<text x="{cx}" y="{cy - 6}" text-anchor="middle" '
        f'font-size="44" font-weight="800" fill="{color}" '
        f'font-family="Inter,sans-serif" letter-spacing="-2">{int(score)}</text>'
        f'<text x="{cx + 30}" y="{cy - 9}" text-anchor="start" '
        f'font-size="17" font-weight="600" fill="{color}" opacity="0.6" '
        f'font-family="Inter,sans-serif">%</text>'
        # State label
        f'<text x="{cx}" y="{cy + 19}" text-anchor="middle" '
        f'font-size="12" font-weight="700" fill="{color}" opacity="0.75" '
        f'font-family="Inter,sans-serif" letter-spacing="1.5">'
        f'{state.upper()}</text>'
        f'</svg></div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Landing
# ─────────────────────────────────────────────────────────────────────────────
def page_landing():
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown("""
        <div style="text-align:center;padding:64px 0 16px">
            <div style="font-size:0.7rem;font-weight:700;letter-spacing:3.5px;
                        color:#2e2e50;text-transform:uppercase;margin-bottom:20px">
                Behavioural Intelligence
            </div>
            <div style="font-size:3.8rem;font-weight:800;letter-spacing:-3.5px;
                        color:#eeeef8;line-height:1">
                VANTAGE
            </div>
            <div style="width:36px;height:3px;
                        background:linear-gradient(90deg,#38388a,#7070c8);
                        border-radius:99px;margin:20px auto 22px"></div>
            <p style="color:#50507a;font-size:0.92rem;line-height:1.8;
                      max-width:380px;margin:0 auto 28px">
                Tracks your gaze, speech, hands, and presence in real time.
                Know your confidence before the room does.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center;padding-bottom:28px">
            <span class="vt-pill">👁  Gaze</span>
            <span class="vt-pill">🎙  Speech</span>
            <span class="vt-pill">✋  Hands</span>
            <span class="vt-pill">🧠  Presence</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Start Session", use_container_width=True, type="primary"):
            goto("live")
            st.rerun()

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        if st.button("View Past Sessions", use_container_width=True):
            goto("history")
            st.rerun()

        st.markdown("""
        <div style="text-align:center;margin-top:44px;padding-top:18px;
                    border-top:1px solid #12122a">
            <span style="font-size:0.72rem;color:#25253a;font-weight:500">
                Shahzeb Alam &nbsp;·&nbsp; DSOC 2026
            </span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — Live Analysis  (all logic unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def _end_session():
    mgr    = st.session_state.get("session_mgr")
    engine = st.session_state.get("engine")
    st.session_state.live_running = False
    summary = mgr.end_session() if mgr else None
    if engine:
        engine.stop()
        # Record the moment the camera was released so page_live() can
        # enforce a re-open delay on the next session.
        st.session_state.engine_stop_time = time.time()
    st.session_state.engine       = None
    st.session_state.session_mgr  = None
    st.session_state.last_summary = summary
    goto("summary")
    st.rerun()


def _end_session_cloud():
    """End session on Streamlit Cloud — close FrameAnalyzer, persist summary."""
    mgr      = st.session_state.get("session_mgr")
    analyzer = st.session_state.get("analyzer")
    st.session_state.live_running = False
    summary = mgr.end_session() if mgr else None
    if analyzer:
        analyzer.close()
    st.session_state.analyzer          = None
    st.session_state.session_mgr       = None
    st.session_state.last_summary      = summary
    st.session_state.last_cloud_result = None
    goto("summary")
    st.rerun()


@st.fragment(run_every=timedelta(milliseconds=100))
def _live_fragment():
    engine = st.session_state.get("engine")
    if not engine or not st.session_state.get("live_running"):
        return

    result = engine.get_latest()
    if not result:
        st.markdown(
            '<p style="color:#2a2a44;font-size:0.85rem">Waiting for camera…</p>',
            unsafe_allow_html=True,
        )
        return

    mgr = st.session_state.get("session_mgr")
    if mgr:
        mgr.record(result)

    left, right = st.columns([3, 2], gap="large")

    with left:
        frame = result["frame"]
        fh, fw = frame.shape[:2]
        if fw > 480:
            scale = 480 / fw
            frame = cv2.resize(
                frame, (480, int(fh * scale)), interpolation=cv2.INTER_AREA
            )
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB", use_container_width=True, output_format="JPEG")

        elapsed = int(time.time() - (st.session_state.get("session_start") or time.time()))
        m, s = divmod(elapsed, 60)
        st.markdown(
            f'<div style="display:flex;gap:14px;padding:6px 2px;'
            f'font-size:0.76rem;font-weight:500;color:#2e2e4a">'
            f'<span>⏱ {m:02d}:{s:02d}</span>'
            f'<span style="color:#1a1a2e">|</span>'
            f'<span>{int(result["fps"])} fps</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with right:
        if result["is_calibrated"]:
            conf = result["confidence"]
            st.markdown(_svg_gauge(conf, result["state"]), unsafe_allow_html=True)
            st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

            tc = {"↑": "#22d47e", "↓": "#e84040", "→": "#f0b429"}.get(
                result["trend"], "#888"
            )
            st.markdown(
                f'<div style="display:flex;align-items:center;'
                f'justify-content:space-between;margin-bottom:14px">'
                f'<div class="vt-section-label">Signal Breakdown</div>'
                f'<div class="vt-trend" style="color:{tc}">'
                f'{result["trend"]}&nbsp; {int(result["avg5"])}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            sp = result["speech_score"]
            st.markdown(
                _signal_bar("Gaze",     result["eye_score"])
                + _signal_bar("Speech", sp if sp is not None else 0, muted=(sp is None))
                + _signal_bar("Hands",  result["hand_score"])
                + _signal_bar("Presence", result["engagement_score"]),
                unsafe_allow_html=True,
            )

            if result["insight"]:
                st.markdown(
                    f'<div class="vt-insight">'
                    f'<div class="vt-insight-dot"></div>'
                    f'{result["insight"]}</div>',
                    unsafe_allow_html=True,
                )

        else:
            pct = int(result["calibration_pct"] * 100)
            st.markdown(
                f'<div style="padding:52px 0;text-align:center">'
                f'<div style="font-size:0.68rem;font-weight:700;letter-spacing:2px;'
                f'text-transform:uppercase;color:#2e2e50;margin-bottom:16px">Calibrating</div>'
                f'<div style="font-size:2.2rem;font-weight:800;color:#5050a0;'
                f'letter-spacing:-1px">{pct}%</div>'
                f'<div class="vt-calib-track" style="max-width:180px;margin:14px auto 8px">'
                f'<div class="vt-calib-fill" style="width:{pct}%"></div></div>'
                f'<p style="font-size:0.76rem;color:#28284a;margin-top:8px">'
                f'Hold still — establishing baseline</p>'
                f'</div>',
                unsafe_allow_html=True,
            )


def page_live():
    if st.session_state.engine is None:
        # Enforce a minimum 1.8 s gap since the last engine stop so that
        # Windows DirectShow has time to release the camera device fully.
        # Without this, rapid stop → start produces "camera already in use".
        _CAM_RELEASE_GRACE = 1.8
        elapsed_since_stop = time.time() - st.session_state.engine_stop_time
        if elapsed_since_stop < _CAM_RELEASE_GRACE:
            wait_remaining = _CAM_RELEASE_GRACE - elapsed_since_stop
            st.markdown(
                f'<p style="color:#28284a;font-size:0.82rem;padding:8px 0">'
                f'Releasing camera…</p>',
                unsafe_allow_html=True,
            )
            time.sleep(wait_remaining)
            st.rerun()
            return

        with st.spinner("Starting camera…"):
            engine = ConfidenceEngine()
            ok = engine.start()
        if not ok:
            st.error("Cannot open camera. Make sure it isn't used by another app.")
            if st.button("← Back"):
                goto("landing"); st.rerun()
            return
        st.session_state.engine        = engine
        st.session_state.session_mgr   = SessionManager()
        st.session_state.session_mgr.start_session()
        st.session_state.session_start = time.time()
        st.session_state.live_running  = True

    hdr_l, hdr_r = st.columns([5, 1])
    with hdr_l:
        st.markdown(
            '<h2 style="margin:0;font-size:1.25rem;font-weight:700;'
            'color:#9090b8;letter-spacing:-0.3px">Live Session</h2>',
            unsafe_allow_html=True,
        )
    with hdr_r:
        if st.button("End Session", type="primary", use_container_width=True):
            _end_session()
            return

    st.markdown('<div class="vt-divider"></div>', unsafe_allow_html=True)
    _live_fragment()


# ─────────────────────────────────────────────────────────────────────────────
# Page 2b — Live Analysis (Cloud — snapshot via st.camera_input)
# ─────────────────────────────────────────────────────────────────────────────
def _render_analysis_panel(result: dict):
    """Shared right-column rendering used by both local and cloud live pages."""
    if result["is_calibrated"]:
        conf = result["confidence"]
        st.markdown(_svg_gauge(conf, result["state"]), unsafe_allow_html=True)
        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

        tc = {"↑": "#22d47e", "↓": "#e84040", "→": "#f0b429"}.get(
            result["trend"], "#888"
        )
        st.markdown(
            f'<div style="display:flex;align-items:center;'
            f'justify-content:space-between;margin-bottom:14px">'
            f'<div class="vt-section-label">Signal Breakdown</div>'
            f'<div class="vt-trend" style="color:{tc}">'
            f'{result["trend"]}&nbsp; {int(result["avg5"])}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        sp = result["speech_score"]
        st.markdown(
            _signal_bar("Gaze",     result["eye_score"])
            + _signal_bar("Speech", sp if sp is not None else 0, muted=(sp is None))
            + _signal_bar("Hands",  result["hand_score"])
            + _signal_bar("Presence", result["engagement_score"]),
            unsafe_allow_html=True,
        )

        if result.get("insight"):
            st.markdown(
                f'<div class="vt-insight">'
                f'<div class="vt-insight-dot"></div>'
                f'{result["insight"]}</div>',
                unsafe_allow_html=True,
            )
    else:
        pct = int(result["calibration_pct"] * 100)
        st.markdown(
            f'<div style="padding:52px 0;text-align:center">'
            f'<div style="font-size:0.68rem;font-weight:700;letter-spacing:2px;'
            f'text-transform:uppercase;color:#2e2e50;margin-bottom:16px">Calibrating</div>'
            f'<div style="font-size:2.2rem;font-weight:800;color:#5050a0;'
            f'letter-spacing:-1px">{pct}%</div>'
            f'<div class="vt-calib-track" style="max-width:180px;margin:14px auto 8px">'
            f'<div class="vt-calib-fill" style="width:{pct}%"></div></div>'
            f'<p style="font-size:0.76rem;color:#28284a;margin-top:8px">'
            f'Take a few photos to calibrate your gaze baseline</p>'
            f'</div>',
            unsafe_allow_html=True,
        )


def page_live_cloud():
    """Snapshot-based live analysis for Streamlit Cloud deployment."""
    # Initialise analyzer + session on first visit
    if st.session_state.analyzer is None:
        st.session_state.analyzer      = FrameAnalyzer(calibration_frames=5)
        st.session_state.session_mgr   = SessionManager()
        st.session_state.session_mgr.start_session()
        st.session_state.session_start = time.time()
        st.session_state.live_running  = True

    # Header
    hdr_l, hdr_r = st.columns([5, 1])
    with hdr_l:
        st.markdown(
            '<h2 style="margin:0;font-size:1.25rem;font-weight:700;'
            'color:#9090b8;letter-spacing:-0.3px">Live Session</h2>',
            unsafe_allow_html=True,
        )
    with hdr_r:
        if st.button("End Session", type="primary", use_container_width=True):
            _end_session_cloud()
            return

    st.markdown('<div class="vt-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        '<p style="font-size:0.78rem;color:#3a3a58;margin-bottom:12px">'
        '📸 &nbsp;Capture snapshots to analyse your confidence. '
        'Speech analysis is unavailable on cloud.</p>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2], gap="large")

    with left:
        img = st.camera_input("Capture for analysis", label_visibility="collapsed")

        if img is not None:
            pil_img   = Image.open(img)
            frame_rgb = np.array(pil_img)
            # Mirror so it matches the user's perspective
            frame_rgb = np.ascontiguousarray(frame_rgb[:, ::-1, :])

            result = st.session_state.analyzer.process(frame_rgb)
            st.session_state.last_cloud_result = result

            mgr = st.session_state.get("session_mgr")
            if mgr:
                mgr.record(result)

        # Show elapsed time
        elapsed = int(time.time() - (st.session_state.get("session_start") or time.time()))
        m, s = divmod(elapsed, 60)
        snaps = st.session_state.analyzer._frame_count if st.session_state.analyzer else 0
        st.markdown(
            f'<div style="display:flex;gap:14px;padding:6px 2px;'
            f'font-size:0.76rem;font-weight:500;color:#2e2e4a">'
            f'<span>⏱ {m:02d}:{s:02d}</span>'
            f'<span style="color:#1a1a2e">|</span>'
            f'<span>{snaps} snapshot{"s" if snaps != 1 else ""}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with right:
        result = st.session_state.get("last_cloud_result")
        if result:
            _render_analysis_panel(result)
        else:
            st.markdown(
                '<div style="padding:52px 0;text-align:center">'
                '<div style="font-size:0.68rem;font-weight:700;letter-spacing:2px;'
                'text-transform:uppercase;color:#2e2e50;margin-bottom:16px">'
                'Ready</div>'
                '<p style="font-size:0.82rem;color:#3a3a58;max-width:220px;'
                'margin:0 auto">Take a photo with the camera widget to begin '
                'your confidence analysis.</p>'
                '</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — Summary  (logic unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def page_summary():
    summary = st.session_state.last_summary
    if not summary:
        st.warning("No session data found.")
        if st.button("← Home"):
            goto("landing"); st.rerun()
        return

    st.markdown(
        '<h2 style="margin:0 0 4px;font-size:1.5rem;font-weight:800;'
        'color:#eeeef8;letter-spacing:-0.5px">Session Summary</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:#28284a;font-size:0.78rem;margin-bottom:22px">'
        f'{summary["timestamp"]} &nbsp;·&nbsp; {summary["duration_s"]}s</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4, gap="small")

    def _stat(col, label, value, color):
        col.markdown(
            f'<div class="vt-stat">'
            f'<div class="vt-stat-val" style="color:{color}">{value}</div>'
            f'<div class="vt-stat-lbl">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    _stat(c1, "Avg Confidence",  f"{summary['avg_confidence']}%",
          _score_color_hex(summary["avg_confidence"]))
    _stat(c2, "Peak",            f"{summary['max_confidence']}%",
          _score_color_hex(summary["max_confidence"]))
    _stat(c3, "High Conf Time",  f"{summary['pct_above_75']}%",  "#22d47e")
    _stat(c4, "Low Conf Time",   f"{summary['pct_below_50']}%",  "#e84040")

    st.markdown('<div style="height:26px"></div>', unsafe_allow_html=True)

    tips = generate_tips(summary)
    if tips:
        st.markdown(
            '<div class="vt-section-label">Improvement Notes</div>',
            unsafe_allow_html=True,
        )
        for tip in tips:
            st.markdown(f'<div class="vt-tip">{tip}</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)
    col_a, col_b, _ = st.columns([1, 1, 2])
    with col_a:
        if st.button("New Session", use_container_width=True, type="primary"):
            goto("live"); st.rerun()
    with col_b:
        if st.button("Past Sessions", use_container_width=True):
            goto("history"); st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 — History  (logic unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def page_history():
    st.markdown(
        '<h2 style="margin:0 0 4px;font-size:1.5rem;font-weight:800;'
        'color:#eeeef8;letter-spacing:-0.5px">Session History</h2>',
        unsafe_allow_html=True,
    )

    history = SessionManager().load_history()

    if not history:
        st.markdown(
            '<p style="color:#28284a;font-size:0.88rem;margin-top:14px">'
            'No sessions recorded yet — start a live session to build your history.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<p style="color:#28284a;font-size:0.76rem;margin-bottom:16px">'
            f'{len(history)} session{"s" if len(history) != 1 else ""} on record</p>',
            unsafe_allow_html=True,
        )
        for i, s in enumerate(history):
            with st.expander(
                f"{s['timestamp']}  ·  {s['avg_confidence']}% avg  ·  {s['duration_s']}s",
                expanded=(i == 0),
            ):
                cols = st.columns(4, gap="small")
                cols[0].metric("Avg",    f"{s['avg_confidence']}%")
                cols[1].metric("Peak",   f"{s['max_confidence']}%")
                cols[2].metric("High %", f"{s['pct_above_75']}%")
                cols[3].metric("Low %",  f"{s['pct_below_50']}%")
                st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
                for tip in generate_tips(s):
                    st.markdown(f'<div class="vt-tip">{tip}</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)
    if st.button("← Home"):
        goto("landing"); st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar  (logic unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def _sidebar():
    if st.session_state.live_running:
        return
    with st.sidebar:
        st.markdown(
            '<div style="padding:10px 0 16px;font-size:1rem;font-weight:800;'
            'letter-spacing:-0.5px;color:#6060a0">VANTAGE</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="vt-divider"></div>', unsafe_allow_html=True)
        for label, pg in [("Home", "landing"), ("Past Sessions", "history")]:
            if st.button(label, use_container_width=True):
                if st.session_state.get("engine"):
                    st.session_state.engine.stop()
                    st.session_state.engine = None
                if st.session_state.get("analyzer"):
                    st.session_state.analyzer.close()
                    st.session_state.analyzer = None
                goto(pg); st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────
_sidebar()

_PAGES = {
    "landing": page_landing,
    "live":    page_live_cloud if IS_CLOUD else page_live,
    "summary": page_summary,
    "history": page_history,
}
_PAGES.get(st.session_state.page, page_landing)()
