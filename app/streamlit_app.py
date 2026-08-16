"""
AisleGuard — Forklift-Pedestrian Conflict Prediction (Streamlit demo)
=====================================================================

A two-tab interface over the AisleGuard pipeline:

  Tab 1 "Risk Model"       Hand-place a pedestrian and forklift and watch the
                           BRIN network forecast SAFE / CAUTION / IMMINENT live.
                           Isolates the model (rasterise -> BRIN) with no video.

  Tab 2 "Full Pipeline"    Upload any warehouse video and run the complete
                           pipeline end to end: YOLO detection -> BoT-SORT
                           tracking -> approximate floor projection -> BRIN risk,
                           with track-stitching, temporal smoothing, a sticky
                           alert, and a large live verdict overlay.

Design language: industrial safety — dark steel base, hazard-amber accent,
traffic-light risk colours. Memory-safe: loads only the 2 MB model and builds
one scene at a time; it never loads the full raster dataset.
"""

import tempfile
from collections import deque, Counter
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent


# =====================================================================
# Model — BRIN (self-contained so the app has no src import dependency)
# =====================================================================
class ResidualBlock(nn.Module):
    """Two 3x3 convs with a skip connection."""

    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class BRIN(nn.Module):
    """Bird's-eye Risk Inference Network: (B,6,64,64) rasters -> (B,3) logits."""

    def __init__(self, in_channels=6, num_classes=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.res1 = ResidualBlock(32)
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.res2 = ResidualBlock(64)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.res3 = ResidualBlock(128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.down1(x); x = self.res2(x)
        x = self.down2(x); x = self.res3(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


# =====================================================================
# Rasteriser — turn agents into the 6-channel bird's-eye tensor
# =====================================================================
GRID = 64
X_MIN, X_MAX = 0.0, 3.5       # across aisle (metres)
Y_MIN, Y_MAX = 0.0, 20.0      # down aisle (metres)


def rasterise(agents):
    """agents -> (6, GRID, GRID): [ped_occ, fork_occ, ped_vx, ped_vy, fork_vx, fork_vy]."""
    r = np.zeros((6, GRID, GRID), dtype=np.float32)
    for a in agents:
        cx = min(max(int((a["x"] - X_MIN) / (X_MAX - X_MIN) * GRID), 0), GRID - 1)
        cy = min(max(int((a["y"] - Y_MIN) / (Y_MAX - Y_MIN) * GRID), 0), GRID - 1)
        if a["cls"] == "person":
            r[0, cy, cx] = 1.0; r[2, cy, cx] = a["vx"]; r[3, cy, cx] = a["vy"]
        else:
            r[1, cy, cx] = 1.0; r[4, cy, cx] = a["vx"]; r[5, cy, cx] = a["vy"]
    return r


# =====================================================================
# Cached loaders
# =====================================================================
@st.cache_resource
def load_brin():
    m = BRIN()
    m.load_state_dict(torch.load(ROOT / "models/brin/brin_final.pt", map_location="cpu"))
    m.eval()
    return m


@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    return YOLO(str(ROOT / "models/yolo/best.pt"))


brin = load_brin()
CLASSES = ["SAFE", "CAUTION", "IMMINENT"]


def predict_risk(agents):
    """Rasterise a scene and return (predicted_class_index, probability_vector)."""
    r = rasterise(agents)
    with torch.no_grad():
        p = F.softmax(brin(torch.tensor(r[None], dtype=torch.float32)), dim=1)[0].numpy()
    return int(p.argmax()), p


# =====================================================================
# Design tokens — industrial safety
# =====================================================================
INK = "#12151a"; PANEL = "#1b1f27"; LINE = "#2b313c"; AMBER = "#f4b41a"
TEXT = "#e8ecf1"; MUTED = "#8a94a3"
SAFE_C = "#2ec26b"; CAUT_C = "#f4b41a"; IMM_C = "#ff4d4d"
RISK_COL = {"SAFE": SAFE_C, "CAUTION": CAUT_C, "IMMINENT": IMM_C}
RISK_BGR = {0: (107, 194, 46), 1: (26, 180, 244), 2: (77, 77, 255)}  # BGR for cv2

st.set_page_config(page_title="AisleGuard", layout="wide", page_icon="⬢")

st.markdown(f"""
<style>
  .stApp {{ background:{INK}; color:{TEXT}; }}
  h1,h2,h3,h4 {{ color:{TEXT}; font-family:'Inter','Segoe UI',sans-serif; letter-spacing:-0.3px; }}
  .block-container {{ padding-top:2.2rem; padding-bottom:1rem; }}
  .hazard-bar {{ height:7px; width:100%;
    background:repeating-linear-gradient(45deg,{AMBER},{AMBER} 13px,{INK} 13px,{INK} 26px);
    margin:0 0 14px 0; border-radius:2px; }}
  .eyebrow {{ color:{AMBER}; font-size:11px; font-weight:700; letter-spacing:2px; text-transform:uppercase; }}
  .brand {{ font-size:30px; font-weight:800; margin:2px 0 0 0; }}
  .sub {{ color:{MUTED}; font-size:13px; margin-top:1px; }}
  .panel {{ background:{PANEL}; border:1px solid {LINE}; border-radius:10px; padding:14px 16px; }}
  .risk-chip {{ padding:16px; border-radius:10px; text-align:center;
    font-size:26px; font-weight:800; letter-spacing:1px; color:#0d0f13; }}
  .metric {{ color:{MUTED}; font-size:12px; margin-bottom:1px; }}
  .note {{ color:{MUTED}; font-size:12px; font-style:italic; }}
  .stTabs [aria-selected="true"] {{ color:{AMBER}; }}
  .footer {{ color:{MUTED}; font-size:11px; border-top:1px solid {LINE}; padding-top:10px; margin-top:18px; }}
  /* tighten sliders so Tab 1 fits on one screen */
  .stSlider {{ padding-top:0; padding-bottom:0; }}
  .stSlider label {{ font-size:12px !important; color:{MUTED} !important; }}
</style>
""", unsafe_allow_html=True)

# ---- header ----
st.markdown('<div class="eyebrow">Warehouse Safety Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="brand">AisleGuard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Forecasting forklift-pedestrian conflict from a single fixed camera.</div>',
            unsafe_allow_html=True)
st.markdown('<div class="hazard-bar"></div>', unsafe_allow_html=True)


# =====================================================================
# Bird's-eye plot for Tab 1 (compact)
# =====================================================================
def bev_plot(px, py, pvx, pvy, fx, fy, fvx, fvy):
    fig, ax = plt.subplots(figsize=(3.6, 5.2))
    fig.patch.set_facecolor(PANEL); ax.set_facecolor("#0f1218")
    ax.set_xlim(-0.3, 3.8); ax.set_ylim(-0.5, 20.5)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.axvspan(0, 3.5, color="#161b22", alpha=0.6)
    ax.axvline(0, color=AMBER, ls=(0, (6, 4)), alpha=.5, lw=1.1)
    ax.axvline(3.5, color=AMBER, ls=(0, (6, 4)), alpha=.5, lw=1.1)
    ax.scatter([px], [py], s=230, c=IMM_C, edgecolor="white", lw=1.4, zorder=4)
    ax.scatter([fx], [fy], s=420, c="#4a9eff", marker="s", edgecolor="white", lw=1.4, zorder=4)
    if abs(pvx) + abs(pvy) > 0.05:
        ax.add_patch(FancyArrowPatch((px, py), (px + pvx * 0.9, py + pvy * 0.9),
                     color=IMM_C, lw=2, arrowstyle="-|>", mutation_scale=15, zorder=3))
    if abs(fvx) + abs(fvy) > 0.05:
        ax.add_patch(FancyArrowPatch((fx, fy), (fx + fvx * 0.9, fy + fvy * 0.9),
                     color="#4a9eff", lw=2, arrowstyle="-|>", mutation_scale=15, zorder=3))
    ax.set_xlabel("across aisle (m)", color=MUTED, fontsize=8)
    ax.set_ylabel("down aisle (m)", color=MUTED, fontsize=8)
    fig.tight_layout()
    return fig


# =====================================================================
# Preset handling for Tab 1 (set values BEFORE sliders are drawn)
# =====================================================================
def apply_preset(vals):
    for k, v in vals.items():
        st.session_state[k] = v


_defaults = dict(px=1.75, py=10.0, pvx=0.2, pvy=0.0, fx=1.75, fy=8.0, fvx=0.0, fvy=3.0)
for k, v in _defaults.items():
    st.session_state.setdefault(k, v)


tab1, tab2 = st.tabs(["  Risk Model  ", "  Full Pipeline (Video)  "])

# ---------------------------------------------------------------------
# TAB 1 — Risk Model (compact single-screen layout)
# ---------------------------------------------------------------------
with tab1:
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        b = st.columns(2)
        b[0].button("Collision course", use_container_width=True, on_click=apply_preset,
                    args=(dict(px=1.75, py=12.0, pvx=0.0, pvy=-1.4, fx=1.75, fy=6.0, fvx=0.0, fvy=3.0),))
        b[1].button("Safe passing", use_container_width=True, on_click=apply_preset,
                    args=(dict(px=0.3, py=5.0, pvx=0.0, pvy=0.0, fx=3.2, fy=15.0, fvx=0.0, fvy=-3.0),))

        st.markdown("**Pedestrian**")
        px = st.slider("across (m)", 0.0, 3.5, key="px")
        py = st.slider("down (m)", 0.0, 20.0, key="py")
        pc = st.columns(2)
        pvx = pc[0].slider("vel across", -2.0, 2.0, key="pvx")
        pvy = pc[1].slider("vel down", -2.0, 2.0, key="pvy")

        st.markdown("**Forklift**")
        fx = st.slider("across (m) ", 0.0, 3.5, key="fx")
        fy = st.slider("down (m) ", 0.0, 20.0, key="fy")
        fc = st.columns(2)
        fvx = fc[0].slider("vel across ", -4.0, 4.0, key="fvx")
        fvy = fc[1].slider("vel down ", -4.0, 4.0, key="fvy")

    agents = [{"cls": "person", "x": px, "y": py, "vx": pvx, "vy": pvy},
              {"cls": "forklift", "x": fx, "y": fy, "vx": fvx, "vy": fvy}]
    pred, probs = predict_risk(agents)

    with right:
        c = RISK_COL[CLASSES[pred]]
        st.markdown(f'<div class="risk-chip" style="background:{c}">{CLASSES[pred]} · {probs[pred]*100:.0f}%</div>',
                    unsafe_allow_html=True)
        pcols = st.columns(3)
        for i, name in enumerate(CLASSES):
            pcols[i].markdown(f'<div class="metric">{name}</div>', unsafe_allow_html=True)
            pcols[i].progress(float(probs[i]))
        st.pyplot(bev_plot(px, py, pvx, pvy, fx, fy, fvx, fvy))


# ---------------------------------------------------------------------
# TAB 2 — Full pipeline on any video
# ---------------------------------------------------------------------
with tab2:
    st.markdown(
        '<div class="panel">Upload any warehouse video. The full pipeline runs live — '
        'YOLO detection -> BoT-SORT tracking -> floor projection -> BRIN risk. Fragmented '
        'tracks are stitched and risk is temporally smoothed.<br><br>'
        '<span class="note">Floor distances are estimated from frame geometry without '
        'per-camera calibration, so risk is approximate; accurate deployment calibrates '
        'each camera.</span></div>', unsafe_allow_html=True)
    st.write("")

    up = st.file_uploader("Warehouse video", type=["mp4", "avi", "mov"], label_visibility="collapsed")
    ctrl = st.columns(2)
    n = ctrl[0].slider("Frames to process", 20, 200, 80, 10)
    aisle_len = ctrl[1].slider("Assumed aisle length (m)", 8, 30, 15, 1)

    if up and st.button("Run pipeline", use_container_width=True):
        yolo = load_yolo()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(up.read()); path = tmp.name

        cap = cv2.VideoCapture(path)
        W = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920
        Hh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080
        fps = cap.get(cv2.CAP_PROP_FPS) or 24

        def to_metres(cx, cy):
            """Approximate pixel -> metre mapping (no calibration): frame spans the aisle."""
            return (cx / W) * 3.5, (1 - cy / Hh) * aisle_len

        # --- track-stitching state: merge a new id into a recent nearby same-class track ---
        STITCH_DIST, STITCH_GAP = 2.5, 25          # metres, frames
        canon, last_seen = {}, {}                  # raw id -> canonical id ; canonical -> (frame,x,y,cls)

        def stitch(raw_id, cls, xm, ym, fidx):
            if raw_id in canon:
                cid = canon[raw_id]; last_seen[cid] = (fidx, xm, ym, cls); return cid
            best, bestd = None, STITCH_DIST
            for cid, (f0, x0, y0, c0) in last_seen.items():
                if c0 != cls or fidx - f0 > STITCH_GAP:
                    continue
                d = ((xm - x0) ** 2 + (ym - y0) ** 2) ** 0.5
                if d < bestd:
                    best, bestd = cid, d
            cid = best if best is not None else f"{cls}_{len(last_seen)}"
            canon[raw_id] = cid; last_seen[cid] = (fidx, xm, ym, cls)
            return cid

        slot = st.empty(); prog = st.progress(0)
        prev = {}                          # canonical id -> (xm, ym) for velocity
        risk_window = deque(maxlen=5)      # last 5 raw risks for temporal smoothing
        risk_hist = []                     # smoothed risk per frame (for summary)
        stitched_ids = set()
        hold, hold_pred = 0, 0             # sticky-alert state
        done = 0

        while done < n:
            ret, frame = cap.read()
            if not ret:
                break
            res = yolo.track(frame, tracker="botsort.yaml", persist=True, conf=0.25, verbose=False)[0]
            ann = frame.copy()
            agents = []

            if res.boxes.id is not None:
                for box, tid, cls in zip(res.boxes.xyxy.cpu().numpy(),
                                         res.boxes.id.cpu().numpy().astype(int),
                                         res.boxes.cls.cpu().numpy().astype(int)):
                    x1, y1, x2, y2 = box.astype(int)
                    kind = "person" if res.names[cls] == "person" else "forklift"
                    xm, ym = to_metres((x1 + x2) / 2, y2)          # bottom-centre = ground point
                    cid = stitch(int(tid), kind, xm, ym, done)      # stitched id
                    stitched_ids.add(cid)
                    vx = vy = 0.0
                    if cid in prev:
                        vx = (xm - prev[cid][0]) * fps
                        vy = (ym - prev[cid][1]) * fps
                    prev[cid] = (xm, ym)
                    agents.append({"cls": kind, "x": xm, "y": ym,
                                   "vx": np.clip(vx, -5, 5), "vy": np.clip(vy, -5, 5)})
                    disp = cid.replace("person", "P").replace("forklift", "F")
                    col = (77, 158, 255) if kind == "forklift" else (46, 194, 107)
                    cv2.rectangle(ann, (x1, y1), (x2, y2), col, 2)
                    lbl = f"{kind} {disp}"
                    cv2.rectangle(ann, (x1, y1 - 22), (x1 + len(lbl) * 10, y1), col, -1)
                    cv2.putText(ann, lbl, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 22, 28), 2)

            has_person = any(a["cls"] == "person" for a in agents)
            has_fork = any(a["cls"] == "forklift" for a in agents)
            raw_pred, _ = predict_risk(agents) if (has_person and has_fork) else (0, None)

            # temporal smoothing: majority vote over the last 5 frames
            risk_window.append(raw_pred)
            smooth_pred = Counter(risk_window).most_common(1)[0][0]
            risk_hist.append(smooth_pred)

            # sticky alert: hold an alert visible for 15 frames after it fires
            if smooth_pred >= 1 and smooth_pred >= hold_pred:
                hold, hold_pred = 60, smooth_pred
            if hold > 0:
                show_pred, held = hold_pred, True
                hold -= 1
                if hold == 0:
                    hold_pred = 0
            else:
                show_pred, held = smooth_pred, False

            # top status bar (auto-scaled font so text never crops)
            cv2.rectangle(ann, (0, 0), (ann.shape[1], 44), RISK_BGR[show_pred], -1)
            tag = "  * ALERT" if held and show_pred >= 1 else ""
            txt = f"RISK: {CLASSES[show_pred]}"
            if not (has_person and has_fork):
                txt += "  (waiting for both)"
            fscale = max(0.5, min(0.9, ann.shape[1] / 900.0))
            cv2.putText(ann, txt + tag, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, fscale, (255, 255, 255), 2)

            # large live verdict overlay burned into the frame (great for recording)
            if show_pred >= 1 and has_person and has_fork:
                H_img, W_img = ann.shape[:2]
                band_h = int(H_img * 0.14)
                y0 = H_img - band_h
                overlay = ann.copy()
                cv2.rectangle(overlay, (0, y0), (W_img, H_img), RISK_BGR[show_pred], -1)
                cv2.addWeighted(overlay, 0.75, ann, 0.25, 0, ann)
                big = CLASSES[show_pred]
                bscale = max(1.0, W_img / 700.0)
                (tw, th), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_DUPLEX, bscale, 3)
                cv2.putText(ann, big, ((W_img - tw) // 2, y0 + (band_h + th) // 2),
                            cv2.FONT_HERSHEY_DUPLEX, bscale, (255, 255, 255), 3)

            # side-by-side: annotated video + live floor projection
            fig, ax = plt.subplots(figsize=(3.0, 5.0))
            fig.patch.set_facecolor(PANEL); ax.set_facecolor("#0f1218")
            ax.set_xlim(-0.3, 3.8); ax.set_ylim(-0.5, aisle_len + 0.5)
            for s in ax.spines.values():
                s.set_color(LINE)
            ax.tick_params(colors=MUTED, labelsize=7)
            ax.axvline(0, color=AMBER, ls=(0, (6, 4)), alpha=.4)
            ax.axvline(3.5, color=AMBER, ls=(0, (6, 4)), alpha=.4)
            for a in agents:
                col = IMM_C if a["cls"] == "person" else "#4a9eff"
                mk = "o" if a["cls"] == "person" else "s"
                ax.scatter([a["x"]], [a["y"]], s=170, c=col, marker=mk, edgecolor="white", lw=1)
            ax.set_title("Floor projection (approx.)", color=TEXT, fontsize=10)
            ax.set_xlabel("across (m)", color=MUTED, fontsize=8)
            ax.set_ylabel("down (m)", color=MUTED, fontsize=8)
            fig.tight_layout()

            with slot.container():
                a_col, b_col = st.columns([2, 1])
                a_col.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                            caption=f"Frame {done+1}", use_container_width=True)
                b_col.pyplot(fig)
            plt.close(fig)

            done += 1
            prog.progress(done / n)

        cap.release()

        # ---- summary ----
        n_fork = len({c for c in stitched_ids if c.startswith("forklift")})
        n_ped = len({c for c in stitched_ids if c.startswith("person")})
        n_imm = sum(1 for r in risk_hist if r == 2)
        n_caut = sum(1 for r in risk_hist if r == 1)
        st.write("")
        m = st.columns(4)
        m[0].metric("Frames", done)
        m[1].metric("Forklifts", n_fork)
        m[2].metric("People", n_ped)
        m[3].metric("Imminent / Caution", f"{n_imm} / {n_caut}")
        overall = "IMMINENT" if n_imm > 0 else ("CAUTION" if n_caut > 0 else "SAFE")
        st.markdown(f'<div class="risk-chip" style="background:{RISK_COL[overall]}">'
                    f'Video verdict: {overall}'
                    f'{"  — conflict detected" if overall != "SAFE" else ""}</div>',
                    unsafe_allow_html=True)


# ---- footer ----
st.markdown(
    '<div class="footer">BRIN — custom residual CNN, 491k parameters, trained from scratch on '
    '6-channel bird\'s-eye velocity rasters. 93% collision recall vs 0% for a naive baseline. '
    'Detection trained on 23k real warehouse images (forklift mAP@50 0.94).</div>',
    unsafe_allow_html=True)