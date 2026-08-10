"""
AisleGuard — Forklift-Pedestrian Conflict Prediction
Tab 1: Risk model demo (BRIN forecasts collision risk from agent positions/motion)
Tab 2: Full pipeline on ANY video — detection, tracking, approximate floor
       projection, and BRIN risk. Risk is approximate on uncalibrated cameras.
"""
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import tempfile
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent

# ======================================================================
# Model
# ======================================================================
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1); self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1); self.bn2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        idn = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + idn)

class BRIN(nn.Module):
    def __init__(self, in_channels=6, num_classes=3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.res1 = ResidualBlock(32)
        self.down1 = nn.Sequential(nn.Conv2d(32,64,3,stride=2,padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.res2 = ResidualBlock(64)
        self.down2 = nn.Sequential(nn.Conv2d(64,128,3,stride=2,padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.res3 = ResidualBlock(128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64,num_classes))
    def forward(self, x):
        x = self.stem(x); x = self.res1(x)
        x = self.down1(x); x = self.res2(x)
        x = self.down2(x); x = self.res3(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

GRID = 64; X_MIN, X_MAX = 0.0, 3.5; Y_MIN, Y_MAX = 0.0, 20.0
def rasterise(agents):
    r = np.zeros((6, GRID, GRID), dtype=np.float32)
    for a in agents:
        cx = min(max(int((a["x"]-X_MIN)/(X_MAX-X_MIN)*GRID), 0), GRID-1)
        cy = min(max(int((a["y"]-Y_MIN)/(Y_MAX-Y_MIN)*GRID), 0), GRID-1)
        if a["cls"] == "person":
            r[0,cy,cx]=1.0; r[2,cy,cx]=a["vx"]; r[3,cy,cx]=a["vy"]
        else:
            r[1,cy,cx]=1.0; r[4,cy,cx]=a["vx"]; r[5,cy,cx]=a["vy"]
    return r

@st.cache_resource
def load_brin():
    m = BRIN(); m.load_state_dict(torch.load(ROOT/"models/brin/brin_final.pt", map_location="cpu")); m.eval()
    return m
@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    return YOLO(str(ROOT/"models/yolo/best.pt"))

brin = load_brin()
CLASSES = ["SAFE", "CAUTION", "IMMINENT"]

def predict_risk(agents):
    r = rasterise(agents)
    with torch.no_grad():
        p = F.softmax(brin(torch.tensor(r[None], dtype=torch.float32)), dim=1)[0].numpy()
    return int(p.argmax()), p

# ======================================================================
# Design tokens
# ======================================================================
INK="#12151a"; PANEL="#1b1f27"; LINE="#2b313c"; AMBER="#f4b41a"
TEXT="#e8ecf1"; MUTED="#8a94a3"
SAFE_C="#2ec26b"; CAUT_C="#f4b41a"; IMM_C="#ff4d4d"
RISK_COL={"SAFE":SAFE_C,"CAUTION":CAUT_C,"IMMINENT":IMM_C}
RISK_BGR={0:(107,194,46),1:(26,180,244),2:(77,77,255)}  # BGR for cv2

st.set_page_config(page_title="AisleGuard", layout="wide", page_icon="⬢")
st.markdown(f"""
<style>
  .stApp {{ background:{INK}; color:{TEXT}; }}
  h1,h2,h3,h4 {{ color:{TEXT}; font-family:'Inter','Segoe UI',sans-serif; letter-spacing:-0.3px; }}
  .hazard-bar {{ height:8px; width:100%;
    background:repeating-linear-gradient(45deg,{AMBER},{AMBER} 14px,{INK} 14px,{INK} 28px);
    margin:0 0 18px 0; border-radius:2px; }}
  .eyebrow {{ color:{AMBER}; font-size:12px; font-weight:700; letter-spacing:2px; text-transform:uppercase; }}
  .brand {{ font-size:34px; font-weight:800; margin:2px 0 0 0; }}
  .sub {{ color:{MUTED}; font-size:14px; margin-top:2px; }}
  .panel {{ background:{PANEL}; border:1px solid {LINE}; border-radius:12px; padding:16px 18px; }}
  .risk-chip {{ padding:20px; border-radius:12px; text-align:center;
    font-size:30px; font-weight:800; letter-spacing:1px; color:#0d0f13; }}
  .metric {{ color:{MUTED}; font-size:13px; }}
  .note {{ color:{MUTED}; font-size:12px; font-style:italic; }}
  .stTabs [aria-selected="true"] {{ color:{AMBER}; }}
  .footer {{ color:{MUTED}; font-size:12px; border-top:1px solid {LINE}; padding-top:12px; margin-top:24px; }}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">Warehouse Safety Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="brand">AisleGuard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Forecasting forklift–pedestrian conflict from a single fixed camera.</div>', unsafe_allow_html=True)
st.markdown('<div class="hazard-bar"></div>', unsafe_allow_html=True)

def bev_plot(px,py,pvx,pvy,fx,fy,fvx,fvy):
    fig, ax = plt.subplots(figsize=(4.2, 6.4))
    fig.patch.set_facecolor(PANEL); ax.set_facecolor("#0f1218")
    ax.set_xlim(-0.3,3.8); ax.set_ylim(-0.5,20.5)
    for s in ["top","right","bottom","left"]: ax.spines[s].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.axvspan(0,3.5, color="#161b22", alpha=0.6)
    ax.axvline(0,color=AMBER,ls=(0,(6,4)),alpha=.5,lw=1.2)
    ax.axvline(3.5,color=AMBER,ls=(0,(6,4)),alpha=.5,lw=1.2)
    ax.scatter([px],[py], s=260, c=IMM_C, edgecolor="white", lw=1.5, zorder=4)
    ax.scatter([fx],[fy], s=460, c="#4a9eff", marker="s", edgecolor="white", lw=1.5, zorder=4)
    if abs(pvx)+abs(pvy)>0.05:
        ax.add_patch(FancyArrowPatch((px,py),(px+pvx*0.9,py+pvy*0.9),color=IMM_C,lw=2,arrowstyle="-|>",mutation_scale=16,zorder=3))
    if abs(fvx)+abs(fvy)>0.05:
        ax.add_patch(FancyArrowPatch((fx,fy),(fx+fvx*0.9,fy+fvy*0.9),color="#4a9eff",lw=2,arrowstyle="-|>",mutation_scale=16,zorder=3))
    ax.set_xlabel("across aisle (m)", color=MUTED, fontsize=9)
    ax.set_ylabel("down aisle (m)", color=MUTED, fontsize=9)
    fig.tight_layout()
    return fig

def apply_preset(vals):
    for k, v in vals.items():
        st.session_state[k] = v
_defaults = dict(px=1.75, py=10.0, pvx=0.2, pvy=0.0, fx=1.75, fy=8.0, fvx=0.0, fvy=3.0)
for k, v in _defaults.items():
    st.session_state.setdefault(k, v)

tab1, tab2 = st.tabs(["  Risk Model  ", "  Full Pipeline (Video)  "])

# ---------------------------------------------------------------- TAB 1
with tab1:
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        b = st.columns(2)
        b[0].button("Collision course", use_container_width=True, on_click=apply_preset,
                    args=(dict(px=1.75,py=12.0,pvx=0.0,pvy=-1.4,fx=1.75,fy=6.0,fvx=0.0,fvy=3.0),))
        b[1].button("Safe passing", use_container_width=True, on_click=apply_preset,
                    args=(dict(px=0.3,py=5.0,pvx=0.0,pvy=0.0,fx=3.2,fy=15.0,fvx=0.0,fvy=-3.0),))
        st.markdown("##### Pedestrian")
        px  = st.slider("position across (m)", 0.0, 3.5, key="px")
        py  = st.slider("position down (m)",   0.0, 20.0, key="py")
        pc  = st.columns(2)
        pvx = pc[0].slider("velocity across", -2.0, 2.0, key="pvx")
        pvy = pc[1].slider("velocity down",   -2.0, 2.0, key="pvy")
        st.markdown("##### Forklift")
        fx  = st.slider("position across (m) ", 0.0, 3.5, key="fx")
        fy  = st.slider("position down (m) ",   0.0, 20.0, key="fy")
        fc  = st.columns(2)
        fvx = fc[0].slider("velocity across ", -4.0, 4.0, key="fvx")
        fvy = fc[1].slider("velocity down ",   -4.0, 4.0, key="fvy")

    agents = [{"cls":"person","x":px,"y":py,"vx":pvx,"vy":pvy},
              {"cls":"forklift","x":fx,"y":fy,"vx":fvx,"vy":fvy}]
    pred, probs = predict_risk(agents)
    with right:
        c = RISK_COL[CLASSES[pred]]
        st.markdown(f'<div class="risk-chip" style="background:{c}">{CLASSES[pred]} · {probs[pred]*100:.0f}%</div>',
                    unsafe_allow_html=True)
        st.write("")
        for i,name in enumerate(CLASSES):
            st.markdown(f'<div class="metric">{name} — {probs[i]*100:.1f}%</div>', unsafe_allow_html=True)
            st.progress(float(probs[i]))
        st.pyplot(bev_plot(px,py,pvx,pvy,fx,fy,fvx,fvy))

# ============ REPLACE the whole "with tab2:" block with this ============
# Two new pieces vs the previous version:
#   1. TRACK-STITCHING: merge a new track id into a recent one of the SAME class
#      whose last known position is close by — fixes one person counted as two.
#   2. TEMPORAL SMOOTHING: keep the last 5 raw risk predictions and use the
#      majority vote, so a single noisy frame can't flip the readout.
with tab2:
    st.markdown(
        '<div class="panel">Upload any warehouse video. The full pipeline runs live — '
        'YOLO detection → BoT-SORT tracking → floor projection → BRIN risk. Fragmented '
        'tracks are stitched and risk is temporally smoothed for a steadier readout.<br><br>'
        '<span class="note">Note: without per-camera calibration, floor distances are '
        'estimated from frame geometry, so risk is approximate — accurate deployment '
        'calibrates each camera (as done for the reference camera in this project).</span></div>',
        unsafe_allow_html=True)
    st.write("")
    up = st.file_uploader("Warehouse video", type=["mp4","avi","mov"], label_visibility="collapsed")
    ncol = st.columns(2)
    n = ncol[0].slider("Frames to process", 20, 200, 80, 10)
    aisle_len = ncol[1].slider("Assumed aisle length (m)", 8, 30, 15, 1)

    if up and st.button("Run pipeline", use_container_width=True):
        from collections import deque, Counter
        yolo = load_yolo()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(up.read()); path = tmp.name
        cap = cv2.VideoCapture(path)
        W  = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920
        Hh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080
        fps = cap.get(cv2.CAP_PROP_FPS) or 24

        def to_metres(cx, cy):
            xm = (cx / W) * 3.5
            ym = (1 - cy / Hh) * aisle_len
            return xm, ym

        # ---- track-stitching state ----
        STITCH_DIST = 2.5      # metres: how close a reappearing agent must be to be "the same"
        STITCH_GAP  = 25       # frames: how long a track can vanish and still be re-linked
        canon = {}             # raw track id -> canonical (stitched) id
        last_seen = {}         # canonical id -> (frame_idx, xm, ym, cls)

        def stitch(raw_id, cls, xm, ym, fidx):
            # already know this raw id
            if raw_id in canon:
                cid = canon[raw_id]
                last_seen[cid] = (fidx, xm, ym, cls)
                return cid
            # try to attach to a recent canonical track of same class, nearby
            best, bestd = None, STITCH_DIST
            for cid,(f0,x0,y0,c0) in last_seen.items():
                if c0 != cls: continue
                if fidx - f0 > STITCH_GAP: continue
                d = ((xm-x0)**2 + (ym-y0)**2) ** 0.5
                if d < bestd:
                    best, bestd = cid, d
            if best is not None:
                canon[raw_id] = best
                last_seen[best] = (fidx, xm, ym, cls)
                return best
            # otherwise this is a genuinely new agent
            cid = f"{cls}_{len(last_seen)}"
            canon[raw_id] = cid
            last_seen[cid] = (fidx, xm, ym, cls)
            return cid

        slot = st.empty(); prog = st.progress(0)
        prev = {}                       # canonical id -> (xm, ym) for velocity
        risk_window = deque(maxlen=5)   # last 5 raw risk classes for smoothing
        risk_hist = []                  # smoothed risk per frame (for summary)
        stitched_ids = set()
        hold = 0; hold_pred = 0
        done = 0

        while done < n:
            ret, frame = cap.read()
            if not ret: break
            res = yolo.track(frame, tracker="botsort.yaml", persist=True, conf=0.25, verbose=False)[0]
            ann = frame.copy()
            agents = []
            if res.boxes.id is not None:
                for box,tid,cls in zip(res.boxes.xyxy.cpu().numpy(),
                                       res.boxes.id.cpu().numpy().astype(int),
                                       res.boxes.cls.cpu().numpy().astype(int)):
                    x1,y1,x2,y2 = box.astype(int)
                    name = res.names[cls]
                    kind = "person" if name=="person" else "forklift"
                    cxp, cyp = (x1+x2)/2, y2
                    xm, ym = to_metres(cxp, cyp)
                    cid = stitch(int(tid), kind, xm, ym, done)   # <-- stitched id
                    stitched_ids.add(cid)
                    vx=vy=0.0
                    if cid in prev:
                        vx=(xm-prev[cid][0])*fps; vy=(ym-prev[cid][1])*fps
                    prev[cid] = (xm,ym)
                    agents.append({"cls":kind,"x":xm,"y":ym,
                                   "vx":np.clip(vx,-5,5),"vy":np.clip(vy,-5,5)})
                    # display the stitched id (short, per-class counter)
                    disp = cid.replace("person","P").replace("forklift","F")
                    col = (77,158,255) if kind=="forklift" else (46,194,107)
                    cv2.rectangle(ann,(x1,y1),(x2,y2),col,2)
                    lbl = f"{kind} {disp}"
                    cv2.rectangle(ann,(x1,y1-22),(x1+len(lbl)*10,y1),col,-1)
                    cv2.putText(ann,lbl,(x1+3,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(20,22,28),2)

            has_person = any(a["cls"]=="person" for a in agents)
            has_fork   = any(a["cls"]=="forklift" for a in agents)
            if has_person and has_fork:
                raw_pred, pp = predict_risk(agents)
            else:
                raw_pred, pp = 0, np.array([1.0,0.0,0.0])

            # ---- temporal smoothing: majority vote over last 5 frames ----
            risk_window.append(raw_pred)
            smooth_pred = Counter(risk_window).most_common(1)[0][0]
            risk_hist.append(smooth_pred)

            # ---- sticky alert on the SMOOTHED prediction ----
            if smooth_pred >= 1 and smooth_pred >= hold_pred:
                hold = 15; hold_pred = smooth_pred
            if hold > 0:
                show_pred = hold_pred; held = True; hold -= 1
                if hold == 0: hold_pred = 0
            else:
                show_pred = smooth_pred; held = False

            lc = RISK_BGR[show_pred]
            cv2.rectangle(ann,(0,0),(ann.shape[1],46),lc,-1)
            tag = "  ⚠ ALERT HELD" if held and show_pred>=1 else ""
            txt = f"RISK: {CLASSES[show_pred]}"
            if not (has_person and has_fork):
                txt += "   [need forklift + person in view]"
            cv2.putText(ann,txt+tag,(12,32),cv2.FONT_HERSHEY_SIMPLEX,0.85,(255,255,255),2)

            fig, ax = plt.subplots(figsize=(3.2,5.2))
            fig.patch.set_facecolor(PANEL); ax.set_facecolor("#0f1218")
            ax.set_xlim(-0.3,3.8); ax.set_ylim(-0.5,aisle_len+0.5)
            for s in ax.spines.values(): s.set_color(LINE)
            ax.tick_params(colors=MUTED, labelsize=7)
            ax.axvline(0,color=AMBER,ls=(0,(6,4)),alpha=.4); ax.axvline(3.5,color=AMBER,ls=(0,(6,4)),alpha=.4)
            for a in agents:
                col = IMM_C if a["cls"]=="person" else "#4a9eff"
                mk  = "o" if a["cls"]=="person" else "s"
                ax.scatter([a["x"]],[a["y"]], s=180, c=col, marker=mk, edgecolor="white", lw=1)
            ax.set_title("Floor projection (approx.)", color=TEXT, fontsize=10)
            ax.set_xlabel("across (m)", color=MUTED, fontsize=8); ax.set_ylabel("down (m)", color=MUTED, fontsize=8)
            fig.tight_layout()

            with slot.container():
                a,b = st.columns([2,1])
                a.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                        caption=f"Frame {done+1}", use_container_width=True)
                b.pyplot(fig)
            plt.close(fig)
            done += 1; prog.progress(done/n)

        cap.release()
        n_fork = len({c for c in stitched_ids if c.startswith("forklift")})
        n_ped  = len({c for c in stitched_ids if c.startswith("person")})
        n_imm  = sum(1 for r in risk_hist if r==2)
        n_caut = sum(1 for r in risk_hist if r==1)
        st.write("")
        m = st.columns(4)
        m[0].metric("Frames", done)
        m[1].metric("Forklifts tracked", n_fork)
        m[2].metric("People tracked", n_ped)
        m[3].metric("Imminent / Caution frames", f"{n_imm} / {n_caut}")
        overall = "IMMINENT" if n_imm>0 else ("CAUTION" if n_caut>0 else "SAFE")
        oc = RISK_COL[overall]
        st.markdown(f'<div class="risk-chip" style="background:{oc}">Video verdict: {overall}'
                    f'{"  — conflict detected" if overall!="SAFE" else ""}</div>',
                    unsafe_allow_html=True)