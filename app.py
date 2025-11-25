import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Core Logic (계산 로직)
# -----------------------------------------------------------------------------
@st.cache_data
def calculate_sasm_metrics(z_mm, wl_nm, px_src_um, W, BLfactor=0.5):
    z = abs(z_mm) * 1e-3
    wl = wl_nm * 1e-9
    px = px_src_um * 1e-6
    
    if W <= 0 or z == 0:
         return {"z_mm": 0, "Sfov_mm": 0, "Valid_FOV_mm": 0, "new_px_um": 0, "Ratio": 0, "theta_deg": 0}

    fov_in = W * px
    sfov_physical = (wl * z) / px if px > 0 else 0
    new_px = sfov_physical / W 
    K = (BLfactor * fov_in) / (2 * z)

    def objective(a):
        if a >= 1.0: return 1e9
        return (a / np.sqrt(1 - a**2) - a) - K

    try:
        grid_limit_a = wl / (2 * px)
        upper_bound = min(0.999, grid_limit_a)
        if objective(upper_bound) < 0:
            a_max = grid_limit_a
        else:
            a_max = brentq(objective, 0, upper_bound)
    except:
        a_max = 0

    theta_deg = np.degrees(np.arcsin(min(1.0, a_max)))
    non_ailiasing_fov = 2 * z * np.tan(np.deg2rad(theta_deg))
    valid_fov = min(non_ailiasing_fov, sfov_physical)

    return {
        "z_mm": z_mm,
        "Sfov_mm": sfov_physical * 1e3,
        "Valid_FOV_mm": valid_fov * 1e3,
        "new_px_um": new_px * 1e6,
        "Ratio": (valid_fov / sfov_physical) * 100 if sfov_physical > 0 else 0,
        "theta_deg": theta_deg,
    }

# -----------------------------------------------------------------------------
# 2. UI Layout (Wide Dashboard)
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Optics Analyzer", page_icon="🔬")

st.markdown("### 🔬 Integrated Optical Analyzer")

# [핵심] 화면을 3분할: [조작부(1)] : [텍스트 리포트(1)] : [그래프(2.2)]
col_ctrl, col_text, col_plot = st.columns([1, 1, 2.2], gap="medium")

# =========================================================
# COLUMN 1: 조작 패널 (Controls)
# =========================================================
with col_ctrl:
    st.info("🎛️ **Control Panel**")
    
    with st.expander("1. Detector Spec", expanded=True):
        w_wav = st.number_input("Wavelength (nm)", 400.0, 1000.0, 532.0, 10.0)
        w_px_det = st.number_input("Det. Pitch (um)", 1.0, 20.0, 3.76, 0.01, format="%.2f")
        w_n = st.number_input("Refractive Index (n)", 1.0, 3.0, 1.0, 0.1)
        w_samp = st.slider("Sampling Rate", 1.0, 10.0, 2.0, 0.1)
        w_angle = st.slider("Max Angle (°)", 5.0, 89.0, 20.0)

    with st.expander("2. Geometry Spec", expanded=True):
        w_ap_out = st.number_input("Ap Out (mm)", 0.1, 50.0, 1.931, 0.1, format="%.3f")
        w_ap_in = st.number_input("Ap In (mm)", 0.1, 50.0, 0.5, 0.1)
        w_size_in = st.number_input("In Mode Size (nm)", 100.0, 5000.0, 350.0, 10.0)

    with st.expander("3. SASM Algo", expanded=True):
        w_px_src = st.number_input("Src Pitch (um)", 0.1, 5.0, 0.35, 0.01)
        w_W = st.slider("Window (W)", 1000, 30000, 10000, 100)
        w_BL = st.slider("BL Factor", 0.1, 1.0, 0.5, 0.05)

# =========================================================
# Calculation (Backend)
# =========================================================
wav = w_wav * 1e-9
px_det = w_px_det * 1e-6
ap_out = w_ap_out * 1e-3
ap_in = w_ap_in * 1e-3
size_in = w_size_in * 1e-9

spk_size = px_det * w_samp
spk_NA = wav / spk_size / 2
spk_NA = min(spk_NA, 1.0)
spk_theta = np.arcsin(spk_NA / w_n) if w_n > 0 else 0

if spk_theta == 0: z_geo = 0
else: z_geo = ap_out / 2 / np.tan(spk_theta)
z_mm = z_geo * 1e3

det_width_m = np.tan(np.deg2rad(w_angle)) * z_geo * 2
n_pixels_det = det_width_m / px_det if px_det > 0 else 0

# Mode Analysis
n_modes_det_1d = det_width_m / spk_size if spk_size > 0 else 0
n_modes_in_1d = (ap_in / size_in) * 2 if size_in > 0 else 0
ratio_1d = n_modes_det_1d / n_modes_in_1d if n_modes_in_1d > 0 else 0

# SASM Prediction
sasm_res = calculate_sasm_metrics(z_mm, w_wav, w_px_src, w_W, w_BL)

# =========================================================
# COLUMN 2: 텍스트 리포트 (Text Report) - 복구됨
# =========================================================
with col_text:
    st.success("📝 **Detailed Report**")
    
    # 예전 스타일의 텍스트 리포트 생성
    report_str = f"""
----------------------------------------
INPUT SUMMARY (SI Units)
----------------------------------------
λ={w_wav:.1f}nm, Px(Det)={w_px_det:.2f}um
Ap(Out)={w_ap_out:.3f}mm, Ap(in)={w_ap_in:.3f}mm
----------------------------------------
GEOMETRY & SPECKLE
----------------------------------------
Spk Size   : {spk_size*1e6:.2f} um
Spk NA     : {spk_NA:.4f}
Dist(Z)    : {z_mm:.2f} mm
Det Width  : {det_width_m*1e3:.2f} mm
Px on Det  : {int(n_pixels_det):,} px
----------------------------------------
[1D] MODE ANALYSIS (Linear)
----------------------------------------
In Modes   : {n_modes_in_1d:.4f}
Det Modes  : {n_modes_det_1d:.4f}
Ratio      : {ratio_1d:.4f}
----------------------------------------
SASM PREDICTION
----------------------------------------
Abs Sfov   : {sasm_res['Sfov_mm']:.2f} mm
Valid FOV  : {sasm_res['Valid_FOV_mm']:.2f} mm
Val Ratio  : {sasm_res['Ratio']:.1f} %
Out Pitch  : {sasm_res['new_px_um']:.2f} um
Max Angle  : {sasm_res['theta_deg']:.2f}°
"""
    # Monospace 폰트로 출력하여 줄맞춤 유지
    st.code(report_str, language="text")

# =========================================================
# COLUMN 3: 그래프 (Plot)
# =========================================================
with col_plot:
    st.warning("📊 **Visualization**")
    
    z_range = np.linspace(1, 60, 100)
    sfov_line = []
    valid_line = []
    det_width_line = []

    for zz in z_range:
        m = calculate_sasm_metrics(zz, w_wav, w_px_src, w_W, w_BL)
        sfov_line.append(m["Sfov_mm"])
        valid_line.append(m["Valid_FOV_mm"])
        dw = np.tan(np.deg2rad(w_angle)) * zz * 1e-3 * 2 * 1e3
        det_width_line.append(dw)

    fig, ax = plt.subplots(figsize=(8, 5.5)) # 세로 길이를 조금 늘림
    
    ax.plot(z_range, sfov_line, "k--", alpha=0.3, label="SASM Raw FOV")
    ax.plot(z_range, valid_line, "r-", linewidth=2.5, label="SASM Valid FOV")
    ax.plot(z_range, det_width_line, "g-.", linewidth=2, label="Physical Det. FOV")
    
    # Points
    curr_valid = sasm_res['Valid_FOV_mm']
    curr_det = det_width_m * 1e3
    
    ax.axvline(z_mm, color="gray", linestyle=":", alpha=0.8)
    
    # 텍스트와 점이 겹치지 않게 어노테이션 추가
    ax.plot(z_mm, curr_valid, "r.", markersize=12, markeredgecolor="w", zorder=5)
    ax.annotate(f"Algo\n{curr_valid:.1f}", (z_mm, curr_valid), xytext=(-20, 5), textcoords='offset points', color='r', fontweight='bold')
    
    ax.plot(z_mm, curr_det, "g.", markersize=10, markeredgecolor="w", zorder=5)
    ax.annotate(f"Det\n{curr_det:.1f}", (z_mm, curr_det), xytext=(10, -15), textcoords='offset points', color='g', fontweight='bold')

    ax.set_title(f"FOV Analysis (Z = {z_mm:.1f} mm)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Propagation Distance (mm)")
    ax.set_ylabel("FOV (mm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 플롯 출력 (반응형)
    st.pyplot(fig, use_container_width=True)
