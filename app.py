import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import streamlit as st


# ==========================================
# 1. SASM Core Logic (Source Domain)
# ==========================================
def calculate_sasm_metrics(z_mm, wl_nm, px_src_um, W, BLfactor=0.5):
    """SASM 계산 로직"""
    z = abs(z_mm) * 1e-3
    wl = wl_nm * 1e-9
    px = px_src_um * 1e-6

    # 예외 처리: W가 0이거나 z가 0일 때
    if W <= 0 or z == 0:
        return {"z_mm": 0, "Sfov_mm": 0, "Valid_FOV_mm": 0, "new_px_um": 0, "Ratio": 0, "is_masked": False, "theta_deg": 0}

    fov_in = W * px
    sfov_physical = (wl * z) / px if px > 0 else 0
    new_px = sfov_physical / W

    K = (BLfactor * fov_in) / (2 * z)

    def objective(a):
        if a >= 1.0:
            return 1e9
        return (a / np.sqrt(1 - a**2) - a) - K

    try:
        grid_limit_a = wl / (2 * px)
        upper_bound = min(0.999, grid_limit_a)

        if objective(upper_bound) < 0:
            a_max = grid_limit_a
            is_masked = False
        else:
            a_max = brentq(objective, 0, upper_bound)
            is_masked = True
    except:
        a_max = 0
        is_masked = True

    theta_deg = np.degrees(np.arcsin(min(1.0, a_max)))
    non_ailiasing_fov = 2 * z * np.tan(np.deg2rad(theta_deg))
    valid_fov = min(non_ailiasing_fov, sfov_physical)

    return {
        "z_mm": z_mm,
        "Sfov_mm": sfov_physical * 1e3,
        "Valid_FOV_mm": valid_fov * 1e3,
        "new_px_um": new_px * 1e6,
        "Ratio": (valid_fov / sfov_physical) * 100 if sfov_physical > 0 else 0,
        "is_masked": is_masked,
        "theta_deg": theta_deg,
    }


# ==========================================
# 2. Streamlit UI Layout
# ==========================================
st.set_page_config(layout="wide", page_title="Optics Analyzer")

st.title("Integrated Optical Analyzer")
st.markdown("Computational Optics System Analyzer for Lensless Imaging")

# --- Layout: 2 Columns for Controls ---
col_ctrl_1, col_ctrl_2 = st.columns([1, 1])

with col_ctrl_1:
    st.subheader("[1] Physical Hardware (Detector)")
    w_wav = st.number_input("Wavelength (nm)", value=532.0, step=1.0, format="%.1f")
    w_px_det = st.number_input("Detector Pitch (um)", value=3.76, step=0.01, format="%.2f")
    w_n = st.number_input("Refractive Index (n)", value=1.0, format="%.1f")
    st.divider()
    w_ap_out = st.number_input("Aperture Out (mm)", value=1.0, step=0.1)
    w_ap_in = st.number_input("Aperture In (mm)", value=1.0, step=0.1)
    w_size_in = st.number_input("Input Mode Size (nm)", value=350.0, step=10.0)
    st.divider()
    w_samp = st.slider("Sampling Rate", 1.0, 10.0, 2.0, 0.1)
    w_angle = st.slider("Detect Angle (°)", 1.0, 89.0, 20.0, 0.5)

with col_ctrl_2:
    st.subheader("[2] Algorithm Settings (Source)")
    w_px_src = st.number_input("Source Pixel (um)", value=0.35, step=0.01, format="%.2f")
    w_W = st.slider("Comp. Window (W)", 100, 30000, 10000, 100)
    w_BL = st.slider("BL Factor", 0.1, 1.0, 0.5, 0.05)
    st.info("* Green Line: Physical Detector FOV limit.\n* Red Line: SASM Algorithm Valid FOV.")

# --- Calculation Logic ---
# Inputs
wav = w_wav * 1e-9
px_det = w_px_det * 1e-6
ap_out = w_ap_out * 1e-3
ap_in = w_ap_in * 1e-3
size_in = w_size_in * 1e-9

# Geometry
spk_size = px_det * w_samp
spk_NA = wav / spk_size / 2 if spk_size > 0 else 0
spk_NA = min(spk_NA, 1.0)
spk_theta = np.arcsin(spk_NA / w_n) if w_n > 0 else 0

if spk_theta == 0:
    z_geo = 0
else:
    z_geo = ap_out / 2 / np.tan(spk_theta)
z_mm = z_geo * 1e3

det_width_m = np.tan(np.deg2rad(w_angle)) * z_geo * 2
n_pixels_det = det_width_m / px_det if px_det > 0 else 0

# Mode Analysis
n_modes_det_1d = det_width_m / spk_size if spk_size > 0 else 0
n_modes_in_1d = (ap_in / size_in) * 2 if size_in > 0 else 0
ratio_1d = n_modes_det_1d / n_modes_in_1d if n_modes_in_1d > 0 else 0

# SASM Prediction
sasm_res = calculate_sasm_metrics(z_mm, w_wav, w_px_src, w_W, w_BL)

# --- Layout: Output (Text & Plot) ---
st.divider()
col_res_text, col_res_plot = st.columns([1, 1.5])

with col_res_text:
    st.markdown("### Analysis Report")
    st.text(
        f"GEOMETRY & SPECKLE\n"
        f"----------------------------------------\n"
        f"Spk Size    : {spk_size * 1e6:.2f} um\n"
        f"Spk NA      : {spk_NA:.4f}\n"
        f"Dist(Z)     : {z_mm:.2f} mm\n"
        f"Det Width   : {det_width_m * 1e3:.2f} mm\n"
        f"Px on Det   : {int(n_pixels_det):,} px"
    )

    st.text(
        f"[1D] MODE ANALYSIS\n"
        f"----------------------------------------\n"
        f"In Modes    : {n_modes_in_1d:.4f}\n"
        f"Det Modes   : {n_modes_det_1d:.4f}\n"
        f"Ratio       : {ratio_1d:.4f}"
    )

    st.text(
        f"SASM PREDICTION\n"
        f"----------------------------------------\n"
        f"Abs Sfov    : {sasm_res['Sfov_mm']:.2f} mm\n"
        f"Valid FOV   : {sasm_res['Valid_FOV_mm']:.2f} mm\n"
        f"Val Ratio   : {sasm_res['Ratio']:.1f} %\n"
        f"Out Pitch   : {sasm_res['new_px_um']:.2f} um\n"
        f"Max Angle   : {sasm_res['theta_deg']:.2f}°"
    )

with col_res_plot:
    # Plotting
    z_range = np.linspace(1, 60, 60)
    sfov_line = []
    valid_line = []
    det_width_line = []

    for zz in z_range:
        m = calculate_sasm_metrics(zz, w_wav, w_px_src, w_W, w_BL)
        sfov_line.append(m["Sfov_mm"])
        valid_line.append(m["Valid_FOV_mm"])
        dw = np.tan(np.deg2rad(w_angle)) * zz * 1e-3 * 2
        det_width_line.append(dw * 1e3)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(z_range, sfov_line, "k--", alpha=0.3, label="SASM Scaled")
    ax.plot(z_range, valid_line, "r-", linewidth=2, label="SASM Valid")
    ax.plot(z_range, det_width_line, "g-.", linewidth=1.5, label="Detector")

    # Points
    valid_fov_curr = sasm_res["Valid_FOV_mm"]
    det_width_curr = det_width_m * 1e3

    ax.axvline(z_mm, color="black", linestyle=":", alpha=0.5)
    ax.plot(z_mm, valid_fov_curr, "r.", markersize=10, markeredgecolor="k")
    ax.plot(z_mm, det_width_curr, "g.", markersize=8, markeredgecolor="k")

    ax.set_title("FOV Analysis")
    ax.set_xlabel("Propagation Distance z [mm]")
    ax.set_ylabel("Field of View [mm]")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
