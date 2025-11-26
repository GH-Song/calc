import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Core Logic
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
# 2. UI Layout
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Optics Analyzer", page_icon="🔬")

# 스타일링: Metric 라벨 크기 조정 및 여백 최소화
st.markdown("""
<style>
    div[data-testid="stMetricValue"] { font-size: 1.2rem; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem; color: #666; }
    .report-box { border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 10px; background-color: #f9f9f9; }
</style>
""", unsafe_allow_html=True)

st.markdown("### 🔬 Integrated Optical Analyzer")

# 비율 조정: [조작부(1)] : [대시보드(1.3)] : [그래프(1.7)]
col_ctrl, col_dash, col_plot = st.columns([1, 1.3, 1.7], gap="medium")

# =========================================================
# COLUMN 1: Inputs (Modified for Compactness)
# =========================================================
with col_ctrl:
    st.info("🎛️ **Settings**")
    
    # 1. Detector Spec
    with st.expander("1. Detector Spec", expanded=True):
        # 2열로 나누어 배치 (gap="small"로 더 밀착)
        c1, c2 = st.columns(2, gap="small")
        
        with c1:
            w_wav = st.number_input("λ (nm)", 400.0, 1000.0, 532.0, 10.0)
            w_n = st.number_input("Ref. Index", 1.0, 3.0, 1.0, 0.1)
        with c2:
            w_px_det = st.number_input("Pitch (um)", 1.0, 20.0, 3.76, 0.01, format="%.2f")
            # Sampling Rate는 Slider 대신 Number Input이 좁은 공간에 더 유리할 수 있으나,
            # 슬라이더 유지를 위해 아래에 배치하거나 여기서 짧게 처리
            pass 

        # 슬라이더들은 조작 편의성을 위해 별도 행이나 꽉 찬 너비가 좋지만, 
        # 컴팩트 요청에 맞춰 2열 혹은 짧은 라벨로 배치
        c3, c4 = st.columns(2, gap="small")
        with c3:
            w_samp = st.slider("Sampling", 1.0, 10.0, 2.0, 0.1)
        with c4:
            w_angle = st.slider("Max Ang(°)", 5.0, 89.0, 20.0)

    # 2. Geometry Spec
    with st.expander("2. Geometry Spec", expanded=True):
        c1, c2 = st.columns(2, gap="small")
        with c1:
            w_ap_out = st.number_input("Ap Out(mm)", 0.1, 50.0, 1.931, 0.1, format="%.3f")
            w_size_in = st.number_input("Mode(nm)", 100.0, 5000.0, 350.0, 10.0)
        with c2:
            w_ap_in = st.number_input("Ap In(mm)", 0.1, 50.0, 0.5, 0.1)
            # 빈 공간 채우기 용 (필요시 추가 위젯)
            st.write("") 

    # 3. SASM Algo
    with st.expander("3. SASM Algo", expanded=True):
        c1, c2 = st.columns(2, gap="small")
        with c1:
            w_px_src = st.number_input("Src Px(um)", 0.1, 5.0, 0.35, 0.01)
            w_BL = st.slider("BL Factor", 0.1, 1.0, 0.5, 0.05)
        with c2:
            # Window 값은 크기 때문에 단독 조절이 편할 수 있으나 2열 배치 시도
            w_W = st.number_input("Win (W)", 1000, 30000, 10000, 100) 
            # (Slider는 공간을 많이 차지하므로 좁은 2열에서는 number_input으로 대체하는 것이 더 깔끔할 수 있음. 
            # 여기선 위 코드의 slider를 number_input으로 교체하거나, slider를 유지하려면 아래처럼 배치)
            
    # Window Slider (Optional: 좁은 폭에서는 Slider 조작이 힘들 수 있어 아래로 뺌)
    # w_W = st.slider("Window (W)", 1000, 30000, 10000, 100) # 위에서 number_input으로 대체함
# =========================================================
# Backend Calculation
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

n_modes_det_1d = det_width_m / spk_size if spk_size > 0 else 0
n_modes_in_1d = (ap_in / size_in) if size_in > 0 else 0
ratio_1d = n_modes_det_1d / n_modes_in_1d if n_modes_in_1d > 0 else 0

n_modes_det_2d = (det_width_m / spk_size)**2 if spk_size > 0 else 0
n_modes_in_2d = (ap_in / size_in)**2 * 2 if size_in > 0 else 0
ratio_2d = n_modes_det_2d / n_modes_in_2d if n_modes_in_2d > 0 else 0

sasm_res = calculate_sasm_metrics(z_mm, w_wav, w_px_src, w_W, w_BL)

# =========================================================
# COLUMN 2: Dashboard (Visualized Text Report)
# =========================================================
with col_dash:
    st.success("📊 **Analysis Dashboard**")

    # 1. Input Summary (Compact)
    with st.container():
        st.caption("🔹 **Input Summary**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("λ (nm)", f"{w_wav:.0f}")
        c2.metric("Px Det", f"{w_px_det}um")
        c3.metric("Ap Out", f"{w_ap_out}mm")
        c4.metric("Ap In", f"{w_ap_in}mm")
    
    st.divider()

    # 2. Geometry & Speckle
    with st.container():
        st.caption("🔹 **Geometry & Speckle**")
        # Grid Layout
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        r1_c1.metric("Prop. Dist (Z)", f"{z_mm:.2f} mm")
        r1_c2.metric("Speckle Size", f"{spk_size*1e6:.2f} um")
        r1_c3.metric("Speckle NA", f"{spk_NA:.4f}")
        
        r2_c1, r2_c2, r2_c3 = st.columns(3)
        r2_c1.metric("Det Width", f"{det_width_m*1e3:.2f} mm")
        r2_c2.metric("Pixels on Det", f"{int(n_pixels_det):,}")
        r2_c3.metric("Max Angle", f"{w_angle}°")

    st.divider()

    # 3. Mode Analysis
    with st.container():
        st.caption("🔹 **Mode Analysis**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Input Modes(1d, intensity)", f"{n_modes_in_1d:.1f}")
        m2.metric("Det. Modes(1d, intensity)", f"{n_modes_det_1d:.1f}")
        # Ratio가 1보다 작으면 빨간색, 크면 기본색
        # m3.metric("Mode Ratio", f"{ratio_1d:.3f}", 
        #           delta="Lossy" if ratio_1d < 1 else "Sufficient",
        #           delta_color="normal" if ratio_1d >= 1 else "inverse")
        
        d1, d2, d3 = st.columns(3)
        d1.metric("Input Modes(2d, complex)", f"{n_modes_in_2d:.1f}")
        d2.metric("Det. Modes(2d, intensity)", f"{n_modes_det_2d:.1f}")
        # Ratio가 1보다 작으면 빨간색, 크면 기본색
        d3.metric("Mode Ratio", f"{ratio_2d:.3f}", 
                  delta="Lossy" if ratio_2d < 1 else "Sufficient",
                  delta_color="normal" if ratio_2d >= 1 else "inverse")
    

    st.divider()

    # 4. SASM Prediction (Highlight)
    with st.container():
        st.caption("🔴 **SASM Algo. Prediction**")
        s1, s2, s3 = st.columns(3)
        s1.metric("Valid FOV", f"{sasm_res['Valid_FOV_mm']:.2f} mm",
                  delta=f"{sasm_res['Ratio']:.1f}% eff.")
        s2.metric("Max Algo Angle", f"{sasm_res['theta_deg']:.2f}°")
        s3.metric("New Pitch", f"{sasm_res['new_px_um']:.2f} um")
        
        # FOV 비교
        diff = sasm_res['Valid_FOV_mm'] - (det_width_m*1e3)
        st.metric("Algo vs Physical FOV", 
                  f"{sasm_res['Valid_FOV_mm']:.2f} vs {det_width_m*1e3:.2f} mm",
                  delta=f"{diff:.2f} mm",
                  delta_color="off")

# =========================================================
# COLUMN 3: Plot
# =========================================================
with col_plot:
    st.warning("📈 **Visualization**")
    
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

    # 그래프 세로 길이를 늘려서 정보가 많아진 가운데 열과 균형 맞춤
    fig, ax = plt.subplots(figsize=(7, 6)) 
    
    ax.plot(z_range, sfov_line, "k--", alpha=0.3, label="SASM Raw FOV")
    ax.plot(z_range, valid_line, "r-", linewidth=2.5, label="SASM Valid FOV")
    ax.plot(z_range, det_width_line, "g-.", linewidth=2, label="Physical Det. FOV")
    
    curr_valid = sasm_res['Valid_FOV_mm']
    curr_det = det_width_m * 1e3
    
    ax.axvline(z_mm, color="gray", linestyle=":", alpha=0.8)
    
    # Annotation
    ax.plot(z_mm, curr_valid, "r.", markersize=14, markeredgecolor="w", zorder=5)
    ax.annotate(f"Algo\n{curr_valid:.1f}", (z_mm, curr_valid), 
                xytext=(-25, 5), textcoords='offset points', 
                color='r', fontweight='bold', fontsize=10)
    
    ax.plot(z_mm, curr_det, "g.", markersize=12, markeredgecolor="w", zorder=5)
    ax.annotate(f"Det\n{curr_det:.1f}", (z_mm, curr_det), 
                xytext=(10, -20), textcoords='offset points', 
                color='g', fontweight='bold', fontsize=10)

    ax.set_title(f"FOV Analysis (Z = {z_mm:.1f} mm)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Propagation Distance (mm)", fontsize=11)
    ax.set_ylabel("FOV (mm)", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig, use_container_width=True)




