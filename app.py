import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Core Logic (계산 로직 - 캐싱 적용)
# -----------------------------------------------------------------------------
@st.cache_data  # 입력값이 같으면 재계산 하지 않고 결과를 불러옴 (속도 향상)
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
# 2. UI Layout (Dashboard Style)
# -----------------------------------------------------------------------------
# [핵심 1] 화면 전체 사용 설정
st.set_page_config(layout="wide", page_title="Optics Analyzer", page_icon="🔬")

st.markdown("### 🔬 Integrated Optical Analyzer")

# [핵심 2] 화면을 좌(입력) : 우(결과) = 1 : 3 비율로 분할
col_ctrl, col_view = st.columns([1, 2.5], gap="medium")

# --- [왼쪽] 조작 패널 (Controls) ---
with col_ctrl:
    st.markdown("#### 🎛️ Settings")
    
    with st.expander("1. Detector Spec", expanded=True):
        w_wav = st.number_input("Wavelength (nm)", 400.0, 1000.0, 532.0, 10.0)
        w_px_det = st.number_input("Det. Pitch (um)", 1.0, 20.0, 3.76, 0.1)
        w_n = st.number_input("Refractive Index (n)", 1.0, 3.0, 1.0, 0.1)
        w_samp = st.slider("Sampling Rate", 1.0, 5.0, 2.0, 0.1)
        w_angle = st.slider("Max Detect Angle (°)", 5.0, 89.0, 20.0)

    with st.expander("2. Geometry Spec", expanded=True):
        w_ap_out = st.number_input("Aperture Out (mm)", 0.1, 50.0, 1.0, 0.5)
        w_ap_in = st.number_input("Aperture In (mm)", 0.1, 50.0, 1.0, 0.5)
        w_size_in = st.number_input("Input Size (nm)", 100.0, 5000.0, 350.0, 50.0)

    with st.expander("3. SASM Algo", expanded=True):
        w_px_src = st.number_input("Source Pitch (um)", 0.1, 5.0, 0.35, 0.05)
        w_W = st.slider("Comp. Window (W)", 1000, 20000, 10000, 500)
        w_BL = st.slider("BL Factor", 0.1, 1.0, 0.5, 0.1)

# --- [오른쪽] 결과 뷰 (Results) ---
with col_view:
    # 1) 실시간 계산
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
    sasm_res = calculate_sasm_metrics(z_mm, w_wav, w_px_src, w_W, w_BL)
    
    # 2) Key Metrics 표시 (가로로 배치)
    st.markdown("#### 📊 Analysis Result")
    
    # 지표를 담을 4개의 컨테이너 박스 생성
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Propagation Z", f"{z_mm:.2f} mm", delta="Auto Calc")
    with m2:
        st.metric("Valid FOV (Algo)", f"{sasm_res['Valid_FOV_mm']:.2f} mm", 
                  delta=f"{sasm_res['Ratio']:.1f}% Use")
    with m3:
        st.metric("Detector FOV", f"{det_width_m*1e3:.2f} mm",
                  delta=f"Diff: {sasm_res['Valid_FOV_mm'] - det_width_m*1e3:.1f} mm",
                  delta_color="inverse")
    with m4:
        st.metric("Algo Pitch (Out)", f"{sasm_res['new_px_um']:.2f} um")

    st.divider()

    # 3) Plot 그리기 (반응형)
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

    # Matplotlib 스타일 조정
    fig, ax = plt.subplots(figsize=(10, 5)) # 비율만 설정 (크기는 streamlit이 늘림)
    
    ax.plot(z_range, sfov_line, "k--", alpha=0.3, label="SASM Raw FOV")
    ax.plot(z_range, valid_line, "r-", linewidth=2.5, label="SASM Valid FOV")
    ax.plot(z_range, det_width_line, "g-.", linewidth=2, label="Physical Det. FOV")
    
    # 현재 포인트
    curr_valid = sasm_res['Valid_FOV_mm']
    curr_det = det_width_m * 1e3
    
    ax.axvline(z_mm, color="gray", linestyle=":", alpha=0.8)
    ax.plot(z_mm, curr_valid, "ro", markersize=10, markeredgecolor="w", zorder=5)
    ax.plot(z_mm, curr_det, "go", markersize=8, markeredgecolor="w", zorder=5)
    
    ax.set_title("Field of View Analysis", fontsize=12, fontweight='bold')
    ax.set_xlabel("Propagation Distance (mm)")
    ax.set_ylabel("FOV (mm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # [핵심 3] use_container_width=True 로 설정하여 컬럼 너비에 꽉 차게 그림
    st.pyplot(fig, use_container_width=True)
