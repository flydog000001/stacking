# -*- coding: utf-8 -*-
"""
GRACE 驱动的孔弹性形变计算引擎
================================================================================
输入: 逐站地下水储量 GWS 时序(月, mm 等效水高 EWH)
处理: GWS -> 水头 h = GWS/Sy -> FFT 分解 -> 半空间孔弹(论文 Eq.3-10)-> 逐日孔弹
这是论文"由地下水储量驱动"的核心(Fig.6/Fig.8 红色 GWS 曲线), 物理内核与 v3 相同。
引用: Ding et al. (2025), Eq.3-10, Table 2。
================================================================================
"""
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
RHO, G = 1000.0, 9.80665
SEC_PER_DAY = 86400.0

MATERIAL_PARAMS = {   # 论文 Table 2 (含 Sy, GRACE 驱动必需)
    "bedrock":     dict(nu=0.25, beta_p=3.24e-10, c=0.10,   Sy=0.15),
    "sand":        dict(nu=0.25, beta_p=5.30e-9,  c=0.50,   Sy=0.15),
    "clay":        dict(nu=0.10, beta_p=7.05e-8,  c=1.0e-5, Sy=0.01),
    "clay_low_sy": dict(nu=0.10, beta_p=1.29e-7,  c=1.0e-7, Sy=0.005),  # <0.01, 可逐站调
}

def seasonal_band(x, lo_days=120, hi_days=600):
    """FFT 带通保留季节带(对应论文小波 D6-D8), 滤掉长期/年际(非弹性驱动)与高频。"""
    n = x.size; X = np.fft.rfft(x - x.mean()); f = np.fft.rfftfreq(n, d=1.0)
    per = np.divide(1.0, f, out=np.full_like(f, 1e12), where=f > 0)
    X[(per < lo_days) | (per > hi_days)] = 0.0
    return np.fft.irfft(X, n=n)

def poroelastic_transfer(head, dt, nu, beta_p, c, b=np.inf):
    """论文 Eq.10: u = (1+nu)/(1-nu)*(beta_p/3)*rho*g*sqrt(c/w)*h*exp(-i*pi/4)。b=inf 为纯半空间。"""
    N = head.size; H = np.fft.rfft(head); w = 2 * np.pi * np.fft.rfftfreq(N, d=dt)
    A = (1 + nu) / (1 - nu) * beta_p / 3
    Gk = np.zeros_like(H); pos = w > 0; k = np.sqrt(w[pos] / (2 * c))
    integ = 1 / ((1 + 1j) * k) if np.isinf(b) else (1 - np.exp(-(1 + 1j) * k * b)) / ((1 + 1j) * k)
    Gk[pos] = A * RHO * G * integ
    return np.fft.irfft(H * Gk, n=N) * 1000.0   # mm

def gws_to_poroelastic(dates, gws_mm, material="clay", Sy=None, seasonal=False, b=np.inf,
                       lo_days=175, hi_days=380, smooth_months=7, c_override=None):
    """
    dates: 日期(月采样即可); gws_mm: GWS 异常(mm EWH)。
    返回逐日 DataFrame[date, gws_mm, head_m, poro_disp_mm, poro_shape_unit]。
    Sy: 覆盖材料默认比给水度(论文对 P304/P566/CUHS 用最小二乘标定 0.01/2e-3/1e-3)。
    b: aquifer thickness (m). b=∞ 为纯半空间; 有限b减小季节振幅(论文 Eq.10).
    c_override: 覆写材料的 hydraulic diffusivity (m²/s). None=用材料默认值.
    """
    m = MATERIAL_PARAMS[material].copy()
    sy = Sy if Sy is not None else m["Sy"]
    if c_override is not None:
        m["c"] = c_override
    s = pd.Series(np.asarray(gws_mm, float), index=pd.to_datetime(dates)).sort_index()
    # 轻度SG平滑: 压住 GRACE 固有 ~1-2cm 月噪声(等效论文小波去噪), 否则硬带通会留下毛刺
    if smooth_months and smooth_months >= 3 and s.size > smooth_months:
        win = int(smooth_months) | 1          # 取奇数窗
        s = pd.Series(savgol_filter(s.values, win, 2), index=s.index)
    daily = s.resample("1D").interpolate("time")           # 月->日
    head = (daily.values / 1000.0) / sy                    # h = EWH/Sy, 单位 m
    head = head - np.nanmean(head)
    # The paper's seasonal extraction applies at the comparison stage. Retain this
    # legacy switch only for sensitivity tests; do not filter GWS by default.
    if seasonal:
        head = seasonal_band(head, lo_days=lo_days, hi_days=hi_days)
    u = poroelastic_transfer(head, SEC_PER_DAY, m["nu"], m["beta_p"], m["c"], b)
    out = pd.DataFrame({"date": daily.index, "gws_mm": daily.values,
                        "head_m": head, "poro_disp_mm": u})
    sd = u.std(); out["poro_shape_unit"] = (u - u.mean()) / sd if sd > 0 else 0.0
    return out
