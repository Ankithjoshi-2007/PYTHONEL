"""
Airfoil Aerodynamics Simulator — Streamlit Dashboard
=====================================================
Two pages:
  1. Dashboard  – sliders, live metrics, CL/CD, Cp, Polar plots
  2. Flow Field – animated particle flow field (auto-refreshing)
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import io

# ─────────────────────────────────────────────────────────────────────────────
#  Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airfoil Aerodynamics Simulator",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
  }
  .stApp { background: #0a0e1a; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1226 0%, #0a0e1a 100%);
    border-right: 1px solid #1e3a5f;
  }
  section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #60a5fa;
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }

  /* Sliders */
  .stSlider [data-baseweb="slider"] { padding: 4px 0; }
  .stSlider [data-testid="stThumbValue"] { color: #60a5fa !important; font-weight: 600; }

  /* Metric cards */
  .metric-card {
    background: linear-gradient(135deg, #0d1932 0%, #0f2044 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
  }
  .metric-card:hover {
    transform: translateY(-2px);
    border-color: #3b82f6;
  }
  .metric-label {
    font-size: 0.70rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 6px;
  }
  .metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.55rem;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1;
  }
  .metric-sub {
    font-size: 0.68rem;
    color: #475569;
    margin-top: 4px;
  }

  /* Section headers */
  .section-header {
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3b82f6;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px;
    margin: 18px 0 12px 0;
    font-weight: 600;
  }

  /* Status badge */
  .badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
  }
  .badge-green  { background: #052e16; color: #34d399; border: 1px solid #065f46; }
  .badge-yellow { background: #2d1a04; color: #fbbf24; border: 1px solid #78350f; }
  .badge-red    { background: #2d0a0a; color: #f87171; border: 1px solid #7f1d1d; }
  .badge-blue   { background: #0c1a3a; color: #60a5fa; border: 1px solid #1e3a5f; }

  /* Plotly chart containers */
  .stPlotlyChart { border-radius: 12px; overflow: hidden; }

  /* Page title */
  .hero-title {
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
  }
  .hero-sub {
    color: #475569;
    font-size: 0.85rem;
  }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    background: #0d1226;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #64748b;
    font-weight: 500;
    padding: 8px 20px;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e3a5f, #0c2d5a);
    color: #60a5fa !important;
  }
  
  /* Stall warning banner */
  .stall-warning {
    background: linear-gradient(90deg, #2d0a0a, #3d1010);
    border: 1px solid #7f1d1d;
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 10px 0;
    animation: pulse 1.5s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.7; }
  }

  /* Flow animation container */
  .flow-container {
    background: #05090f;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 4px;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  GEOMETRY EXTRACTION & PARSING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def parse_airfoil_file(file_content: bytes, file_name: str):
    text = file_content.decode("utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    if file_name.endswith(".csv") or "," in lines[0]:
        xs, ys = [], []
        start_idx = 1 if ("x" in lines[0].lower() or "y" in lines[0].lower()) else 0
        for line in lines[start_idx:]:
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    xs.append(float(parts[0]))
                    ys.append(float(parts[1]))
                except ValueError:
                    continue
    else:
        xs, ys = [], []
        start_idx = 1
        parts_first = lines[0].split()
        if len(parts_first) >= 2:
            try:
                float(parts_first[0])
                float(parts_first[1])
                start_idx = 0
            except ValueError:
                pass
        
        for line in lines[start_idx:]:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    xs.append(float(parts[0]))
                    ys.append(float(parts[1]))
                except ValueError:
                    continue

    if len(xs) < 10:
        raise ValueError("Insufficient coordinates parsed.")

    x = np.array(xs)
    y = np.array(ys)
    le_idx = np.argmin(x)
    
    xu = x[:le_idx+1]
    yu = y[:le_idx+1]
    xl = x[le_idx:]
    yl = y[le_idx:]

    if xu[0] > xu[-1]:
        xu = xu[::-1]
        yu = yu[::-1]
    if xl[-1] < xl[0]:
        xl = xl[::-1]
        yl = yl[::-1]

    min_x = min(xu[0], xl[0])
    max_x = max(xu[-1], xl[-1])
    chord_len = max_x - min_x
    if chord_len > 0:
        xu = (xu - min_x) / chord_len
        xl = (xl - min_x) / chord_len
        yu /= chord_len
        yl /= chord_len

    grid_x = (1 - np.cos(np.linspace(0, np.pi, 120))) / 2
    yu_interp = np.interp(grid_x, xu, yu)
    yl_interp = np.interp(grid_x, xl, yl)

    return grid_x, yu_interp, grid_x, yl_interp


def analyze_geometry(xu, yu, xl, yl):
    thickness = yu - yl
    max_t = float(np.max(thickness))
    camber_line = 0.5 * (yu + yl)
    max_m = float(np.max(np.abs(camber_line)))
    
    if max_m > 1e-4:
        max_p_idx = np.argmax(np.abs(camber_line))
        max_p = float(xu[max_p_idx])
    else:
        max_m = 0.0
        max_p = 0.0

    return max_m, max_p, max_t


# ─────────────────────────────────────────────────────────────────────────────
#  NACA & AERODYNAMICS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def naca_profile(code: str, n: int = 120, custom_coords=None):
    if custom_coords is not None:
        return custom_coords

    code = str(code).zfill(4)
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0

    beta = np.linspace(0, np.pi, n)
    x = (1 - np.cos(beta)) / 2

    yt = (t / 0.2) * (0.2969 * np.sqrt(x)
                      - 0.1260 * x
                      - 0.3516 * x**2
                      + 0.2843 * x**3
                      - 0.1015 * x**4)

    if p > 0:
        yc = np.where(x <= p,
                      m / p**2 * (2*p*x - x**2),
                      m / (1-p)**2 * ((1 - 2*p) + 2*p*x - x**2))
        dyc_dx = np.where(x <= p,
                          2*m / p**2 * (p - x),
                          2*m / (1-p)**2 * (p - x))
    else:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)

    theta = np.arctan(dyc_dx)
    xu = x  - yt * np.sin(theta);  yu = yc + yt * np.cos(theta)
    xl = x  + yt * np.sin(theta);  yl = yc - yt * np.cos(theta)
    return xu, yu, xl, yl


def rotate_airfoil(xu, yu, xl, yl, alpha_deg):
    alpha = np.radians(alpha_deg)
    pivot = 0.25
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    def rot(x, y):
        xp = pivot + (x - pivot) * cos_a - y * sin_a
        yp =         (x - pivot) * sin_a + y * cos_a
        return xp, yp
    return (*rot(xu, yu), *rot(xl, yl))


class AirfoilAero:
    def __init__(self, naca="2412", alpha=5.0, Re=3e6, mach=0.15, wind=50.0, custom_geom=None):
        self.naca = str(naca).zfill(4)
        self.alpha = alpha
        self.Re    = Re
        self.mach  = mach
        self.wind  = wind
        self.custom_geom = custom_geom
        self._parse_naca()

    def _parse_naca(self):
        if self.custom_geom is not None:
            self.m, self.p, self.t = self.custom_geom
        else:
            self.m = int(self.naca[0]) / 100.0
            self.p = int(self.naca[1]) / 10.0
            self.t = int(self.naca[2:]) / 100.0

    @property
    def pg_factor(self):
        M = min(self.mach, 0.99)
        return 1.0 / np.sqrt(max(1e-6, 1 - M**2))

    @property
    def alpha_0(self):
        if self.p == 0: return 0.0
        return np.degrees(-self.m * (1 - 2*self.p) * np.pi)

    @property
    def cl_slope(self):
        return 2 * np.pi * (1 + 0.77 * self.t)

    @property
    def stall_angle(self):
        return 10.0 + 25*self.t + 15*self.m + 200*self.t*self.m

    def cl(self, alpha=None):
        a = self.alpha if alpha is None else alpha
        alpha_rad  = np.radians(a)
        alpha0_rad = np.radians(self.alpha_0)
        re_factor  = 1 + 0.10 * np.log10(max(self.Re, 1e5) / 1e6)
        raw = self.cl_slope * (alpha_rad - alpha0_rad) * self.pg_factor * re_factor
        stall = np.radians(self.stall_angle)
        alpha_arr = np.atleast_1d(alpha_rad)
        raw_arr   = np.atleast_1d(raw)
        over  = np.where(alpha_arr >  stall, (alpha_arr - stall)**2, 0)
        under = np.where(alpha_arr < -stall * 0.85, (alpha_arr + stall*0.85)**2, 0)
        result = raw_arr * np.exp(-2.8 * (over + under))
        return float(result[0]) if np.ndim(a) == 0 else result

    def cd(self, alpha=None):
        a = self.alpha if alpha is None else alpha
        scalar = np.ndim(a) == 0
        a_arr  = np.atleast_1d(a)
        cl_val = self.cl(a_arr)
        cf  = 0.074 / max(self.Re, 1e4)**0.2
        cd0 = 2 * cf * (1 + 2.7*self.t + 100*self.t**4)
        e   = 0.85 + 0.10*self.m - 0.05*abs(self.p - 0.4)
        cdi = np.atleast_1d(cl_val)**2 / (np.pi * e * 8)
        M   = self.mach
        cd_wave  = 20 * max(0, M - 0.72)**3 if M > 0.72 else 0
        a_rad    = np.radians(a_arr)
        stall_r  = np.radians(self.stall_angle)
        cd_stall = 0.1 * np.maximum(0, np.abs(a_rad) - stall_r)**2
        result   = cd0 + cdi + cd_wave + cd_stall
        return float(result[0]) if scalar else result

    def cp_distribution(self, n=80):
        x = np.linspace(0, 1, n)
        a = self.alpha
        cl_val = self.cl()
        pg = self.pg_factor
        cp_upper = (-cl_val*(1-x)*np.exp(-5.5*x) - 0.28*np.sin(np.pi*x)
                    - np.radians(a)*0.4 + self.m*0.25) * pg * 0.55
        cp_lower = (cl_val*x*(1-x)*0.7 + np.radians(a)*0.18
                    - self.m*0.18) * pg * 0.55
        return x, cp_upper, cp_lower

    @property
    def q_inf(self):
        return 0.5 * 1.225 * self.wind**2

    def ld_ratio(self):
        c = self.cd()
        return self.cl() / max(c, 1e-9)


def compute_flow_field(aero: AirfoilAero, Nx=55, Ny=38):
    alpha_rad = np.radians(aero.alpha)
    V_inf = max(aero.wind, 1.0)
    cl = aero.cl()
    x = np.linspace(-0.6, 1.8, Nx)
    y = np.linspace(-0.7, 0.7, Ny)
    X, Y = np.meshgrid(x, y)
    U = V_inf * np.ones_like(X)
    V = np.zeros_like(X)

    mu = 0.045 * (1 + 4 * aero.t)
    Gamma = 0.5 * V_inf * cl

    cos_a = np.cos(-alpha_rad); sin_a = np.sin(-alpha_rad)
    pivot = 0.25
    def rot_pt(xp, yp):
        xr = pivot + (xp-pivot)*cos_a - yp*sin_a
        yr =         (xp-pivot)*sin_a + yp*cos_a
        return xr, yr

    raw_panels = [(0.25,0.00,mu*1.2),(0.50,0.00,mu*0.9),
                  (0.10,0.00,mu*0.7),(0.75,0.00,mu*0.5)]
    for (xp, yp, strength) in raw_panels:
        rxp, ryp = rot_pt(xp, yp)
        dx = X-rxp; dy = Y-ryp
        r2 = dx**2 + dy**2 + 1e-6
        U += -strength*(dx**2 - dy**2)/r2**2
        V += -strength*2*dx*dy/r2**2

    vxp, vyp = rot_pt(0.25, 0.0)
    dx = X-vxp; dy = Y-vyp
    r2 = dx**2+dy**2+1e-6
    U += -Gamma/(2*np.pi)*(-dy/r2)
    V += -Gamma/(2*np.pi)*( dx/r2)

    pg = aero.pg_factor
    U *= pg; V *= pg
    speed = np.sqrt(U**2+V**2)
    return X, Y, U, V, speed


def evaluate_velocity_at(aero: AirfoilAero, px, py):
    V_inf = max(aero.wind, 1.0)
    cl = aero.cl()
    u = V_inf * np.ones_like(px, dtype=float)
    v = np.zeros_like(py, dtype=float)
    mu = 0.045*(1+4*aero.t)
    Gamma = 0.5*V_inf*cl
    alpha_rad = np.radians(aero.alpha)
    cos_a = np.cos(-alpha_rad); sin_a = np.sin(-alpha_rad)
    pivot = 0.25
    def rot_pt(xp, yp):
        xr = pivot+(xp-pivot)*cos_a - yp*sin_a
        yr =       (xp-pivot)*sin_a + yp*cos_a
        return xr, yr
    raw_panels = [(0.25,0.00,mu*1.2),(0.50,0.00,mu*0.9),
                  (0.10,0.00,mu*0.7),(0.75,0.00,mu*0.5)]
    for (xp,yp,strength) in raw_panels:
        rxp,ryp = rot_pt(xp,yp)
        dx=px-rxp; dy=py-ryp
        r2=dx**2+dy**2+1e-6
        u += -strength*(dx**2-dy**2)/r2**2
        v += -strength*2*dx*dy/r2**2
    vxp,vyp = rot_pt(0.25,0.0)
    dx=px-vxp; dy=py-vyp
    r2=dx**2+dy**2+1e-6
    u += -Gamma/(2*np.pi)*(-dy/r2)
    v += -Gamma/(2*np.pi)*( dx/r2)
    pg = aero.pg_factor
    u*=pg; v*=pg
    return u, v


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE  (particles for animation)
# ─────────────────────────────────────────────────────────────────────────────

def _init_particles(Re: float, n_rows_half: int = None):
    if n_rows_half is None:
        re_m = Re / 1e6
        n_rows_half = int(np.clip(re_m * 2.5, 6, 18))
    per_row = 14
    top_rows = np.linspace(0.62, 0.04, n_rows_half)
    bot_rows = -top_rows
    rows_y = np.empty(n_rows_half * 2)
    rows_y[0::2] = top_rows
    rows_y[1::2] = bot_rows
    xs, ys, row_ids = [], [], []
    for r_idx, ry in enumerate(rows_y):
        for j in range(per_row):
            xs.append(-0.6 + j * (2.4 / per_row))
            ys.append(ry)
            row_ids.append(r_idx)
    return (np.array(xs, dtype=float),
            np.array(ys, dtype=float),
            rows_y,
            np.array(row_ids, dtype=int))


def _is_inside_airfoil(px, py, aero: AirfoilAero, custom_coords=None):
    alpha_rad = np.radians(aero.alpha)
    pivot = 0.25
    cos_a = np.cos(alpha_rad); sin_a = np.sin(alpha_rad)
    dx = px - pivot
    lx = pivot + dx*cos_a + py*sin_a
    ly =        -dx*sin_a + py*cos_a
    inside = np.zeros(len(px), dtype=bool)
    chord_mask = (lx >= -0.01) & (lx <= 1.01)
    
    if np.any(chord_mask):
        lxc = np.clip(lx[chord_mask], 0.0, 1.0)
        lyc = ly[chord_mask]
        
        if custom_coords is not None:
            cxu, cyu, _, cyl = custom_coords
            bound_upper = np.interp(lxc, cxu, cyu)
            bound_lower = np.interp(lxc, cxu, cyl)
            inside[chord_mask] = (lyc > bound_lower - 0.005) & (lyc < bound_upper + 0.005)
        else:
            t = aero.t; m = aero.m; p = aero.p
            yt = (t/0.2)*(0.2969*np.sqrt(np.maximum(lxc,0))-0.1260*lxc
                          -0.3516*lxc**2+0.2843*lxc**3-0.1015*lxc**4)
            if p > 0:
                yc = np.where(lxc <= p,
                              m/p**2*(2*p*lxc-lxc**2),
                              m/(1-p)**2*((1-2*p)+2*p*lxc-lxc**2))
            else:
                yc = np.zeros_like(lxc)
            inside[chord_mask] = np.abs(lyc - yc) < (yt + 0.015)
            
    return inside


def _step_particles(aero: AirfoilAero, px, py, rows_y, row_ids, custom_coords=None):
    u, v = evaluate_velocity_at(aero, px, py)
    dt = 0.0005 * (100.0 / max(aero.wind, 5.0))
    px = px + u * dt
    py = py + v * dt
    out_bounds = (px > 1.8) | (px < -0.6) | (py > 0.7) | (py < -0.7)
    in_foil    = _is_inside_airfoil(px, py, aero, custom_coords)
    reset = out_bounds | in_foil
    if np.any(reset):
        px[reset] = -0.6
        py[reset] = rows_y[row_ids[reset]]
    return px, py


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTLY CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG   = "#070d1a"
PANEL_BG  = "#0d1226"
GRID_COL  = "#1a2540"
TEXT_COL  = "#94a3b8"
TITLE_COL = "#e2e8f0"

def _plotly_layout(title="", height=320):
    return dict(
        title=dict(text=title, font=dict(color=TITLE_COL, size=13, family="Inter"),
                   x=0.5, xanchor="center"),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COL, family="Inter"),
        height=height,
        margin=dict(l=50, r=20, t=45, b=45),
        xaxis=dict(gridcolor=GRID_COL, zerolinecolor="#2d3748", tickfont=dict(size=10)),
        yaxis=dict(gridcolor=GRID_COL, zerolinecolor="#2d3748", tickfont=dict(size=10)),
    )


def build_clcd_chart(aero: AirfoilAero):
    aoas = np.linspace(-20, 28, 300)
    cls  = aero.cl(aoas)
    cds  = aero.cd(aoas)
    cl_c = aero.cl(); cd_c = aero.cd()
    stall = aero.stall_angle

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=aoas, y=cls, fill="tozeroy",
                             fillcolor="rgba(59,130,246,0.08)",
                             line=dict(color="#3b82f6", width=2.5),
                             name="CL (Lift Coeff.)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=aoas, y=cds, fill="tozeroy",
                             fillcolor="rgba(245,158,11,0.07)",
                             line=dict(color="#f59e0b", width=2.0, dash="dash"),
                             name="CD (Drag Coeff.)"), secondary_y=True)

    fig.add_trace(go.Scatter(x=[aero.alpha], y=[cl_c],
                             mode="markers",
                             marker=dict(color="#34d399", size=10, symbol="circle",
                                         line=dict(color="white", width=1.5)),
                             name=f"α = {aero.alpha:.1f}°"), secondary_y=False)
    fig.add_trace(go.Scatter(x=[aero.alpha], y=[cd_c],
                             mode="markers",
                             marker=dict(color="#fbbf24", size=10, symbol="diamond",
                                         line=dict(color="white", width=1.5)),
                             showlegend=False), secondary_y=True)

    for sv, txt_x in [(stall, stall+0.4), (-stall*0.85, -stall*0.85+0.4)]:
        fig.add_vline(x=sv, line=dict(color="#ef4444", dash="dot", width=1.0))
    fig.add_annotation(x=stall+0.4, y=max(cls)*0.85, text="stall",
                       font=dict(color="#ef4444", size=9), showarrow=False)

    fig.add_vline(x=aero.alpha, line=dict(color="#ef4444", dash="dot", width=1.2))

    layout = _plotly_layout("CL & CD vs Angle of Attack", height=330)
    layout.update(
        yaxis=dict(title="CL", color="#3b82f6", gridcolor=GRID_COL),
        yaxis2=dict(title="CD", color="#f59e0b", gridcolor=GRID_COL,
                    overlaying="y", side="right"),
        xaxis=dict(title="Angle of Attack α (°)", gridcolor=GRID_COL),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10))
    )
    fig.update_layout(**layout)
    return fig


def build_cp_chart(aero: AirfoilAero):
    x, cpu, cpl = aero.cp_distribution(n=100)
    peak_i = np.argmin(cpu)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=cpu, fill="tozeroy",
                             fillcolor="rgba(239,68,68,0.12)",
                             line=dict(color="#ef4444", width=2.5),
                             name="Upper surface (suction)"))
    fig.add_trace(go.Scatter(x=x, y=cpl, fill="tozeroy",
                             fillcolor="rgba(59,130,246,0.12)",
                             line=dict(color="#3b82f6", width=2.5),
                             name="Lower surface (pressure)"))
    fig.add_hline(y=0, line=dict(color="#475569", width=0.8))

    fig.add_annotation(x=x[peak_i], y=cpu[peak_i],
                       text=f"Cp={cpu[peak_i]:.2f}",
                       font=dict(color="#ef4444", size=9),
                       arrowcolor="#ef4444", arrowwidth=1.2,
                       ax=40, ay=-30)

    layout = _plotly_layout("Pressure Coefficient (Cp) Distribution", height=330)
    layout.update(
        xaxis=dict(title="x/c (Chord fraction)", gridcolor=GRID_COL),
        yaxis=dict(title="−Cp", gridcolor=GRID_COL, autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10))
    )
    fig.update_layout(**layout)
    return fig


def build_polar_chart(aero: AirfoilAero):
    aoas = np.linspace(-18, 28, 300)
    cls  = aero.cl(aoas)
    cds  = aero.cd(aoas)
    cl_c = aero.cl(); cd_c = aero.cd()

    ld = cls / np.where(cds < 1e-9, 1e-9, cds)
    bi = np.argmax(ld)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cds, y=cls,
        mode="lines",
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=cds, y=cls,
        mode="lines+markers",
        marker=dict(size=4, color=aoas, colorscale="RdBu_r",
                    colorbar=dict(title=dict(text="AoA °", font=dict(color=TEXT_COL, size=9)),
                                  thickness=10,
                                  tickfont=dict(size=9, color=TEXT_COL)),
                    showscale=True),
        line=dict(color="rgba(0,0,0,0)"),
        name="Polar curve",
        hovertemplate="α=%{marker.color:.1f}°<br>CD=%{x:.5f}<br>CL=%{y:.4f}"
    ))

    fig.add_trace(go.Scatter(x=[0, cds[bi]], y=[0, cls[bi]],
                             mode="lines",
                             line=dict(color="#34d399", dash="dash", width=1.5),
                             name=f"Best L/D = {ld[bi]:.1f}"))
    fig.add_annotation(x=cds[bi]*1.1, y=cls[bi],
                       text=f"L/D={ld[bi]:.1f}",
                       font=dict(color="#34d399", size=9), showarrow=False)

    fig.add_trace(go.Scatter(x=[cd_c], y=[cl_c],
                             mode="markers",
                             marker=dict(color="#f87171", size=12, symbol="star",
                                         line=dict(color="white", width=1.5)),
                             name=f"Operating α={aero.alpha:.1f}°"))

    layout = _plotly_layout("Polar Diagram (CL vs CD)", height=330)
    layout.update(
        xaxis=dict(title="CD (Drag Coefficient)", gridcolor=GRID_COL),
        yaxis=dict(title="CL (Lift Coefficient)", gridcolor=GRID_COL),
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)")
    )
    fig.update_layout(**layout)
    return fig


def build_airfoil_shape_chart(aero: AirfoilAero, height=200, custom_coords=None):
    xu, yu, xl, yl = naca_profile(aero.naca, custom_coords=custom_coords)
    rxu, ryu, rxl, ryl = rotate_airfoil(xu, yu, xl, yl, aero.alpha)
    foil_x = np.concatenate([rxu, rxl[::-1], [rxu[0]]])
    foil_y = np.concatenate([ryu, ryl[::-1], [ryu[0]]])
    cl  = aero.cl(); cd = aero.cd()
    mid = (float(np.mean(rxu)), float(np.mean(ryu)))
    lift_scale = 0.22 * cl
    drag_scale = 0.8 * cd

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=foil_x, y=foil_y,
                             fill="toself", fillcolor="rgba(30,64,175,0.85)",
                             line=dict(color="#93c5fd", width=1.5),
                             name="Airfoil Shape"))
    # Lift arrow
    fig.add_annotation(x=mid[0], y=mid[1]+lift_scale,
                       ax=mid[0], ay=mid[1],
                       xref="x", yref="y", axref="x", ayref="y",
                       arrowhead=2, arrowwidth=2.5, arrowcolor="#34d399",
                       text="", showarrow=True)
    fig.add_annotation(x=mid[0]+0.06, y=mid[1]+lift_scale*0.55,
                       text=f"L CL={cl:.3f}", font=dict(color="#34d399", size=9),
                       showarrow=False, xref="x", yref="y")
    # Drag arrow
    if drag_scale > 0.005:
        fig.add_annotation(x=mid[0]+drag_scale, y=mid[1],
                           ax=mid[0], ay=mid[1],
                           xref="x", yref="y", axref="x", ayref="y",
                           arrowhead=2, arrowwidth=2, arrowcolor="#f87171",
                           text="", showarrow=True)
        fig.add_annotation(x=mid[0]+drag_scale*0.5, y=mid[1]-0.04,
                           text=f"D CD={cd:.4f}", font=dict(color="#f87171", size=9),
                           showarrow=False, xref="x", yref="y")
    # Freestream arrow
    le_x = float(min(rxu[0], rxl[0]))
    fig.add_annotation(x=le_x, y=0.0,
                       ax=le_x-0.25, ay=0.0,
                       xref="x", yref="y", axref="x", ayref="y",
                       arrowhead=2, arrowwidth=1.5, arrowcolor="#60a5fa",
                       text="V∞", font=dict(color="#60a5fa", size=9),
                       showarrow=True)

    layout = dict(
        paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COL, family="Inter"),
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[-0.4, 1.35], gridcolor=GRID_COL, showticklabels=False,
                   zeroline=False, scaleanchor="y"),
        yaxis=dict(range=[-0.35, 0.45], gridcolor=GRID_COL, showticklabels=False,
                   zeroline=False),
        showlegend=False
    )
    fig.update_layout(**layout)
    return fig


def _build_quiver_trace(U, V, X, Y, step=4):
    Xs = X[::step, ::step].flatten()
    Ys = Y[::step, ::step].flatten()
    Us = U[::step, ::step].flatten()
    Vs = V[::step, ::step].flatten()
    mag = np.sqrt(Us**2 + Vs**2) + 1e-9
    scale = 0.045
    qx, qy = [], []
    for i in range(0, len(Xs), 2):
        x0, y0 = Xs[i], Ys[i]
        x1 = x0 + Us[i] / mag[i] * scale
        y1 = y0 + Vs[i] / mag[i] * scale
        qx.extend([x0, x1, None])
        qy.extend([y0, y1, None])
    return go.Scatter(
        x=qx, y=qy,
        mode="lines",
        line=dict(color="rgba(147,197,253,0.28)", width=0.9),
        showlegend=False, hoverinfo="skip"
    )


def build_animated_flow_chart(aero: AirfoilAero, n_frames: int = 70,
                              steps_per_frame: int = 4, custom_coords=None,
                              frame_duration: int = 50):
    X, Y, U, V, speed = compute_flow_field(aero, Nx=50, Ny=34)
    speed_n = speed / (speed.max() + 1e-9)

    xu, yu, xl, yl = naca_profile(aero.naca, custom_coords=custom_coords)
    rxu, ryu, rxl, ryl = rotate_airfoil(xu, yu, xl, yl, aero.alpha)
    foil_x = np.concatenate([rxu, rxl[::-1], [rxu[0]]])
    foil_y = np.concatenate([ryu, ryl[::-1], [ryu[0]]])

    cl = aero.cl(); cd = aero.cd()

    contour_trace = go.Contour(
        x=X[0], y=Y[:, 0], z=speed_n,
        colorscale="RdBu_r",
        contours=dict(showlines=False),
        showscale=True,
        colorbar=dict(
            title=dict(text="Speed (norm.)", font=dict(color=TEXT_COL, size=9)),
            thickness=10, tickfont=dict(size=9, color=TEXT_COL)
        ),
        opacity=0.50,
        name="",
        hoverinfo="skip"
    )

    quiver_trace = _build_quiver_trace(U, V, X, Y, step=4)

    px, py, rows_y, row_ids = _init_particles(aero.Re)
    n_total = len(px)
    n_half  = n_total // 2
    p_colors = (["rgba(56,189,248,0.9)"] * n_half +
                ["rgba(192,132,252,0.9)"] * (n_total - n_half))

    frames = []
    for f in range(n_frames):
        for _ in range(steps_per_frame):
            px, py = _step_particles(aero, px, py, rows_y, row_ids, custom_coords)
        frames.append(go.Frame(
            data=[go.Scatter(
                x=px.tolist(), y=py.tolist(),
                mode="markers",
                marker=dict(size=4, color=p_colors, opacity=0.9),
                hoverinfo="skip"
            )],
            traces=[2],
            name=str(f)
        ))

    particle_trace = go.Scatter(
        x=frames[0].data[0].x,
        y=frames[0].data[0].y,
        mode="markers",
        marker=dict(size=4, color=p_colors, opacity=0.9),
        showlegend=False, hoverinfo="skip"
    )

    airfoil_trace = go.Scatter(
        x=foil_x, y=foil_y,
        fill="toself", fillcolor="rgba(30,64,175,0.95)",
        line=dict(color="#93c5fd", width=1.8),
        showlegend=False, hoverinfo="skip"
    )

    fig = go.Figure(
        data=[contour_trace, quiver_trace, particle_trace, airfoil_trace],
        frames=frames
    )

    if abs(aero.alpha) > aero.stall_angle * 0.85:
        fig.add_annotation(x=0.6, y=0.4,
                           text="⚠ FLOW SEPARATION",
                           font=dict(color="#ef4444", size=12, family="Inter"),
                           bgcolor="rgba(45,10,10,0.85)",
                           bordercolor="#ef4444", borderwidth=1.2,
                           showarrow=False)

    info = (f"Shape: {aero.naca} | α = {aero.alpha:.1f}° | "
            f"Ma = {aero.mach:.2f} | Re = {aero.Re/1e6:.2f}M | "
            f"CL = {cl:.4f} | CD = {cd:.5f}")
    fig.add_annotation(x=0.5, y=-0.62, text=info,
                       font=dict(color="#64748b", size=9),
                       showarrow=False, xref="x", yref="y")

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.08, x=0.5, xanchor="center",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, dict(
                            frame=dict(duration=frame_duration, redraw=False),
                            transition=dict(duration=0),
                            fromcurrent=True,
                            mode="immediate"
                        )]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            transition=dict(duration=0),
                            mode="immediate"
                        )]
                    )
                ],
                bgcolor="#0d1932",
                bordercolor="#1e3a5f",
                font=dict(color="#93c5fd", size=11)
            )
        ],
        paper_bgcolor="#05090f",
        plot_bgcolor="#070d1a",
        font=dict(color=TEXT_COL, family="Inter"),
        height=510,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis=dict(range=[-0.6, 1.8], gridcolor=GRID_COL, showticklabels=False,
                   zeroline=False, scaleanchor="y"),
        yaxis=dict(range=[-0.7, 0.7], gridcolor=GRID_COL, showticklabels=False,
                   zeroline=False),
        showlegend=False,
        title=dict(
            text="Flow Field – Particle Stream Animation",
            font=dict(color=TITLE_COL, size=13), x=0.5, xanchor="center"
        )
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR – shared controls
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 12px 0 8px 0;">
          <div style="font-size:2rem;">✈️</div>
          <div style="font-size:1rem; font-weight:700; color:#60a5fa; letter-spacing:0.05em;">
            AIRFOIL SIMULATOR
          </div>
          <div style="font-size:0.72rem; color:#475569; margin-top:2px;">
            2D Aerodynamics · Thin Airfoil Theory
          </div>
        </div>
        <hr style="border-color:#1e3a5f; margin:10px 0;">
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Shape Definition</div>', unsafe_allow_html=True)
        shape_type = st.radio("Shape Mode", ["Standard NACA Code", "Import Custom Shape"], index=0)
        
        custom_coords = None
        custom_geom = None
        naca = "2412"

        if shape_type == "Standard NACA Code":
            naca = st.selectbox("Profile", ["0009","0012","2412","4412","6412","2415","4415"],
                                index=2, label_visibility="collapsed")
        else:
            uploaded_file = st.file_uploader("Upload Coordinate File (.dat, .txt, .csv)", type=["dat", "txt", "csv"])
            if uploaded_file is not None:
                try:
                    file_bytes = uploaded_file.read()
                    cxu, cyu, cxl, cyl = parse_airfoil_file(file_bytes, uploaded_file.name)
                    custom_coords = (cxu, cyu, cxl, cyl)
                    m, p, t = analyze_geometry(cxu, cyu, cxl, cyl)
                    custom_geom = (m, p, t)
                    naca = uploaded_file.name.split(".")[0][:8]
                    st.success(f"Loaded: {naca}")
                    st.info(f"Geometry Analysed:\n- Max Camber: {m*100:.1f}%\n- Position: {p*100:.1f}%\n- Thickness: {t*100:.1f}%")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
                    st.warning("Falling back to standard NACA 2412 profile shape.")
            else:
                st.info("Upload Selig or CSV coordinate format airfoil profile to test.")

        st.markdown('<div class="section-header">Fluid Medium</div>', unsafe_allow_html=True)
        medium = st.selectbox("Medium Preset", 
                              ["Normal Flight (Air)", "Water", "Engine Oil (SAE 30)", "Custom Fluid"],
                              index=0)

        presets = {
            "Normal Flight (Air)": (1.225, 1.81e-5, 340.0),
            "Water": (998.0, 1.002e-3, 1482.0),
            "Engine Oil (SAE 30)": (890.0, 0.29, 1740.0),
            "Custom Fluid": (1.0, 1e-5, 300.0)
        }

        default_rho, default_mu, default_a = presets[medium]

        if medium == "Custom Fluid":
            rho = st.number_input("Density ρ (kg/m³)", min_value=0.01, max_value=20000.0, value=1.0, step=0.1)
            visc = st.number_input("Dynamic Viscosity μ (Pa·s)", min_value=1e-7, max_value=10.0, value=1e-5, format="%.2e")
            speed_of_sound = st.number_input("Speed of Sound (m/s)", min_value=1.0, max_value=10000.0, value=300.0, step=10.0)
        else:
            rho = default_rho
            visc = default_mu
            speed_of_sound = default_a
            st.text(f"Density ρ: {rho} kg/m³")
            st.text(f"Viscosity μ: {visc:.2e} Pa·s")
            st.text(f"Speed of Sound: {speed_of_sound} m/s")

        st.markdown('<div class="section-header">Flight Parameters</div>', unsafe_allow_html=True)

        alpha = st.slider("α  Angle of Attack (°)", -15.0, 25.0, 5.0, 0.5,
                          help="Angle between chord line and flow direction")
        
        # Scale slider to ~0.9× speed of sound so Mach is always meaningful
        max_vel = round(speed_of_sound * 0.90, 0)
        default_vel = min(50.0, max_vel * 0.15)   # ~15% of max as sensible default
        vel_step = max(0.1, round(max_vel / 500, 2))
        wind  = st.slider("V∞  Velocity (m/s)", 1.0, float(max_vel),
                          float(default_vel), float(vel_step),
                          help="Free-stream flow speed relative to airfoil")

        chord = 1.0
        Re = (rho * wind * chord) / visc
        mach = wind / speed_of_sound

        st.markdown('<div class="section-header">Derived Metrics</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.8rem; color:#94a3b8; line-height:1.6;">
          🧪 Reynolds Number: <b>{Re:.2e}</b><br>
          ⚡ Mach Number: <b>{mach:.3f}</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Physics Features</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.72rem; color:#475569; line-height:1.9;">
          📐 Thin Airfoil Theory<br>
          🔧 Thickness/Camber correction<br>
          🌊 Prandtl-Glauert (compressibility)<br>
          🔬 Turbulent BL skin friction<br>
          🪂 Induced drag (AR = 8)<br>
          💥 Soft stall model<br>
          🌐 Wave drag (M > 0.72)
        </div>
        """, unsafe_allow_html=True)

    return naca, alpha, Re, mach, wind, custom_coords, custom_geom


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER — metric card HTML
# ─────────────────────────────────────────────────────────────────────────────

def metric_card(label: str, value: str, sub: str = "", color: str = "#e2e8f0"):
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value" style="color:{color};">{value}</div>
      {f'<div class="metric-sub">{sub}</div>' if sub else ''}
    </div>
    """


def flight_badge(aero: AirfoilAero):
    alpha_abs = abs(aero.alpha)
    if alpha_abs >= aero.stall_angle:
        return '<span class="badge badge-red">⚠ STALL</span>'
    elif alpha_abs >= 0.85 * aero.stall_angle:
        return '<span class="badge badge-yellow">⚡ STALL WARNING</span>'
    elif aero.cl() > 0.05:
        return '<span class="badge badge-green">✅ STABLE LIFT</span>'
    else:
        return '<span class="badge badge-blue">🔵 LOW LIFT</span>'


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def page_dashboard(aero: AirfoilAero, custom_coords=None):
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:16px; margin-bottom:4px;">
      <div>
        <div class="hero-title">Airfoil Aerodynamics Dashboard</div>
        <div class="hero-sub">
          Shape: {aero.naca} &nbsp;·&nbsp; α = {aero.alpha:.1f}°
          &nbsp;·&nbsp; Ma = {aero.mach:.2f}
          &nbsp;·&nbsp; Re = {aero.Re/1e6:.2f}M
          &nbsp;&nbsp;{flight_badge(aero)}
        </div>
      </div>
    </div>
    <hr style="border-color:#1e3a5f; margin: 8px 0 20px 0;">
    """, unsafe_allow_html=True)

    if abs(aero.alpha) >= aero.stall_angle * 0.85:
        st.markdown(f"""
        <div class="stall-warning">
          <strong style="color:#ef4444;">⚠ STALL {"" if abs(aero.alpha) < aero.stall_angle else "— CRITICAL"}</strong>
          &nbsp; Angle of attack {aero.alpha:.1f}° is
          {"approaching" if abs(aero.alpha) < aero.stall_angle else "beyond"} the
          stall limit of {aero.stall_angle:.1f}°. Expect flow separation and lift loss.
        </div>
        """, unsafe_allow_html=True)

    cl  = aero.cl(); cd  = aero.cd()
    ld  = cl / max(cd, 1e-9); q = aero.q_inf
    M   = aero.mach

    cols = st.columns(5)
    cards = [
        ("CL Lift Coeff.", f"{cl:+.4f}", f"Stall at {aero.stall_angle:.1f}°", "#34d399"),
        ("CD Drag Coeff.", f"{cd:.5f}",  "Total 2D drag",                      "#f59e0b"),
        ("L/D Efficiency", f"{ld:.2f}",  "Higher = more efficient",            "#60a5fa"),
        ("q∞ Dyn. Press.", f"{q:.1f} Pa","0.5 ρ V²",                           "#c084fc"),
        ("Ma Mach",        f"{M:.3f}",   "Compressibility" if M>0.3 else "Subsonic", "#f472b6"),
    ]
    for col, (lbl, val, sub, clr) in zip(cols, cards):
        with col:
            st.markdown(metric_card(lbl, val, sub, clr), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    col_shape, col_clcd = st.columns([1, 2])
    with col_shape:
        st.markdown('<div class="section-header">Airfoil Shape</div>', unsafe_allow_html=True)
        st.plotly_chart(build_airfoil_shape_chart(aero, height=260, custom_coords=custom_coords),
                        use_container_width=True, config={"displayModeBar": False})
    with col_clcd:
        st.markdown('<div class="section-header">CL & CD vs Angle of Attack</div>', unsafe_allow_html=True)
        st.plotly_chart(build_clcd_chart(aero),
                        use_container_width=True, config={"displayModeBar": False})

    col_cp, col_polar = st.columns(2)
    with col_cp:
        st.markdown('<div class="section-header">Pressure Coefficient Distribution</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(build_cp_chart(aero),
                        use_container_width=True, config={"displayModeBar": False})
    with col_polar:
        st.markdown('<div class="section-header">Polar Diagram (CL vs CD)</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(build_polar_chart(aero),
                        use_container_width=True, config={"displayModeBar": False})

    with st.expander("📋 Full Aerodynamic Breakdown", expanded=False):
        cf    = 0.074 / max(aero.Re, 1e4)**0.2
        cd0   = 2*cf*(1+2.7*aero.t+100*aero.t**4)
        e     = 0.85+0.10*aero.m-0.05*abs(aero.p-0.4)
        cdi   = cl**2/(np.pi*e*8)
        M     = aero.mach
        cd_w  = 20*max(0,M-0.72)**3 if M>0.72 else 0.0
        a_r   = abs(np.radians(aero.alpha)); s_r=np.radians(aero.stall_angle)
        cd_s  = 0.1*max(0,a_r-s_r)**2

        rows = {
            "Airfoil Designation":    aero.naca,
            "Angle of Attack":         f"{aero.alpha:.2f}°",
            "Zero-Lift Angle":         f"{aero.alpha_0:.2f}°",
            "Estimated Stall Angle":   f"{aero.stall_angle:.2f}°",
            "Prandtl-Glauert Factor":  f"{aero.pg_factor:.4f}",
            "Lift Coefficient (CL)":   f"{cl:.6f}",
            "Profile Drag (CD0)":      f"{cd0:.6f}",
            "Induced Drag (CDi)":      f"{cdi:.6f}",
            "Wave Drag (CD_wave)":     f"{cd_w:.6f}",
            "Stall Drag (CD_stall)":   f"{cd_s:.6f}",
            "Total Drag (CD)":         f"{cd:.6f}",
            "L/D Ratio":               f"{ld:.4f}",
            "Dynamic Pressure q∞":     f"{q:.2f} Pa",
            "Wind Speed V∞":           f"{aero.wind:.1f} m/s",
            "Reynolds Number":         f"{aero.Re:.3e}",
            "Mach Number":             f"{aero.mach:.4f}",
        }
        col_k, col_v = st.columns([1,1])
        for i,(k,v) in enumerate(rows.items()):
            with (col_k if i%2==0 else col_k):
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:5px 0;border-bottom:1px solid #1e3a5f;font-size:0.82rem;'>"
                    f"<span style='color:#64748b;'>{k}</span>"
                    f"<span style='color:#e2e8f0;font-family:JetBrains Mono,monospace;"
                    f"font-size:0.82rem;'>{v}</span></div>",
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 2 — FLOW FIELD ANIMATION
# ─────────────────────────────────────────────────────────────────────────────

def page_flow_animation(aero: AirfoilAero, custom_coords=None):
    st.markdown("""
    <div class="hero-title">Flow Field Animation</div>
    <div class="hero-sub">Particle stream visualization — animation runs fully in the browser</div>
    <hr style="border-color:#1e3a5f; margin: 8px 0 12px 0;">
    """, unsafe_allow_html=True)

    # ── Inline controls row ────────────────────────────────────────────────
    vc, ac, _ = st.columns([2, 1, 1])

    with vc:
        # V∞ slider: range mirrors the sidebar medium's speed-of-sound cap
        max_v = round(aero.wind * 6, 0)   # give plenty of headroom above current value
        max_v = max(max_v, 100.0)
        local_wind = st.slider(
            "✈️  Airfoil Speed  V∞ (m/s)",
            min_value=1.0,
            max_value=float(max_v),
            value=float(aero.wind),
            step=max(0.5, round(max_v / 300, 1)),
            help="Free-stream velocity of the airfoil through the fluid"
        )

    with ac:
        speed_mult = st.select_slider(
            "🎞️ Anim Speed",
            options=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
            value=1.0,
            format_func=lambda v: f"{v}×",
            help="Playback speed of the particle animation"
        )

    frame_duration = max(10, int(50 / speed_mult))

    # Rebuild aero with the locally overridden wind speed
    # (Re and Mach update automatically)
    local_aero = AirfoilAero(
        naca=aero.naca,
        alpha=aero.alpha,
        Re=(aero.Re / aero.wind) * local_wind,   # scale Re linearly with V
        mach=local_wind / (aero.wind / aero.mach) if aero.mach > 0 else 0.0,
        wind=local_wind,
        custom_geom=aero.custom_geom
    )

    cl = local_aero.cl(); cd = local_aero.cd()
    local_mach = local_aero.mach
    local_re   = local_aero.Re

    st.markdown(
        f'<div style="color:#64748b; font-size:0.8rem; margin-bottom:8px;">'
        f'Shape: <b style="color:#93c5fd">{local_aero.naca}</b>'
        f' &nbsp;·&nbsp; α = <b style="color:#93c5fd">{local_aero.alpha:.1f}°</b>'
        f' &nbsp;·&nbsp; V∞ = <b style="color:#34d399">{local_wind:.1f} m/s</b>'
        f' &nbsp;·&nbsp; Ma = <b style="color:#f472b6">{local_mach:.3f}</b>'
        f' &nbsp;·&nbsp; Re = <b style="color:#a78bfa">{local_re/1e6:.2f}M</b>'
        f' &nbsp;·&nbsp; CL = <b style="color:#34d399">{cl:+.4f}</b>'
        f' &nbsp;·&nbsp; CD = <b style="color:#f59e0b">{cd:.5f}</b>'
        f'&nbsp;&nbsp;<span style="color:#475569; font-size:0.72rem;">(▶ Play / ⏸ Pause inside chart)</span>'
        f'</div>',
        unsafe_allow_html=True)

    cache_key = (local_aero.naca, round(local_aero.alpha, 1),
                 round(local_mach, 3), int(local_wind * 10), frame_duration)
    if st.session_state.get("anim_cache_key") != cache_key:
        with st.spinner("Computing animation frames…"):
            fig = build_animated_flow_chart(
                local_aero, n_frames=70, steps_per_frame=4,
                custom_coords=custom_coords, frame_duration=frame_duration
            )
        st.session_state["anim_fig"]       = fig
        st.session_state["anim_cache_key"] = cache_key
    else:
        fig = st.session_state["anim_fig"]

    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False})

    cl = aero.cl(); cd = aero.cd()
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("CL", f"{cl:+.4f}")
    with m2: st.metric("CD", f"{cd:.5f}")
    with m3: st.metric("L/D", f"{cl/max(cd,1e-9):.2f}")
    with m4: st.metric("q∞ (Pa)", f"{aero.q_inf:.1f}")

    if abs(aero.alpha) >= aero.stall_angle * 0.85:
        st.markdown("""
        <div class="stall-warning">
          ⚠ <strong style="color:#ef4444;">Flow Separation Detected</strong>
          — At this angle of attack the boundary layer separates from the
          upper surface. Lift drops sharply and drag increases significantly.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    naca, alpha, Re, mach, wind, custom_coords, custom_geom = render_sidebar()
    aero = AirfoilAero(naca=naca, alpha=alpha, Re=Re, mach=mach, wind=wind, custom_geom=custom_geom)

    tab1, tab2 = st.tabs(["📊  Dashboard", "🌊  Flow Field Animation"])

    with tab1:
        page_dashboard(aero, custom_coords=custom_coords)

    with tab2:
        page_flow_animation(aero, custom_coords=custom_coords)


if __name__ == "__main__":
    main()
