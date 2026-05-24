"""
Airfoil Aerodynamics Simulation
================================
2D airfoil simulation with:
  - NACA profile geometry
  - CL/CD vs AoA graph
  - Pressure coefficient distribution
  - CL-CD Polar diagram
  - Animated flow field with streamlines

Dependencies: numpy, matplotlib, scipy
Install:  pip install numpy matplotlib scipy
Run:      python airfoil_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  NACA 4-DIGIT GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

def naca_profile(code: str, n: int = 120):
    """Return upper/lower surface (x,y) coords for a NACA 4-digit airfoil."""
    code = str(code).zfill(4)
    m = int(code[0]) / 100.0   # max camber
    p = int(code[1]) / 10.0    # max camber position
    t = int(code[2:]) / 100.0  # thickness

    beta = np.linspace(0, np.pi, n)
    x = (1 - np.cos(beta)) / 2  # cosine spacing for LE accuracy

    # Thickness distribution (NACA 4-digit formula)
    yt = (t / 0.2) * (0.2969 * np.sqrt(x)
                      - 0.1260 * x
                      - 0.3516 * x**2
                      + 0.2843 * x**3
                      - 0.1015 * x**4)

    # Camber line and slope
    yc = np.where(x <= p,
                  m / p**2 * (2*p*x - x**2) if p > 0 else np.zeros_like(x),
                  m / (1-p)**2 * ((1 - 2*p) + 2*p*x - x**2) if p > 0 else np.zeros_like(x))

    if p > 0:
        dyc_dx = np.where(x <= p,
                          2*m / p**2 * (p - x),
                          2*m / (1-p)**2 * (p - x))
    else:
        dyc_dx = np.zeros_like(x)

    theta = np.arctan(dyc_dx)

    xu = x  - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x  + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return xu, yu, xl, yl


def rotate_airfoil(xu, yu, xl, yl, alpha_deg):
    """Rotate airfoil coordinates by angle of attack about quarter-chord."""
    alpha = np.radians(alpha_deg)
    pivot = 0.25  # quarter chord
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)

    def rot(x, y):
        xp = pivot + (x - pivot) * cos_a - y * sin_a
        yp =         (x - pivot) * sin_a + y * cos_a
        return xp, yp

    return (*rot(xu, yu), *rot(xl, yl))


# ─────────────────────────────────────────────────────────────────────────────
#  AERODYNAMICS MODEL
# ─────────────────────────────────────────────────────────────────────────────

class AirfoilAero:
    """
    Physics model combining:
      - Thin airfoil theory (2π lift slope)
      - Thickness & camber corrections
      - Prandtl-Glauert compressibility
      - Reynolds number skin friction
      - Stall onset (soft post-stall model)
      - Induced drag (finite span, AR=8)
      - Wave drag (supersonic onset)
    """

    def __init__(self, naca="2412", alpha=5.0, Re=3e6,
                 mach=0.15, wind=50.0, viscosity=1.81e-5):
        self.naca = str(naca).zfill(4)
        self.alpha = alpha
        self.Re = Re
        self.mach = mach
        self.wind = wind
        self.viscosity = viscosity
        self._parse_naca()

    def _parse_naca(self):
        self.m = int(self.naca[0]) / 100.0
        self.p = int(self.naca[1]) / 10.0
        self.t = int(self.naca[2:]) / 100.0

    # ── Prandtl-Glauert compressibility factor ───────────────────────────────
    @property
    def pg_factor(self):
        M = min(self.mach, 0.99)
        return 1.0 / np.sqrt(max(1e-6, 1 - M**2))

    # ── Zero-lift angle ──────────────────────────────────────────────────────
    @property
    def alpha_0(self):
        if self.p == 0:
            return 0.0
        return np.degrees(-self.m * (1 - 2*self.p) * np.pi / 1.0)

    # ── Lift curve slope (2π with corrections) ───────────────────────────────
    @property
    def cl_slope(self):
        return 2 * np.pi * (1 + 0.77 * self.t)

    # ── Stall angle estimate ─────────────────────────────────────────────────
    @property
    def stall_angle(self):
        return 10.0 + 25*self.t + 15*self.m + 200*self.t*self.m

    # ── Lift coefficient ─────────────────────────────────────────────────────
    def cl(self, alpha=None):
        a = self.alpha if alpha is None else alpha
        alpha_rad = np.radians(a)
        alpha0_rad = np.radians(self.alpha_0)
        re_factor = 1 + 0.10 * np.log10(max(self.Re, 1e5) / 1e6)
        raw = self.cl_slope * (alpha_rad - alpha0_rad) * self.pg_factor * re_factor
        # Stall model: Gaussian roll-off past stall angle
        stall = np.radians(self.stall_angle)
        alpha_arr = np.atleast_1d(alpha_rad)
        raw_arr   = np.atleast_1d(raw)
        over  = np.where(alpha_arr >  stall, (alpha_arr - stall)**2, 0)
        under = np.where(alpha_arr < -stall * 0.85,
                         (alpha_arr + stall * 0.85)**2, 0)
        result = raw_arr * np.exp(-2.8 * (over + under))
        scalar_input = np.ndim(a) == 0
        return float(result[0]) if scalar_input else result

    # ── Drag coefficient ─────────────────────────────────────────────────────
    def cd(self, alpha=None):
        a = self.alpha if alpha is None else alpha
        scalar_input = np.ndim(a) == 0
        a_arr = np.atleast_1d(a)
        cl_val = self.cl(a_arr)
        # Skin friction (Prandtl turbulent BL)
        cf = 0.074 / max(self.Re, 1e4)**0.2
        cd0 = 2 * cf * (1 + 2.7*self.t + 100*self.t**4)
        # Induced drag (finite wing, AR=8)
        e = 0.85 + 0.10*self.m - 0.05*abs(self.p - 0.4)
        AR = 8
        cdi = np.atleast_1d(cl_val)**2 / (np.pi * e * AR)
        # Wave drag
        M = self.mach
        cd_wave = 20 * max(0, M - 0.72)**3 if M > 0.72 else 0
        # Post-stall pressure drag
        a_rad = np.radians(a_arr)
        stall_rad = np.radians(self.stall_angle)
        cd_stall = 0.1 * np.maximum(0, np.abs(a_rad) - stall_rad)**2
        result = cd0 + cdi + cd_wave + cd_stall
        return float(result[0]) if scalar_input else result

    # ── Pressure coefficient distribution ────────────────────────────────────
    def cp_distribution(self, n=80):
        """Returns x, cp_upper, cp_lower arrays along chord."""
        x = np.linspace(0, 1, n)
        a = self.alpha
        cl_val = self.cl()
        pg = self.pg_factor

        # Upper: suction peak near LE, rolls off to TE
        cp_upper = (-cl_val * (1 - x) * np.exp(-5.5 * x)
                    - 0.28 * np.sin(np.pi * x)
                    - np.radians(a) * 0.4
                    + self.m * 0.25) * pg * 0.55

        # Lower: positive pressure
        cp_lower = (cl_val * x * (1 - x) * 0.7
                    + np.radians(a) * 0.18
                    - self.m * 0.18) * pg * 0.55

        return x, cp_upper, cp_lower

    # ── Dynamic pressure ─────────────────────────────────────────────────────
    @property
    def q_inf(self):
        rho = 1.225  # kg/m³ ISA sea level
        return 0.5 * rho * self.wind**2

    # ── L/D ratio ────────────────────────────────────────────────────────────
    def ld_ratio(self, alpha=None):
        a = self.alpha if alpha is None else alpha
        c = self.cd(a)
        return self.cl(a) / np.where(c == 0, 1e-9, c)


# ─────────────────────────────────────────────────────────────────────────────
#  FLOW FIELD (Panel-Method-Inspired Velocity Field)
# ─────────────────────────────────────────────────────────────────────────────

def compute_flow_field(aero: AirfoilAero, Nx=60, Ny=40):
    """
    Build a 2D velocity field around the airfoil using superposition of:
      - Uniform freestream (at angle alpha)
      - Doublet (simulates airfoil body)
      - Vortex (simulates circulation / lift)
    Returns X, Y, U, V grids and speed magnitude.
    """
    alpha = np.radians(aero.alpha)
    V_inf = max(aero.wind, 1.0)
    cl = aero.cl()

    x = np.linspace(-0.6, 1.8, Nx)
    y = np.linspace(-0.7, 0.7, Ny)
    X, Y = np.meshgrid(x, y)

    # Freestream
    U = V_inf * np.cos(alpha) * np.ones_like(X)
    V = V_inf * np.sin(alpha) * np.ones_like(X)

    # Doublet strength proportional to chord + thickness
    mu = 0.045 * (1 + 4 * aero.t)
    # Vortex strength from Kutta-Joukowski: Gamma = 0.5 * V_inf * cl * chord
    Gamma = 0.5 * V_inf * cl * 1.0

    # Multiple source panels to approximate body
    panels = [
        (0.25, 0.00, mu * 1.2),
        (0.50, 0.00, mu * 0.9),
        (0.10, 0.00, mu * 0.7),
        (0.75, 0.00, mu * 0.5),
    ]
    for (xp, yp, strength) in panels:
        dx = X - xp; dy = Y - yp
        r2 = dx**2 + dy**2 + 1e-6
        U += -strength * (dx**2 - dy**2) / r2**2
        V += -strength * 2 * dx * dy      / r2**2

    # Vortex at quarter chord
    dx = X - 0.25; dy = Y - 0.0
    r2 = dx**2 + dy**2 + 1e-6
    U += -Gamma / (2*np.pi) * (-dy / r2)
    V += -Gamma / (2*np.pi) * ( dx / r2)

    # Prandtl-Glauert scaling
    pg = aero.pg_factor
    U *= pg; V *= pg

    speed = np.sqrt(U**2 + V**2)
    return X, Y, U, V, speed


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

class AirfoilSimApp:

    NACA_OPTIONS = ["0009", "0012", "2412", "4412", "6412", "2415", "4415"]

    def __init__(self):
        self.aero = AirfoilAero()
        self._build_ui()
        self._init_animation()
        self._update_all()

    # ─────────────────────────────────────────────────────────────────────────
    #  UI LAYOUT
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.fig = plt.figure(figsize=(17, 10), facecolor="#0f1117")
        self.fig.canvas.manager.set_window_title("Airfoil Aerodynamics Simulator")

        # ── Outer grid: controls (left) | plots (right) ─────────────────────
        outer = gridspec.GridSpec(1, 2, figure=self.fig,
                                  width_ratios=[1, 3.4],
                                  left=0.01, right=0.99,
                                  top=0.97, bottom=0.04,
                                  wspace=0.04)

        # ── Left panel: sliders + NACA selector ─────────────────────────────
        ctrl_ax = self.fig.add_subplot(outer[0])
        ctrl_ax.set_facecolor("#0f1117")
        ctrl_ax.axis("off")
        self._build_controls()

        # ── Right panel: 2×2 grid of plots ──────────────────────────────────
        right_gs = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[1],
            hspace=0.38, wspace=0.32)

        dark = "#0f1117"
        axes_kw = dict(facecolor="#181c27")

        self.ax_clcd  = self.fig.add_subplot(right_gs[0, 0], **axes_kw)
        self.ax_cp    = self.fig.add_subplot(right_gs[0, 1], **axes_kw)
        self.ax_polar = self.fig.add_subplot(right_gs[1, 0], **axes_kw)
        self.ax_flow  = self.fig.add_subplot(right_gs[1, 1], **axes_kw)

        for ax in [self.ax_clcd, self.ax_cp, self.ax_polar, self.ax_flow]:
            ax.tick_params(colors="#9ca3af", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2d3748")
            ax.grid(True, color="#1f2937", linewidth=0.5, alpha=0.8)

        # Titles
        titles = ["CL & CD vs Angle of Attack",
                  "Pressure Coefficient (Cp) Distribution",
                  "Polar Diagram  (CL vs CD)",
                  "Flow Field Animation"]
        for ax, t in zip([self.ax_clcd, self.ax_cp, self.ax_polar, self.ax_flow], titles):
            ax.set_title(t, color="#e2e8f0", fontsize=9, pad=5, fontweight="bold")

        # ── Metric boxes (top-right) ─────────────────────────────────────────
        self.metric_ax = self.fig.add_axes([0.68, 0.93, 0.31, 0.06])
        self.metric_ax.axis("off")
        self.metric_ax.set_facecolor("#0f1117")
        self.metric_texts = {}

    # ─────────────────────────────────────────────────────────────────────────
    #  CONTROL PANEL
    # ─────────────────────────────────────────────────────────────────────────

    def _build_controls(self):
        fig = self.fig
        SLIDER_COLOR  = "#1e3a5f"
        LABEL_COLOR   = "#94a3b8"
        VAL_COLOR     = "#60a5fa"

        # Title
        fig.text(0.015, 0.965, "AIRFOIL SIMULATOR",
                 color="#60a5fa", fontsize=11, fontweight="bold", va="top")
        fig.text(0.015, 0.945, "2D Aerodynamics · Thin Airfoil Theory",
                 color="#475569", fontsize=7.5, va="top")

        # ── NACA Radio Buttons ───────────────────────────────────────────────
        fig.text(0.015, 0.915, "NACA PROFILE",
                 color=LABEL_COLOR, fontsize=8, fontweight="bold", va="top")
        radio_ax = fig.add_axes([0.012, 0.81, 0.19, 0.10])
        radio_ax.set_facecolor("#0f1117")
        self.naca_radio = RadioButtons(
            radio_ax, self.NACA_OPTIONS,
            active=self.NACA_OPTIONS.index("2412"),
            activecolor="#3b82f6")
        for lbl in self.naca_radio.labels:
            lbl.set_color(LABEL_COLOR)
            lbl.set_fontsize(8)
        self.naca_radio.on_clicked(self._on_naca_change)

        # ── Parameter Sliders ────────────────────────────────────────────────
        slider_specs = [
            ("α  Angle of Attack (°)",    "aoa",    -15, 25,  5.0,   0.5),
            ("Re  Reynolds Number (M)",   "re",      0.5, 10, 3.0,  0.1),
            ("Ma  Mach Number",           "mach",   0.01, 0.85, 0.15, 0.01),
            ("V∞  Wind Speed (m/s)",      "wind",     5, 340, 50.0,  1.0),
            ("μ  Viscosity (×10⁻⁵ Pa·s)", "visc",  1.00, 3.00, 1.81, 0.01),
        ]

        self.sliders = {}
        tops = [0.76, 0.69, 0.62, 0.55, 0.48]
        for (label, key, vmin, vmax, vinit, vstep), top in zip(slider_specs, tops):
            fig.text(0.015, top + 0.025, label,
                     color=LABEL_COLOR, fontsize=7.5, va="bottom")
            ax_sl = fig.add_axes([0.015, top, 0.175, 0.018])
            sl = Slider(ax_sl, "", vmin, vmax, valinit=vinit,
                        color="#3b82f6", track_color=SLIDER_COLOR)
            sl.valstep = vstep
            sl.label.set_color(LABEL_COLOR)
            sl.valtext.set_color(VAL_COLOR)
            sl.valtext.set_fontsize(8)
            sl.on_changed(lambda val, k=key: self._on_slider(k, val))
            self.sliders[key] = sl

        # ── Metrics display ──────────────────────────────────────────────────
        fig.text(0.015, 0.44, "LIVE METRICS",
                 color=LABEL_COLOR, fontsize=8, fontweight="bold")

        metric_labels = ["CL", "CD", "L/D", "q∞ (Pa)"]
        metric_colors = ["#34d399", "#f59e0b", "#60a5fa", "#c084fc"]
        self.metric_vals = {}
        for i, (lbl, col) in enumerate(zip(metric_labels, metric_colors)):
            ypos = 0.41 - i * 0.045
            fig.text(0.015, ypos, lbl + ":", color="#64748b", fontsize=8)
            t = fig.text(0.09, ypos, "—", color=col, fontsize=9, fontweight="bold")
            self.metric_vals[lbl] = t

        # ── Buttons ──────────────────────────────────────────────────────────
        btn_ax1 = fig.add_axes([0.015, 0.19, 0.085, 0.03])
        btn_ax2 = fig.add_axes([0.108, 0.19, 0.085, 0.03])
        self.btn_anim  = Button(btn_ax1, "⏸ Pause", color="#1e3a5f", hovercolor="#2d4a6f")
        self.btn_reset = Button(btn_ax2, "↺ Reset",  color="#1e3a5f", hovercolor="#2d4a6f")
        for btn in [self.btn_anim, self.btn_reset]:
            btn.label.set_color("#93c5fd")
            btn.label.set_fontsize(8)
        self.btn_anim.on_clicked(self._toggle_animation)
        self.btn_reset.on_clicked(self._reset_params)

        # ── Info block ───────────────────────────────────────────────────────
        info = (
            "Model: Thin Airfoil Theory\n"
            "  + Thickness / Camber correction\n"
            "  + Prandtl-Glauert (compressibility)\n"
            "  + Turbulent BL (skin friction)\n"
            "  + Induced drag  (AR = 8)\n"
            "  + Soft stall model\n"
            "  + Wave drag (M > 0.72)"
        )
        fig.text(0.015, 0.17, info, color="#475569", fontsize=6.8,
                 va="top", linespacing=1.6,
                 bbox=dict(boxstyle="round,pad=0.4", fc="#0d1117", ec="#1e3a5f", lw=0.8))

    # ─────────────────────────────────────────────────────────────────────────
    #  GRAPH PANEL 1 — CL & CD vs AoA
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_clcd(self):
        ax = self.ax_clcd
        ax.cla()
        ax.set_facecolor("#181c27")
        ax.grid(True, color="#1f2937", linewidth=0.5, alpha=0.8)

        aoas = np.linspace(-20, 28, 300)
        cls  = self.aero.cl(aoas)
        cds  = self.aero.cd(aoas)
        alpha_cur = self.aero.alpha

        # Fill regions
        ax.fill_between(aoas, cls, alpha=0.06, color="#3b82f6")
        ax.fill_between(aoas, cds * 5, alpha=0.05, color="#f59e0b")

        # Main curves
        l1, = ax.plot(aoas, cls,  color="#3b82f6", lw=2.0, label="CL (Lift Coeff.)")
        ax2  = ax.twinx()
        ax2.set_facecolor("#181c27")
        ax2.tick_params(colors="#f59e0b", labelsize=8)
        ax2.set_ylabel("CD", color="#f59e0b", fontsize=8)
        l2,  = ax2.plot(aoas, cds, color="#f59e0b", lw=1.8,
                        linestyle="--", label="CD (Drag Coeff.)")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#2d3748")

        # Current AoA marker
        cl_cur = self.aero.cl(alpha_cur)
        cd_cur = self.aero.cd(alpha_cur)
        ax.axvline(alpha_cur, color="#ef4444", lw=1.0, linestyle=":", alpha=0.8)
        ax.scatter([alpha_cur], [cl_cur],  color="#34d399", s=45, zorder=5)
        ax2.scatter([alpha_cur], [cd_cur], color="#fbbf24", s=45, zorder=5)

        # Stall annotation
        stall = self.aero.stall_angle
        ax.axvline( stall, color="#ef4444", lw=0.7, alpha=0.4, linestyle="-.")
        ax.axvline(-stall * 0.85, color="#ef4444", lw=0.7, alpha=0.4, linestyle="-.")
        ax.text(stall + 0.3, ax.get_ylim()[0] + 0.05, "stall",
                color="#ef4444", fontsize=6.5, alpha=0.7)

        ax.set_xlabel("Angle of Attack α (°)", color="#9ca3af", fontsize=8)
        ax.set_ylabel("CL",                    color="#3b82f6",  fontsize=8)
        ax.set_title("CL & CD vs Angle of Attack",
                     color="#e2e8f0", fontsize=9, pad=5, fontweight="bold")
        ax.tick_params(colors="#9ca3af", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3748")

        lines  = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper left",
                  fontsize=7, facecolor="#0f1117", edgecolor="#2d3748",
                  labelcolor="white")

    # ─────────────────────────────────────────────────────────────────────────
    #  GRAPH PANEL 2 — Pressure Distribution
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_cp(self):
        ax = self.ax_cp
        ax.cla()
        ax.set_facecolor("#181c27")
        ax.grid(True, color="#1f2937", linewidth=0.5, alpha=0.8)

        x, cpu, cpl = self.aero.cp_distribution(n=100)

        ax.fill_between(x, cpu, 0,  color="#ef4444", alpha=0.18, label="_")
        ax.fill_between(x, cpl, 0,  color="#3b82f6", alpha=0.18, label="_")
        ax.plot(x, cpu, color="#ef4444", lw=2.0, label="Upper surface (suction)")
        ax.plot(x, cpl, color="#3b82f6", lw=2.0, label="Lower surface (pressure)")
        ax.axhline(0, color="#475569", lw=0.8)

        # Peak suction label
        peak_idx = np.argmin(cpu)
        ax.annotate(f"Cp={cpu[peak_idx]:.2f}",
                    xy=(x[peak_idx], cpu[peak_idx]),
                    xytext=(x[peak_idx]+0.12, cpu[peak_idx]-0.05),
                    color="#ef4444", fontsize=7,
                    arrowprops=dict(arrowstyle="->", color="#ef4444", lw=0.8))

        ax.invert_yaxis()
        ax.set_xlabel("x/c  (Chord fraction)", color="#9ca3af", fontsize=8)
        ax.set_ylabel("−Cp",                   color="#9ca3af", fontsize=8)
        ax.set_title("Pressure Coefficient Distribution",
                     color="#e2e8f0", fontsize=9, pad=5, fontweight="bold")
        ax.tick_params(colors="#9ca3af", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3748")
        ax.legend(fontsize=7, facecolor="#0f1117",
                  edgecolor="#2d3748", labelcolor="white")

        # Mini airfoil inset
        self._draw_airfoil_inset(ax)

    def _draw_airfoil_inset(self, parent_ax):
        """Small airfoil shape inset in corner of Cp panel."""
        inset = parent_ax.inset_axes([0.62, 0.02, 0.36, 0.28])
        inset.set_facecolor("#0f1117")
        xu, yu, xl, yl = naca_profile(self.aero.naca)
        rxu, ryu, rxl, ryl = rotate_airfoil(xu, yu, xl, yl, self.aero.alpha)
        inset.fill(np.concatenate([rxu, rxl[::-1]]),
                   np.concatenate([ryu, ryl[::-1]]),
                   color="#1e40af", alpha=0.8)
        inset.plot(np.concatenate([rxu, rxl[::-1]]),
                   np.concatenate([ryu, ryl[::-1]]),
                   color="#60a5fa", lw=0.8)
        inset.set_aspect("equal"); inset.axis("off")
        inset.set_title(f"NACA {self.aero.naca}",
                        color="#94a3b8", fontsize=6.5, pad=2)
        # Flow arrow
        inset.annotate("", xy=(rxu[0], ryu[0]),
                       xytext=(rxu[0]-0.25, ryu[0]),
                       arrowprops=dict(arrowstyle="->", color="#60a5fa", lw=0.8))

    # ─────────────────────────────────────────────────────────────────────────
    #  GRAPH PANEL 3 — Polar Diagram
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_polar(self):
        ax = self.ax_polar
        ax.cla()
        ax.set_facecolor("#181c27")
        ax.grid(True, color="#1f2937", linewidth=0.5, alpha=0.8)

        aoas = np.linspace(-18, 28, 400)
        cls  = self.aero.cl(aoas)
        cds  = self.aero.cd(aoas)

        # Color-code the polar by AoA
        points  = np.array([cds, cls]).T.reshape(-1, 1, 2)
        segs    = np.concatenate([points[:-1], points[1:]], axis=1)
        norm    = plt.Normalize(-15, 25)
        lc      = LineCollection(segs, cmap="coolwarm", norm=norm, linewidth=2.2)
        lc.set_array(aoas[:-1])
        ax.add_collection(lc)
        plt.colorbar(lc, ax=ax, label="AoA (°)", shrink=0.7,
                     pad=0.01).ax.tick_params(colors="#9ca3af", labelsize=7)

        # Best L/D tangent from origin
        ld_arr = cls / np.where(cds < 1e-9, 1e-9, cds)
        best_i = np.argmax(ld_arr)
        ax.plot([0, cds[best_i]], [0, cls[best_i]],
                color="#34d399", lw=1.0, linestyle="--", alpha=0.7)
        ax.text(cds[best_i] * 1.05, cls[best_i],
                f"L/D={ld_arr[best_i]:.1f}", color="#34d399", fontsize=7)

        # Current point
        cl_c = self.aero.cl(); cd_c = self.aero.cd()
        ax.scatter([cd_c], [cl_c], color="#f87171", s=60, zorder=6,
                   edgecolors="white", linewidths=0.8, label=f"α={self.aero.alpha:.1f}°")
        ax.axhline(0, color="#475569", lw=0.6)
        ax.axvline(0, color="#475569", lw=0.6)

        ax.set_xlabel("CD  (Drag Coefficient)", color="#9ca3af", fontsize=8)
        ax.set_ylabel("CL  (Lift Coefficient)", color="#9ca3af", fontsize=8)
        ax.set_title("Polar Diagram  (CL vs CD)",
                     color="#e2e8f0", fontsize=9, pad=5, fontweight="bold")
        ax.tick_params(colors="#9ca3af", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3748")
        ax.legend(fontsize=7, facecolor="#0f1117",
                  edgecolor="#2d3748", labelcolor="white")

    # ─────────────────────────────────────────────────────────────────────────
    #  GRAPH PANEL 4 — Flow Animation
    # ─────────────────────────────────────────────────────────────────────────

    def _init_animation(self):
        """Set up static elements for the flow animation panel."""
        self.anim_running = True
        self._anim_obj    = None
        self._stream_seed  = None

    def _draw_flow_static(self):
        """Draw the airfoil body and flow field (called once per param change)."""
        ax = self.ax_flow
        ax.cla()
        ax.set_facecolor("#070d1a")
        ax.set_aspect("equal")
        ax.set_xlim(-0.6, 1.8); ax.set_ylim(-0.7, 0.7)
        ax.tick_params(colors="#9ca3af", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3748")

        # ── Flow field ───────────────────────────────────────────────────────
        X, Y, U, V, speed = compute_flow_field(self.aero, Nx=55, Ny=38)
        speed_n = speed / (speed.max() + 1e-9)

        # Contour of speed (Bernoulli pressure)
        cs = ax.contourf(X, Y, speed_n, levels=30,
                         cmap="coolwarm", alpha=0.45, zorder=1)

        # Streamlines
        seed_y = np.linspace(-0.55, 0.55, 16)
        seed_x = np.full_like(seed_y, -0.55)
        ax.streamplot(X, Y, U, V,
                      color=speed_n, cmap="Blues",
                      density=1.6, linewidth=0.7,
                      arrowsize=0.7, arrowstyle="->",
                      start_points=np.column_stack([seed_x, seed_y]),
                      zorder=2)

        # ── Airfoil body ─────────────────────────────────────────────────────
        xu, yu, xl, yl = naca_profile(self.aero.naca)
        rxu, ryu, rxl, ryl = rotate_airfoil(xu, yu, xl, yl, self.aero.alpha)
        foil_x = np.concatenate([rxu, rxl[::-1], [rxu[0]]])
        foil_y = np.concatenate([ryu, ryl[::-1], [ryu[0]]])
        ax.fill(foil_x, foil_y, color="#1e40af", alpha=0.95, zorder=5)
        ax.plot(foil_x, foil_y, color="#93c5fd", lw=1.2, zorder=6)

        # ── Lift arrow ───────────────────────────────────────────────────────
        cl  = self.aero.cl()
        mid = (np.mean(rxu), np.mean(ryu))
        lift_scale = 0.20 * cl
        ax.annotate("", xy=(mid[0], mid[1] + lift_scale),
                    xytext=mid,
                    arrowprops=dict(arrowstyle="->",
                                    color="#34d399", lw=1.8))
        ax.text(mid[0]+0.03, mid[1]+lift_scale/2,
                f"L (CL={cl:.3f})", color="#34d399", fontsize=6.5)

        # ── Drag arrow ───────────────────────────────────────────────────────
        cd  = self.aero.cd()
        drag_scale = 0.25 * cd * 30
        ax.annotate("", xy=(mid[0] + drag_scale, mid[1]),
                    xytext=mid,
                    arrowprops=dict(arrowstyle="->",
                                    color="#f87171", lw=1.5))

        # ── Velocity vectors (subset) ─────────────────────────────────────────
        step = 4
        ax.quiver(X[::step, ::step], Y[::step, ::step],
                  U[::step, ::step], V[::step, ::step],
                  speed_n[::step, ::step],
                  cmap="winter", scale=30, alpha=0.35,
                  width=0.003, headwidth=3, zorder=3)

        # ── Stagnation / separation labels ──────────────────────────────────
        alpha_abs = abs(self.aero.alpha)
        if alpha_abs > self.aero.stall_angle * 0.85:
            ax.text(0.6, 0.35, "⚠ FLOW\nSEPARATION",
                    color="#ef4444", fontsize=7.5, fontweight="bold",
                    ha="center", alpha=0.9,
                    bbox=dict(boxstyle="round", fc="#1a0000", ec="#ef4444", lw=0.8))

        # ── Info overlay ─────────────────────────────────────────────────────
        info = (f"NACA {self.aero.naca}  |  α = {self.aero.alpha:.1f}°  |  "
                f"Ma = {self.aero.mach:.2f}  |  Re = {self.aero.Re/1e6:.2f}M")
        ax.set_title("Flow Field Animation",
                     color="#e2e8f0", fontsize=9, pad=5, fontweight="bold")
        ax.text(0.5, -0.62, info, transform=ax.transData,
                color="#64748b", fontsize=6.5, ha="center")

    # ─────────────────────────────────────────────────────────────────────────
    #  METRICS DISPLAY
    # ─────────────────────────────────────────────────────────────────────────

    def _update_metrics(self):
        cl  = self.aero.cl()
        cd  = self.aero.cd()
        ld  = cl / max(cd, 1e-9)
        q   = self.aero.q_inf

        values  = {"CL": f"{cl:+.4f}", "CD": f"{cd:.5f}",
                   "L/D": f"{ld:.2f}", "q∞ (Pa)": f"{q:.1f}"}
        for key, val in values.items():
            if key in self.metric_vals:
                self.metric_vals[key].set_text(val)

    # ─────────────────────────────────────────────────────────────────────────
    #  MAIN UPDATE
    # ─────────────────────────────────────────────────────────────────────────

    def _update_all(self):
        self._draw_clcd()
        self._draw_cp()
        self._draw_polar()
        self._draw_flow_static()
        self._update_metrics()
        self.fig.canvas.draw_idle()

    # ─────────────────────────────────────────────────────────────────────────
    #  CALLBACKS
    # ─────────────────────────────────────────────────────────────────────────

    def _on_slider(self, key, val):
        if   key == "aoa":  self.aero.alpha     = val
        elif key == "re":   self.aero.Re        = val * 1e6
        elif key == "mach": self.aero.mach      = val
        elif key == "wind": self.aero.wind      = val
        elif key == "visc": self.aero.viscosity = val * 1e-5
        self._update_all()

    def _on_naca_change(self, label):
        self.aero.naca = label
        self.aero._parse_naca()
        self._update_all()

    def _toggle_animation(self, event):
        self.anim_running = not self.anim_running
        label = "⏸ Pause" if self.anim_running else "▶ Resume"
        self.btn_anim.label.set_text(label)
        self.fig.canvas.draw_idle()

    def _reset_params(self, event):
        defaults = dict(aoa=5.0, re=3.0, mach=0.15, wind=50.0, visc=1.81)
        for key, val in defaults.items():
            self.sliders[key].set_val(val)
        self.naca_radio.set_active(self.NACA_OPTIONS.index("2412"))
        self._update_all()

    def run(self):
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Airfoil Aerodynamics Simulator")
    print("  NACA 4-digit · 2D · Matplotlib interactive GUI")
    print("=" * 60)
    print()
    app = AirfoilSimApp()
    app.run()
