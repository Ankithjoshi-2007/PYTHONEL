


import math
import sys


class Aircraft:
    """
    Models the 3D physical wing geometry and weight parameters of the aircraft.
    """
    def __init__(self, wingspan: float, mean_chord: float, weight: float, oswald_efficiency: float):
        """
        Initializes the Aircraft physical profile.

        Args:
            wingspan (float): Wingspan (b) in meters.
            mean_chord (float): Mean aerodynamic chord length (c) in meters.
            weight (float): Aircraft total weight (W) in Newtons.
            oswald_efficiency (float): Oswald efficiency factor (e) (dimensionless).
        """
        self.wingspan = float(wingspan)
        self.mean_chord = float(mean_chord)
        self.weight = float(weight)
        self.oswald_efficiency = float(oswald_efficiency)

    @property
    def wing_area(self) -> float:
        """
        Calculates reference wing area (S).
        Formula: S = b * c
        """
        return self.wingspan * self.mean_chord

    @property
    def aspect_ratio(self) -> float:
        """
        Calculates Aspect Ratio (AR).
        Formula: AR = b^2 / S = b / c
        """
        area = self.wing_area
        if area == 0:
            return 0.0
        return (self.wingspan ** 2) / area

    @property
    def mass_kg(self) -> float:
        """Calculates aircraft mass in kg under standard gravity (g = 9.81 m/s^2)."""
        return self.weight / 9.81


class Aerodynamics:
    """
    Performs multi-fidelity aerodynamic solver calculations, combining NACA shape profiles
    or custom 2D coefficients with full 3D aspect ratio and wave-drag corrections.
    """
    DYNAMIC_VISCOSITY = 1.81e-5  # Dynamic viscosity of air (mu) in Pa*s

    def __init__(self, aircraft: Aircraft, rho: float, velocity: float, viscosity: float = 1.81e-5):
        """
        Initializes the Aerodynamic State Solver.

        Args:
            aircraft (Aircraft): Physical aircraft geometry.
            rho (float): Ambient air density (rho) in kg/m^3.
            velocity (float): True Airspeed (V) in m/s.
            viscosity (float): Dynamic viscosity (mu) in Pa*s.
        """
        self.aircraft = aircraft
        self.rho = float(rho)
        self.velocity = float(velocity)
        self.viscosity = float(viscosity)
        
        # Flight state variables (resolved dynamically in solver)
        self.cl_2d = 0.0
        self.cd0_custom = None
        self.alpha_deg = 0.0
        
        # NACA 4-Digit Profile geometry parameters (initialized to 2412 as default)
        self.naca = "2412"
        self.m = 0.02
        self.p = 0.4
        self.t = 0.12
        self.use_naca_solver = False

    def setup_custom_coefficients(self, cl_2d: float, cd0: float):
        """Sets solver to use direct 2D Lift and zero-lift Drag coefficient inputs."""
        self.cl_2d = float(cl_2d)
        self.cd0_custom = float(cd0)
        self.use_naca_solver = False
        
        # Approximate equivalent Angle of Attack (AoA) for wave/stall corrections
        # Thin airfoil theory lift slope (2*pi) approximation: Cl = 2 * pi * alpha_rad
        self.alpha_deg = math.degrees(self.cl_2d / (2.0 * math.pi))

    def setup_naca_profile(self, naca: str, alpha_deg: float):
        """Configures solver to physically compute coefficients from NACA 4-digit geometry."""
        self.naca = str(naca).zfill(4)
        self.alpha_deg = float(alpha_deg)
        self.use_naca_solver = True
        
        # Parse NACA 4-digit parameters:
        # Digit 1: max camber (m)
        self.m = int(self.naca[0]) / 100.0
        # Digit 2: position of max camber (p)
        self.p = int(self.naca[1]) / 10.0
        # Digit 3 & 4: thickness (t)
        self.t = int(self.naca[2:]) / 100.0

    @property
    def mach_number(self) -> float:
        """Calculates flight Mach number assuming standard speed of sound (340.0 m/s)."""
        return self.velocity / 340.0

    @property
    def reynolds_number(self) -> float:
        """
        Calculates flight Reynolds Number.
        Formula: Re = (rho * V * c) / mu
        """
        c = self.aircraft.mean_chord
        if self.viscosity == 0:
            return 1e6
        return (self.rho * self.velocity * c) / self.viscosity

    @property
    def dynamic_pressure(self) -> float:
        """Calculates ambient dynamic pressure (q = 0.5 * rho * V^2) in Pascals (Pa)."""
        return 0.5 * self.rho * (self.velocity ** 2)

    @property
    def pg_factor(self) -> float:
        """
        Calculates Prandtl-Glauert compressibility correction factor.
        Formula: pg = 1.0 / sqrt(1 - M^2)
        """
        M = min(self.mach_number, 0.99)
        return 1.0 / math.sqrt(max(1e-6, 1.0 - M**2))

    @property
    def zero_lift_angle_deg(self) -> float:
        """
        Calculates the zero-lift angle of attack (alpha_0) in degrees from camber profile.
        Formula (from airfoil_simulation.py): alpha_0 = -m * (1 - 2*p) * pi (converted to deg)
        """
        if self.p == 0:
            return 0.0
        # Formula matches Line 126 in airfoil_simulation (1).py
        val_rad = -self.m * (1.0 - 2.0 * self.p) * math.pi
        return math.degrees(val_rad)

    @property
    def cl_slope_2d(self) -> float:
        """
        Calculates 2D Lift curve slope with thickness correction.
        Formula: cl_slope = 2 * pi * (1 + 0.77 * t)
        """
        return 2.0 * math.pi * (1.0 + 0.77 * self.t)

    @property
    def stall_angle_deg(self) -> float:
        """
        Estimates airfoil stall angle in degrees from geometry parameters.
        Formula: 10 + 25*t + 15*m + 200*t*m
        """
        return 10.0 + 25.0 * self.t + 15.0 * self.m + 200.0 * self.t * self.m

    def solve_2d_cl(self) -> float:
        """
        Calculates 2D Lift Coefficient.
        If using NACA solver, applies thin airfoil theory, Prandtl-Glauert, 
        Reynolds number factor, and soft stall roll-off.
        """
        if not self.use_naca_solver:
            return self.cl_2d

        alpha_rad = math.radians(self.alpha_deg)
        alpha0_rad = math.radians(self.zero_lift_angle_deg)
        
        # Reynolds correction factor: 1 + 0.10 * log10(Re / 1e6)
        re_factor = 1.0 + 0.10 * math.log10(max(self.reynolds_number, 1e5) / 1e6)
        
        # Raw linear lift coefficient
        raw_cl = self.cl_slope_2d * (alpha_rad - alpha0_rad) * self.pg_factor * re_factor
        
        # Stall model: Gaussian roll-off past stall angle (Line 149 in airfoil_simulation.py)
        stall_rad = math.radians(self.stall_angle_deg)
        over = (alpha_rad - stall_rad)**2 if alpha_rad > stall_rad else 0.0
        under = (alpha_rad + stall_rad * 0.85)**2 if alpha_rad < -stall_rad * 0.85 else 0.0
        
        return raw_cl * math.exp(-2.8 * (over + under))

    def solve_cd0(self) -> float:
        """
        Calculates zero-lift profile drag coefficient (CD0).
        If using NACA solver, derives it from boundary layer skin friction & thickness.
        Formula: cd0 = 2 * cf * (1 + 2.7*t + 100*t^4)
        """
        if not self.use_naca_solver and self.cd0_custom is not None:
            return self.cd0_custom

        # Skin friction coefficient (Prandtl turbulent boundary layer)
        re = max(self.reynolds_number, 1e4)
        cf = 0.074 / (re ** 0.2)
        
        # Zero-lift profile drag formula (Line 164 in airfoil_simulation (1).py)
        return 2.0 * cf * (1.0 + 2.7 * self.t + 100.0 * (self.t ** 4))

    def calculate_3d_cl(self) -> float:
        """
        Calculates corrected 3D Lift Coefficient for finite wing.
        Formula: CL_3D = CL_2D / (1 + (CL_2D / (pi * e * AR)))
        """
        cl_2d = self.solve_2d_cl()
        ar = self.aircraft.aspect_ratio
        e = self.aircraft.oswald_efficiency
        
        denom = 1.0 + (cl_2d / (math.pi * e * ar)) if ar > 0 and e > 0 else 1.0
        return cl_2d / denom

    def calculate_induced_drag_coefficient(self) -> float:
        """
        Calculates lift-induced drag coefficient.
        Formula: Cdi = CL_3D^2 / (pi * e * AR)
        """
        cl_3d = self.calculate_3d_cl()
        ar = self.aircraft.aspect_ratio
        e = self.aircraft.oswald_efficiency
        denom = math.pi * e * ar
        if denom == 0:
            return 0.0
        return (cl_3d ** 2) / denom

    def calculate_wave_drag_coefficient(self) -> float:
        """
        Calculates compressibility wave drag coefficient past drag divergence Mach limit.
        Formula: cd_wave = 20 * max(0, M - 0.72)^3 if M > 0.72 else 0
        """
        M = self.mach_number
        if M > 0.72:
            return 20.0 * ((M - 0.72) ** 3)
        return 0.0

    def calculate_stall_drag_coefficient(self) -> float:
        """
        Calculates post-stall pressure separation drag.
        Formula: cd_stall = 0.1 * max(0, abs(alpha_rad) - stall_rad)^2
        """
        alpha_rad = math.radians(self.alpha_deg)
        stall_rad = math.radians(self.stall_angle_deg)
        
        diff = abs(alpha_rad) - stall_rad
        if diff > 0:
            return 0.1 * (diff ** 2)
        return 0.0

    def calculate_total_drag_coefficient(self) -> float:
        """
        Calculates total 3D Drag coefficient.
        Formula: CD = CD0 + Cdi + Cd_wave + Cd_stall
        """
        cd0 = self.solve_cd0()
        cdi = self.calculate_induced_drag_coefficient()
        cd_wave = self.calculate_wave_drag_coefficient()
        cd_stall = self.calculate_stall_drag_coefficient()
        
        return cd0 + cdi + cd_wave + cd_stall

    def calculate_lift_force(self) -> float:
        """
        Calculates total Aerodynamic Lift force (L).
        Formula: L = 0.5 * rho * V^2 * S * CL_3D
        """
        cl_3d = self.calculate_3d_cl()
        return self.dynamic_pressure * self.aircraft.wing_area * cl_3d

    def calculate_drag_force(self) -> float:
        """
        Calculates total Aerodynamic Drag force (D).
        Formula: D = 0.5 * rho * V^2 * S * CD
        """
        cd = self.calculate_total_drag_coefficient()
        return self.dynamic_pressure * self.aircraft.wing_area * cd

    def calculate_power_required(self) -> float:
        """
        Calculates power required to sustain airspeed.
        Formula: P = D * V (Watts)
        """
        return self.calculate_drag_force() * self.velocity

    def calculate_lift_to_drag_ratio(self) -> float:
        """Calculates Lift-to-Drag Ratio (L/D) representing aerodynamic efficiency."""
        cd = self.calculate_total_drag_coefficient()
        if cd == 0:
            return 0.0
        return self.calculate_3d_cl() / cd


class FlightAnalyzer:
    """
    Performs flight mechanics, force balances, safety, and equilibrium analysis.
    """
    STABILITY_THRESHOLD_N = 10.0  # Force balance range in Newtons

    def __init__(self, aero: Aerodynamics):
        self.aero = aero
        self.aircraft = aero.aircraft

    def calculate_net_vertical_force(self) -> float:
        """
        Calculates net vertical force (Fz).
        Formula: Fz = Lift - Weight
        """
        return self.aero.calculate_lift_force() - self.aircraft.weight

    def calculate_thrust_to_weight_ratio(self) -> float:
        """
        Calculates thrust-to-weight ratio (T/W).
        Under steady flight, Thrust Required = Drag (T = D).
        """
        thrust_required = self.aero.calculate_drag_force()
        return thrust_required / self.aircraft.weight

    def determine_flight_state(self) -> str:
        """
        Determines current flight state and evaluates stall tendencies.
        """
        # 1. Evaluate Stall Tendency
        # High 2D Lift Coefficient or High angle of attack triggers warnings
        cl_2d = self.aero.solve_2d_cl()
        if self.aero.alpha_deg >= self.aero.stall_angle_deg:
            return "STALL (Angle of Attack exceeds stall limit)"
        elif self.aero.alpha_deg >= 0.85 * self.aero.stall_angle_deg or cl_2d >= 1.4:
            return "STALL WARNING (High 2D Lift Coefficient / Imminent Stall Tendency)"

        # 2. Evaluate Trajectory Balance
        net_vertical = self.calculate_net_vertical_force()
        
        if abs(net_vertical) < self.STABILITY_THRESHOLD_N:
            return "STABLE FLIGHT (Cruise Equilibrium)"
        elif net_vertical > 0:
            return "CLIMBING"
        else:
            return "DESCENDING"

    def generate_report(self) -> str:
        """
        Compiles the aerodynamic corrections and force balances into a detailed report.
        """
        cl_3d = self.aero.calculate_3d_cl()
        cd0 = self.aero.solve_cd0()
        cdi = self.aero.calculate_induced_drag_coefficient()
        cd_wave = self.aero.calculate_wave_drag_coefficient()
        cd_stall = self.aero.calculate_stall_drag_coefficient()
        cd_total = self.aero.calculate_total_drag_coefficient()
        
        lift = self.aero.calculate_lift_force()
        drag = self.aero.calculate_drag_force()
        net_vertical = self.calculate_net_vertical_force()
        ld_ratio = self.aero.calculate_lift_to_drag_ratio()
        tw_ratio = self.calculate_thrust_to_weight_ratio()
        
        power_w = self.aero.calculate_power_required()
        power_kw = power_w / 1000.0
        power_hp = power_w / 745.7
        state = self.determine_flight_state()
        
        induced_percent = (cdi / cd_total) * 100 if cd_total > 0 else 0.0
        wave_percent = (cd_wave / cd_total) * 100 if cd_total > 0 else 0.0
        stall_percent = (cd_stall / cd_total) * 100 if cd_total > 0 else 0.0
        profile_percent = (cd0 / cd_total) * 100 if cd_total > 0 else 0.0
        
        report = []
        report.append("=" * 80)
        report.append("             3D FINITE-WING AIRCRAFT FLIGHT PERFORMANCE REPORT             ")
        report.append("=" * 80)
        
        # Section 1: Wing Geometry
        report.append("\n[1] 3D WING GEOMETRIC CONFIGURATION")
        report.append("-" * 80)
        report.append(f"  * Wingspan (b)           : {self.aircraft.wingspan:<10.2f} m")
        report.append(f"  * Mean Chord Length (c)  : {self.aircraft.mean_chord:<10.2f} m")
        report.append(f"  * Wing Reference Area (S): {self.aircraft.wing_area:<10.2f} m^2        (S = b * c)")
        report.append(f"  * Aspect Ratio (AR)      : {self.aircraft.aspect_ratio:<10.2f}          (AR = b^2 / S = b/c)")
        report.append(f"  * Oswald Efficiency (e)  : {self.aircraft.oswald_efficiency:<10.2f}")
        report.append(f"  * Aircraft Weight (W)    : {self.aircraft.weight:<10.2f} N          (Mass: {self.aircraft.mass_kg:.1f} kg)")
        report.append("-" * 80)
        
        # Section 2: Environment
        report.append("\n[2] ATMOSPHERIC & OPERATING FLOW FIELD")
        report.append("-" * 80)
        report.append(f"  * Air Density (rho)      : {self.aero.rho:<10.4f} kg/m^3")
        report.append(f"  * Velocity (V)           : {self.aero.velocity:<10.2f} m/s        (TAS)")
        report.append(f"  * Mach Number (M)        : {self.aero.mach_number:<10.4f}          (Speed of sound: 340.0 m/s)")
        report.append(f"  * Reynolds Number (Re)   : {self.aero.reynolds_number:<10.2e}          (Dynamic Viscosity: 1.81e-5 Pa*s)")
        report.append(f"  * Dynamic Pressure (q)   : {self.aero.dynamic_pressure:<10.2f} Pa         (0.5 * rho * V^2)")
        
        if self.aero.use_naca_solver:
            report.append(f"  * Operating NACA Profile : {self.aero.naca:<10s}          (Camber: {self.aero.m*100}%, Position: {self.aero.p*10}%, Thickness: {self.aero.t*100}%)")
            report.append(f"  * Angle of Attack (AoA)  : {self.aero.alpha_deg:<10.2f}°")
            report.append(f"  * Zero-Lift Angle (a0)   : {self.aero.zero_lift_angle_deg:<+10.2f}°")
            report.append(f"  * Est. Stall Angle       : {self.aero.stall_angle_deg:<10.2f}°")
        else:
            report.append(f"  * Operating 2D CL        : {self.aero.cl_2d:<10.4f}          (User defined airfoil coefficient)")
            report.append(f"  * Zero-Lift Drag Coeff   : {self.aero.cd0_custom:<10.4f}          (User defined profile coefficient)")
        report.append("-" * 80)
        
        # Section 3: Coefficient Decomposition
        report.append("\n[3] 3D AERODYNAMIC COEFFICIENTS & DRAG DECOMPOSITION")
        report.append("-" * 80)
        report.append(f"  * Corrected 3D CL        : {cl_3d:<10.4f}          (CL_3D = CL_2D / [1 + (CL_2D / [pi * e * AR])])")
        report.append(f"  * Parasite Drag Coeff CD0: {cd0:<10.4f}          ({profile_percent:.1f}% of total drag)")
        report.append(f"  * Induced Drag Coeff Cdi : {cdi:<10.4f}          ({induced_percent:.1f}% of total drag - finite wing)")
        
        if cd_wave > 0:
            report.append(f"  * Wave Drag Coeff CdWave : {cd_wave:<10.4f}          ({wave_percent:.1f}% of total drag - wave divergence)")
        if cd_stall > 0:
            report.append(f"  * Stall Drag Coeff CdStal: {cd_stall:<10.4f}          ({stall_percent:.1f}% of total drag - stall separation)")
            
        report.append(f"  * Total 3D Drag Coeff CD : {cd_total:<10.4f}          (CD = CD0 + Cdi + Cd_wave + Cd_stall)")
        report.append("-" * 80)
        
        # Section 4: Forces and Equilibrium
        report.append("\n[4] FORCE BALANCE & ENGINE METRICS")
        report.append("-" * 80)
        report.append(f"  * Aerodynamic Lift (L)   : {lift:<10.2f} N")
        report.append(f"  * Aerodynamic Drag (D)   : {drag:<10.2f} N")
        report.append(f"  * Required Thrust (T)    : {drag:<10.2f} N          (T = D for constant velocity)")
        report.append(f"  * Net Vertical Force     : {net_vertical:<+10.2f} N          (Fz = Lift - Weight)")
        report.append(f"  * Lift-to-Drag (L/D)     : {ld_ratio:<10.2f}          (Efficiency metric)")
        report.append(f"  * Thrust-to-Weight (T/W) : {tw_ratio:<10.4f}")
        report.append(f"  * Power Required (P)     : {power_kw:<10.2f} kW         ({power_hp:.1f} HP, P = D * V)")
        report.append("-" * 80)
        
        # Section 5: Flight State
        report.append("\n[5] FLIGHT STATE ANALYSIS")
        report.append("-" * 80)
        report.append(f"  * Determined Condition   : {state}")
        
        equilibrium = "YES (Balanced flight forces)" if abs(net_vertical) < self.STABILITY_THRESHOLD_N else "NO (Aircraft climbing or descending)"
        report.append(f"  * Static Equilibrium?    : {equilibrium}")
        report.append("=" * 80)
        
        # Section 6: Architecture Hook for Extensibility
        report.append("\n[6] EXTENSIBILITY HOOKS (FUTURE CFD/ENGINE MODULES)")
        report.append("-" * 80)
        report.append("  >> stability_analysis(): Hook for spanwise moment pitching derivatives.")
        report.append("  >> cfd_mesh_integrator(): Hook compatible with spanwise chord variation.")
        report.append("  >> engine_simulation(): Integrate standard TSFC curves at T = D.")
        report.append("=" * 80)
        report.append("                       END OF 3D ENGINEERING REPORT                        ")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    print("=" * 80)
    print("      ✈️   HIGH-FIDELITY 3D finite-WING FLIGHT PERFORMANCE SIMULATOR  ✈️")
    print("=" * 80)
    print("  Models 3D finite wingspan aerodynamics, lift-induced drag corrections,")
    print("  Prandtl-Glauert compressibility, turbulent skin friction, and wave-drag")
    print("  derived directly from 'airfoil_simulation (1).py' physics engine.")
    print("=" * 80)
    
    # Mode selection to accommodate either custom coefficients or NACA shape physical solvers
    print("\nSelect Simulation Input Mode:")
    print("----------------------------")
    print("1) Enter Custom 2D Coefficients (Cl_2D, Cd0)")
    print("2) Enter NACA 4-Digit Airfoil Profile (NACA shape, AoA, physical Re & Mach solve)")
    
    while True:
        mode = input("\nSelect mode (1-2) [Default: 1]: ").strip()
        if mode == "" or mode == "1":
            use_naca = False
            break
        elif mode == "2":
            use_naca = True
            break
        else:
            print("  [Error] Invalid choice. Please enter 1 or 2.")
            
    try:
        print("\n--- Enter Environment and Aircraft Wing Parameters ---")
        rho = float(input("1. Air Density (rho) [kg/m^3]: "))
        velocity = float(input("2. Velocity (V) [m/s]: "))
        wingspan = float(input("3. Wingspan (b) [meters]: "))
        mean_chord = float(input("4. Mean Chord Length (c) [meters]: "))
        weight = float(input("5. Aircraft Weight (W) [Newtons]: "))
        oswald_e = float(input("6. Oswald Efficiency Factor (e) [0.1 - 1.0]: "))
        
        # Instantiate base geometry & aerodynamics
        aircraft = Aircraft(
            wingspan=wingspan,
            mean_chord=mean_chord,
            weight=weight,
            oswald_efficiency=oswald_e
        )
        
        aero = Aerodynamics(
            aircraft=aircraft,
            rho=rho,
            velocity=velocity
        )
        
        if use_naca:
            naca = input("\n7. Enter NACA 4-Digit Profile Code [e.g. 2412]: ").strip()
            if not naca:
                naca = "2412"
            aoa = float(input("8. Enter Angle of Attack (AoA) [degrees]: "))
            aero.setup_naca_profile(naca=naca, alpha_deg=aoa)
        else:
            cl_2d = float(input("\n7. Enter 2D Lift Coefficient (Cl_2D): "))
            cd0 = float(input("8. Enter Zero-Lift Drag Coefficient (Cd0): "))
            aero.setup_custom_coefficients(cl_2d=cl_2d, cd0=cd0)
            
        # Verify inputs are physically valid
        if rho < 0 or velocity < 0 or wingspan <= 0 or mean_chord <= 0 or weight <= 0 or oswald_e <= 0:
            print("\n[Error] Invalid inputs. Dimensions, weights, and environments must be non-negative.")
            return

        # Perform Flight Mechanics Analysis
        analyzer = FlightAnalyzer(aero)
        
        # Print High-Fidelity Report
        print("\nSolving 3D Aerodynamics & Flight States...")
        print(analyzer.generate_report())

        # Dynamic Connection to airfoil_simulation (1).py
        print("\n" + "=" * 80)
        print("                 INTERACTIVE WIND TUNNEL SIMULATOR INTERFACE                ")
        print("=" * 80)
        launch_sim = input("Would you like to launch the interactive 2D Wind Tunnel simulation\nusing these exact calculated flight variables? (y/n) [Default: n]: ").strip().lower()
        
        if launch_sim in ('y', 'yes'):
            print("\nPreparing aerodynamic flow-field visualizer...")
            try:
                import importlib
                # Dynamically import the airfoil_simulation module handling space in name
                airfoil_sim = importlib.import_module("airfoil_simulation (1)")
                
                # Resolve parameters
                naca_profile = aero.naca if aero.use_naca_solver else "2412"
                alpha = aero.alpha_deg
                reynolds = aero.reynolds_number
                mach = aero.mach_number
                wind = aero.velocity
                
                print(f"Launching interactive GUI with:")
                print(f"  * Profile   : NACA {naca_profile}")
                print(f"  * AoA       : {alpha:.2f}°")
                print(f"  * Reynolds  : {reynolds:.2e}")
                print(f"  * Mach      : {mach:.4f}")
                print(f"  * Wind Speed: {wind:.2f} m/s")
                print("\nClose the simulation window to return to the analyzer.")
                
                # Instantiate and run visual application
                app = airfoil_sim.AirfoilSimApp(
                    naca=naca_profile,
                    alpha=alpha,
                    Re=reynolds,
                    mach=mach,
                    wind=wind
                )
                app.run()
                print("\nVisual simulation closed. Returning to main program.")
            except ImportError as err:
                print(f"\n[Error] Dependency import error: {err}")
                missing_mod = None
                for mod in ["matplotlib", "numpy", "scipy"]:
                    if mod in str(err):
                        missing_mod = mod
                        break
                if missing_mod or "airfoil_simulation" not in str(err):
                    print("  --> The required graphic dependencies for the wind tunnel visualizer are not installed.")
                    print("  --> Please install them by running the following command in your terminal:")
                    print("      pip install numpy matplotlib scipy")
                else:
                    print("Please ensure 'airfoil_simulation (1).py' is in the current directory.")
            except Exception as ex:
                print(f"\n[Error] Failed to initialize visualizer: {ex}")
                
    except ValueError:
        print("\n[Error] Invalid entry. Please enter numbers only.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAerodynamic suite interrupted. Exiting.")
        sys.exit(0)
