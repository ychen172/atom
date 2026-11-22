"""
Supporting functions for the TAB (Taylor Analogy Breakup) droplet atomization simulation
(O'Rourke & Amsden 1987, SAE 872089).

Common arguments of the analytic TAB functions:
    y0, ydot0: TAB state at the reference time t = 0
               (y = x/(Cb*r): equator displacement scaled so breakup is y = 1; ydot = dy/dt)
    a: CF/Cb * rhog/rhol * u^2/r^2   aerodynamic forcing; a/b = We/12 is the equilibrium y
    b: Ck*sigma/(rhol*r^3)            surface-tension spring; undamped omega0^2
    c: Cd*mul/(rhol*r^2)              viscous damping; c/2 = 1/td
    a, b, c must be unchanged since the reference time; re-reference after any update
"""
import math as mh
from bisect import bisect_left, bisect_right
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
import scipy.special as sc
from typing import NamedTuple
from collections.abc import Callable
import numpy as np
from numba import njit
import CoolProp as cpr
import cantera as ct
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

#### Functions
def y_calc(t,y0,ydot0,a,b,c):
    """
    Analytic TAB distortion y(t) at constant a, b, c (O'Rourke Eq. 4).
    Valid in all damping regimes, with omega^2 = b - c^2/4:
    underdamped (omega^2 > 0), critical (omega^2 = 0), overdamped (omega^2 < 0).
    t: time elapsed since the reference state (pass t = 0 if y0, ydot0 are the current state)
    y0, ydot0: TAB state at the reference time t = 0
    a: CF/Cb * rhog/rhol * u^2/r^2
    b: (Ck*sigma)/(rhol*r^3)
    c: (Cd*mul)/(rhol*r^2)
    a, b, c must be unchanged since the reference time; re-reference after any update
    return: y(t)
    """
    # Quick return
    if t==0.0:
        return y0
    alpha,beta,omega_sqr = alpha_beta_omega_sqr_calc(y0,ydot0,a,b,c)
    if omega_sqr == 0:
        # print(f'Critical damp detected: omega^2:{omega_sqr} detected at t:{t}, y0:{y0}, ydot0:{ydot0}')
        y = a/b + mh.exp(-c*t/2)*(alpha + beta*t)
    elif omega_sqr < 0:
        # print(f'Overdamp detected: omega^2:{omega_sqr} detected at t:{t}, y0:{y0}, ydot0:{ydot0}')
        omega_mag = mh.sqrt(-omega_sqr)
        cos_omega_t = mh.cosh(omega_mag*t) #cos(omega*t)
        sin_omega_t_omega = mh.sinh(omega_mag*t)/omega_mag #sin(omega*t)/omega
        y = a/b + mh.exp(-c*t/2)*(alpha*cos_omega_t + beta*sin_omega_t_omega)
    else:
        omega = mh.sqrt(omega_sqr)
        y = a/b + mh.exp(-c*t/2)*(alpha*mh.cos(omega*t) + beta*mh.sin(omega*t)/omega)
    return y
    

def alpha_beta_omega_sqr_calc(y0,ydot0,a,b,c):
    """
    Constants of the analytic TAB solution
        y(t) = a/b + exp(-c*t/2)*(alpha*cos(omega*t) + beta*sin(omega*t)/omega)
    y0, ydot0: TAB state at the reference time t = 0
    a: CF/Cb * rhog/rhol * u^2/r^2
    b: (Ck*sigma)/(rhol*r^3)
    c: (Cd*mul)/(rhol*r^2)
    return: alpha = y0 - a/b, beta = ydot0 + c*alpha/2, omega_sqr = b - c^2/4
    """
    alpha = y0-(a/b)
    beta = ydot0 + c*y0/2 - a*c/2/b
    omega_sqr = b-c**2/4
    return alpha,beta,omega_sqr


def dt_peak_calc(t_cur,y0,ydot0,a,b,c):
    """
    Time from t_cur to the next maximum of y (next peak of the damped oscillation).
    Only valid for underdamped oscillation (omega_sqr > 0); raises ValueError otherwise.
    Peaks satisfy omega*t - delta - psi = 3*pi/2 + 2*n*pi, with
    delta = atan2(beta/omega, alpha) and psi = atan2(2*omega, c).
    t_cur: time elapsed since the reference state at which to start looking
           (pass t_cur = 0 if y0, ydot0 are the current state)
    y0, ydot0: TAB state at the reference time t = 0
    a: CF/Cb * rhog/rhol * u^2/r^2
    b: (Ck*sigma)/(rhol*r^3)
    c: (Cd*mul)/(rhol*r^2)
    a, b, c must be unchanged since the reference time; re-reference after any update
    return: dt_peak >= 0, time from t_cur to the next peak
    """
    alpha,beta,omega_sqr = alpha_beta_omega_sqr_calc(y0,ydot0,a,b,c)
    if omega_sqr <= 0: raise ValueError(f'Overdamp detected in dt_peak_calc calculation: omega^2:{omega_sqr} detected at t:{t_cur}, y0:{y0}, ydot0:{ydot0}')
    omega = mh.sqrt(omega_sqr)
    delta = mh.atan2(beta/omega,alpha)
    psi = mh.atan2(2*omega,c)
    return ((3*mh.pi/2 - (omega*t_cur-delta-psi))%(2*mh.pi))/omega

def ydot_calc(t,y0,ydot0,a,b,c):
    """
    Analytic TAB distortion rate dy/dt at constant a, b, c (time derivative of y_calc).
    Valid in all damping regimes (underdamped, critical, overdamped), like y_calc.
    t: time elapsed since the reference state (pass t = 0 if y0, ydot0 are the current state)
    y0, ydot0: TAB state at the reference time t = 0
    a: CF/Cb * rhog/rhol * u^2/r^2
    b: (Ck*sigma)/(rhol*r^3)
    c: (Cd*mul)/(rhol*r^2)
    a, b, c must be unchanged since the reference time; re-reference after any update
    return: ydot(t) [1/s]
    """
    # Quick return
    if t==0.0:
        return ydot0
    alpha,beta,omega_sqr = alpha_beta_omega_sqr_calc(y0,ydot0,a,b,c)
    if omega_sqr == 0:
        # print(f'ydot_calc: critical damp detected: omega^2:{omega_sqr} detected at t:{t}, y0:{y0}, ydot0:{ydot0}')
        ydot = (-c/2)*mh.exp(-c*t/2)*(alpha + beta*t) + mh.exp(-c*t/2)*beta
    elif omega_sqr < 0:
        # print(f'Overdamp detected: omega^2:{omega_sqr} detected at t:{t}, y0:{y0}, ydot0:{ydot0}')
        omega_mag = mh.sqrt(-omega_sqr)
        cos_omega_t = mh.cosh(omega_mag*t) #cos(omega*t)
        sin_omega_t_omega = mh.sinh(omega_mag*t)/omega_mag #sin(omega*t)/omega
        ydot = (-c/2)*mh.exp(-c*t/2)*(alpha*cos_omega_t + beta*sin_omega_t_omega) + \
               mh.exp(-c*t/2)*(-alpha*omega_sqr*sin_omega_t_omega + beta*cos_omega_t)
    else:
        coeff = mh.sqrt(alpha**2 + beta**2/omega_sqr)
        coeff2 = mh.sqrt(c**2/4+omega_sqr)
        omega = mh.sqrt(omega_sqr)
        angle = omega*t - mh.atan2(beta/omega, alpha) - mh.atan2(2*omega,c)
        ydot = -coeff*mh.exp(-c*t/2)*coeff2*mh.cos(angle)
    return ydot

def r32_N_calc(rp,ydot,rhol,sig,Ck,K,Cb,Cv):
    """
    Child drop size from O'Rourke's breakup energy balance (Eq. 22, written with general constants),
    evaluated at the moment of breakup (y = 1).
    rp: parent drop radius [m]
    ydot: dy/dt at breakup [1/s], with y = x/(Cb*r)
    rhol, sig: liquid density [kg/m3] and surface tension [N/m]
    Ck: TAB spring constant (8, matches Lamb's n = 2 frequency)
    K: ratio of total distortion/oscillation energy to fundamental-mode energy (10/3)
    Cb: breakup amplitude coefficient (1/2)
    Cv: child normal-velocity coefficient (1)
    return: r32, N
        r32: Sauter mean radius of the children [m]
        N: (rp/r32)^3, children per parent drop (multiply the parcel's N by this; mass conserved)
    """
    rp_r32 = 1 + (Ck*K*Cb**2)/5 + ((rhol*rp**3*ydot**2)/sig) * (K*Cb**2/5 - Cb**2*Cv**2/6)
    r32 = rp/rp_r32
    N = rp_r32**3
    return r32,N

#### Grouped inputs
class TABConst(NamedTuple):
    """TAB model constants (O'Rourke & Amsden 1987); defaults are the published values."""
    CF: float = 1/3     # aerodynamic forcing coefficient
    Cb: float = 0.5     # breakup amplitude coefficient (equator moves r/2 at breakup)
    Ck: float = 8.0     # spring constant (Lamb n = 2 frequency)
    Cd: float = 5.0     # damping constant (Lamb viscous decay)
    K: float = 10/3     # total / fundamental-mode oscillation energy ratio
    Cv: float = 1.0     # child normal-velocity coefficient

class Props(NamedTuple):
    """Liquid and gas properties used by TAB."""
    rhol: float         # liquid density [kg/m3]
    sig: float          # surface tension [N/m]
    mul: float          # liquid dynamic viscosity [Pa s]
    rhog: float         # gas density [kg/m3]

class DropState(NamedTuple):
    """State of one drop parcel."""
    N: float            # drops represented by the parcel (> 0)
    r: float            # drop radius [m]
    y: float            # TAB distortion, y = x/(Cb*r); breakup at y = 1
    ydot: float         # dy/dt [1/s]

#### Step output and updates
class StepResult(NamedTuple):
    """Result of one atomize_one_step call; all values at the end of the advanced time dt_done."""
    dt_done: float    # time actually advanced (<= dt_tar if breakup)
    flag_breakup: bool # Whether breakup has happened
    r_new: float          # drop radius after the step
    N_new: float          # drops represented (multiplied at breakup)
    y_new: float          # TAB distortion at end of step (0 after breakup)
    ydot_new: float
    vel_break: float  # V_perp of children at breakup (0 if none)

def nonbreakup_update(dt_tar,y0,ydot0,a,b,c,N0,r0):
    """
    Advance the TAB oscillator by dt_tar with no breakup (constant a, b, c).
    dt_tar: time to advance [s] from the reference state y0, ydot0
    N0, r0: parcel drop count and radius (unchanged)
    return: StepResult with dt_done = dt_tar, r and N unchanged, y and ydot at dt_tar, vel_break = 0, flag_breakup=False
    """
    y_new = y_calc(dt_tar,y0,ydot0,a,b,c)
    ydot_new = ydot_calc(dt_tar,y0,ydot0,a,b,c)
    dt_done, N_new, r_new  = dt_tar, N0, r0
    vel_break = 0.0
    return StepResult(dt_done=dt_done, flag_breakup=False, r_new=r_new, N_new=N_new, 
                      y_new=y_new, ydot_new=ydot_new, vel_break=vel_break)

def breakup_update(dt_breakup,y0,ydot0,a,b,c,N0,r0,rhol,sig,Ck,K,Cb,Cv):
    """
    Apply breakup at dt_breakup, the time from the reference state at which y first reaches 1.
    Children get radius r32 and the parcel's N is multiplied by (r0/r32)^3 (see r32_N_calc);
    children start undistorted, y = ydot = 0.
    vel_break = Cv*Cb*r0*ydot at breakup: child velocity normal to the relative velocity [m/s]
    (uses the parent radius r0)
    return: StepResult with dt_done = dt_breakup, flag_breakup=True
    """
    ydot_new = ydot_calc(dt_breakup,y0,ydot0,a,b,c)
    r_new,N_new = r32_N_calc(r0,ydot_new,rhol,sig,Ck,K,Cb,Cv)
    N_new *= N0 # Each parent drop breakup into more
    dt_done = dt_breakup
    vel_break = Cv*Cb*r0*ydot_new
    ydot_new = 0.0 #Reset state after break
    y_new = 0.0 #Reset state after breakup
    return StepResult(dt_done=dt_done, flag_breakup=True, r_new=r_new, N_new=N_new, 
                      y_new=y_new, ydot_new=ydot_new, vel_break=vel_break)

def atomize_one_step(dt_tar,drop,u,props,tc=TABConst(),atol_frac=1e-12):
    """
    Advance one drop parcel's TAB oscillator by up to dt_tar at constant relative velocity u,
    stopping early at the first breakup (y reaching 1).
    - Overdamped/critically damped (omega^2 <= 0): advanced analytically, no breakup.
    - Underdamped: if the next peak exceeds 1 and the crossing y = 1 lies within the step,
      the crossing time is found with brentq on [0, min(dt_peak, dt_tar)].
    dt_tar: target step size [s] from the current state (t = 0)
    drop: DropState, current parcel state (N > 0 and y <= 1; raises ValueError otherwise)
    u: relative gas-drop speed [m/s], held constant during the step
    props: Props (liquid and gas properties)
    tc: TABConst (defaults: O'Rourke's values)
    atol_frac: brentq time tolerance as a fraction of the search interval
    return: StepResult(dt_done, flag_breakup, r_new, N_new, y_new, ydot_new, vel_break)
        flag_breakup being true means a breakup happened at dt_done: call again for the remaining time
    """
    # Decompose grouped inputs
    N0,r0,y0,ydot0 = drop
    rhol,sig,mul,rhog = props
    CF,Cb,Ck,Cd,K,Cv = tc
    # TAB coefficients for the current radius and relative velocity
    a = CF/Cb * rhog/rhol * u**2/r0**2
    b = Ck*sig/(rhol*r0**3)
    c = Cd*mul/(rhol*r0**2)
    # Size check
    if N0<=0: raise ValueError(f'Number of droplets starting {N0} need to be at least bigger than 0')
    if y0>1: raise ValueError(f'y_init {y0} larger than one')
    # Check on overdamped and critically damped cases
    alpha,beta,omega_sqr = alpha_beta_omega_sqr_calc(y0,ydot0,a,b,c)
    if omega_sqr<=0.0:
        state_new = nonbreakup_update(dt_tar,y0,ydot0,a,b,c,N0,r0)
        return state_new
    # underdamped cases
    dt_peak = dt_peak_calc(0.0,y0,ydot0,a,b,c)
    y_peak = y_calc(dt_peak,y0,ydot0,a,b,c)
    if y_peak<=1:
        # No breakup
        state_new = nonbreakup_update(dt_tar,y0,ydot0,a,b,c,N0,r0)
    else:
        # Find the time of breakup
        if dt_peak <= dt_tar: #The breakup will happen within peak time
            dt_breakup = brentq(lambda t_test: y_calc(t_test,y0,ydot0,a,b,c)-1.0, 0.0, dt_peak, xtol=atol_frac*dt_peak)
            state_new = breakup_update(dt_breakup,y0,ydot0,a,b,c,N0,r0,rhol,sig,Ck,K,Cb,Cv)
        else:
            y_tar = y_calc(dt_tar,y0,ydot0,a,b,c)
            if y_tar > 1: #Breakup will happen
                dt_breakup = brentq(lambda t_test: y_calc(t_test,y0,ydot0,a,b,c)-1.0, 0.0, dt_tar, xtol=atol_frac*dt_tar)
                state_new = breakup_update(dt_breakup,y0,ydot0,a,b,c,N0,r0,rhol,sig,Ck,K,Cb,Cv)
            else: #Breakup will not happen
                state_new = nonbreakup_update(dt_tar,y0,ydot0,a,b,c,N0,r0)
    return state_new

#### Drop drag (velocity relaxation toward the gas)
def u_drop_update_Putnam(dt_tar,rho_drop,d_drop,mu_gas,rho_gas,u_rel_mag0,u_rel_dir,u_gas):
    """
    Exact drag update for Re <= 1000 with the Putnam (KIVA) drag law Cd = 24/Re*(1 + Re^(2/3)/6),
    at constant drop diameter and gas state:
        d|w|/dt = -(|w|/tau0)*(1 + c*|w|^(2/3)),  w = u_gas - u_drop
    solved with s = |w|^(2/3): s/(1 + c*s) = s0/(1 + c*s0)*exp(-(2/3)*t/tau0)
    dt_tar: time to advance [s]
    rho_drop, d_drop: drop density [kg/m3] and diameter [m]
    mu_gas, rho_gas: gas viscosity [Pa s] and density [kg/m3]
    u_rel_mag0: initial slip speed |u_gas - u_drop| [m/s] (must give Re <= 1000)
    u_rel_dir: unit vector of the slip, unchanged by drag
    u_gas: gas velocity vector [m/s]
    return: u_drop_new, u_rel_new (vectors [m/s])
    """
    tau0 = (rho_drop*d_drop**2)/(18*mu_gas)
    c = (1/6)*(rho_gas*d_drop/mu_gas)**(2/3)
    S0 = u_rel_mag0**(2/3)
    G = (S0/(1+c*S0))*mh.exp((-2/3)*(dt_tar/tau0))
    u_rel_mag_new = mh.sqrt((G/(1-c*G))**3)
    u_rel_new = u_rel_dir*u_rel_mag_new
    u_drop_new = u_gas - u_rel_new
    return u_drop_new,u_rel_new

class DragArgs(NamedTuple):
    """Fixed inputs of vel_change_one_step over one step; see vel_change_one_step for each field."""
    d_drop: float
    rho_gas: float
    rho_drop: float
    mu_gas: float

def vel_change_one_step(dt_tar,u0_drop,u_gas,drag_args):
    """
    Advance one drop's velocity by dt_tar under drag (KIVA-II drag law), exactly, at constant drop
    diameter and gas state:
        Re <= 1000: Cd = 24/Re*(1 + Re^(2/3)/6) (Putnam);  Re > 1000: Cd = 0.424
        Re = |u_gas - u_drop|*d_drop*rho_gas/mu_gas
    Drag acts along the slip w = u_gas - u_drop, so the slip keeps its direction and only its
    magnitude decays. A drop starting above Re = 1000 is advanced on the constant-Cd branch until
    Re = 1000, then on the Putnam branch for the rest of the step (Cd is continuous there).
    dt_tar: time to advance [s]; always advanced in full
    u0_drop: drop velocity vector [m/s] (any dimension, e.g. 3)
    u_gas: gas velocity vector [m/s], same dimension as u0_drop
    drag_args: DragArgs:
        d_drop: drop diameter [m]
        rho_gas, rho_drop: gas and drop densities [kg/m3]
        mu_gas: gas dynamic viscosity [Pa s]
    return: dt_tar, u_drop_new, u_rel_new (u_rel_new = u_gas - u_drop_new) [m/s]
    """
    (d_drop,rho_gas,rho_drop,mu_gas) = drag_args
    u_rel0 = u_gas - u0_drop # (u_rel,v_rel,w_rel)
    if np.all(u_rel0 == 0.0):
        return dt_tar, u_gas.copy(), u_rel0
    u_rel_mag0 = np.sqrt(np.sum((u_rel0)**2)) # initial absolute mag
    u_rel_dir = u_rel0/u_rel_mag0 # unit directional vector
    Re0 = u_rel_mag0*d_drop*rho_gas/mu_gas #Initial Re number
    # Attempt step
    if Re0 < 1000:
        u_drop_new,u_rel_new = u_drop_update_Putnam(dt_tar,rho_drop,d_drop,mu_gas,rho_gas,u_rel_mag0,u_rel_dir,u_gas)
    else:
        # Constant Cd (Newton regime): d|w|/dt = -C1*|w|^2, so |w| = |w0|/(1 + C1*|w0|*t)
        Cd = 0.424
        C1 = (3/4)*(rho_gas/rho_drop)*(Cd/d_drop) # [1/m]
        # Critical time where Reynolds number drops to 1000
        dt_critical = max((u_rel_mag0/(1000*mu_gas/rho_gas/d_drop) - 1)/(C1*u_rel_mag0),0.0) # clamp: round-off can push it below 0 at Re0 = 1000
        dt_used = min(dt_critical,dt_tar)
        u_rel_mag_new = u_rel_mag0/(1+C1*u_rel_mag0*dt_used) # direction unchanged
        if dt_used<dt_tar: # Re reached 1000: Putnam branch for the rest of the step
            u_drop_new,u_rel_new = u_drop_update_Putnam(dt_tar-dt_used,rho_drop,d_drop,mu_gas,rho_gas,u_rel_mag_new,u_rel_dir,u_gas)
        else:
            u_rel_new = u_rel_dir*u_rel_mag_new
            u_drop_new = u_gas - u_rel_new
    return dt_tar, u_drop_new, u_rel_new

def _u_gas_fun_def(x_drop):
    """
    Default gas velocity field for pos_vel_upd: quiescent gas
    x_drop: drop position vector [m] (3D)
    return: gas velocity vector [m/s]: zeros(3)
    """
    return np.zeros(3)

def pos_vel_upd_step(dt_tar,x0_drop,u0_drop,drag_args,u_gas_fun=_u_gas_fun_def,u0_gas=None,uf_gas=None):
    """
    This function advances one drop's position and velocity by one predictor-corrector step through a gas velocity field
    Predict the end position with the old velocity, drag (vel_change_one_step) toward the gas velocity averaged
    between the start and the predicted end, then correct the position with the trapezoid rule on the drop velocity
    args:
        dt_tar: float: time to advance [s]; always advanced in full
        x0_drop: array: drop position at the start of the step [m]
        u0_drop: array: drop velocity at the start of the step [m/s]
        drag_args: DragArgs: fixed drop and gas properties over the step (see vel_change_one_step)
        u_gas_fun: function: u_gas_fun(x) -> gas velocity vector [m/s] at position x, same dimension as x
        u0_gas=None,uf_gas=None: optional: start and end gas velocity (Provided then u_gas_fun will not be called)
    return: xf_drop, uf_drop
        xf_drop: array: drop position at the end of the step [m]
        uf_drop: array: drop velocity at the end of the step [m/s]
    Behavior:
        Second order for drops with inertia; first order in the tracer limit (drag time << dt_tar)
        Accuracy depends on how much u_gas changes over the step: the caller must keep dt_tar small enough
    """
    # Predict position
    if (u0_gas is None) or (uf_gas is None):
        xf_drop = x0_drop + dt_tar*u0_drop #array: estimated droplet location
        u_gas = (u_gas_fun(x0_drop)+u_gas_fun(xf_drop))/2 #array: average gas velocity [m/s]
    else:
        u_gas = (u0_gas+uf_gas)/2#array: average gas velocity [m/s]
    # Predict velocity
    _, uf_drop, _ = vel_change_one_step(dt_tar,u0_drop,u_gas,drag_args)
    # Correct position
    u_drop = (u0_drop+uf_drop)/2
    xf_drop = x0_drop + dt_tar*u_drop
    return xf_drop,uf_drop

def pos_vel_upd(dt_tar,dt_ini,x0_drop,u0_drop,drag_args,u_gas_fun=_u_gas_fun_def,rel_du_gas_max=5e-4,frac_du_gas_tar=0.9,frac_dt_min=1e-10,eps_u_mag=1e-10,eps_rel_du_gas=1e-10,sca_up_dt_max=2):
    """
    This function advances one drop's position and velocity over dt_tar by sub-cycling pos_vel_upd_step
    Each sub-step is sized by the relative change of the gas velocity between its start and predicted end:
        rel_du_gas = |u_gas(x_end) - u_gas(x_start)| / max(|u_gas(x_start)|, |u_gas(x_end)|, eps_u_mag)
        rel_du_gas > rel_du_gas_max: reject, shrink to 0.5*frac_du_gas_tar*rel_du_gas_max, retry
        accepted: next sub-step rescaled toward frac_du_gas_tar*rel_du_gas_max, growth capped at sca_up_dt_max
    args:
        dt_tar: float: time to advance [s]; always advanced in full
        dt_ini: float: first trial sub-step [s]; pass dt_rec from the previous call (or dt_tar)
        x0_drop: array: drop position at the start [m]
        u0_drop: array: drop velocity at the start [m/s]
        drag_args: DragArgs: fixed drop and gas properties (see vel_change_one_step)
        u_gas_fun: function: u_gas_fun(x) -> gas velocity vector [m/s] at position x, same dimension as x
        rel_du_gas_max: float: maximum relative gas velocity change per sub-step
        frac_du_gas_tar: float: target fraction of rel_du_gas_max for sizing the next sub-step (< 1)
        frac_dt_min: float: smallest sub-step as a fraction of dt_tar; accepted without check (gas discontinuities)
        eps_u_mag: float: floor of the reference gas speed in rel_du_gas [m/s]
        eps_rel_du_gas: float: floor of rel_du_gas when rescaling the next sub-step (uniform gas: rel_du_gas = 0)
        sca_up_dt_max: float: maximum growth factor of the sub-step between accepted sub-steps
    return: x_drop, u_drop, dt_rec
        x_drop: array: drop position at the end [m]
        u_drop: array: drop velocity at the end [m/s]
        dt_rec: float: planned sub-step before the end clip [s]; pass as dt_ini of the next call
    Behavior:
        Second order for drops with inertia; first order in the tracer limit (accuracy set by rel_du_gas_max)
    """
    # Initialization
    t = 0
    dt_cur = dt_ini
    dt_rec = dt_ini # returned unchanged if dt_tar <= 0
    x_drop = x0_drop
    u_drop = u0_drop
    dt_min = frac_dt_min*dt_tar
    # Simulation
    while t<dt_tar:
        # Compute tentative step
        dt_rec = dt_cur #Record the last unclip dt as starting guess for next run
        dt_cur = min(dt_cur,dt_tar-t)
        u0_gas = u_gas_fun(x_drop)
        uf_gas = u_gas_fun(x_drop+dt_cur*u_drop)
        # Get tentative relative velocity change
        u0_gas_mag = np.sqrt(np.sum(u0_gas**2))
        uf_gas_mag = np.sqrt(np.sum(uf_gas**2))
        rel_du_gas = np.sqrt(np.sum((u0_gas-uf_gas)**2))/max(u0_gas_mag,uf_gas_mag,eps_u_mag)
        # Reduce time step and rewind if gas velocity change too much
        if (rel_du_gas>rel_du_gas_max) and (dt_cur>dt_min):
            dt_cur = max(dt_cur*rel_du_gas_max*frac_du_gas_tar*0.5/rel_du_gas,dt_min)
            continue
        # Run one step update
        x_drop,u_drop=pos_vel_upd_step(dt_cur,x_drop,u_drop,drag_args,u0_gas=u0_gas,uf_gas=uf_gas)
        t+=dt_cur
        # Rescale next sub-step toward the target (shrink or grow, growth capped)
        dt_cur *= min(sca_up_dt_max,rel_du_gas_max*frac_du_gas_tar/max(rel_du_gas,eps_rel_du_gas))
    return x_drop,u_drop,dt_rec

#### Domain Limiter
def dom_limiter_none(xi,yi,zi,u_i,v_i,w_i,ax_i,ay_i,az_i):
    """
    Placeholder domain limiter (no wall): returns velocity and acceleration unchanged
    """
    return u_i,v_i,w_i,ax_i,ay_i,az_i

def dom_lim_axi_sym_init(xlst,Rlst,Rref,eps_vlim=1e-3,eps_Rlim=1e-2,tau_rest=5e-6,flg_inner=False):
    """
    This function make a domain limiter that will constrain the droplet movement to be within
    a certain domain by clipping droplet velocity when droplet location is outside of a defined domain.
    - Only design for axially symmetrical domain (around axis 'x')
    - list of x and R is ordered from upstream to downstream: non-decreasing
    - Outer wall (flow on the axis side of the wall) by default, inner wall (flow on the larger R side) with flg_inner
    - Droplets outside the x range of xlst aren't constrained
    - Slip wall boundary condition
    - R domain boundary is larger than 0. Not blocking the flow.
    - Avoid very sharp corner (wall turns < 180 deg)
    - Droplets farther than the distance where the correction is < 1e-16 return immediately;
    - Near the wall only the segments within the distance bound of the segment spanning xi are checked (bisect)
    args:
        xlst: array: [m] domain boundary x coordinates (non-decreasing)
        Rlst: array: [m] domain boundary R coordinates (>0)
        Rref: float: [m] reference radius to define film thickness
        eps_vlim: float: at a boundary film height, vel_correction/vel_full_correction (<1)
        eps_Rlim: float: the boundary height fraction, height_film/Rref for eps_vlim (<1)
        tau_rest: float: [s] response time of the critically damped penalty acceleration inside the wall (>0)
        flg_inner: bool: False for an outer wall (e.g. liner), True for an inner wall (e.g. hub); xlst still upstream to downstream
    return:
        dom_limiter: function: dom_limiter(x,y,z,u,v,w,ax,ay,az) -> u,v,w,ax,ay,az wall limited velocity [m/s] and acceleration [m/s2] for x,y,z [m]
                     Non-inertial transport: update dx/dt, dy/dt, dz/dt with u,v,w output
                     Inertial transport: update du/dx, dv/dt, dw/dt with ax,ay,az output
    """
    #### For inward pointing unit vectors
    if np.any(np.diff(xlst) < 0):
        raise ValueError(f'dom_lim_axi_sym_init: xlst must be non-decreasing (ordered from upstream to downstream)')
    dx_dom = np.diff(xlst)
    dx_dom = np.concatenate((dx_dom,[dx_dom[-1]])) #last vertex relies on the padding 
    dR_dom = np.diff(Rlst)
    dR_dom = np.concatenate((dR_dom,[dR_dom[-1]])) #same a size as xlst
    C1 = np.sqrt(dR_dom**2 + dx_dom**2) #same a size as xlst
    if np.any(C1 == 0):
        raise ValueError(f'dom_lim_axi_sym_init: zero-length boundary segment(s)')
    Cx = dR_dom/C1
    Cyz = -dx_dom/C1 #same a size as xlst
    # Flip the inward velocity vector for inner hub case
    if flg_inner:
        Cx *= -1.0
        Cyz *= -1.0
    # Vin = {Cx, Cyz*yi/Ri, Cyz*zi/Ri} where Ri = sqrt(yi^2 + zi^2)
    
    #### Domain boundary starting and ending point
    xlims = [np.min(xlst),np.max(xlst)]
    
    #### Compute boundary velocity correction strength
    if not (0 < eps_vlim < 1 and 0 < eps_Rlim < 1 and Rref > 0):
        raise ValueError(f'dom_lim_axi_sym_init: need 0<eps_vlim<1, 0<eps_Rlim<1, Rref>0; '
                         f'got eps_vlim={eps_vlim}, eps_Rlim={eps_Rlim}, Rref={Rref}')
    k_corr = mh.log(eps_vlim)/(-eps_Rlim*Rref)
    if not (tau_rest > 0):
        raise ValueError(f'dom_lim_axi_sym_init: need tau_rest>0; got tau_rest={tau_rest}')
    k_rest = 1.0/tau_rest**2 #restoring spring [1/s2] from unforced oscillation
    c_rest = 2.0/tau_rest #critical damping [1/s]

    #### Plain python lists for the per call scalar math (numpy per call overhead dominates for few points)
    xlst = [float(x_cur) for x_cur in xlst]
    Rlst = [float(R_cur) for R_cur in Rlst]
    dx_dom, dR_dom, C1, Cx, Cyz = dx_dom.tolist(), dR_dom.tolist(), C1.tolist(), Cx.tolist(), Cyz.tolist()
    num_seg = len(xlst)-1
    Tx = [dx_dom[idx]/C1[idx] for idx in range(num_seg)] # unit tangent along each segment (x, R components)
    TR = [dR_dom[idx]/C1[idx] for idx in range(num_seg)]
    
    #### Below this radius every wall point is farther than the distance where the correction drops under 1e-16
    if flg_inner:
        R_far = max(Rlst) + mh.log(1e16)/k_corr #Inner wall: above this radius every wall point is that far
    else:
        R_far = min(Rlst) - mh.log(1e16)/k_corr #Subtract out the film thickness where velocity is less than 1e-16 affected

    #### For projection distance determination
    def seg_dist(idx,xi,Ri):
        """
        Distance from (xi,Ri) to segment idx (clamped projection, the clamped end is the vertex) and the projection along it
        - The first return is the absolute distance of the projection
            - for projection within segment, it is just the distance as clip and norm cancel out
            - for projection outside segmement, it is the distance from point to vertex by trigonometry
        - The second return is the unclipped projection along the tangential, can be <0 or >full length
        """
        dx_i = xi-xlst[idx]
        dR_i = Ri-Rlst[idx] #vector component from the first vertex to the point
        proj_norm = dx_i*Tx[idx] + dR_i*TR[idx] #projected distance from the first point of a segment along tangential
        proj_clip = min(max(proj_norm,0.0),C1[idx]) #If projection within segment, then unchange, if outside then clip at 0 or full tangential length
        return mh.hypot(proj_norm-proj_clip, dx_i*Cx[idx] + dR_i*Cyz[idx]), proj_norm

    #### Create the limiting function
    def dom_limiter(xi,yi,zi,u_i,v_i,w_i,ax_i,ay_i,az_i):
        #### no-opt
        Ri = mh.sqrt(yi**2 + zi**2)
        if (xi<xlims[0]) or (xi>xlims[-1]):
            return u_i,v_i,w_i,ax_i,ay_i,az_i
        if flg_inner:
            if (Ri>R_far) or (Ri==0):
                return u_i,v_i,w_i,ax_i,ay_i,az_i
        else:
            if (Ri<R_far) or (Ri==0):
                return u_i,v_i,w_i,ax_i,ay_i,az_i

        #### Compute inward vector and sign distance
        # Identify the closest segment, clipped normal distance to the segment/vertex, and unclipped projection along the tangential direction
        idx_span = min(max(bisect_right(xlst,xi)-1,0),num_seg-1) # Segment with closest x
        dist_bound = seg_dist(idx_span,xi,Ri)[0] # Distance to that segment or vertex
        idx_lo = max(bisect_left(xlst,xi-dist_bound)-1,0) # Lower bound to search for the shorest segment
        idx_hi = min(bisect_right(xlst,xi+dist_bound),num_seg) #Upper bound to search for the shorest segment
        dist_norm_min = mh.inf
        for idx in range(idx_lo,idx_hi):
            dist_cur,proj_cur = seg_dist(idx,xi,Ri)
            if dist_cur < dist_norm_min:
                dist_norm_min = dist_cur # Minimum distance (clipped by the distance to the vertex)
                idx_seg_min_dist = idx # Index of this segment
                proj_min = proj_cur # Unclipped projection along the tangential from the first vertex, can be <0 or >length
        # Get the inward pointing vector and the side of the point to the domain
        if 0.0 <= proj_min <= C1[idx_seg_min_dist]: #Direct normal contact
            vec_in_x = Cx[idx_seg_min_dist]
            vec_in_y = Cyz[idx_seg_min_dist]*yi/Ri
            vec_in_z = Cyz[idx_seg_min_dist]*zi/Ri
            # Get the side of the point wrt to the segment
            cross_min = dx_dom[idx_seg_min_dist]*(Ri-Rlst[idx_seg_min_dist])-dR_dom[idx_seg_min_dist]*(xi-xlst[idx_seg_min_dist]) #Crossing tangential vector with vertex1->point, if positive k means outward (to the left of the segment)
            if flg_inner:
                cross_min *= -1.0 #domain to the left of segment (Turn to positive outside of the domain)
            # Create sign
            if cross_min > 0:
                sign_min = 1.0 #postive outside
            elif cross_min < 0:
                sign_min = -1.0
            else:
                sign_min = 0.0 #Full effect with 0 distance, sign does not matter
        else:
            # First detect which vertex
            if proj_min < 0.0:
                idx_vertex = idx_seg_min_dist #First vertex clipped
            else:
                idx_vertex = idx_seg_min_dist+1 #Second vertex clipped
            # Setup the index for surrounding wall normal vectors
            if idx_vertex==0 or idx_vertex==num_seg:
                idx_vertex_m1 = idx_vertex #Both the first and the last vertex use padding normal vector from vertex itself
                idx_vertex_p1 = idx_vertex
            else:
                idx_vertex_m1 = idx_vertex-1 #Or else use the two normal vectors around the vertex
                idx_vertex_p1 = idx_vertex
            # Side and inward vector determination
            dx_i = xi-xlst[idx_vertex]
            dR_i = Ri-Rlst[idx_vertex] #Vector from the vertex pointing to the point
            if (dx_i==0) and (dR_i==0): #Right at the vertex. Use averaged of the two wall normal
                dx_i = Cx[idx_vertex_m1] +Cx[idx_vertex_p1]
                dR_i = Cyz[idx_vertex_m1]+Cyz[idx_vertex_p1] #Should have non-zero norm if not 180 deg turn
                sign_min = 0.0 #Sign does not matter as distance is 0 and correction is full strength
            else: #Need side detection
                proj_to_vec_in_m1 = Cx[idx_vertex_m1]*dx_i + Cyz[idx_vertex_m1]*dR_i
                proj_to_vec_in_p1 = Cx[idx_vertex_p1]*dx_i + Cyz[idx_vertex_p1]*dR_i
                proj_sum = proj_to_vec_in_m1+proj_to_vec_in_p1 # If not 180 deg turn. <0 means point is outside, and the direction should be revert.
                if proj_sum < 0: # Pointing outside with no 180 deg turn
                    sign_min = 1.0 # Postive outside
                    dx_i *= -1
                    dR_i *= -1 #Flip to inward pointing vector (or at least a direct project applicable domain)
                else:
                    sign_min = -1.0 # Pointing inside
            vec_in_i_norm = mh.hypot(dx_i,dR_i) # magnitude for normalization
            vec_in_x = dx_i/vec_in_i_norm
            vec_in_y = (yi/Ri)*dR_i/vec_in_i_norm
            vec_in_z = (zi/Ri)*dR_i/vec_in_i_norm
        # Get the signed distance
        dist_i = dist_norm_min*sign_min #float: Positive means outside of the domain

        #### Apply droplet velocity correction
        scale_corr = 1.0 #Postive distance (outside the domain) do full correction
        if dist_i<0:
            scale_corr = mh.exp(k_corr*dist_i) #Negative distance (inside the domain) with diminishing correction as distance goes
        # Velocity correction
        comp_vec_out = -(u_i*vec_in_x + v_i*vec_in_y + w_i*vec_in_z) #Component of velocity along outward movement vector
        if comp_vec_out > 0: #Correct the velocity only if outward movement
            u_i -= scale_corr*comp_vec_out*(-vec_in_x) #Subtract out the outward component
            v_i -= scale_corr*comp_vec_out*(-vec_in_y)
            w_i -= scale_corr*comp_vec_out*(-vec_in_z)
        # Acceleration correction
        if dist_i>0: #Inside the wall: spring on penetration + damper on outward velocity
            a_wall_in = k_rest*dist_i + c_rest*comp_vec_out #Inward acceleration magnitude
            ax_i += a_wall_in*vec_in_x
            ay_i += a_wall_in*vec_in_y
            az_i += a_wall_in*vec_in_z
        return u_i,v_i,w_i,ax_i,ay_i,az_i

    return dom_limiter

#### Evaporation and heating
class EvapArgs(NamedTuple):
    """Fixed inputs of evap_heat_fun over one integration step; see evap_heat_fun for each field."""
    fuel_cpr: object
    gas_ct: object
    Y_air_pure_arr: np.ndarray
    Y_fuel_pure_arr: np.ndarray
    i_fuel_ct: int
    P_gas_tot: float
    T_gas_inf: float
    Y_vap_inf: float
    u_rel_mag: float
    R_drop_floor: float
    MW_air_pure: float
    MW_fuel_pure: float

def evap_heat_fun(t,y,evap_args):
    """
    This function compute the evaporation and heating process of a single droplet
    Tracking the temperature and size of a droplet by output the local rate of change of them
    Warp this solver in an integrator to get the full evolution
    y: (R_drop^2,T_drop) -> dy/dt: (d(R_drop^2)/dt,dT_drop/dt)
    y: solution state:
        R_drop2: R_drop^2: droplet radius square [m^2]
        T_drop: droplet temperature [K]
    evap_args: EvapArgs: Fixed parameters for this function call. Passed to solve_ivp as args=(evap_args,):
        fuel_cpr: abstract state of coolprop for fuel: cpr.AbstractState('HEOS','n-Dodecane')
        gas_ct: cantera solution to get vapor property, reduced to the film species for speed
                (transport cost ~ species^2; ~200x faster than the full mechanism, identical properties):
                    full = ct.Solution('X.yaml', transport_model='mixture-averaged')
                    idx = np.nonzero(Y_air_full + Y_fuel_full)[0] # mass fractions on the full mechanism
                    gas_ct = ct.Solution(thermo='ideal-gas', transport_model='mixture-averaged',
                                         species=[full.species(full.species_names[k]) for k in idx])
                    Y_air_pure_arr, Y_fuel_pure_arr = Y_air_full[idx], Y_fuel_full[idx]
        Y_air_pure_arr: array: pure air mass fraction (Cannot contain fuel), indexed to gas_ct
        Y_fuel_pure_arr: array: pure fuel mass fraction, indexed to gas_ct
        i_fuel_ct: fuel species_index from gas_ct to extract diffusivity
        P_gas_tot: float: ambient presure. assume uniform and constant. [Pa]
        T_gas_inf: float: far field temperature [K]
        Y_vap_inf: float: far field vapor mass fraction
        u_rel_mag: float: relative velocity magnitude [m/s]
        R_drop_floor: float: minimum droplet radius for termination [m]
        MW_air_pure: float: pure air molecular weight [kg/kmol]
        MW_fuel_pure: float: pure fuel molecular weight [kg/kmol]
    Behavior:
        All SI unit
        Fuel used in fuel_cpr is single component therefore vapor pressure is independent of vapor quantity
    """
    # Default Fixed Parameter
    rtol_Tcrit = 0.95
    atol_Y = 1e-6
    frac_ave_prop = 1/3
    # Expand State
    (R_drop2,T_drop)=y
    (fuel_cpr,gas_ct,Y_air_pure_arr,Y_fuel_pure_arr,i_fuel_ct,P_gas_tot,
     T_gas_inf,Y_vap_inf,u_rel_mag,R_drop_floor,MW_air_pure,MW_fuel_pure)=evap_args
    R_drop2 = max(R_drop2,R_drop_floor**2)
    if (T_drop <= 0): raise ValueError(f'Droplet Temperature: {T_drop} smaller or equal to 0')
    R_drop = mh.sqrt(R_drop2) #Droplet radius [m]
    if (T_drop > rtol_Tcrit*fuel_cpr.T_critical()): raise ValueError(f'Droplet Temperature: {T_drop}  close to Critical Temperature: {fuel_cpr.T_critical()}')
    # Get surface vaport mass fraction
    fuel_cpr.update(cpr.PT_INPUTS, P_gas_tot, T_drop) #Droplet liquid form
    rho_liq = fuel_cpr.rhomass() #[kg/m3]
    cp_liq = fuel_cpr.cpmass() #[J/kg/K]
    h_liq = fuel_cpr.hmass() #[J/kg]
    drho_liq_dT = fuel_cpr.first_partial_deriv(cpr.iDmass, cpr.iT, cpr.iP) #Partial derivative for density change
    fuel_cpr.update(cpr.QT_INPUTS, 1.0, T_drop) #Droplet vapor form
    P_vap_sur = fuel_cpr.p() #surface vapor pressure [Pa]
    h_vap = fuel_cpr.hmass() #[J/kg]
    lhv_vapori = h_vap - h_liq #[J/kg] droplet heat of vaporization
    X_vap_sur = P_vap_sur/P_gas_tot
    X_vap_sur = min(X_vap_sur,1-atol_Y) #float
    Y_vap_sur = X_vap_sur/(X_vap_sur + (1-X_vap_sur)*(MW_air_pure/MW_fuel_pure)) #float
    # Use surface vaport mass fraction to get a reference gas mixture state for average vapor and mixture transport and thermal properties
    T_gas_ref = T_drop + (T_gas_inf-T_drop)*(frac_ave_prop)
    Y_vap_ref = Y_vap_sur + (Y_vap_inf-Y_vap_sur)*(frac_ave_prop)
    Y_mix_ref_arr = Y_vap_ref*Y_fuel_pure_arr + (1-Y_vap_ref)*Y_air_pure_arr #[array] the average mixture mass fraction at reference location
    # Get reference mixture properties
    gas_ct.TPY = T_gas_ref,P_gas_tot,Y_mix_ref_arr
    mu_gas_ref = gas_ct.viscosity #[Pa*s]
    rho_gas_ref = gas_ct.density_mass #[kg/m3]
    k_gas_ref = gas_ct.thermal_conductivity #[W/m/K]
    cp_gas_ref = gas_ct.cp_mass #[J/kg/K]
    D_gas_ref = gas_ct.mix_diff_coeffs[i_fuel_ct] #[m2/s]
    # Get reference vapor properties
    gas_ct.TPY = T_gas_ref,P_gas_tot,Y_fuel_pure_arr
    cp_vap_ref = gas_ct.cp_mass #[J/kg/K]
    # Get reference non-dimensional transport properties
    Re_ref = u_rel_mag*2*R_drop*rho_gas_ref/mu_gas_ref
    Pr_ref = cp_gas_ref*mu_gas_ref/k_gas_ref
    Sc_ref = mu_gas_ref/(rho_gas_ref*D_gas_ref)
    # Evaporation
    Sh_0 = 2 + 0.6*Re_ref**(1/2)*Sc_ref**(1/3)
    Bm = (Y_vap_sur-Y_vap_inf)/(1-Y_vap_sur)
    mdot_vap_sur_out = 2*mh.pi*R_drop*Sh_0*rho_gas_ref*D_gas_ref*mh.log1p(Bm)
    # Heating
    Nu_0 = 2 + 0.6*Re_ref**(1/2)*Pr_ref**(1/3)
    Cheat = (mdot_vap_sur_out*cp_vap_ref)/(2*mh.pi*R_drop*k_gas_ref*Nu_0)
    Q_dot_sur_in = 2*mh.pi*R_drop*Nu_0*k_gas_ref*(T_gas_inf-T_drop)
    if Cheat != 0:
        Q_dot_sur_in *= Cheat/mh.expm1(Cheat)
    # Update state
    dR_drop2_dt = - Sh_0 * D_gas_ref * (rho_gas_ref/rho_liq) * mh.log1p(Bm)
    dT_drop_dt = (Q_dot_sur_in - mdot_vap_sur_out*lhv_vapori)/((4/3)*mh.pi*(R_drop**3)*rho_liq*cp_liq)
    # Additional term for density change
    dR_drop2_dt -= (2/3)*(R_drop2/rho_liq)*drho_liq_dT*dT_drop_dt
    return [dR_drop2_dt,dT_drop_dt]

def evap_heat_finish(t,y,evap_args):
    """
    This function is the solve_ivp event that ends the integration once the droplet is fully evaporated
    y[0]: R_drop^2
    evap_args: Contains attribute R_drop_floor
    return: float: R_drop^2 - R_drop_floor^2 [m^2]; crosses zero from + to - when R_drop reaches R_drop_floor
    """
    return y[0]-evap_args.R_drop_floor**2 #Goes from + to - terminate simulation
evap_heat_finish.terminal = True #Triggered lead to termination
evap_heat_finish.direction = -1 #Triggered when crossed from positive to negative

def evap_heat_solve(dt_tar,R_drop_ini,T_drop_ini,evap_args,rtol_sol=1e-6,rtol_fin=1e-6):
    """
    This function integrates evap_heat_fun over one step with LSODA
    Stops early when the droplet radius reaches evap_args.R_drop_floor (evap_heat_finish event)
    args:
        dt_tar: float: target time to advance [s]
        R_drop_ini: float: droplet radius at the start of the step [m]
        T_drop_ini: float: droplet temperature at the start of the step [K]
        evap_args: EvapArgs: fixed inputs over the step (see evap_heat_fun)
        rtol_sol: float: solver relative tolerance; absolute tolerance is 0.1*rtol_sol times each initial state
        rtol_fin: float: relative margin above R_drop_floor below which the droplet counts as evaporated
    return: dt_used, R_drop_new, T_drop_new
        dt_used: float: time actually advanced [s]; dt_used < dt_tar (0.0 included) means fully evaporated
        R_drop_new: float: droplet radius at the end of the step [m] (R_drop_floor if evaporated)
        T_drop_new: float: droplet temperature at the end of the step [K]
    Behavior:
        Keep R_drop_floor/R_drop_ini >= sqrt(0.1*rtol_sol) so the R^2 absolute tolerance stays below R_drop_floor^2
        Raises RuntimeError if the solver fails
    """
    # Check for too small droplet radius
    if R_drop_ini <= evap_args.R_drop_floor*(1 + rtol_fin):
        return 0.0,evap_args.R_drop_floor,T_drop_ini
    # Solve for one step
    sol = solve_ivp(
        evap_heat_fun,
        (0.0, dt_tar),
        [R_drop_ini**2, T_drop_ini],
        method = 'LSODA',
        args=(evap_args,),
        events=evap_heat_finish, #Terminate simulation when droplet too small
        rtol=rtol_sol,
        atol=[0.1*rtol_sol*R_drop_ini**2, 0.1*rtol_sol*T_drop_ini],
        dense_output=False)
    # Check for solution failiure
    if sol.status == -1: raise RuntimeError(f'evap_heat_solve failed: {sol.message}')
    # Extract results
    dt_used = sol.t[-1]
    R_drop_new = mh.sqrt(sol.y[0,-1])
    T_drop_new = sol.y[1,-1]
    return dt_used,R_drop_new,T_drop_new

class EvapDragArgs(NamedTuple):
    """Fixed inputs of evap_heat_drag_fun over one integration step; see evap_heat_drag_fun for each field."""
    fuel_cpr: object
    gas_ct: object
    Y_air_pure_arr: np.ndarray
    Y_fuel_pure_arr: np.ndarray
    i_fuel_ct: int
    P_gas_tot: float
    T_gas_inf: float
    Y_vap_inf: float
    R_drop_floor: float
    MW_air_pure: float
    MW_fuel_pure: float
    u_gas_fun: Callable[[np.ndarray],np.ndarray]
    dom_limiters: tuple = (dom_limiter_none,)

def evap_heat_drag_fun(t,y,evap_drag_args):
    """
    This function compute the evaporation, heating, droplet movement process of a single droplet
    Tracking the temperature, size, position, and velocity of a droplet by output the local rate of change of them
    Warp this solver in an integrator to get the full evolution
    y: (R_drop^2,T_drop,x_drop,y_drop,z_drop,u_drop,v_drop,w_drop) -> dy/dt: (d(R_drop^2)/dt,dT_drop/dt,dx_drop/dt,dy_drop/dt,dz_drop/dt,du_drop/dt,dv_drop/dt,dw_drop/dt)
    y: solution state:
        R_drop2: R_drop^2: droplet radius square [m^2]
        T_drop: droplet temperature [K]
        x_drop,y_drop,z_drop: x,y,z position of the droplet [m]
        u_drop,v_drop,w_drop: x,y,z velocity of the droplet [m/s]
    evap_drag_args: EvapDragArgs: Fixed parameters for this function call. Passed to solve_ivp as args=(evap_drag_args,):
        fuel_cpr: abstract state of coolprop for fuel: cpr.AbstractState('HEOS','n-Dodecane')
        gas_ct: cantera solution to get vapor property, reduced to the film species for speed:
                full = ct.Solution('X.yaml', transport_model='mixture-averaged')
                idx = np.nonzero(Y_air_full + Y_fuel_full)[0] # mass fractions on the full mechanism
                gas_ct = ct.Solution(thermo='ideal-gas', transport_model='mixture-averaged',
                                     species=[full.species(full.species_names[k]) for k in idx])
                Y_air_pure_arr, Y_fuel_pure_arr = Y_air_full[idx], Y_fuel_full[idx]
        Y_air_pure_arr: array: pure air mass fraction (Cannot contain fuel. Represent fuel concentration by Y_vap_inf.), size reduced to be consistent with gas_ct
        Y_fuel_pure_arr: array: pure fuel mass fraction, size reduced to be consistent with gas_ct
        i_fuel_ct: fuel species_index from gas_ct to extract diffusivity
        P_gas_tot: float: ambient presure. assume uniform and constant. [Pa]
        T_gas_inf: float: far field temperature [K]
        Y_vap_inf: float: far field vapor mass fraction
        R_drop_floor: float: minimum droplet radius for termination [m]
        MW_air_pure: float: pure air molecular weight [kg/kmol]
        MW_fuel_pure: float: pure fuel molecular weight [kg/kmol]
        u_gas_fun: function: u_gas_fun([x,y,z]) -> gas velocity vector [u,v,w] [m/s]
        dom_limiters: tuple-function: wall limiters for acceleration
    Behavior:
        All SI unit
        Fuel used in fuel_cpr is single component therefore vapor pressure is independent of vapor quantity
        Assume thermaldynamics state at infinite remain constant over the integration time step
        Assume gas velocity remain unaffected by the the droplet momentum change
        No blowing correct to film thickness effect
    """
    # Default Fixed Parameter
    rtol_Tcrit = 0.95
    atol_Y = 1e-6
    frac_ave_prop = 1/3
    # Expand State
    (R_drop2,T_drop,x_drop,y_drop,z_drop,u_drop,v_drop,w_drop)=y
    (fuel_cpr,gas_ct,Y_air_pure_arr,Y_fuel_pure_arr,i_fuel_ct,P_gas_tot,
     T_gas_inf,Y_vap_inf,R_drop_floor,MW_air_pure,MW_fuel_pure,u_gas_fun,dom_limiters)=evap_drag_args
    #### Droplet Movement First Part
    loc_drop = np.array([x_drop,y_drop,z_drop])
    vel_drop = np.array([u_drop,v_drop,w_drop])
    vel_gas = u_gas_fun(loc_drop) #[array] [m/s]
    vel_rel = vel_gas-vel_drop #[array] [m/s]
    vel_rel_mag = np.sqrt(np.sum(vel_rel**2)) #[float] [m/s] Magnitude of the relative velocity droplet is seeing
    if vel_rel_mag != 0.0:
        vel_rel_dir = vel_rel/vel_rel_mag #[array] [N/A] Unit direction of the acceleration the droplet is seeing
    else:
        vel_rel_dir = np.zeros(3)
    #### Evaporation and Heating Section Below
    R_drop2 = max(R_drop2,R_drop_floor**2)
    if (T_drop <= 0): raise ValueError(f'Droplet Temperature: {T_drop} smaller or equal to 0')
    R_drop = mh.sqrt(R_drop2) #Droplet radius [m]
    if (T_drop > rtol_Tcrit*fuel_cpr.T_critical()): raise ValueError(f'Droplet Temperature: {T_drop}  close to Critical Temperature: {fuel_cpr.T_critical()}')
    # Get surface vaport mass fraction
    fuel_cpr.update(cpr.PT_INPUTS, P_gas_tot, T_drop) #Droplet liquid form
    rho_liq = fuel_cpr.rhomass() #[kg/m3]
    cp_liq = fuel_cpr.cpmass() #[J/kg/K]
    h_liq = fuel_cpr.hmass() #[J/kg]
    drho_liq_dT = fuel_cpr.first_partial_deriv(cpr.iDmass, cpr.iT, cpr.iP) #Partial derivative for density change
    fuel_cpr.update(cpr.QT_INPUTS, 1.0, T_drop) #Droplet vapor form
    P_vap_sur = fuel_cpr.p() #surface vapor pressure [Pa]
    h_vap = fuel_cpr.hmass() #[J/kg]
    lhv_vapori = h_vap - h_liq #[J/kg] droplet heat of vaporization
    X_vap_sur = P_vap_sur/P_gas_tot
    X_vap_sur = min(X_vap_sur,1-atol_Y) #float
    Y_vap_sur = X_vap_sur/(X_vap_sur + (1-X_vap_sur)*(MW_air_pure/MW_fuel_pure)) #float
    # Use surface vaport mass fraction to get a reference gas mixture state for average vapor and mixture transport and thermal properties
    T_gas_ref = T_drop + (T_gas_inf-T_drop)*(frac_ave_prop)
    Y_vap_ref = Y_vap_sur + (Y_vap_inf-Y_vap_sur)*(frac_ave_prop)
    Y_mix_ref_arr = Y_vap_ref*Y_fuel_pure_arr + (1-Y_vap_ref)*Y_air_pure_arr #[array] the average mixture mass fraction at reference location
    # Get reference mixture properties
    gas_ct.TPY = T_gas_ref,P_gas_tot,Y_mix_ref_arr
    mu_gas_ref = gas_ct.viscosity #[Pa*s]
    rho_gas_ref = gas_ct.density_mass #[kg/m3]
    k_gas_ref = gas_ct.thermal_conductivity #[W/m/K]
    cp_gas_ref = gas_ct.cp_mass #[J/kg/K]
    D_gas_ref = gas_ct.mix_diff_coeffs[i_fuel_ct] #[m2/s]
    # Get reference vapor properties
    gas_ct.TPY = T_gas_ref,P_gas_tot,Y_fuel_pure_arr
    cp_vap_ref = gas_ct.cp_mass #[J/kg/K]
    # Get reference non-dimensional transport properties
    Re_ref = vel_rel_mag*2*R_drop*rho_gas_ref/mu_gas_ref
    Pr_ref = cp_gas_ref*mu_gas_ref/k_gas_ref
    Sc_ref = mu_gas_ref/(rho_gas_ref*D_gas_ref)
    # Evaporation
    Sh_0 = 2 + 0.6*Re_ref**(1/2)*Sc_ref**(1/3)
    Bm = (Y_vap_sur-Y_vap_inf)/(1-Y_vap_sur)
    mdot_vap_sur_out = 2*mh.pi*R_drop*Sh_0*rho_gas_ref*D_gas_ref*mh.log1p(Bm)
    # Heating
    Nu_0 = 2 + 0.6*Re_ref**(1/2)*Pr_ref**(1/3)
    Cheat = (mdot_vap_sur_out*cp_vap_ref)/(2*mh.pi*R_drop*k_gas_ref*Nu_0)
    Q_dot_sur_in = 2*mh.pi*R_drop*Nu_0*k_gas_ref*(T_gas_inf-T_drop)
    if Cheat != 0:
        Q_dot_sur_in *= Cheat/mh.expm1(Cheat)
    # Update state
    dR_drop2_dt = - Sh_0 * D_gas_ref * (rho_gas_ref/rho_liq) * mh.log1p(Bm)
    dT_drop_dt = (Q_dot_sur_in - mdot_vap_sur_out*lhv_vapori)/((4/3)*mh.pi*(R_drop**3)*rho_liq*cp_liq)
    # Additional term for density change
    dR_drop2_dt -= (2/3)*(R_drop2/rho_liq)*drho_liq_dT*dT_drop_dt
    #### Droplet Movement Second Part
    gas_ct.TPY = T_gas_inf,P_gas_tot,(Y_air_pure_arr*(1-Y_vap_inf) + Y_fuel_pure_arr*(Y_vap_inf))
    rho_gas_inf = gas_ct.density_mass
    Re_drag = rho_gas_inf*vel_rel_mag*2*R_drop/mu_gas_ref # Special combine Re for drag calculation
    if Re_drag <= 1000: #Putnam (KIVA) drag law
        if Re_drag != 0:
            Cd_drop = (24/Re_drag)*(1+Re_drag**(2/3)/6)
        else:
            Cd_drop = 0.0
    else: #High Reynolds number limit
        Cd_drop = 0.424
    dvel_drop_dt_mag = (3/8)*(rho_gas_inf/rho_liq)*vel_rel_mag**2*Cd_drop/R_drop
    (du_drop_dt,dv_drop_dt,dw_drop_dt) = vel_rel_dir*dvel_drop_dt_mag #[array] [m/s2] [u_com,v_com,w_com]
    #### Apply wall acceleration clipping
    for lim_cur in dom_limiters:
        (_,_,_,du_drop_dt,dv_drop_dt,dw_drop_dt) = \
        lim_cur(x_drop,y_drop,z_drop,u_drop,v_drop,w_drop,\
                du_drop_dt,dv_drop_dt,dw_drop_dt)
    return [dR_drop2_dt,dT_drop_dt,u_drop,v_drop,w_drop,du_drop_dt,dv_drop_dt,dw_drop_dt]

def only_u_gas_transport_fun(t,y,u_gas_fun,dom_limiters):
    """
    When evaporation ends with no more droplet radius and temperature update.
    Assume the drop/mixture trajectory is passively transport by the steady external gas velocity field
    inputs:
        t: float: time [s] not used (no time varying flow field)
        y: array: droplet location [xi,yi,zi] [m]
        u_gas_fun: function: u_gas_fun(x) -> gas velocity vector [m/s] at position x, same dimension as x. Inputted as args=(u_gas_fun,dom_limiters).
        dom_limiters: tuple-function: wall velocity limiters. Input as args=(u_gas_fun,dom_limiters).
    return:
        dy/dt: array: droplet velocity at [xi,yi,zi] [u,v,w] [m/s]
    """
    (u,v,w) = u_gas_fun(y)
    (xi,yi,zi) = y
    for lim_cur in dom_limiters:
        (u,v,w,_,_,_) = lim_cur(xi,yi,zi,u,v,w,0.,0.,0.)
    return [u,v,w]

def evap_heat_trans_solve(dt_tar,R_drop_ini,T_drop_ini,loc_drop_ini,vel_drop_ini,evap_drag_args,len_scale=1,vel_scale=1,rtol_sol=1e-6,atol_sol_ratio=0.1,eps_Rf=1e-6,flag_dense_outputs=False):
    """
    This function integrates evap_heat_drag_fun over one step with LSODA
    Split simulation when the droplet radius reaches evap_drag_args.R_drop_floor. Then, passive gas convection.
    args:
        dt_tar: float: target time to advance [s]
        R_drop_ini: float: droplet radius at the start of the step [m]
        T_drop_ini: float: droplet temperature at the start of the step [K]
        loc_drop_ini: array: droplet initial location [x,y,z] [m]
        vel_drop_ini: array: droplet initial velocity [u,v,w] [m/s]
        evap_drag_args: EvapDragArgs: fixed inputs over the step (see evap_heat_drag_fun)
        rtol_sol: float: solver relative tolerance; absolute tolerance is atol_sol_ratio*rtol_sol times each reference/initial scales
        atol_sol_ratio: absolute tolerance ratio derived from relative
        eps_Rf: float: relative margin above R_drop_floor below which the droplet counts as evaporation finish
        len_scale: float: length scale to set tolerance [m]
        vel_scale: float: velocity scale to set tolerance [m/s]
        flag_dense_outputs: bool: true to get the interpolation function for state retracting (sol_interp)
    return:
        R_drop_fin: float: droplet radius at the end of the step [m]
        T_drop_fin: float: droplet temperature at the end of the step [K]
        loc_drop_fin: array: [x,y,z] [m]
        vel_drop_fin: array: [u,v,w] [m/s]
        dt_split: float: time when evaporation stops [s] [exactly same as dt_tar inputted if did not happen] (Same radius and temperature state as final)
        loc_drop_split: array: [x,y,z] [m] [exactly same as loc_drop_fin if did not happen]
        vel_drop_split: array: [u,v,w] [m/s] [exactly same as vel_drop_fin if did not happen]
        evap_ended: bool: True if evaporation stops in this integration or before.
        sol_interp: solution interpolation function or None: Only return from the evaporation on branch (valid in [0, dt_split]) (for low cost way of retracting step)
                    sol_interp(t) -> [R_drop^2,T_drop,x,y,z,u,v,w]
    Behavior:
        atol_sol_act = (atol_sol_ratio*rtol_sol)*R_drop_ini^2 for droplet radius
        Thus R_drop_floor^2 need to be > atol_sol_act
        Raises RuntimeError if the solver fails
    """
    # Too small starting droplet radius
    atol_sol = rtol_sol*atol_sol_ratio
    sol_interp = None
    if R_drop_ini <= evap_drag_args.R_drop_floor*(1+eps_Rf):
        sol = solve_ivp(only_u_gas_transport_fun,
                        (0.0, dt_tar),
                        loc_drop_ini,
                        method = 'LSODA',
                        args=(evap_drag_args.u_gas_fun,evap_drag_args.dom_limiters),
                        rtol=rtol_sol,
                        atol=[atol_sol*len_scale,atol_sol*len_scale,atol_sol*len_scale],
                        dense_output=False)
        if sol.status == -1: raise RuntimeError(f'evap_heat_trans_solve failed (stage 1): {sol.message}')
        R_drop_fin = R_drop_ini
        T_drop_fin = T_drop_ini
        loc_drop_fin = sol.y[:,-1].copy()
        vel_drop_fin = evap_drag_args.u_gas_fun(loc_drop_fin).copy()
        dt_split = 0.0
        loc_drop_split = loc_drop_ini.copy()
        vel_drop_split = vel_drop_ini.copy()
        evap_ended = True
    else:
        # Evaporation on
        sol = solve_ivp(evap_heat_drag_fun,
                        (0.0, dt_tar),
                        np.concatenate(([R_drop_ini**2, T_drop_ini],loc_drop_ini,vel_drop_ini)),
                        method = 'LSODA',
                        args=(evap_drag_args,),
                        events=evap_heat_finish, #Terminate simulation when droplet too small
                        rtol=rtol_sol,
                        atol=[atol_sol*R_drop_ini**2,atol_sol*T_drop_ini,
                              atol_sol*len_scale,atol_sol*len_scale,atol_sol*len_scale,
                              atol_sol*vel_scale,atol_sol*vel_scale,atol_sol*vel_scale],
                        dense_output=flag_dense_outputs)
        if sol.status == -1: raise RuntimeError(f'evap_heat_trans_solve failed (stage 2): {sol.message}')
        if flag_dense_outputs:
            sol_interp = sol.sol
        # Output or split
        if (sol.status == 0):
            R_drop_fin = mh.sqrt(sol.y[0,-1])
            T_drop_fin = sol.y[1,-1]
            loc_drop_fin = sol.y[2:5,-1].copy()
            vel_drop_fin = sol.y[5:8,-1].copy()
            dt_split = dt_tar
            loc_drop_split = loc_drop_fin.copy()
            vel_drop_split = vel_drop_fin.copy()
            evap_ended = False
        else: # Split and run passive transport
            R_drop_fin = mh.sqrt(sol.y[0,-1])
            T_drop_fin = sol.y[1,-1]
            loc_drop_split = sol.y[2:5,-1].copy()
            vel_drop_split = sol.y[5:8,-1].copy()
            evap_ended = True
            if (sol.t[-1]>=dt_tar):
                dt_split = dt_tar
                loc_drop_fin = sol.y[2:5,-1].copy()
                vel_drop_fin = sol.y[5:8,-1].copy()
            else: # Passive transport
                dt_split = sol.t[-1]
                sol = solve_ivp(only_u_gas_transport_fun,
                                (0.0, dt_tar-sol.t[-1]),
                                loc_drop_split,
                                method = 'LSODA',
                                args=(evap_drag_args.u_gas_fun,evap_drag_args.dom_limiters),
                                rtol=rtol_sol,
                                atol=[atol_sol*len_scale,atol_sol*len_scale,atol_sol*len_scale],
                                dense_output=False)
                if sol.status == -1: raise RuntimeError(f'evap_heat_trans_solve failed (stage 3): {sol.message}')
                loc_drop_fin = sol.y[:,-1].copy()
                vel_drop_fin = evap_drag_args.u_gas_fun(loc_drop_fin).copy()
    return R_drop_fin,T_drop_fin,loc_drop_fin,vel_drop_fin,dt_split,loc_drop_split,vel_drop_split,evap_ended,sol_interp

def HED_to_Atom_step(dt,sol_interp,N_drop_ini,y_distort_ini,ydot_distort_ini,evap_drag_args):
    """
    This function interpolates a state from HED with (dt), and perform atomization calculation with (dt*2)
    TAB coefficients are frozen at the interpolated state (midpoint of the atomization interval [0, dt*2])
    args:
        dt: float: interpolation time (half of the atomization run time) [s]; must lie within the range of sol_interp
        sol_interp: OdeSolution: HED interpolation function from evap_heat_trans_solve (flag_dense_outputs=True)
        N_drop_ini: float: drops represented by the parcel
        y_distort_ini, ydot_distort_ini: float: TAB distortion and its rate at the start of the step (t = 0)
        evap_drag_args: EvapDragArgs: same object used for the HED solve
    return: res_atom, R_drop_ini, T_drop_ini, loc_drop_ini, vel_drop_ini
        res_atom: StepResult: atomize_one_step result over dt*2 (r_new is the interpolated radius if no breakup)
        R_drop_ini, T_drop_ini, loc_drop_ini, vel_drop_ini: interpolated HED state at dt used for the TAB coefficients
    Behavior:
        rhog of TAB is the far-field air-vapor mixture density (same as drag)
    """
    #### Expand fixed parameters
    (fuel_cpr,gas_ct,Y_air_pure_arr,Y_fuel_pure_arr,i_fuel_ct,P_gas_tot,
     T_gas_inf,Y_vap_inf,R_drop_floor,MW_air_pure,MW_fuel_pure,u_gas_fun,dom_limiters)=evap_drag_args
    #### Obtain an interpolated HED state
    sol_ini = sol_interp(dt)
    R_drop_ini = mh.sqrt(sol_ini[0])
    T_drop_ini = sol_ini[1]
    loc_drop_ini = sol_ini[2:5] #[x,y,z]
    vel_drop_ini = sol_ini[5:8] #[u,v,w]
    #### Atomization step
    # Get middle state relative velocity for atomization forcing
    vel_gas_ini = u_gas_fun(loc_drop_ini)
    vel_gas_ini_mag = np.sqrt(np.sum((vel_drop_ini-vel_gas_ini)**2)) #[m/s]
    # Get initial droplet thermal property
    fuel_cpr.update(cpr.PT_INPUTS, P_gas_tot, T_drop_ini) #Droplet liquid form
    rho_liq_ini = fuel_cpr.rhomass() #[kg/m3]
    mu_liq_ini  = fuel_cpr.viscosity() #[Pa*s]
    fuel_cpr.update(cpr.QT_INPUTS, 0.0, T_drop_ini)
    sig_liq_ini = fuel_cpr.surface_tension() #[N/m]
    gas_ct.TPY = T_gas_inf,P_gas_tot,Y_vap_inf*Y_fuel_pure_arr+(1-Y_vap_inf)*Y_air_pure_arr
    rho_gas_ini = gas_ct.density_mass #[kg/m3] Far-field density for TAB aerodynamic forcing (same as drag)
    # Atomization run
    props = Props(rhol=rho_liq_ini,sig=sig_liq_ini,mul=mu_liq_ini,rhog=rho_gas_ini)
    drop = DropState(N=N_drop_ini,r=R_drop_ini,y=y_distort_ini,ydot=ydot_distort_ini)
    res_atom = atomize_one_step(dt*2,drop,vel_gas_ini_mag,props)
    return res_atom,R_drop_ini,T_drop_ini,loc_drop_ini,vel_drop_ini

def HED_to_Atom_step_wrap(dt,sol_interp,N_drop_ini,y_distort_ini,ydot_distort_ini,evap_drag_args):
    """
    wrapper around HED_to_Atom_step to return a residual 0.5 for atomization occurred and -0.5 for atomization did not occur
    """
    res_atom,_,_,_,_ = HED_to_Atom_step(dt,sol_interp,N_drop_ini,y_distort_ini,ydot_distort_ini,evap_drag_args)
    return res_atom.flag_breakup-0.5

#### Velocity field construction
def make_vel_field_Lamb_Oseen(r_peak,vel_theta_peak,vel_axis,axis='x'):
    """
    Build a steady gas velocity field: a Lamb-Oseen swirl about a coordinate axis plus a uniform axial velocity
        vel_theta(r) = gamma/(2*pi*r)*(1 - exp(-r^2/(4*nu_t))), r = distance from the axis, nu_t = (kinematic viscosity)*t
    The swirl peaks at r = 1.120906*sqrt(4*nu_t) = 2.241813*sqrt(nu_t), with vel_theta_peak = 0.113849*gamma/r_peak, so
    nu_t and gamma follow from the requested peak. Solid-body rotation near the axis, free vortex (gamma/(2*pi*r)) far out.
    The core size is fixed (no growth along the axis).
    args:
        r_peak: float: radius of the peak swirl velocity [m]
        vel_theta_peak: float: peak swirl velocity [m/s]; positive = counterclockwise about the +axis direction
                        (right-handed: about +x, swirl from +y toward +z)
        vel_axis: float: uniform axial velocity along the chosen axis [m/s]
        axis: str: 'x', 'y' or 'z', the vortex axis (through the origin)
    return: function vel_gas_fun(loc) -> np.array([u,v,w]) [m/s] for a location loc = [x,y,z] [m]
    Behavior:
        On the axis the swirl is zero and the velocity is purely axial
        'y' and 'z' are the 'x' field with coordinates rotated cyclically (x,y,z) -> (y,z,x) / (z,x,y), same swirl sense
    """
    # Process parameters
    nu_t = (r_peak/2.241813)**2 # (kinematic viscosity)*t [m^2] from r_peak = 2.241813*sqrt(nu_t)
    gamma = vel_theta_peak*r_peak/0.113849 # circulation [m^2/s] from vel_theta_peak = 0.113849*gamma/r_peak
    # Get function
    def vel_gas_fun_x_Lamb_Oseen(loc):
        (x,y,z) = loc
        r = mh.sqrt(y**2 + z**2)
        if r==0:
            return np.array([vel_axis,0,0])
        else:
            vel_x = vel_axis
            vel_the = -gamma/(2*mh.pi*r)*(1-mh.exp(-r**2/4/nu_t))
            vel_y = vel_the*z/r
            vel_z = -vel_the*y/r
            return np.array([vel_x,vel_y,vel_z])
    # Get funcion for other direction
    if axis.lower() == 'x':
        return vel_gas_fun_x_Lamb_Oseen
    elif axis.lower() == 'y':
        def vel_gas_fun_y_Lamb_Oseen(loc):
            (z,x,y) = loc
            (vel_y,vel_z,vel_x) = vel_gas_fun_x_Lamb_Oseen((x,y,z))
            return np.array([vel_x,vel_y,vel_z])
        return vel_gas_fun_y_Lamb_Oseen
    elif axis.lower() == 'z':
        def vel_gas_fun_z_Lamb_Oseen(loc):
            (y,z,x) = loc
            (vel_z,vel_x,vel_y) = vel_gas_fun_x_Lamb_Oseen((x,y,z))
            return np.array([vel_x,vel_y,vel_z])
        return vel_gas_fun_z_Lamb_Oseen
    else:
        raise ValueError(f'Axis {axis} is not recognized')

def make_vel_field_Lamb_Oseen_evo(xlst,Rout_lst,Rin_lst,mdot,rho,swirl_num,frac_peak,axis='x'):
    """
    Build a steady swirling gas velocity field in an axisymmetric passage Rin(x) < r < Rout(x) (piecewise linear in x),
    conserving the mass flow and the circulation (angular momentum) of every streamtube (inviscid, no torque)
        Inlet station xlst[0]: uniform axial velocity and a Lamb-Oseen swirl
            vel_axis_0 = mdot/(rho*pi*(Rout0^2 - Rin0^2))
            r_peak_0 = Rin0 + frac_peak*(Rout0 - Rin0), nu_0 = (r_peak_0/2.241813)^2
            gamma from the swirl number S = G_theta/(Rout0*G_x) (no pressure term), with uniform axial velocity:
            G_theta = rho*vel_axis_0*gamma*[r^2/2 + 2*nu_0*exp(-r^2/(4*nu_0))]_Rin0^Rout0, G_x = rho*vel_axis_0^2*pi*(Rout0^2 - Rin0^2)
        Station x: uniform axial velocity vel_axis = mdot/(rho*pi*(Rout^2 - Rin^2))
            a streamline at r started at r0 with the same flow area fraction: (r0^2 - Rin0^2)/(Rout0^2 - Rin0^2) = (r^2 - Rin^2)/(Rout^2 - Rin^2)
            its circulation is kept: vel_theta(r,x) = gamma/(2*pi*r)*(1 - exp(-r0^2/(4*nu_0)))
            radial velocity from the streamline slope (continuity holds exactly): vel_r = vel_axis*dr/dx along the streamline
        No hub (Rin = 0) and Rout scaled by s(x): Lamb-Oseen with the same gamma and r_peak scaled by s(x)
    args:
        xlst: array: [m] axial stations (strictly increasing); geometry held constant outside [xlst[0], xlst[-1]]
        Rout_lst: array: [m] outer (tip) radius at xlst
        Rin_lst: array: [m] inner (hub) radius at xlst (>=0, 0: no hub), Rin < Rout
        mdot: float: [kg/s] mass flow through the passage
        rho: float: [kg/m3] gas density (constant)
        swirl_num: float: swirl number at the inlet station; positive = counterclockwise about the +axis direction
        frac_peak: float: (r_peak - Rin)/(Rout - Rin) of the Lamb-Oseen swirl peak at the inlet station (r_peak > 0)
        axis: str: 'x', 'y' or 'z', the passage axis (through the origin)
    return: function vel_gas_fun(loc) -> np.array([u,v,w]) [m/s] for a location loc = [x,y,z] [m]
    Behavior:
        Inside the hub (r < Rin) the field is extrapolated with r0 clipped at 0 (no swirl on the inlet axis)
        Radial velocity jumps where the wall slope changes (piecewise linear walls)
        No vertical walls or sudden steps: the streamtube mapping assumes attached flow following the walls;
        represent a step as a ramp over its reattachment length, or as a sudden expansion between two passages
        'y' and 'z' are the 'x' field with coordinates rotated cyclically (x,y,z) -> (y,z,x) / (z,x,y), same swirl sense
    """
    #### Check and process the geometry
    xlst = [float(x_cur) for x_cur in xlst]
    Rout_lst = [float(R_cur) for R_cur in Rout_lst]
    Rin_lst = [float(R_cur) for R_cur in Rin_lst]
    num_seg = len(xlst)-1
    if num_seg < 1 or any(xlst[idx+1] <= xlst[idx] for idx in range(num_seg)):
        raise ValueError(f'make_vel_field_Lamb_Oseen_evo: xlst must have >= 2 strictly increasing points')
    if any((R_in < 0) or (R_in >= R_out) for R_in,R_out in zip(Rin_lst,Rout_lst)):
        raise ValueError(f'make_vel_field_Lamb_Oseen_evo: need 0 <= Rin < Rout at every station')
    slope_out = [(Rout_lst[idx+1]-Rout_lst[idx])/(xlst[idx+1]-xlst[idx]) for idx in range(num_seg)] # dRout/dx per segment
    slope_in = [(Rin_lst[idx+1]-Rin_lst[idx])/(xlst[idx+1]-xlst[idx]) for idx in range(num_seg)] # dRin/dx per segment

    #### Inlet station Lamb-Oseen swirl from mass flow and swirl number
    R_in0, R_out0 = Rin_lst[0], Rout_lst[0]
    area_0 = R_out0**2 - R_in0**2 # flow area / pi
    flow_pi = mdot/(rho*mh.pi) # vel_axis*area/pi, constant along x
    vel_axis_0 = flow_pi/area_0 # uniform axial velocity at the inlet
    r_peak_0 = R_in0 + frac_peak*(R_out0 - R_in0)
    if not (r_peak_0 > 0):
        raise ValueError(f'make_vel_field_Lamb_Oseen_evo: r_peak at the inlet must be > 0; got {r_peak_0}')
    nu_0 = (r_peak_0/2.241813)**2 # (kinematic viscosity)*t [m^2] from r_peak = 2.241813*sqrt(nu_t)
    bracket_0 = (R_out0**2/2 + 2*nu_0*mh.exp(-R_out0**2/(4*nu_0))) - (R_in0**2/2 + 2*nu_0*mh.exp(-R_in0**2/(4*nu_0))) # integral of (1-exp(-r^2/4nu))*r dr
    gamma = swirl_num*R_out0*vel_axis_0*mh.pi*area_0/bracket_0 # circulation [m^2/s] giving the inlet swirl number

    #### Get function
    xlst, Rin_lst, Rout_lst = np.array(xlst), np.array(Rin_lst), np.array(Rout_lst)
    slope_in, slope_out = np.array(slope_in), np.array(slope_out)
    @njit
    def vel_gas_fun_x_Lamb_Oseen_evo(loc):
        (x,y,z) = loc
        #### Passage geometry at x (held constant outside the x range)
        if x <= xlst[0]:
            R_in, R_out, dR_in, dR_out = Rin_lst[0], Rout_lst[0], 0.0, 0.0
        elif x >= xlst[-1]:
            R_in, R_out, dR_in, dR_out = Rin_lst[-1], Rout_lst[-1], 0.0, 0.0
        else:
            idx = np.searchsorted(xlst,x,side='right')-1 #Segment index
            R_in = Rin_lst[idx] + (x-xlst[idx])*slope_in[idx]
            R_out = Rout_lst[idx] + (x-xlst[idx])*slope_out[idx]
            dR_in, dR_out = slope_in[idx], slope_out[idx]
        
        #### Local area
        area = R_out**2 - R_in**2 # flow area / pi
        
        #### Calculate axial velocity based on mass conservation
        vel_x = flow_pi/area # axial velocity assume constant mass flow rate and density
        r = mh.sqrt(y**2 + z**2)
        
        #### Velocity along the center axis (outside of the domain) (Not use but in case droplet penetrate wall)
        if r==0:
            return np.array([vel_x, 0.0, 0.0])
        
        #### Streamline origin at the inlet station
        r0_sq = max(R_in0**2 + (r**2 - R_in**2)*area_0/area, 0.0) #r0^2 the same stream position back to the starting point
        vel_the = -gamma/(2*mh.pi*r)*(1-mh.exp(-r0_sq/(4*nu_0)))
        
        #### Radial velocity from the streamline slope: r*dr/dx = R_in*dR_in + (r^2 - R_in^2)*darea/area, darea = d(area)/dx/2
        darea = R_out*dR_out - R_in*dR_in
        vel_r = vel_x*(R_in*dR_in + (r**2 - R_in**2)*darea/area)/r
        vel_y = vel_the*z/r + vel_r*y/r
        vel_z = -vel_the*y/r + vel_r*z/r
        return np.array([vel_x,vel_y,vel_z])
    
    #### Get funcion for other direction
    if axis.lower() == 'x':
        return vel_gas_fun_x_Lamb_Oseen_evo
    elif axis.lower() == 'y':
        def vel_gas_fun_y_Lamb_Oseen_evo(loc):
            (z,x,y) = loc
            (vel_y,vel_z,vel_x) = vel_gas_fun_x_Lamb_Oseen_evo((x,y,z))
            return np.array([vel_x,vel_y,vel_z])
        return vel_gas_fun_y_Lamb_Oseen_evo
    elif axis.lower() == 'z':
        def vel_gas_fun_z_Lamb_Oseen_evo(loc):
            (y,z,x) = loc
            (vel_z,vel_x,vel_y) = vel_gas_fun_x_Lamb_Oseen_evo((x,y,z))
            return np.array([vel_x,vel_y,vel_z])
        return vel_gas_fun_z_Lamb_Oseen_evo
    else:
        raise ValueError(f'Axis {axis} is not recognized')

#### Parcel generation
def LJICF_parcel_init(num_pars,num_injs,mdot_fuel,loc_drop_rel,y_break,vel_air_fun,gas_air,cpr_fuel,dd32lb=2.771265e-1,dd32ub=4.240989):
    """
    This function initializes droplet parcels for LJICF in swirler
        Droplet size distribution: Nukiyama–Tanasawa-type droplet size distribution
            - CDF: Vi_Vt(di_d32) = gammainc(18.75, 16.98 * di_d32**0.4)
            - Limits: 98% region by mass/by volume: di_d32: 2.771265e-1 to 4.240989
        Droplet mean size: "Atomization and Sprays" (Lefebvre; McDonell)
            - Song et al.: d0/SMD = 0.267 * We**0.44 * q**0.08 * (rhol/rhoa)**0.30 * (mul/mua)**(-0.16)
            - Momentum Flux Ratio: q = (rhol * ul**2) / (rhoa * ua**2)
            - Wu et al.: Breakpoint: yb/d0 = 3.44 * q**0.50, xb/d0 = 8.06
        Droplets of different sizes all release at the same half channel height.
        Use breakpoint to calculate average droplet velocity
        Use air velocity quested at droplet release location to approximate air velocity for LJICF calculation
        cpr_fuel will be modified in place
        args:
            num_pars: int: number of parcels (different sizes) to create
            num_injs: int: number of injectors (will determine the injector diameter)
            mdot_fuel: float: total fuel mass flow rate for the whole swirler [kg/s]
            loc_drop_rel: array: [x,y,z] Location of droplet release. Corresponds to the half channel height position after primary breakup.
            y_break: float: Primary breakup height in LJICF. Vertical distance from injector.
            vel_air_fun: function: [u,v,w] air velocity as f([x,y,z])
            gas_air: ct solution: contain air mass fraction, air temperature, environment pressure
            cpr_fuel: coolprop object: fuel
            dd32lb, dd32ub: float: lower and upper bound of Di/D32 based on cdf will give 98% of droplets by mass
        return:
            pars_lst: list-Parcel: the list of initial parcels generated    
    """
    #### Obtain air and fuel properties
    # Get properties for parcel calculation
    rho_air = gas_air.density_mass #[kg/m3]
    mu_air = gas_air.viscosity #[Pa*s]
    if not mh.isclose(cpr_fuel.p(), gas_air.P, rel_tol=1e-6):
        raise ValueError(f'LJICF_parcel_init: cpr_fuel pressure {cpr_fuel.p():.6e} Pa differs from gas_air pressure {gas_air.P:.6e} Pa')
    cpr_fuel.update(cpr.QT_INPUTS, 0.0, cpr_fuel.T())
    sig_fuel = cpr_fuel.surface_tension() #[N/m]
    cpr_fuel.update(cpr.PT_INPUTS, gas_air.P, cpr_fuel.T())
    rho_fuel = cpr_fuel.rhomass() #[kg/m3]
    mu_fuel  = cpr_fuel.viscosity() #[Pa*s]
    
    #### Compute air velocity
    vel_air = vel_air_fun(loc_drop_rel)
    vel_air_mag = np.linalg.norm(vel_air)
    vel_air_dir = vel_air/vel_air_mag

    #### Compute droplet starting velocity
    mdot_fuel_i = mdot_fuel/num_injs
    d_inj = (3.44*4.0/mh.pi) * (mdot_fuel_i/(vel_air_mag * y_break * mh.sqrt(rho_air*rho_fuel)))
    rat_mom_flux = (y_break/(3.44*d_inj))**2 #Jet to crossflow momentum ratio q
    vel_jet = mdot_fuel_i/(0.25*mh.pi*rho_fuel*d_inj**2)
    tau_break = y_break/vel_jet
    vel_drop_sta = 8.06*d_inj/tau_break #Starting droplet velocity estimated from breakpoint height and distance

    #### Compute SMD
    We = rho_air*d_inj*vel_air_mag**2/sig_fuel
    d32_drop = d_inj / (0.267 * We**0.44 * rat_mom_flux**0.08 * (rho_fuel/rho_air)**0.3 * (mu_fuel/mu_air)**(-0.16))

    #### Get droplet size distribution
    dd32bound_pow = np.linspace(dd32lb**2 , dd32ub**2 , num_pars+1)
    dd32bound = np.sqrt(dd32bound_pow)
    dd32ave = np.sqrt((dd32bound_pow[:-1]+dd32bound_pow[1:])*0.5)
    d_drop_ave = d32_drop * dd32ave # averaged droplet diameter [m]
    Voli_Volt = lambda Di_D32: sc.gammainc(18.75, 16.98 * Di_D32**0.4)
    mdot_drop_dist_raw  = Voli_Volt(dd32bound[1:]) - Voli_Volt(dd32bound[:-1])
    num_drop_lst = (mdot_fuel_i * mdot_drop_dist_raw / np.sum(mdot_drop_dist_raw)) / \
                   (mh.pi * rho_fuel * d_drop_ave**3 / 6.0)

    #### Construct parcels
    pars_lst = []
    for idx_par in range(num_pars):
        pars_lst.append(Parcel(t=0.0, R=d_drop_ave[idx_par]/2., \
                               N=num_drop_lst[idx_par], y=0.0, ydot=0.0, \
                               T=cpr_fuel.T(), \
                               loc=np.array(loc_drop_rel), \
                               vel=np.array(vel_drop_sta)*vel_air_dir, \
                               gen=0, pid=str(idx_par), dt=None))

    return pars_lst

#### Combine all steps
def HEAD_one_step(dt_tar,R_drop_ini,N_drop_ini,y_distort_ini,ydot_distort_ini,T_drop_ini,loc_drop_ini,vel_drop_ini,
                  evap_drag_args,len_scale=1,vel_scale=1,rtol_bis=1e-6,n_parcels=2,flg_atom=True):
    """
    For one parcel, perform single step of heating(H), evaporation(E), atomization(A), and drag(D) calculation
    Current model does not capture the variation of gas thermodynamics state variation within one step
    Scheme:
        HED is solved once over dt_tar with dense output; TAB (A) then follows the HED trajectory with coefficients
        frozen at the HED state at dt_tar/2 (valid because HED uses a spherical drop and does not depend on y, ydot)
        If A breaks up within dt_tar, the midpoint time is bisected so that breakup lands at the end of a shortened
        step [0, dt_used]; the parent HED state is interpolated at dt_used and the breakup is applied there
    args:
        dt_tar: float: target time step [s]
        R_drop_ini: float: droplet radius at the start of the step [m]
        N_drop_ini: float: drops represented by the parcel
        y_distort_ini, ydot_distort_ini: float: TAB distortion and its rate [1/s] at the start of the step (y < 1)
        T_drop_ini: float: droplet temperature at the start of the step [K]
        loc_drop_ini: array: droplet location [x,y,z] [m]
        vel_drop_ini: array: droplet velocity [u,v,w] [m/s]
        evap_drag_args: EvapDragArgs: fixed inputs over the step (see evap_heat_drag_fun)
        len_scale, vel_scale: float: length [m] and velocity [m/s] scales for the HED absolute tolerances
        rtol_bis: float: bisection tolerance on the critical midpoint time, as a fraction of dt_tar/2
        n_parcels: int: child parcels created at breakup, spread evenly on a ring normal to the slip velocity
                   1 gives no splitting velocity kick
        flg_atom: bool: turns this to false if want to turn off atomization process
    return: has_atomized, dt_used, R_drop_fin_lst, T_drop_fin_lst, loc_drop_fin_lst, vel_drop_fin_lst,
            N_drop_fin_lst, y_distort_fin_lst, ydot_distort_fin_lst, m_vap_rel, dt_vap_rel, loc_vap_rel
        has_atomized: bool: True if breakup happened in this step
        dt_used: float: time actually advanced [s]; dt_tar without breakup, breakup time (<= dt_tar) with breakup
        *_lst: list: end-of-step state of each output parcel (one entry without breakup, n_parcels with breakup)
               radius [m], temperature [K], location [m], velocity [m/s], drops represented, y, ydot [1/s]
        m_vap_rel: float: vapor mass released by the all parcels collectively (N drops) during this step [kg]
                   = N*(4/3)*pi*(rho_l*R^3 at the step start - rho_l*R^3 at dt_evap_end_before_atomization), rho_l at (T, P_gas_tot)
        dt_vap_rel: float: release time relative to the step start [s] (midpoint of the evaporating part of the step)
        loc_vap_rel: array: release location [x,y,z] [m] (droplet position at dt_vap_rel)
    Behavior:
        Without breakup: R, T, loc, vel from HED at dt_tar; y, ydot from A; N unchanged
        With breakup: children share the parent HED state at dt_used; child radius = (r32/R_mid)*R(dt_used) and
            N from A, so mass is conserved at dt_used; children start with y = ydot = 0
        Splitting velocity (vel_break) is added along n_parcels directions normal to the slip at dt_used
            (fixed ring phase, so deterministic); skipped when n_parcels <= 1, the slip is exactly 0, or vel_break is 0
        If evaporation finishes within the step (evap_ended), atomization is not computed for that step
        Raises ValueError if the bisection does not land on a breakup state
    """
    #### Evaporation Step
    R_drop_HED_fin,T_drop_HED_fin,loc_drop_HED_fin,vel_drop_HED_fin,\
    dt_split,loc_drop_split,vel_drop_split,evap_ended,sol_interp =\
    evap_heat_trans_solve(dt_tar,R_drop_ini,T_drop_ini,loc_drop_ini,vel_drop_ini,evap_drag_args,
                          len_scale=len_scale,vel_scale=vel_scale,flag_dense_outputs=True)
    #### Atomization Step
    if evap_ended or not flg_atom: #Evaporation finishes within current full time step. Ignore atomization.
        # Trivial: Directly take R_drop_HED_fin,T_drop_HED_fin,loc_drop_HED_fin,vel_drop_HED_fin (From the full dt HED simulation) (Freeze N,y,ydot)
        has_atomized = False
        dt_used = dt_tar #TODO: when evaporation ends within the step (evap_ended), this still completes the full dt_tar with passive gas transport after dt_split, so the leftover floor liquid is logged by parcels_simulate at the end-of-step location/time instead of loc_drop_split/dt_split
        dt_evap_end = dt_split #Evaporation stops at dt_split
        R_drop_fin_lst = [R_drop_HED_fin]
        T_drop_fin_lst = [T_drop_HED_fin]
        loc_drop_fin_lst = [loc_drop_HED_fin.copy()]
        vel_drop_fin_lst = [vel_drop_HED_fin.copy()]
        N_drop_fin_lst = [N_drop_ini]
        y_distort_fin_lst = [y_distort_ini]
        ydot_distort_fin_lst = [ydot_distort_ini]
    else:
        # Attempt the full step atomization from the half HED state
        res_atom,_,_,_,_ = HED_to_Atom_step(dt_tar/2,sol_interp,N_drop_ini,y_distort_ini,ydot_distort_ini,evap_drag_args) #This return the atomization process after two times of the input time step (in this case full dt_tar)
        if not res_atom.flag_breakup: #No atomization, finish the second half of HED right away (Merge solution with HED dt results above)
            has_atomized = False
            dt_used = dt_tar
            dt_evap_end = dt_tar
            # Merge solution
            R_drop_fin_lst = [R_drop_HED_fin] #Without atomization, radius changes only by evaporation (full dt HED state)
            T_drop_fin_lst = [T_drop_HED_fin]
            loc_drop_fin_lst = [loc_drop_HED_fin.copy()]
            vel_drop_fin_lst = [vel_drop_HED_fin.copy()]
            N_drop_fin_lst = [N_drop_ini] #Without atomization, number of drops does not change
            y_distort_fin_lst = [res_atom.y_new] #Real update
            ydot_distort_fin_lst = [res_atom.ydot_new] #Real update
        else: #Has atomization, need to bisect out the limited dt_tar that will allow atomization happen right at the end of dt
            tol_dt_bis = rtol_bis*dt_tar/2
            dt_HED_critical = brentq(HED_to_Atom_step_wrap, 0.0, dt_tar/2, args=(sol_interp,N_drop_ini,y_distort_ini,ydot_distort_ini,evap_drag_args), xtol=tol_dt_bis)
            dt_HED_critical = min(dt_HED_critical+tol_dt_bis*3,dt_tar/2) # Add 3 time tolerance margin to force final solution to an atomized state
            res_atom,_R_drop_bef_atom,_,_,_ = HED_to_Atom_step(dt_HED_critical,sol_interp,N_drop_ini,y_distort_ini,ydot_distort_ini,evap_drag_args)
            R_drop_atom_ratio = res_atom.r_new/_R_drop_bef_atom #Radius reduction fraciton after atomization
            has_atomized = res_atom.flag_breakup
            if not has_atomized: raise ValueError(f'After bumping the critical HEDdt after bisection search {res_atom.dt_done}, still no atomization')
            dt_used = res_atom.dt_done #Actaul atomization time run through within critical requirement
            dt_evap_end = dt_used
            # Merge solution and create atomized state
            # Because the critical time might be different from the original target time, need to refine the HED state to merge
            sol_fin = sol_interp(dt_used)
            R_drop_HED_fin = mh.sqrt(sol_fin[0]) #Get updated droplet radius before atomization, as res_atom has radius in half HED state if no atomization
            T_drop_HED_fin = sol_fin[1]
            loc_drop_HED_fin = sol_fin[2:5] #[x,y,z]
            vel_drop_HED_fin = sol_fin[5:8] #[u,v,w]
            # Scale down droplet size due to atomization (mirro the ratio for mass conservation)
            R_drop_fin = R_drop_HED_fin*R_drop_atom_ratio #Final droplet size with all four processes
            N_drop_fin = res_atom.N_new
            vel_brk_fin = res_atom.vel_break #Keep splitting velocity the same (Normal to the relative velocity)
            # Add the split velocity
            vel_gas_HED_fin = evap_drag_args.u_gas_fun(loc_drop_HED_fin)
            vel_rel_HED_fin = vel_gas_HED_fin-vel_drop_HED_fin
            vel_rel_HED_fin_mag = np.linalg.norm(vel_rel_HED_fin)
            if (vel_rel_HED_fin_mag == 0) or (n_parcels<=1) or (vel_brk_fin==0):
                R_drop_fin_lst = [R_drop_fin] 
                T_drop_fin_lst = [T_drop_HED_fin]
                loc_drop_fin_lst = [loc_drop_HED_fin.copy()]
                vel_drop_fin_lst = [vel_drop_HED_fin.copy()]
                N_drop_fin_lst = [N_drop_fin]
                y_distort_fin_lst = [0] # Should be 0
                ydot_distort_fin_lst = [0] # Should be 0
            else:
                vel_rel_HED_fin_dir = vel_rel_HED_fin/vel_rel_HED_fin_mag
                helper = np.eye(3)[np.argmin(np.abs(vel_rel_HED_fin_dir))] # axis least aligned with n_rel
                e1 = helper - np.dot(helper,vel_rel_HED_fin_dir)*vel_rel_HED_fin_dir
                e1 /= np.linalg.norm(e1)
                e2 = np.cross(vel_rel_HED_fin_dir, e1) # already unit length
                # break down into number of parcels
                R_drop_fin_lst = []
                T_drop_fin_lst = []
                loc_drop_fin_lst = []
                vel_drop_fin_lst = []
                N_drop_fin_lst = []
                y_distort_fin_lst = []
                ydot_distort_fin_lst = []
                for idx_par in range(n_parcels):   
                    theta = 2*np.pi*idx_par/n_parcels
                    dir_break = np.cos(theta)*e1 + np.sin(theta)*e2
                    vel_drop_fin = vel_drop_HED_fin + dir_break*vel_brk_fin
                    # Collect each branch
                    R_drop_fin_lst.append(R_drop_fin)
                    T_drop_fin_lst.append(T_drop_HED_fin)
                    loc_drop_fin_lst.append(loc_drop_HED_fin.copy())
                    vel_drop_fin_lst.append(vel_drop_fin.copy())
                    N_drop_fin_lst.append(N_drop_fin/n_parcels)
                    y_distort_fin_lst.append(0)
                    ydot_distort_fin_lst.append(0)
    #### Vapor release over the evaporating part of the step [0, dt_evap_end] (parent parcel, N_drop_ini drops)
    if sol_interp is None: #Droplet already at the floor: no evaporation
        m_vap_rel = 0.0
        dt_vap_rel = 0.0
        loc_vap_rel = loc_drop_ini.copy()
    else:
        sol_evap_end = sol_interp(dt_evap_end)
        # Solve for vapor release
        fuel_cpr = evap_drag_args.fuel_cpr
        fuel_cpr.update(cpr.PT_INPUTS, evap_drag_args.P_gas_tot, T_drop_ini) #Droplet liquid form at step start
        m_drop_sta = (4/3)*mh.pi*fuel_cpr.rhomass()*R_drop_ini**3 #[kg] per droplet
        fuel_cpr.update(cpr.PT_INPUTS, evap_drag_args.P_gas_tot, sol_evap_end[1]) #Droplet liquid form at evaporation end (Before breakup)
        m_drop_end = (4/3)*mh.pi*fuel_cpr.rhomass()*sol_evap_end[0]**1.5 #[kg] per droplet
        m_vap_rel = N_drop_ini*(m_drop_sta-m_drop_end)
        # Use mid point rule to approximate vapor release location and time
        dt_vap_rel = dt_evap_end/2
        loc_vap_rel = sol_interp(dt_vap_rel)[2:5]
    return has_atomized,dt_used,R_drop_fin_lst,T_drop_fin_lst,loc_drop_fin_lst,vel_drop_fin_lst,N_drop_fin_lst,y_distort_fin_lst,ydot_distort_fin_lst,\
           m_vap_rel,dt_vap_rel,loc_vap_rel

#### Turbulence
def kin_turb_gen_simple(l_cha,u_mag_ref,I_turb,d_drop_min,rho_drop_ref,mu_gas_inlet,num_modes,rng_seed):
    """
    Frozen kinematic-simulation (random Fourier mode) velocity perturbation, toy version (after Fung et al. 1992):
        u'(x) = sum_n( A_n*sin(k_n . x) + B_n*cos(k_n . x) )
    A_n, B_n, k_n are 3-component vectors with A_n, B_n normal to k_n, so u' is divergence free.
    Mode directions k_hat are uniform on the unit sphere; |k_n| is geometrically spaced from k_min to k_max.
    Every mode carries the same expected energy (E(k) ~ 1/k on geometric spacing, not a von Karman spectrum), scaled so the
    ensemble (seed) average of <|u'|^2> is 3*u_rms^2; a single realization deviates by ~ +-16% (16 modes), +-9% (64 modes).
    No time dependence (no omega_n): the field is frozen for the whole simulation.
    Scales (wavenumber convention k = 1/l):
        L11 = 0.2*l_cha (integral length scale), u_rms = I_turb*u_mag_ref, eps = 0.8*u_rms^3/L11
        k_min = 0.3/L11 (largest eddies)
        k_max = 1/l_min, l_min = sqrt(eps*tau_p^3): eddy whose turnover time equals the Stokes relaxation time
                tau_p = rho_drop*d^2/(18*mu) of the smallest drop that matters (smaller eddies barely move it).
                Assumes l_min > Kolmogorov scale eta = (nu^3/eps)^(1/4), i.e. tau_p > (nu/eps)^(1/2) (not checked)
    args:
        l_cha: float: channel height or characteristic length scale [m]
        u_mag_ref: float: a reference inlet velocity to define perturbation velocity strength [m/s]
        I_turb: float: turbulence intensity
        d_drop_min: float: smallest droplet size that matters [m]
        rho_drop_ref: float: reference droplet density [kg/m3]
        mu_gas_inlet: float: inlet gas dynamic viscosity [kg/(m*s)]
        num_modes: int: number of modes to generate
        rng_seed: int: seed for random number generator
    return:
        kAB_lst: array (num_modes, 9): one row per mode [k1,k2,k3, A1,A2,A3, B1,B2,B3] ([1/m], [m/s], [m/s])
    raises:
        ValueError: k_min >= k_max (smallest drop responds to eddies larger than the integral scale)
    """
    l_11 = 0.2*l_cha # integral length scale [m]
    u_rms = u_mag_ref*I_turb # rms of one velocity component [m/s]
    eps = 0.8*u_rms**3/l_11 # turbulent dissipation rate [m2/s3]
    k_min = 0.3/l_11 # smallest wavenumber (largest eddies) [1/m]
    tau_p_min = rho_drop_ref*d_drop_min**2/(18.*mu_gas_inlet) # Stokes relaxation time of the smallest drop [s]
    l_min = mh.sqrt(eps*tau_p_min**3) # eddy size with turnover time (l^2/eps)^(1/3) = tau_p_min [m]
    k_max = 1/l_min # largest wavenumber [1/m]
    if k_min >= k_max:
        raise ValueError(f'k_min {k_min} larger or equal to k_max {k_max}')
    k_mag_lst = np.geomspace(k_min, k_max, num=num_modes)
    rng = np.random.default_rng(seed=rng_seed)
    kAB_lst = np.empty((num_modes,9)) #modes * [k1,k2,k3,A1,A2,A3,B1,B2,B3]
    SD = u_rms * mh.sqrt(1.5/num_modes) # per-component sd of the raw amplitudes: E<|u'|^2> = 2*N*SD^2 = 3*u_rms^2
    for idx,k_mag in enumerate(k_mag_lst):
        # Mode direction: normalized 3D Gaussian -> uniform on the unit sphere
        k_hat = rng.normal(loc=0.0, scale=1.0, size=3)
        while np.linalg.norm(k_hat) == 0.0:
            k_hat = rng.normal(loc=0.0, scale=1.0, size=3)
        k_hat /= np.linalg.norm(k_hat)
        # Amplitudes normal to k_hat (cross product) -> divergence free
        A_raw = rng.normal(loc=0.0, scale=SD, size=3)
        B_raw = rng.normal(loc=0.0, scale=SD, size=3)
        A = np.cross(A_raw,k_hat)
        B = np.cross(B_raw,k_hat)
        k = k_mag*k_hat
        kAB_lst[idx,:3] = k[:]
        kAB_lst[idx,3:6] = A[:]
        kAB_lst[idx,6:9] = B[:]
    return kAB_lst

def vel_field_comb(vel_func_bulk,kAB_lst):
    """
    Combine an averaged laminar bulk velocity field with a frozen turbulent field for a realization
    args:
        [u,v,w]=vel_func_bulk(loc): laminar velocity field
        kAB_lst: frozen turbulence field, see 'kin_turb_gen_simple'
    return:
        [u,v,w]=vel_fld(loc), combined velocity field
    """
    def vel_fld(loc):
        # Perturbed velocity field
        return vel_func_bulk(loc)+vel_turb_sum(loc[0],loc[1],loc[2],kAB_lst)
    return vel_fld

@njit(cache=True)
def vel_turb_sum(x,y,z,kAB_lst):
    """
    Perturbation velocity sum_n( A_n*sin(k_n . x) + B_n*cos(k_n . x) ) at (x,y,z); kAB_lst rows [k1,k2,k3,A1,A2,A3,B1,B2,B3]
    args:
        x,y,z: locations [m]
        kAB_lst: frozen turbulence field, see 'kin_turb_gen_simple'
    return:
        vel_prim: array: u',v',w'
    """
    vel_prim = np.zeros(3)
    for idx in range(kAB_lst.shape[0]):
        phase = kAB_lst[idx,0]*x + kAB_lst[idx,1]*y + kAB_lst[idx,2]*z
        sin_ph = mh.sin(phase)
        cos_ph = mh.cos(phase)
        for idx_dir in range(3):
            vel_prim[idx_dir] += kAB_lst[idx,3+idx_dir]*sin_ph + kAB_lst[idx,6+idx_dir]*cos_ph
    return vel_prim

#### Parcel Tracking
class Parcel(NamedTuple): #This is immutable so always passed by copy when append
    """State of one parcel at time t (all drops in the parcel share it)."""
    t: float            # time [s]
    R: float            # droplet radius [m]
    N: float            # drops represented by the parcel
    y: float            # TAB distortion
    ydot: float         # TAB distortion rate [1/s]
    T: float            # droplet temperature [K]
    loc: np.ndarray     # droplet location [x,y,z] [m]
    vel: np.ndarray     # droplet velocity [u,v,w] [m/s]
    gen: int            # breakup generation (0: not broken up yet)
    pid: str            # lineage id: children of '0' are '0.0', '0.1', ...
    dt: float = None    # next time step to try [s] (parcels_simulate); None: start from its initial guess

class VapLog(NamedTuple):
    """Vapor release log (one row per parcel step with evaporation)."""
    t: np.ndarray       # release time [s], shape (M,)
    loc: np.ndarray     # release location [x,y,z] [m], shape (M,3)
    m: np.ndarray       # released vapor mass [kg], shape (M,)
    pid: list           # lineage id of the releasing parcel, length M

def parcels_simulate(parcel_ini_lst,t_end,evap_drag_args,cha_rel_tar,n_ring=1,len_scale=1,vel_scale=1,rtol_bis=1e-6,
                     eps_Rf=1e-6,max_steps=1000000,eps_dt=1e-12,eps_vel_fra=0.03,iter_rerun_max=20,dt_max=1e-4,dt_min=5e-7,frac_rerun_overshoot=0.9,
                     sca_up_dt_max=2.0,flg_atom=True):
    """
    Advance a list of parcels to t_end with HEAD_one_step.
    Following every breakup lineage, and log the vapor release.
    Gas state (T_gas_inf, Y_vap_inf, u_gas_fun) is fixed in evap_drag_args
    args:
        parcel_ini_lst: list of Parcel: initial parcels (gen = 0 for drops that have not broken up)
        t_end: float: end time [s]
        evap_drag_args: EvapDragArgs: fixed inputs (see evap_heat_drag_fun)
        cha_rel_tar: array-like (3,): per-step limits [slip, T, dist]: relative change of the slip velocity magnitude and of
                     the drop temperature (TAB accuracy), and largest distance travelled as a fraction of len_scale
                     sets the spatial resolution of the vapor log and when atomization condition vary much slower than evaporation
        n_ring: int: child parcels created at every breakup, spread on a ring normal to the slip (1: no velocity kick)
        len_scale: float: length scale for evap_heat_trans_solve to set tolerance for location simulation
        vel_scale: float: velocity scale for evap_heat_trans_solve to set tolerance for velocity simulation
        rtol_bis: float: bisection tolerance used by HEAD_one_step to push the atomization process to the end of one full step
        eps_Rf: float: relative margin above R_drop_floor that counts as fully evaporated
        max_steps: int: step limit per parcel segment (raises RuntimeError if exceeded)
        eps_dt: float: relative tol check for final simulation time reach
        eps_vel_fra: float: slip floor for the relative slip change, as a fraction of vel_scale
                            below it the slip change is measured against the floor
        iter_rerun_max: int: maximum number of rerun per step to get to target relative change
        dt_max: float: maximum time step to take [s]
        dt_min: float: minimum time step to take [s]
        frac_rerun_overshoot: float: fraction of cha_rel_tar to overshoot the time step scaling to avoid repeated clipping
        sca_up_dt_max: float: largest factor the time step may grow by from one accepted step to the next
        flg_atom: bool: turns this to false if want to turn off atomization process
    return: traj_lst, vap_log
        traj_lst: list of list of Parcel: one history per lineage segment (a segment ends at breakup, at t_end,
                  or when evaporated); a child segment starts with the breakup state
        vap_log: VapLog: all vapor releases
    Behavior:
        A parcel stops at the end of the step in which its drops reach R_drop_floor; the liquid left at the floor is
        released as vapor at that time and location (vapor log holds all fuel for any floor size)
    """
    #### Droplet states
    traj_lst = [] # list-list-parcel: number parcels to track * number of iterations * parcel variables
    #### Vapor states
    vap_t = []
    vap_loc = []
    vap_m = [] 
    vap_pid = []
    #### Initial parcels states and conditions
    parcel_run_lst = list(parcel_ini_lst) # Finish one parcel path at a time, by grabing one parcel from the running list and finish it, and append new child parcel to the same list
    dt_guess = mh.sqrt(dt_max*dt_min) # Geo-mean
    while parcel_run_lst: #Keep on running while the list is not empty (some parcel still need processing)
        # Get one parcel to run
        par = parcel_run_lst.pop() #Get rid of the last parcel
        hist = [par] #Place the last parcel into a history log for simulation
        # Start from the step carried by the parcel (children inherit the parent's), else from the guess
        dt_cur = par.dt if par.dt is not None else dt_guess
        for _ in range(max_steps): #Simulate this many iterations at most or else report error
            #### Once a droplet reaches radius floor, assume all its mass become vapor release and the simulation stop
            if par.R <= evap_drag_args.R_drop_floor*(1+eps_Rf):
                evap_drag_args.fuel_cpr.update(cpr.PT_INPUTS, evap_drag_args.P_gas_tot, par.T) #Droplet liquid form
                vap_t.append(par.t)
                vap_loc.append(par.loc.copy())
                vap_m.append(par.N*(4/3)*mh.pi*evap_drag_args.fuel_cpr.rhomass()*par.R**3)
                vap_pid.append(par.pid) #TODO: Might need to record down vapor temperature with hvap taken out in the future
                break
            #### Check if simulation finished
            if (t_end - par.t) <= eps_dt*t_end:
                break
            #### Take one step #TODO: Later when fully coupled remember to udpate the gas velocity, gas temperature, gas composition in evap_drag_args
            # Parameters to check relative change
            vel_gas_sta = evap_drag_args.u_gas_fun(par.loc)
            vel_rel_sta_mag = np.linalg.norm(vel_gas_sta-par.vel)
            vel_floor_mag = eps_vel_fra*vel_scale #only for relative difference calcualtion
            # Run simulation
            for _ in range(iter_rerun_max):
                # Run simulation
                dt_cur = min(dt_cur, t_end-par.t)
                (has_atomized,dt_used,R_lst,T_lst,loc_lst,vel_lst,N_lst,y_lst,ydot_lst,\
                m_vap_rel,dt_vap_rel,loc_vap_rel) = \
                HEAD_one_step(dt_cur,par.R,par.N,par.y,par.ydot,par.T,par.loc,par.vel,\
                            evap_drag_args,len_scale=len_scale,vel_scale=vel_scale,\
                            rtol_bis=rtol_bis,n_parcels=n_ring,flg_atom=flg_atom)
                # Fully evaporated within the step: the end state is passive gas transport (zero slip), nothing left to resolve
                if R_lst[0] <= evap_drag_args.R_drop_floor*(1+eps_Rf):
                    break
                # Relative change of slip magnitude and drop temperature, each as a fraction of its own target
                vel_gas_end = evap_drag_args.u_gas_fun(loc_lst[0])
                vel_drop_end = np.mean(vel_lst,axis=0) # Parent velocity: breakup kicks are evenly spread on a ring with equal N per child and cancel in the mean
                cha_vel = abs((np.linalg.norm(vel_gas_end-vel_drop_end)-vel_rel_sta_mag)/max(vel_rel_sta_mag,vel_floor_mag))
                cha_T = abs((T_lst[0]-par.T)/par.T)
                cha_x = np.linalg.norm(loc_lst[0]-par.loc)/len_scale # Distance travelled as a fraction of len_scale
                cha_rel = max(cha_vel/cha_rel_tar[0],cha_T/cha_rel_tar[1],cha_x/cha_rel_tar[2]) # 1 = the most limiting quantity is exactly at its target
                # Adjust time step
                if cha_rel <= 0:
                    dt_cur = min(dt_cur*sca_up_dt_max,dt_max)
                    break
                elif cha_rel < frac_rerun_overshoot:
                    dt_cur = min(dt_cur*min(frac_rerun_overshoot/cha_rel,sca_up_dt_max),dt_max)
                    break
                elif (cha_rel > 1) and (dt_cur>dt_min*(1+eps_dt)):
                    dt_cur = max(dt_cur*(frac_rerun_overshoot/cha_rel),dt_min)
                else:
                    break
            else: #Run out of retry
                warnings.warn(f'parcels_simulate: parcel {par.pid} at t = {par.t:.3e} s: no acceptable step after {iter_rerun_max} retries '
                              f'(dt = {dt_cur:.3e} s, change/target = {cha_rel:.3g}); using the last one', RuntimeWarning)
            #### Log vapor release
            if m_vap_rel != 0.0:
                vap_t.append(par.t+dt_vap_rel)
                vap_loc.append(loc_vap_rel)
                vap_m.append(m_vap_rel)
                vap_pid.append(par.pid) #TODO: Might need to record down vapor temperature with hvap taken out in the future
            #### Update parcel(s)
            t_new = par.t + dt_used
            # With breakup
            if has_atomized:
                for k in range(len(R_lst)): #Appending the new particle to simulation with new starting time for them
                    parNew = Parcel(t=t_new        ,R=R_lst[k]       ,N=N_lst[k],
                                    y=y_lst[k]     ,ydot=ydot_lst[k] ,T=T_lst[k],
                                    loc=loc_lst[k] ,vel=vel_lst[k]   ,gen=par.gen+1,
                                    pid=f'{par.pid}.{k}',dt=dt_cur)
                    parcel_run_lst.append(parNew)
                #### Once a breakup happens, history for parent ends here. Go back to top loop for child simulation.
                break
            # Without breakup, did not or just reach final time, did not fully become vapor (Then have droplet history)
            par = Parcel(t=t_new        ,R=R_lst[0]       ,N=N_lst[0],
                         y=y_lst[0]     ,ydot=ydot_lst[0] ,T=T_lst[0],
                         loc=loc_lst[0] ,vel=vel_lst[0]   ,gen=par.gen,
                         pid=par.pid,dt=dt_cur)
            hist.append(par)
        else:
            raise RuntimeError(f'parcels_simulate: parcel {par.pid} exceeded max_steps = {max_steps} at t = {par.t}')
        #### Record down the history of current path (end at final time or breakup or become fully vapor)
        traj_lst.append(hist)
    #### Collect vapor states
    vap_log = VapLog(t=np.array(vap_t),loc=np.array(vap_loc).reshape(-1,3),m=np.array(vap_m),pid=vap_pid)
    return traj_lst, vap_log

#### Plotting
def plot_parcel_hist_3d(traj_lst,n_sample,seed=0,len_unit=1e-3,elev=30,azim=-60,x_lim=None,wall_lst=None,n_theta_wall=24):
    """
    Plot the 3D path of a few randomly sampled parcel histories in two separate figures,
    one colored by droplet diameter and one by droplet temperature
    args:
        traj_lst: list of list of Parcel: parcel histories from parcels_simulate
        n_sample: int: number of histories to sample (all of them if n_sample >= len(traj_lst))
        seed: int: random seed for the sampling
        len_unit: float: plotting length unit [m] (1e-3: axes in mm)
        elev, azim: float: viewing elevation and azimuth angles [deg] (matplotlib view_init; defaults are matplotlib's)
        x_lim: (float, float) or None: [m] only plot segments with both ends within this x range (None: all)
        wall_lst: list of (x_arr, R_arr) or None: [m] axisymmetric walls (about the x axis) drawn as gray wireframes, clipped to x_lim
        n_theta_wall: int: number of azimuthal wireframe lines per wall
    return: figs, axs
        figs: list of the two matplotlib figures [diameter, temperature]
        axs: list of their 3D axes [diameter, temperature]
    Behavior:
        Each segment between two logged states is colored by the mean of its end values; histories with a single state are skipped
        Diameter in um, temperature in K; color limits are shared over the sampled histories
        Equal scaling on all axes (true geometry)
    """
    # Sample histories
    rng = np.random.default_rng(seed)
    n_sample = min(n_sample,len(traj_lst))
    idx_sample = rng.choice(len(traj_lst),size=n_sample,replace=False)
    hist_lst = [traj_lst[i] for i in idx_sample if len(traj_lst[i])>1]
    # Collect line segments and their values
    seg_lst = []
    dia_lst = []
    T_lst = []
    for hist in hist_lst:
        loc = np.array([par.loc for par in hist])/len_unit #[n,3]
        dia = np.array([2*par.R for par in hist])*1e6 #[um]
        T = np.array([par.T for par in hist]) #[K]
        msk_seg = np.ones(len(loc)-1,dtype=bool) # segments to keep
        if x_lim is not None: # 3D axes do not clip, so drop segments outside the x range
            x_seg_min = np.minimum(loc[:-1,0], loc[1:,0])
            x_seg_max = np.maximum(loc[:-1,0], loc[1:,0])
            msk_seg = (x_seg_min >= x_lim[0]/len_unit) & (x_seg_max <= x_lim[1]/len_unit)
        seg_lst.append(np.stack([loc[:-1],loc[1:]],axis=1)[msk_seg]) #[n-1,2,3]
        dia_lst.append(((dia[:-1]+dia[1:])/2)[msk_seg])
        T_lst.append(((T[:-1]+T[1:])/2)[msk_seg])
    if not seg_lst: raise ValueError('plot_parcel_hist_3d: no sampled history has more than one state')
    seg_all = np.concatenate(seg_lst)
    if len(seg_all)==0: raise ValueError(f'plot_parcel_hist_3d: no sampled segment within x_lim = {x_lim}')
    loc_all = seg_all.reshape(-1,3)
    # Wall surfaces of revolution about the x axis (clipped to x_lim), included in the axis scaling
    wall_surf_lst = []
    if wall_lst is not None:
        theta = np.linspace(0.0, 2*np.pi, n_theta_wall+1)
        for (x_wall, R_wall) in wall_lst:
            x_wall = np.asarray(x_wall, dtype=float)
            R_wall = np.asarray(R_wall, dtype=float)
            x_plot = x_wall
            if x_lim is not None:
                x_in = x_wall[(x_wall > x_lim[0]) & (x_wall < x_lim[1])]
                x_plot = np.concatenate(([x_lim[0]], x_in, [x_lim[1]]))
            R_plot = np.interp(x_plot, x_wall, R_wall)
            X = np.outer(x_plot, np.ones(len(theta)))/len_unit
            Y = np.outer(R_plot, np.cos(theta))/len_unit
            Z = np.outer(R_plot, np.sin(theta))/len_unit
            wall_surf_lst.append((X, Y, Z))
            loc_all = np.concatenate((loc_all, np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)))
    # Plot
    figs = []
    axs = []
    for val,label,cmap in [(np.concatenate(dia_lst),'Droplet diameter [um]','viridis'),
                           (np.concatenate(T_lst),'Droplet temperature [K]','inferno')]:
        fig = plt.figure(figsize=(7,5.5))
        ax = fig.add_subplot(projection='3d')
        lc = Line3DCollection(seg_all,cmap=cmap,norm=plt.Normalize(val.min(),val.max()),linewidths=1.5)
        lc.set_array(val)
        ax.add_collection(lc)
        for (X, Y, Z) in wall_surf_lst:
            ax.plot_wireframe(X, Y, Z, rcount=min(X.shape[0], 20), ccount=n_theta_wall+1, color='gray', linewidth=0.3, alpha=0.5)
        ax.auto_scale_xyz(loc_all[:,0],loc_all[:,1],loc_all[:,2]) # add_collection does not autoscale in 3D
        ax.set_box_aspect(np.ptp(loc_all,axis=0)+1e-12) # equal scaling on all axes
        ax.view_init(elev=elev,azim=azim)
        unit = 'mm' if len_unit==1e-3 else f'{len_unit:g} m'
        ax.set_xlabel(f'x [{unit}]')
        ax.set_ylabel(f'y [{unit}]')
        ax.set_zlabel(f'z [{unit}]')
        fig.colorbar(lc,ax=ax,label=label,shrink=0.7,pad=0.1)
        ax.set_title(f'Sampled parcel histories: {len(hist_lst)}')
        fig.tight_layout()
        figs.append(fig)
        axs.append(ax)
    return figs,axs
