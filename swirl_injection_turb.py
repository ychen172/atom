"""
This script test out the injection of a liquid droplet or a spray of droplets 
without atomization through a swirl flow channel modeled by Lamb-Oseen vortex  
"""

#### Import Link Module
import os
import sys
BaseDir = os.path.dirname(os.path.abspath(__file__))
#### Module imports
from utility import *
import CoolProp as cpr
import cantera as ct
import pickle
ct.add_directory(BaseDir)

#### Setup
# Flow Information
axis_flow = 'x'
vel_axis = 65.0 #[m/s] axial velocity at the venturi inlet (sets the mass flow)
x_venturi_pts = np.array([0.0, 25.0, 50.0])*1e-3 #[m] venturi tip radius control points
R_venturi_pts = np.array([20.0, 10.0, 15.0])*1e-3 #[m] (quadratic R(x) = a*x^2 + b*x + c through them)
num_venturi = 51 # number of points sampling the venturi wall
x_tail = 0.5 #[m] constant radius tail after the venturi exit (walls must cover the droplet x range)
r_swirl_out = R_venturi_pts[0] #[m] swirler tip radius at the venturi inlet
swirl_num = 0.7 # swirl number at the venturi inlet (x = 0)
frac_peak = 1.0 # (r_peak - r_hub)/(r_tip - r_hub) of the Lamb-Oseen swirl peak at the inlet (1.0: r_peak = r_tip)
ratio_hub_to_tip = 0.3
tau_rest_wall = 1e-6 #[s] wall penalty response time (wall penetration ~ impact speed*tau_rest_wall)
T_gas = 884.0 #[K]
P_gas = 15.9*1e5 #[Pa]
X_gas_pure = 'O2:0.2079,N2:0.7821,H2O:0.01'
# Fuel Information
ratio_fuel_air = 0.055 #pilot flame fuel air ratio (Assume all injected by jet in crossflow)
reac_mech = 'TOT_HT_NOX_413_14922_wTrans.yaml'
X_fuel_pure = 'NC12H26:1.0' #TODO: Only single component fuel allowed
spe_fuel_diffu = 'NC12H26' #single species used to extract diffusivity
spe_fuel_liquid = 'n-Dodecane' #single species used to establish liquid object
T_fuel = 298.0 #[K]
x_inj = 25.0e-3 #[m] injection station (droplets released at half channel height there)
num_drop_pars = 10 # number of droplet parcels/sizes to initialize
num_injectors = 6 # number of injector per swirler
# Simulation setup
flg_atom = False
t_end = 5e-3 #[s]
cha_rel_tar = np.array([5e-2,1.5e-2,4e-2])
n_ring = 1 #number of droplets after each secondary breakup
len_scale=r_swirl_out
vel_scale=vel_axis
r_floor_frac = 1e-3
# Turbulence
I_turb = 0.05 # Turbulence intensity
num_realization = 1 # Number of perturbed frozen turbulent flow field to simulate
num_turb_modes = 30 # Number of turbulent velocity modes
# Save setup
save_name = 'nDod_Vel65_S07_h2t03_T884P15_9_nPar10_nInj6_ITurb5_nTMod30'
save_dir = './'

#### Create domain (venturi)
coef_venturi = np.polyfit(x_venturi_pts, R_venturi_pts, 2) # quadratic through the 3 control points
x_venturi_pts_fine = np.concatenate((np.linspace(x_venturi_pts[0], x_venturi_pts[-1], num_venturi), [x_tail])) #[m] wall x coordinates
R_tip_venturi_fine = np.polyval(coef_venturi, np.minimum(x_venturi_pts_fine, x_venturi_pts[-1])) #[m] swirler tip (outer wall) radius at x_venturi_pts_fine, constant in the tail
R_hub_venturi_fine = ratio_hub_to_tip*R_tip_venturi_fine #[m] swirler hub (inner wall) radius at x_venturi_pts_fine
R_tip_inj = np.polyval(coef_venturi, x_inj) #[m] tip radius at the injection station
R_hub_inj = ratio_hub_to_tip*R_tip_inj #[m] hub radius at the injection station
loc_drop_ini = np.array([x_inj, 0.0, 0.5*(R_tip_inj+R_hub_inj)]) # Half channel height at the injection station
dom_limiter_tip = dom_lim_axi_sym_init(x_venturi_pts_fine,R_tip_venturi_fine,np.max(R_tip_venturi_fine),tau_rest=tau_rest_wall)
dom_limiter_hub = dom_lim_axi_sym_init(x_venturi_pts_fine,R_hub_venturi_fine,np.max(R_hub_venturi_fine),tau_rest=tau_rest_wall,flg_inner=True)

#### Create chemistry object
# Gas phase
gas_ct = ct.Solution(reac_mech, transport_model='mixture-averaged')
gas_ct.TPX  = 298.0 , 101325.0 , X_fuel_pure
Y_fuel_pure = gas_ct.Y
MW_fuel_pure = gas_ct.mean_molecular_weight #TODO: At this point, this only work for single component fuel
gas_ct.TPX  = 298.0 , 101325.0 , X_gas_pure
Y_gas_pure = gas_ct.Y
MW_gas_pure = gas_ct.mean_molecular_weight
# Simplify cantera solution
idx_nontrivial_spe = np.nonzero(Y_fuel_pure+Y_gas_pure)[0]
gas_ct = ct.Solution(thermo='ideal-gas', transport_model='mixture-averaged',
                  species=[gas_ct.species(gas_ct.species_names[k]) for k in idx_nontrivial_spe])
Y_fuel_pure = Y_fuel_pure[idx_nontrivial_spe]
Y_gas_pure = Y_gas_pure[idx_nontrivial_spe]
i_fuel_diffu = gas_ct.species_index(spe_fuel_diffu) # index for single species used to extract diffusivity
# Fuel phase
fuel_cpr = cpr.AbstractState('HEOS',spe_fuel_liquid)
fuel_cpr.update(cpr.PT_INPUTS, P_gas, T_fuel)
rho_drop_ref = fuel_cpr.rhomass() #[kg/m3]

#### Calculate droplet initial condition
# Initial droplet condition
gas_ct.TPY = T_gas, P_gas, Y_gas_pure
mdot_gas = mh.pi*r_swirl_out**2.*(1.-ratio_hub_to_tip**2.) * gas_ct.density_mass * vel_axis #[kg/s]
mdot_fuel_tot = ratio_fuel_air*mdot_gas #[kg/s] Total fuel flow rate for the whole swirler
mu_gas_inlet = gas_ct.viscosity #[Pa*s]
# Velocity field: Lamb-Oseen swirl through the venturi, conserving mass and streamtube circulation
vel_gas_fun = make_vel_field_Lamb_Oseen_evo(x_venturi_pts_fine,R_tip_venturi_fine,R_hub_venturi_fine,mdot_gas,gas_ct.density_mass,swirl_num,frac_peak,axis=axis_flow)
y_break = 0.5*(R_tip_inj-R_hub_inj) #[m] half channel height at the injection station
parcel_ini_lst = LJICF_parcel_init(num_drop_pars,num_injectors,mdot_fuel_tot,loc_drop_ini, \
                                   y_break,vel_gas_fun,gas_ct,fuel_cpr)
min_ini_R_drop = min([par_cur.R for par_cur in parcel_ini_lst])

#### Sweep through realizations
l_cha = max(R_venturi_pts)*(1-ratio_hub_to_tip)
vel_mag_ref = vel_axis
d_drop_min = 2*min_ini_R_drop
fuel_cpr.update(cpr.PT_INPUTS, P_gas, T_fuel)
# Save the shared setup once: mean field inputs (make_vel_field_Lamb_Oseen_evo(**mean_inputs)) and run conditions
setup_path = save_dir+save_name+'_Turb_Setup.pkl'
setup_res = {'mean_inputs':dict(xlst=x_venturi_pts_fine, Rout_lst=R_tip_venturi_fine, Rin_lst=R_hub_venturi_fine, mdot=mdot_gas,
                                rho=gas_ct.density_mass, swirl_num=swirl_num, frac_peak=frac_peak, axis=axis_flow),
             'run_inputs':dict(T_gas=T_gas, P_gas=P_gas, X_gas_pure=X_gas_pure, ratio_fuel_air=ratio_fuel_air, spe_fuel_liquid=spe_fuel_liquid,
                               T_fuel=T_fuel, t_end=t_end, flg_atom=flg_atom, num_realization=num_realization,
                               num_injectors=num_injectors, loc_drop_ini=loc_drop_ini, mdot_fuel_tot=mdot_fuel_tot, tau_rest_wall=tau_rest_wall)}
with open(setup_path,'wb') as file_out:
    pickle.dump(setup_res,file_out)
print(f'Saved {setup_path}')
# simulate through realization
for seed_turb in range(num_realization):
    # Get parameters for velocity perturbation
    kAB_cur_lst = kin_turb_gen_simple(l_cha,vel_mag_ref,I_turb,d_drop_min,rho_drop_ref,mu_gas_inlet,num_turb_modes,seed_turb)
    # Incorporate turbulent vleocity field into mean flow
    vel_gas_fun_turb = vel_field_comb(vel_gas_fun,kAB_cur_lst)
    # Evaporation parameters
    evap_drag_args = EvapDragArgs(fuel_cpr=fuel_cpr, gas_ct=gas_ct,
                                  Y_air_pure_arr=Y_gas_pure, Y_fuel_pure_arr=Y_fuel_pure, i_fuel_ct=i_fuel_diffu,
                                  P_gas_tot=P_gas, T_gas_inf=T_gas, Y_vap_inf=0.0,
                                  R_drop_floor=min_ini_R_drop*r_floor_frac, MW_air_pure=MW_gas_pure, MW_fuel_pure=MW_fuel_pure,
                                  u_gas_fun=vel_gas_fun_turb, dom_limiters=(dom_limiter_tip,dom_limiter_hub))
    # Simulation
    print(f'Simulate realization #{seed_turb}')
    drop_hist,vapo_hist = parcels_simulate(parcel_ini_lst,t_end,evap_drag_args,cha_rel_tar,n_ring=n_ring,len_scale=len_scale,vel_scale=vel_scale,flg_atom=flg_atom)

    #### Save simulation condition and results (turbulent field saved as its modes; rebuild with vel_field_comb)
    save_path = save_dir+save_name+f'_Turb_Seed{seed_turb}.pkl'
    save_res = {'seed_turb':seed_turb, 'kAB_lst':kAB_cur_lst,
                'turb_inputs':dict(l_cha=l_cha, vel_mag_ref=vel_mag_ref, I_turb=I_turb, d_drop_min=d_drop_min,
                                   rho_drop_ref=rho_drop_ref, mu_gas_inlet=mu_gas_inlet, num_turb_modes=num_turb_modes),
                'drop_hist':drop_hist, 'vapo_hist':vapo_hist}
    with open(save_path,'wb') as file_out:
        pickle.dump(save_res,file_out)
    print(f'Saved {save_path}')
