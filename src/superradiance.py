import numpy as np
import scipy.optimize as optim

def superradiant_rate(n, l, m, axion_mass, BH_mass, BH_spin):
    # define the outer horizon radius $r_+$
    outer_horizon_radius = BH_mass * (1 + np.sqrt(1 - BH_spin**2))
    
    # first line of equation (1)
    first_line = axion_mass * (axion_mass * BH_mass)**(4*l+4) * (m * BH_spin - 2 * axion_mass * outer_horizon_radius)
    # second line of equation (1)
    combinatorics = (2**(4*l+2) * np.factorial(2*l + 1 + n)) / ((l + 1 + n)**(2*l+4) * np.factorial(n)) * (
                        np.factorial(l) / (np.factorial(2*l) * np.factorial(2*l + 1)))**2
    # third line of equation (1)
    k_values = np.arange(1, l+1)
    product_over_k = np.power(k_values, 2) * (1-BH_spin)**2 + np.ones(l) * (m * BH_spin - 2 * axion_mass * outer_horizon_radius)**2
    
    return first_line * combinatorics * np.prod(product_over_k)

def emission_timescale(n, l, m, axion_mass, BH_mass, BH_spin): return 1 / superradiant_rate(n, l, m, axion_mass, BH_mass, BH_spin)

def mode_spin(n, l, m, axion_mass, BH_mass, merger_timescale):
    return optim.fsolve(lambda chi: superradiant_rate(n, l, m, axion_mass, BH_mass, chi) - 1/merger_timescale, 0.5)

# return an integer corresponding to l=m of the highest accessible mode [0, l, m] = [0, m, m]
def find_highest_achievable_mode(n, axion_mass, BH_mass, initial_BH_spin, merger_timescale):
    maximum_checked_mode = 3

    # check successively higher nodes until growth timescale < merger timescale
    for m in range(1, maximum_checked_mode+1):
        if m == 1:
            growth_timescale_m = 180 * emission_timescale(n, m, m, axion_mass, BH_mass, initial_BH_spin)
        else:
            growth_timescale_m = 180 * emission_timescale(n, m, m, axion_mass, BH_mass,
                                                          mode_spin(n, m-1, m-1, axion_mass, BH_mass, merger_timescale))
        if growth_timescale_m < merger_timescale: return m-1
    
    return maximum_checked_mode

def final_BH_spin(axion_mass, BH_mass, initial_BH_spin, merger_timescale):
    l_m_max = find_highest_achievable_mode(0, axion_mass, BH_mass, initial_BH_spin, merger_timescale)
    return mode_spin(0, l_m_max, l_m_max, axion_mass, BH_mass, merger_timescale)