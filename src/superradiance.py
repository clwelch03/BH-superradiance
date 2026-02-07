import numpy as np 
import scipy.optimize as optim 
import astropy.constants as consts
import astropy.units as units
import math 


def superradiant_instability_rate(n, l, m, axion_geometric_mass, BH_geometric_mass, BH_dimensionless_spin) -> float:
    """
    Calculate the superradiant instability rate for a black hole. Based on Equation (1) of arxiv/1908.02312.

    Args:
        n (int): The adjusted principal quantum number n-l-m.
        l (int): The azimuthal quantum number.
        m (int): The magnetic quantum number.
        axion_geometric_mass (float): The axion mass, expressed as its Compton frequency m_eV/hbar. 
        BH_geometric_mass (float): The black hole mass, expressed in units of time G*M_BH/c^3.
        BH_dimensionless_spin (float): The BH spin, expressed as a dimensionless constant chi between 0 and 1.

    Returns:
        float: The calculated rate of instability.
    """
    # Define the outer horizon radius $r_+$
    outer_horizon_radius = BH_geometric_mass * (1 + np.sqrt(1 - BH_dimensionless_spin**2))
    
    # First line of equation (1)
    mass_terms = axion_geometric_mass * (axion_geometric_mass * BH_geometric_mass)**(4*l+4) \
                        * (m * BH_dimensionless_spin - 2 * axion_geometric_mass * outer_horizon_radius)
    
    # Second line of equation (1)
    combinatorics = (2**(4*l+2) * math.factorial(2*l + 1 + n)) / ((l + 1 + n)**(2*l+4) * math.factorial(n)) \
                        * (math.factorial(l) / (math.factorial(2*l) * math.factorial(2*l + 1)))**2
    
    # Third line of equation (1)
    k_values = np.arange(1, l+1)
    product_over_k = np.power(k_values, 2) * (1-BH_dimensionless_spin**2) \
                        + np.ones(l) * (m * BH_dimensionless_spin - 2 * axion_geometric_mass * outer_horizon_radius)**2
    
    return mass_terms * combinatorics * np.prod(product_over_k)


def superradiant_instability_timescale(n, l, m, axion_mass, BH_mass, BH_spin) -> float:
    """
    Calculates the characteristic timescale of superradiance -- the reciprocal of the superradiant instability rate.

    Returns:
        float: the instability timescale in seconds.
    """
    return 1 / superradiant_instability_rate(n, l, m, axion_mass, BH_mass, BH_spin)


def critical_spin(n, l, m, axion_geometric_mass, BH_geometric_mass, merger_timescale):
    """
    Calculates the spin below which the superradiant growth of the axion cloud cannot occur.

    Args:
        n (int): The adjusted principal quantum number n-l-m.
        l (int): The azimuthal quantum number.
        m (int): The magnetic quantum number.
        axion_geometric_mass (float): The axion mass, expressed as its Compton frequency m_eV/hbar. 
        BH_geometric_mass (float): The black hole mass, expressed in units of time G*M_BH/c^3.
        merger_timescale (float): The timescale on which the two black holes merge, in seconds.

    Returns:
        float: The critical spin for the given mode.
    """
    return optim.fsolve(lambda chi: superradiant_instability_rate(n, l, m, axion_geometric_mass, BH_geometric_mass, chi) - 1/merger_timescale, 0.5)


def find_highest_achievable_mode(n, axion_geometric_mass, BH_geometric_mass, initial_BH_spin, merger_timescale) -> int:
    """
    Calculates the highest mode of the axion cloud that can be populated.
    
    Only considers contributions from the dominant modes, i.e. modes of the form [n, l, m] = [n, m, m].

    Args:
        n (int): The adjusted principal quantum number n-l-m. For the dominant modes, we generically have n=0.
        axion_geometric_mass (float): The axion mass, expressed as its Compton frequency m_eV/hbar. 
        BH_geometric_mass (float): The black hole mass, expressed in units of time G*M_BH/c^3.
        initial_BH_spin (float): The initial BH spin, expressed as a dimensionless constant chi between 0 and 1.
        merger_timescale (float): The timescale on which the two black holes merge, in seconds.

    Returns:
        int: The maximum value of m such that [n, m, m] is an accessible superradiant mode.
    """
    maximum_checked_mode = 5
    
    for m in range(1, maximum_checked_mode+1):
        # For the first mode (l=m=1), the growth timescale is based off the initial BH spin
        # For higher modes, the growth timescale is based off the critical spin of the previous mode
        if m == 1:
            growth_timescale_m = 180 * superradiant_instability_timescale(n, m, m, axion_geometric_mass, BH_geometric_mass, initial_BH_spin)
        else:
            growth_timescale_m = 180 * superradiant_instability_timescale(n, m, m, axion_geometric_mass, BH_geometric_mass,
                                                          critical_spin(n, m-1, m-1, axion_geometric_mass, BH_geometric_mass, merger_timescale))
        
        if growth_timescale_m > merger_timescale:
            return m-1
    
    return maximum_checked_mode


def final_BH_spin(axion_geometric_mass, BH_geometric_mass, initial_BH_spin, merger_timescale) -> float:
    """
    Calcluates the final spin of the black hole post-merger, assuming superradiance occurs.

    Args:
        axion_geometric_mass (float): The axion mass, expressed as its Compton frequency m_eV/hbar. 
        BH_geometric_mass (float): The black hole mass, expressed in units of time G*M_BH/c^3.
        initial_BH_spin (float): The initial BH spin, expressed as a dimensionless constant chi between 0 and 1.
        merger_timescale (float): The timescale on which the two black holes merge, in seconds.

    Returns:
        float: The final dimensionless spin of the black hole.
    """
    l_m_max = find_highest_achievable_mode(0, axion_geometric_mass, BH_geometric_mass, initial_BH_spin, merger_timescale)
    return critical_spin(0, l_m_max, l_m_max, axion_geometric_mass, BH_geometric_mass, merger_timescale)[0]


TEN_BILLION_YEARS_IN_SECONDS = 3.15457e17
M_SOL_TO_GEOMETRIC = (consts.G  * consts.M_sun / consts.c**3).to(units.s).value # type: ignore
EV_TO_GEOMETRIC = (1 * units.eV / consts.hbar).to(units.Hz).value # type: ignore

def calculate_everything(BH_mass_M_sol, axion_mass_eV, initial_BH_spin, merger_timescale):
    if not (0 <= initial_BH_spin <= 1):
        raise ValueError("Initial dimensionless BH spin must be between 0 and 1.")
    
    BH_geometric_mass = BH_mass_M_sol * M_SOL_TO_GEOMETRIC
    axion_geometric_mass = axion_mass_eV * EV_TO_GEOMETRIC
    highest_achievable_mode = find_highest_achievable_mode(0, axion_geometric_mass, BH_geometric_mass, initial_BH_spin, merger_timescale)
    
    print(f"BH mass = {BH_geometric_mass} M_sol")
    print(f"axion mass = {axion_mass_eV} eV")
    print(f"initial spin chi_F = {initial_BH_spin}\n\n")
    superradiance_gamma = superradiant_instability_rate(0, 1, 1, axion_geometric_mass, BH_geometric_mass, initial_BH_spin)

    print(f"Superradiant rate: {superradiance_gamma}")
    print(f"Instability timescale: {1/superradiance_gamma}")
    print(f"Highest achievable mode: {highest_achievable_mode}")
    print(f"Final BH spin: {final_BH_spin(axion_geometric_mass, BH_geometric_mass, initial_BH_spin, merger_timescale)}")
    
# calculate_everything(5.0, 1e-12, 0.99, TEN_BILLION_YEARS_IN_SECONDS)

###

#let's test things

M = 60.0 #5 M_sol
mu = 1.0e-12 #eV
chi = 1.0
s_to_yr = 3.17e-8
yr_to_s = 1.0/s_to_yr
tau_s = 1e10 * yr_to_s

calculate_everything(M, mu, chi, tau_s)

# print("M (M_sol): ", M)
# print("mu (eV): ", mu)
# print("chi_F: ", chi)
# print("n, l, m: ", np.array([n, l, m]))
# print("\n")

# Gamma_test = superradiant_instability_rate(n, l, m, mu, M, chi)

# print("Gamma: ", Gamma_test)

# tau_inst_test = superradiant_instability_timescale(n, l, m, mu, M, chi)
# print("tau_inst (s): ", tau_inst_test) #seconds
# print("tau_inst (yr): ", tau_inst_test*s_to_yr)

# print("tau_s (s): ", tau_s)

# chi_test = solve_chi(n, l, m, mu, M, tau_s)

# print("chi_test: ", chi_test)

# tau_grow_test = 180*tau_inst(n, 2, 2, mu, M, chi_test)
# print("tau_grow after first phase (m=2): ", tau_grow_test)

# chi_F = 1.0
# m_max_test = get_highest_mode(n, mu, M, chi_F, tau_s)

# print("m_max: ", m_max_test)

# final_spin_test = get_merger_spin(mu, M, chi_F, tau_s)

# print("final spin: ", final_spin_test)

# print(np.shape(final_spin_test[0]))