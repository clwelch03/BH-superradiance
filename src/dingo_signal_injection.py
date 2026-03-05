import numpy as np
from scipy.stats import beta
import superradiance as sr
import gwpopulation

num_events = 1000

## ----- Generate mock data ----- ##

# - Spins - #
# Draw from the GWTC-3 "Default" spin model (Beta distribution)
# Let's just pretend this is the right distribution for now
# I should really extract my own priors from the big-ass dataset here:
# https://zenodo.org/records/11254021
a_1_mock = beta.rvs(a=2.0, b=7.0, size=num_events)
a_2_mock = beta.rvs(a=2.0, b=7.0, size=num_events)

# - Masses - #
#TODO: make mass distributions PowerLaw+Peak
MIN_BH_MASS = 5.0 * sr.M_SOL_TO_GEOMETRIC
MAX_BH_MASS = 50.0 * sr.M_SOL_TO_GEOMETRIC
MASS_DIST_EXPONENT = -1.35

# Create array of BH spins according to a uniform distribution
num_events = 10000
BH_initial_spins = np.random.uniform(0, 1, num_events)

# Create array of BH masses according to a power-law distribution
# Values come from Orion's code
BH_uniform = np.random.uniform(0, 1, num_events)
mass_1 = ( (MAX_BH_MASS**MASS_DIST_EXPONENT - MIN_BH_MASS**MASS_DIST_EXPONENT)*BH_uniform + MIN_BH_MASS**MASS_DIST_EXPONENT) ** (1.0/MASS_DIST_EXPONENT)
mass_ratio = np.random.uniform(0.1, 1.0, num_events)
mass_2 = mass_1 * mass_ratio

# Dingo will want chirp mass
chirp_mass = (mass_1 * mass_2)**(3./5.) / (mass_1 + mass_2)**(1./5.)

# - Extrinsic parameters - #
#TODO: this all came from Gemini. double check it
tilt_1 = np.arccos(np.random.uniform(-1.0, 1.0, num_events))
tilt_2 = np.arccos(np.random.uniform(-1.0, 1.0, num_events))
phi_12 = np.random.uniform(0.0, 2 * np.pi, num_events)
phi_jl = np.random.uniform(0.0, 2 * np.pi, num_events)

phase = np.random.uniform(0.0, 2 * np.pi, num_events)
psi = np.random.uniform(0.0, np.pi, num_events) # Polarization angle
theta_jn = np.arccos(np.random.uniform(-1.0, 1.0, num_events)) # Inclination

# Sky Location (Isotropic across the sky)
ra = np.random.uniform(0.0, 2 * np.pi, num_events)
dec = np.arcsin(np.random.uniform(-1.0, 1.0, num_events))

# Distance (Uniform in comoving volume, approximating roughly 100 to 1500 Mpc)
# We draw D^3 uniformly to simulate 3D space correctly
luminosity_distance = np.cbrt(np.random.uniform(100**3, 1500**3, num_events))

# Time (Randomly distributed across the LIGO O3 observing run epochs)
geocent_time = np.random.uniform(1238166018, 1269363618, num_events)

# 3. Superradiance Physics Block goes here!
# a_1_superradiant = apply_superradiance(a_1_mock, ...)