from __future__ import annotations
from typing import Tuple, Dict, List
from gwpopulation.models.redshift import PowerLawRedshift
from bilby.core.result import read_in_result
import numpy as np
import h5py
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

def bounds_to_edges(min_val: float, max_val: float, n_centers: int) -> np.ndarray:
    """
	Convert bounds and a length to an array of grid edges in a dimension.

    Args:
        min_val (float): Minimum value along the axis.
        max_val (float): Maximum value along the axis.
		n_centers (int): Number of grid centers along the axis.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays containing the edges corresponding to grid centers.
    """
    centers = np.linspace(min_val, max_val, n_centers)
    edges = np.empty(n_centers + 1)
	
    # All non-end edges are the average between their two adjacent centers
	# First and last edge have value equal to their corresponding center
    edges[1:-1] = 0.5 * (centers[1:] + centers[:-1])
    edges[0] = centers[0]
    edges[-1] = centers[-1]
	
    return edges


def sample_from_2D_ppd(h5_path: str,
                       num_samples: int,
                       x_bounds: Tuple[float, float],
                       y_bounds: Tuple[float, float],
					   rng: np.random.Generator | None = None) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Sample x, y from a 2D density stored as an HDF5 dataset.
	
	Assumes the dataset is tabulated on a rectilinear grid with shape (n_y, n_x),
    i.e. axis-0 corresponds to y and axis-1 corresponds to x.
	(why? idk man that's just how the data we work with is structured)

	Args:
		h5_path (str): Path to the density file
		num_samples (int): Number of samples to generate
		x_bounds (Tuple[float, float]): (x_min, x_max)
		y_bounds (Tuple[float, float]): (y_min, y_max)

	Returns:
		Tuple[np.ndarray, np.ndarray]: x, y arrays of shape (num_samples,)
	"""
	rng = np.random.default_rng() if rng is None else rng

	# Load PPD and get its shape
	with h5py.File(h5_path, "r") as file:
		ppd = file["ppd"][...] # type: ignore
	if ppd.ndim != 2: # type: ignore
		raise ValueError(f"PPD must be 2D, got shape {ppd.shape}") # type: ignore
	
	n_y, n_x = ppd.shape # type: ignore

	# Infer centers from provided bounds, then calculate edges and distances
	x_edges = bounds_to_edges(x_bounds[0], x_bounds[1], n_x)
	y_edges = bounds_to_edges(y_bounds[0], y_bounds[1], n_y)
	x_bin_widths = np.diff(x_edges)  # (n_x,)
	y_bin_widths = np.diff(y_edges)  # (n_y,)

	# Construct cell probability weights
	cell_weights = np.nan_to_num(ppd, nan=0.0, posinf=0.0, neginf=0.0) # type: ignore
	cell_weights = np.clip(cell_weights, 0.0, np.inf)
	cell_weights = cell_weights * (y_bin_widths[:, None] * x_bin_widths[None, :])  # probability mass per cell

	# Normalize cell probabilities
	total_probability_mass = float(cell_weights.sum())
	if total_probability_mass <= 0.0:
		raise ValueError("PPD has non-positive total probability mass after cleanup.")

	flattened_cell_probabilities = (cell_weights / total_probability_mass).ravel()

	# Sample cells
	flattened_cell_indices = rng.choice(
		flattened_cell_probabilities.size,
		size=num_samples,
		replace=True,
		p=flattened_cell_probabilities,
	)
	y_indices, x_indices = np.unravel_index(
		flattened_cell_indices,
		(n_y, n_x),
	)

	# Jitter uniformly within the selected cells
	x_jitter = rng.random(num_samples)
	y_jitter = rng.random(num_samples)

	x_samples = (
		x_edges[x_indices]
		+ x_jitter * x_bin_widths[x_indices]
	)
	y_samples = (
		y_edges[y_indices]
		+ y_jitter * y_bin_widths[y_indices]
	)

	# Clip values and return
	x_samples = np.clip(x_samples, x_edges[0], x_edges[-1])
	y_samples = np.clip(y_samples, y_edges[0], y_edges[-1])
	return x_samples, y_samples



def sample_redshifts(
	num_samples: int,
	rate_evolution_index: float,
	max_redshift: float = 2.0,
	redshift_grid_size: int = 20000,
	rng: np.random.Generator | None = None,
) -> np.ndarray:
	"""
	Sample redshifts z from the gwpopulation PowerLawRedshift model via inverse-CDF sampling.

	The model corresponds to an astrophysical merger-rate density evolving as
		R(z) ∝ (1+z)^{rate_evolution_index}
	and includes the standard factors dVc/dz and 1/(1+z).

	Args:
		num_samples (int): Number of redshift samples to draw.
		rate_evolution_index (float): The power-law index (often called 'lamb' in gwpopulation).
		max_redshift (float, optional): Maximum redshift. Defaults to 2.0.
		redshift_grid_size (int, optional): Size of sampling grid. Defaults to 20000.
		rng (np.random.Generator | None, optional): RNG for reproducibility. Defaults to None.

	Returns:
		np.ndarray: Array of sampled redshifts.

	Raises:
		ValueError: If inputs are invalid or the PDF cannot be normalized.
	"""
	if num_samples < 0:
		raise ValueError("num_samples must be >= 0")
	if max_redshift <= 0:
		raise ValueError("max_redshift must be > 0")
	if redshift_grid_size < 2:
		raise ValueError("redshift_grid_size must be >= 2")
	if not np.isfinite(rate_evolution_index):
		raise ValueError("rate_evolution_index must be finite")

	rng = np.random.default_rng() if rng is None else rng
	if num_samples == 0:
		return np.empty(0, dtype=float)

	redshift_model = PowerLawRedshift(z_max=max_redshift)

	redshift_grid = np.linspace(0.0, max_redshift, int(redshift_grid_size), dtype=float)
	redshift_grid[0] = 1e-8  # avoid exactly 0, just in case anything weird happens

	dataset = {"redshift": redshift_grid}

	# Get probability density on the grid
	pdf = redshift_model.probability(dataset=dataset, lamb=float(rate_evolution_index))
	pdf = np.asarray(pdf, dtype=float)

	# Defensive cleaning: remove NaNs/infs and forbid negative density
	pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
	pdf = np.clip(pdf, 0.0, np.inf)

	# Build CDF using cumulative trapezoidal integration (stable and accurate on a grid)
	dz = np.diff(redshift_grid)
	cdf = np.empty_like(redshift_grid)
	cdf[0] = 0.0
	cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * dz)

	total = cdf[-1]
	if not np.isfinite(total) or total <= 0.0:
		raise ValueError("Redshift PDF could not be normalized; integral is non-positive.")

	cdf /= total
	cdf = np.clip(cdf, 0.0, 1.0)

	# Ensure a valid inverse-CDF mapping: np.interp requires strictly increasing xp
	# (flat regions can happen if pdf==0 over an interval).
	cdf, unique_idx = np.unique(cdf, return_index=True)
	redshift_grid = redshift_grid[unique_idx]

	# If everything collapsed (pathological), fail explicitly.
	if cdf.size < 2:
		raise ValueError("CDF is degenerate; cannot sample redshifts from this configuration.")

	uniform_draws = rng.random(num_samples)
	return np.interp(uniform_draws, cdf, redshift_grid)


# The following code handles sampling redshift values from an existing Bilby result #
def load_lambda_samples(result_path: str) -> np.ndarray:
	"""
	Load posterior samples of the redshift-evolution index ('lamb') from a Bilby result file.

	Args:
		result_path (str): Path to a Bilby result file (e.g., .json).

	Returns:
		np.ndarray: 1D array of posterior samples for 'lamb'.

	Raises:
		FileNotFoundError: If the result file cannot be found/read.
		KeyError: If the 'lamb' parameter is not present in the result posterior.
		ValueError: If no finite samples are available.
	"""
	pop_result = read_in_result(result_path)

	if "lamb" not in pop_result.posterior:
		raise KeyError(f"'lamb' not found in result posterior. Available keys: {list(pop_result.posterior.columns)}")

	lambda_samples = np.asarray(pop_result.posterior["lamb"].to_numpy()).reshape(-1)
	lambda_samples = lambda_samples[np.isfinite(lambda_samples)]

	if lambda_samples.size == 0:
		raise ValueError("No finite 'lamb' samples found in result posterior.")

	return lambda_samples


def draw_lambda(lambda_samples: np.ndarray, rng: np.random.Generator | None = None) -> float:
	"""
	Pick a rate evolution index lambda given an array of samples.

	Args:
		lambda_samples (np.ndarray): Array of samples.
		rng (np.random.Generator): Random number generator used for selection.

	Returns:
		float: Randomly selected rate evolution index.

	Raises:
		ValueError: If lambda_samples is empty or contains no finite values.
	"""
	rng = np.random.default_rng() if rng is None else rng

	lambda_samples = np.asarray(lambda_samples).reshape(-1)
	lambda_samples = lambda_samples[np.isfinite(lambda_samples)]

	if lambda_samples.size == 0:
		raise ValueError("lambda_samples is empty or contains no finite values.")

	return float(rng.choice(lambda_samples))


#src/analyses/PowerLawPeak/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json

def sample_redshifts_from_result(result_path: str,
								num_samples: int,
								max_redshift: float = 2.0,
								redshift_grid_size: int = 20000,
								rng: np.random.Generator | None = None) -> np.ndarray:
	"""
	Sample redshift values from a result file.

	Args:
		result_path (str): Path to file containing redshift results
		num_samples (int): Number of samples desired
		max_redshift (float, optional): Maximum redshift. Defaults to 2.0.
		redshift_grid_size (int, optional): Size of sampling grid. Defaults to 20000.
		rng (np.random.Generator | None, optional): RNG for reproducibility. Defaults to None.

	Returns:
		np.ndarray: Array of sampled redshifts.
	"""
	rng = np.random.default_rng() if rng is None else rng
	lambda_samples = load_lambda_samples(result_path)
	lamb = draw_lambda(lambda_samples, rng)
	return sample_redshifts(
		num_samples=num_samples,
		rate_evolution_index=lamb,
		max_redshift=max_redshift,
		redshift_grid_size=redshift_grid_size,
		rng=rng,
	)


def chirp_mass_from_mass_1_and_ratio(mass_1: float, mass_ratio: float) -> float:
	mass_2 = mass_1 * mass_ratio
	return (mass_1 * mass_2) ** (3.0 / 5.0) / (mass_1 + mass_2) ** (1.0 / 5.0)

def masses_from_chirp_mass_and_mass_ratio(chirp_mass: np.ndarray, mass_ratio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # m1 = Mc * (1+q)^(1/5) / q^(3/5), m2 = q*m1
    mass_1 = chirp_mass * (1.0 + mass_ratio) ** (1.0 / 5.0) / np.maximum(mass_ratio, 1e-12) ** (3.0 / 5.0)
    mass_2 = mass_ratio * mass_1
    return mass_1, mass_2


# def simulate_mass_measurement_posterior(
#     true_mass_1: float,
#     true_mass_ratio: float,
#     num_posterior_samples: int = 4000,
#     chirp_mass_fractional_sigma: float = 0.03,     # ~3% in Mc (toy)
#     mass_ratio_sigma: float = 0.08,                # absolute sigma in q (toy)
#     mc_q_correlation: float = -0.6,
#     mass_1_bounds: Tuple[float, float] = (2.0, 100.0),
#     mass_ratio_bounds: Tuple[float, float] = (0.1, 1.0),
#     random_number_generator: np.random.Generator | None = None,
# ) -> Dict[str, np.ndarray]:
# 	return


def draw_extrinsics(geocenter_time: float, rng: np.random.Generator | None = None,) -> Dict[str, float]:
	"""
	Randomly generate extrinsic parameters for a merger event.

	Args:
		geocenter_time (float): GPS time of the merger.
		rng (np.random.Generator | None, optional): RNG for reproducibility. Defaults to None.

	Returns:
		dict: Extrinsic parameters [right ascension,
							declination,
							inclination,
							polarization,
							phase_angle,
							geocenter_time]
	"""
	rng = np.random.default_rng() if rng is None else rng

	return dict(
        ra=rng.uniform(0, 2*np.pi),
        dec=np.arcsin(rng.uniform(-1, 1)),
        theta_jn=np.arccos(rng.uniform(-1, 1)),
        psi=rng.uniform(0, np.pi),
        phase=rng.uniform(0, 2*np.pi),
        geocent_time=geocenter_time,
    )


def build_injection_from_mc(mass_1_source: float,
							mass_ratio: float,
							spin_magnitude_1: float, 
							spin_magnitude_2: float,
							redshift: float,
                            geocent_time=1126259462.4,
							rng: np.random.Generator | None = None) -> Dict[str, float]:
	"""
	Create a Bilby injection dict given source parameters.

	Args:
		mass_1_source (float): Source-frame mass of the larger black hole.
		mass_ratio (float): Black hole mass ratio.
		spin_magnitude_1 (float): Magnitude of the first black hole's spin.
		spin_magnitude_2 (float): Magnitude of the second black hole's spin.
		redshift (float): Redshift value of the merger.
		geocent_time (float, optional): Time of the merger.. Defaults to 1126259462.4.
		rng (np.random.Generator | None, optional): RNG for reproducibility. Defaults to None.

	Raises:
		ValueError: need to make sure the values are good. man idk what you want here

	Returns:
		Dict[str, float]: Dictionary describing the injection for Bilby.
	"""
	# finiteness checks
	for name, val in [
		("mass_1_source", mass_1_source),
		("mass_ratio", mass_ratio),
		("spin_magnitude_1", spin_magnitude_1),
		("spin_magnitude_2", spin_magnitude_2),
		("redshift", redshift),
		("geocent_time", geocent_time),
	]:
		if not np.isfinite(val):
			raise ValueError(f"{name} must be finite, got {val}")

	# range/physics checks
	if redshift < 0:
		raise ValueError(f"redshift must be >= 0, got {redshift}")

	if not (0.0 <= spin_magnitude_1 <= 1.0):
		raise ValueError(f"spin_magnitude_1 must be in [0,1], got {spin_magnitude_1}")
	if not (0.0 <= spin_magnitude_2 <= 1.0):
		raise ValueError(f"spin_magnitude_2 must be in [0,1], got {spin_magnitude_2}")

	if mass_ratio <= 0.0 or mass_ratio > 1.0:
		raise ValueError(f"mass_ratio must be in (0,1], got {mass_ratio}")

	rng = np.random.default_rng() if rng is None else rng
	mass_2_source = mass_ratio * mass_1_source

	# calculate distance from redshift
	dL = cosmo.luminosity_distance(redshift).to(u.Mpc).value # type: ignore

	# source -> detector frame
	#TODO: do we need to do this????
	mass_1_det = mass_1_source * (1 + redshift)
	mass_2_det = mass_2_source * (1 + redshift)

	# isotropic spin directions (needed for generic-precessing models)
	tilt_1 = np.arccos(rng.uniform(-1, 1))
	tilt_2 = np.arccos(rng.uniform(-1, 1))
	phi_12 = rng.uniform(0, 2*np.pi)
	phi_jl = rng.uniform(0, 2*np.pi)

	inj = dict(
		mass_1=mass_1_det,
		mass_2=mass_2_det,
		a_1=float(spin_magnitude_1),
		a_2=float(spin_magnitude_2),
		tilt_1=float(tilt_1),
		tilt_2=float(tilt_2),
		phi_12=float(phi_12),
		phi_jl=float(phi_jl),
		luminosity_distance=float(dL),
		**draw_extrinsics(geocent_time, rng)
	)
	return inj


Injection = Dict[str, float]
Bounds = Tuple[float, float]

def generate_injection_catalog(
	n_events: int,
	mass_ppd_path: str,
	spin_ppd_path: str,
	bilby_result_path_for_lambda: str,
	m1_bounds: Bounds = (2.0, 100.0),
	q_bounds: Bounds = (0.1, 1.0),
	a_bounds: Bounds = (0.0, 1.0),
	z_max: float = 2.0,
	time0: float = 1126259462.4,
	dt: float = 10.0,
	rng: np.random.Generator | None = None,
) -> Tuple[List[Injection], float]:
	"""
	Generate a Monte Carlo catalog of bilby-compatible injection dictionaries.

	This function performs the "population-level" Monte Carlo:
		1) Draw a redshift-evolution index (lambda) from a Bilby population result posterior
		2) Sample (m1_source, q) from a 2D PPD grid stored in an HDF5 file
		3) Sample (a1, a2) from a 2D PPD grid stored in an HDF5 file
		4) Sample redshifts z from a PowerLawRedshift model conditioned on lambda
		5) Convert each draw into an injection dictionary via build_injection_from_mc

	Note:
		This draws ONE lambda for the entire catalog. If you want to marginalize
		over lambda per-event, draw lambda inside the loop.

	Args:
		n_events (int): Number of injections to generate.
		mass_ppd_path (str): Path to HDF5 file containing a 2D PPD for (m1_source, q).
		spin_ppd_path (str): Path to HDF5 file containing a 2D PPD for (a1, a2).
		bilby_result_path_for_lambda (str): Path to a Bilby result file containing posterior samples of 'lamb'.
		m1_bounds (Tuple[float, float], optional): Bounds for m1_source used to interpret the mass PPD grid. Defaults to (2.0, 100.0).
		q_bounds (Tuple[float, float], optional): Bounds for mass ratio q used to interpret the mass PPD grid. Defaults to (0.1, 1.0).
		a_bounds (Tuple[float, float], optional): Bounds for spin magnitudes used to interpret the spin PPD grid. Defaults to (0.0, 1.0).
		z_max (float, optional): Maximum redshift for redshift sampling. Defaults to 2.0.
		time0 (float, optional): Reference geocent_time (GPS seconds) for the first injection. Defaults to 1126259462.4.
		dt (float, optional): Time spacing (seconds) between consecutive injections. Defaults to 10.0.
		rng (np.random.Generator | None, optional): RNG for reproducibility. Defaults to None.

	Returns:
		Tuple[List[Dict[str, float]], float]:
			- injections: List of bilby-compatible injection dictionaries (length n_events)
			- lamb: The lambda value used to sample all redshifts in this catalog

	Raises:
		ValueError: If n_events is negative, or if z_max/dt are not positive finite values.
		KeyError / FileNotFoundError / ValueError: If the lambda posterior cannot be loaded.
		Any exception raised by sample_from_2D_ppd(), sample_redshifts(), draw_lambda(),
		or build_injection_from_mc() will propagate.
	"""
	rng = np.random.default_rng() if rng is None else rng

	# Draw one lambda for the whole catalog for now
	#TODO: marginalize over lambda?
	lambda_samples = load_lambda_samples(bilby_result_path_for_lambda)
	lamb = draw_lambda(lambda_samples, rng=rng)

	# Draw intrinsic parameters from your PPDs
	mass_1_source, mass_ratios = sample_from_2D_ppd(mass_ppd_path, n_events, x_bounds=m1_bounds, y_bounds=q_bounds)
	spin_magnitudes_1, spin_magnitudes_2 = sample_from_2D_ppd(spin_ppd_path, n_events, x_bounds=a_bounds, y_bounds=a_bounds)

	# Draw redshifts conditional on lambda
	redshifts = sample_redshifts(n_events, rate_evolution_index=lamb, max_redshift=z_max, rng=rng)

	# Build bilby-compatible injection dicts
	injections = []
	for i in range(n_events):
		geocent_time = time0 + i * dt
		inj = build_injection_from_mc(
			mass_1_source=float(mass_1_source[i]),
			mass_ratio=float(mass_ratios[i]),
			spin_magnitude_1=float(spin_magnitudes_1[i]),
			spin_magnitude_2=float(spin_magnitudes_2[i]),
			redshift=float(redshifts[i]),
			geocent_time=geocent_time,
			rng=rng,
		)
		injections.append(inj)

	return injections, lamb