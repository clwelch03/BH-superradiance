from __future__ import annotations
from typing import Tuple, Dict
from gwpopulation.models.redshift import PowerLawRedshift
from bilby.core.result import read_in_result
import numpy as np
import h5py
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

rng=np.random.default_rng(0)

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


def draw_extrinsics(geocenter_time: float) -> Dict[str, float]:
	"""
	Randomly generate extrinsic parameters for a merger event.

	Args:
		geocenter_time (_type_): GPS time of the merger.

	Returns:
		dict: Extrinsic parameters [right ascension,
							declination,
							inclination,
							polarization,
							phase_angle,
							geocenter_time]
	"""
	return dict(
        ra=rng.uniform(0, 2*np.pi),
        dec=np.arcsin(rng.uniform(-1, 1)),
        theta_jn=np.arccos(rng.uniform(-1, 1)),
        psi=rng.uniform(0, np.pi),
        phase=rng.uniform(0, 2*np.pi),
        geocent_time=geocenter_time,
    )


def build_injection_from_mc(mass_1_source,
							mass_ratio,
							spin_magnitude_1, 
							spin_magnitude_2,
							redshift,
                            geocenter_time=1126259462.4):
	"""_summary_

	Args:
		mass_1_source (_type_): _description_
		mass_ratio (_type_): _description_
		spin_magnitude_1 (_type_): _description_
		spin_magnitude_2 (_type_): _description_
		redshift (_type_): _description_
		geocenter_time (float, optional): _description_. Defaults to 1126259462.4.

	Returns:
		_type_: _description_
	"""
	mass_2_source = mass_ratio * mass_1_source

	# enforce basic physical bounds we will also use in priors
	if mass_1_source < 2 or mass_1_source > 100:
		return None
	if mass_2_source < 2 or mass_2_source > mass_1_source:
		return None

	# choose redshift+distance (or just choose distance directly)
	dL = cosmo.luminosity_distance(redshift).to(u.Mpc).value # type: ignore

	# source -> detector frame
	#TODO: do we need to do this????
	m1_det = mass_1_source * (1 + redshift)
	m2_det = mass_2_source * (1 + redshift)

	# isotropic spin directions (needed for generic-precessing models)
	tilt_1 = np.arccos(rng.uniform(-1, 1))
	tilt_2 = np.arccos(rng.uniform(-1, 1))
	phi_12 = rng.uniform(0, 2*np.pi)
	phi_jl = rng.uniform(0, 2*np.pi)

	inj = dict(
		mass_1=m1_det,
		mass_2=m2_det,
		a_1=float(spin_magnitude_1),
		a_2=float(spin_magnitude_2),
		tilt_1=float(tilt_1),
		tilt_2=float(tilt_2),
		phi_12=float(phi_12),
		phi_jl=float(phi_jl),
		luminosity_distance=float(dL),
		**draw_extrinsics(geocenter_time),
	)
	return inj