import numpy as np
from numpy.typing import NDArray

from gwpopulation.models.redshift import PowerLawRedshift
from bilby.core.result import read_in_result
from typing import Tuple


def sample_redshifts(num_samples: int,
					 rate_evolution_index: float,
					 max_redshift: float = 2.0,
					 redshift_grid_size: int = 20000,
					 rng: np.random.Generator | None = None) -> NDArray[np.float64]:
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
		NDArray[np.float64]: Array of sampled redshifts.

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

	if num_samples == 0:
		return np.empty(0, dtype=float)

	# Instantiate rng if not passed
	rng = np.random.default_rng() if rng is None else rng

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
	# (flat regions can happen if pdf==0 over an interval)
	cdf, unique_idx = np.unique(cdf, return_index=True)
	redshift_grid = redshift_grid[unique_idx]

	# If everything collapsed (pathological), fail explicitly
	if cdf.size < 2:
		raise ValueError("CDF is degenerate; cannot sample redshifts from this configuration.")

	uniform_draws = rng.random(num_samples)
	return np.interp(uniform_draws, cdf, redshift_grid)

def load_lambda_samples(result_path: str) -> NDArray[np.float64]:
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


def draw_lambda(lambda_samples: NDArray[np.float64], rng: np.random.Generator = np.random.default_rng()) -> float:
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

	lambda_samples = np.asarray(lambda_samples).reshape(-1)
	lambda_samples = lambda_samples[np.isfinite(lambda_samples)]

	if lambda_samples.size == 0:
		raise ValueError("lambda_samples is empty or contains no finite values.")

	return float(rng.choice(lambda_samples))

def sample_redshifts_from_result(result_path: str,
								num_samples: int,
								max_redshift: float = 2.0,
								redshift_grid_size: int = 20000,
								rng: np.random.Generator | None = None) -> Tuple[NDArray[np.float64], float]:
	"""
	Sample redshift values from a result file.

	Args:
		result_path (str): Path to file containing redshift results
		num_samples (int): Number of samples desired
		max_redshift (float, optional): Maximum redshift. Defaults to 2.0.
		redshift_grid_size (int, optional): Size of sampling grid. Defaults to 20000.
		rng (np.random.Generator | None, optional): RNG for reproducibility.

	Returns:
		np.ndarray: Array of sampled redshifts.
		float: Sampled lambda value.
	"""
	# Instantiate rng if not passed
	rng = np.random.default_rng() if rng is None else rng
	
	lambda_samples = load_lambda_samples(result_path)
	lamb = draw_lambda(lambda_samples, rng)
	return sample_redshifts(
		num_samples=num_samples,
		rate_evolution_index=lamb,
		max_redshift=max_redshift,
		redshift_grid_size=redshift_grid_size,
		rng=rng,
	), lamb