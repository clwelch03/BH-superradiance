from __future__ import annotations
from typing import Tuple
from gwpopulation.models.redshift import PowerLawRedshift
from bilby.core.result import read_in_result
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import h5py


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
                       y_bounds: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
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

	rng = np.random.default_rng() 

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

