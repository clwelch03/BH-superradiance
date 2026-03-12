import numpy as np
from typing import Dict, Tuple, List

from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

from mc_pipeline.ppd import sample_from_2D_ppd
from mc_pipeline.redshift import sample_redshifts_from_result
from mc_pipeline.snr import GWContext, GWSettings

Injection = Dict[str, float]
Bounds = Tuple[float, float]

def draw_extrinsics(geocenter_time: float, rng: np.random.Generator | None = None) -> Dict[str, float]:
	"""
	Randomly generate extrinsic parameters for a merger event.

	Args:
		geocenter_time (float): GPS time of the merger.
		rng (np.random.Generator, optional): RNG for reproducibility.

	Returns:
		dict: Extrinsic parameters [right ascension,
							declination,
							inclination,
							polarization,
							phase_angle,
							geocenter_time]
	"""
	# Instantiate rng if not passed
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
							rng: np.random.Generator | None = None) -> Injection:
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
		ValueError: Values must be physically sensible (positive, finite, etc.)

	Returns:
		Dict[str, float]: Dictionary describing the injection for Bilby.
	"""
	# Finiteness checks
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

	# Instantiate rng if not passed
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

	injection = dict(
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
	return injection

def generate_injection_catalog(n_events: int,
							   mass_ppd_path: str,
							   spin_ppd_path: str,
							   bilby_result_path_for_lambda: str,
							   gw_context_settings: GWSettings,
							   m1_bounds: Bounds = (2.0, 100.0),
							   q_bounds: Bounds = (0.1, 1.0),
							   a_bounds: Bounds = (0.0, 1.0),
							   z_max: float = 1.0,
							   time0: float = 1126259462.4,
							   dt: float = 10.0,
							   snr_threshold: float = 10.0,
							   rng: np.random.Generator | None = None,
							   verbose: bool = False) -> Tuple[List[Injection], List[float], float]:
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
		gw_context_settings (GWSettings): Class containing duration, sampling frequency, waveform frequencies, approximant, and interferometers.
		m1_bounds (Tuple[float, float], optional): Bounds for m1_source used to interpret the mass PPD grid. Defaults to (2.0, 100.0).
		q_bounds (Tuple[float, float], optional): Bounds for mass ratio q used to interpret the mass PPD grid. Defaults to (0.1, 1.0).
		a_bounds (Tuple[float, float], optional): Bounds for spin magnitudes used to interpret the spin PPD grid. Defaults to (0.0, 1.0).
		z_max (float, optional): Maximum redshift for redshift sampling. Defaults to 2.0.
		time0 (float, optional): Reference geocent_time (GPS seconds) for the first injection. Defaults to 1126259462.4.
		dt (float, optional): Time spacing (seconds) between consecutive injections. Defaults to 10.0.
		snr_threshold (float, optional): Signal-to-noise ratio threshold; all injections will be above this.
		rng (np.random.Generator | None, optional): RNG for reproducibility.
		verbose (bool): Toggle for extra debug output. Defaults to False.

	Raises:
		RuntimeError: If the SNR cut is too high (or the max redshift is too high) to generate events that meet our criteria.

	Returns:
		Tuple[List[Injection], List[float], float]:
			- injections: Bilby-compatible injection dictionaries (length n_events)
			- snrs: Signal-to-noise ratios for each injection
			- lamb: Lambda value used to sample all redshifts in this catalog
	"""
	# Instantiate rng if not passed
	rng = np.random.default_rng() if rng is None else rng
	
	# Build GWContext from provided GW settings
	gw_context = GWContext(gw_context_settings)

	# Calculate the maximum allowed draws based on n_events
	max_draws = 100*n_events

	# Draw a pool of candidate full draws, then reject/accept until we have n_events
	# One lambda for the whole catalog (kept consistent because we call this once)
	redshifts, lamb = sample_redshifts_from_result(
		bilby_result_path_for_lambda,
		max_draws,
		max_redshift=z_max,
		rng=rng,
	)

	# Candidate intrinsic parameters
	mass_1_source, mass_ratios = sample_from_2D_ppd(mass_ppd_path, max_draws, x_bounds=m1_bounds, y_bounds=q_bounds)
	spin_magnitudes_1, spin_magnitudes_2 = sample_from_2D_ppd(spin_ppd_path, max_draws, x_bounds=a_bounds, y_bounds=a_bounds)

	# Build bilby-compatible injection dicts
	injections: List[Injection] = []
	snrs: List[float] = []

	candidate_idx = 0

	#TODO: later, we should chunk proposal pools (draw 256–1024 candidates at a time)
	for i in range(n_events):
		geocent_time = time0 + i * dt
		success = False # we repeat until we generate a high-SNR event

		while not success:
			if candidate_idx >= max_draws:
				raise RuntimeError(
					f"Ran out of candidate draws (max_draws={max_draws}) before collecting "
					f"{n_events} injections with SNR >= {snr_threshold}. "
					f"Lower snr_threshold or z_max."
				)

			# Build a candidate injection
			injection = build_injection_from_mc(
				mass_1_source=float(mass_1_source[candidate_idx]),
				mass_ratio=float(mass_ratios[candidate_idx]),
				spin_magnitude_1=float(spin_magnitudes_1[candidate_idx]),
				spin_magnitude_2=float(spin_magnitudes_2[candidate_idx]),
				redshift=float(redshifts[candidate_idx]),
				geocent_time=geocent_time,
				rng=rng
			)
			candidate_idx += 1

			# Calculate SNR for the injection candidate
			injection_snr = gw_context.network_optimal_snr(injection)
			if verbose:
				print(f"Injection candidate has SNR {injection_snr}")

			# If the injection is above the SNR threshold, add it to the list
			if injection_snr > snr_threshold:
				injections.append(injection)
				snrs.append(injection_snr)
				success = True

	return injections, snrs, lamb