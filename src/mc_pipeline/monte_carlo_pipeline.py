from __future__ import annotations
from typing import Tuple, Dict, List, Mapping
from gwpopulation.models.redshift import PowerLawRedshift

from bilby.core.result import read_in_result
from bilby.gw.detector import InterferometerList
from bilby.gw.waveform_generator import WaveformGenerator
from bilby.gw.source import lal_binary_black_hole

import numpy as np
import h5py
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

from mc_pipeline.ppd import sample_from_2D_ppd
from mc_pipeline.redshift import sample_redshifts_from_result
from mc_pipeline.injection import draw_extrinsics, build_injection_from_mc

# def chirp_mass_from_mass_1_and_ratio(mass_1: float, mass_ratio: float) -> float:
# 	mass_2 = mass_1 * mass_ratio
# 	return (mass_1 * mass_2) ** (3.0 / 5.0) / (mass_1 + mass_2) ** (1.0 / 5.0)

# def masses_from_chirp_mass_and_mass_ratio(chirp_mass: np.ndarray, mass_ratio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     # m1 = Mc * (1+q)^(1/5) / q^(3/5), m2 = q*m1
#     mass_1 = chirp_mass * (1.0 + mass_ratio) ** (1.0 / 5.0) / np.maximum(mass_ratio, 1e-12) ** (3.0 / 5.0)
#     mass_2 = mass_ratio * mass_1
#     return mass_1, mass_2


Injection = Dict[str, float]
Bounds = Tuple[float, float]

### check SNR ###
def _make_snr_context(
	duration: float,
	fs: float,
	fmin: float,
	approximant: str,
	detectors: Tuple[str, ...],
):
	import bilby
	from bilby.gw.detector import InterferometerList
	from bilby.gw.waveform_generator import WaveformGenerator

	ifos = InterferometerList(list(detectors))
	ifos.set_strain_data_from_power_spectral_densities(
		sampling_frequency=fs,
		duration=duration,
		start_time=0,  # arbitrary; SNR uses |h(f)| so absolute time origin doesn't matter
	)
	for ifo in ifos:
		ifo.minimum_frequency = fmin

	wfg = WaveformGenerator(
		duration=duration,
		sampling_frequency=fs,
		frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
		waveform_arguments=dict(
			waveform_approximant=approximant,
			reference_frequency=50.0,
			minimum_frequency=fmin,
		),
	)
	return ifos, wfg


def _network_optimal_snr(inj: Injection, ifos, wfg) -> float:
	polarizations = wfg.frequency_domain_strain(parameters=inj)

	snr2 = 0.0
	for ifo in ifos:
		signal = ifo.get_detector_response(polarizations=polarizations, parameters=inj)
		snr2 += float(ifo.optimal_snr_squared(signal))
	return float(np.sqrt(snr2))


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
	snr_threshold: float = 10.0,
	rng: np.random.Generator | None = None,
) -> Tuple[List[Dict[str, float]], float]:
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
		snr_threshold (float, optional): Signal-to-noise ratio threshold; all injections will be above this.
		rng (np.random.Generator | None, optional): RNG for reproducibility. Defaults to None.

	Returns:
		Tuple[List[Dict[str, float]], float]:
			- injections: List of bilby-compatible injection dictionaries (length n_events)
			- lamb: The lambda value used to sample all redshifts in this catalog
	"""
	rng = np.random.default_rng() if rng is None else rng

	# Draw intrinsic parameters from your PPDs
	mass_1_source, mass_ratios = sample_from_2D_ppd(mass_ppd_path, n_events, x_bounds=m1_bounds, y_bounds=q_bounds)
	spin_magnitudes_1, spin_magnitudes_2 = sample_from_2D_ppd(spin_ppd_path, n_events, x_bounds=a_bounds, y_bounds=a_bounds)

	# Draw redshifts conditional on lambda
	# Draw one lambda for the whole catalog for now
	#TODO: marginalize over lambda?
	redshifts, lamb = sample_redshifts_from_result(bilby_result_path_for_lambda, n_events, max_redshift=z_max, rng=rng)

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