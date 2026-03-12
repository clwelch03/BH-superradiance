import numpy as np
from typing import Dict

from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

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