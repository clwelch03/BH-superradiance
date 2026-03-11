import numpy as np
import pytest
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

from monte_carlo_pipeline import draw_extrinsics, build_injection_from_mc


REQUIRED_EXTRINSIC_KEYS = {"ra", "dec", "theta_jn", "psi", "phase", "geocent_time"}
REQUIRED_INJECTION_KEYS = {
    "mass_1", "mass_2",
    "a_1", "a_2",
    "tilt_1", "tilt_2", "phi_12", "phi_jl",
    "luminosity_distance",
    "ra", "dec", "theta_jn", "psi", "phase", "geocent_time",
}


def test_draw_extrinsics_keys_and_ranges():
    rng = np.random.default_rng(123)
    t = 1126259462.4
    ext = draw_extrinsics(t, rng=rng)

    assert REQUIRED_EXTRINSIC_KEYS.issubset(ext.keys())
    assert ext["geocent_time"] == t

    assert 0.0 <= ext["ra"] < 2*np.pi
    assert -np.pi/2 <= ext["dec"] <= np.pi/2
    assert 0.0 <= ext["theta_jn"] <= np.pi
    assert 0.0 <= ext["psi"] <= np.pi
    assert 0.0 <= ext["phase"] < 2*np.pi

    for k, v in ext.items():
        assert np.isfinite(v), f"{k} not finite"


def test_build_injection_valid_event_has_keys_and_is_finite():
    rng = np.random.default_rng(1)
    z = 0.5
    m1_src = 30.0
    q = 0.8

    inj = build_injection_from_mc(
        mass_1_source=m1_src,
        mass_ratio=q,
        spin_magnitude_1=0.4,
        spin_magnitude_2=0.2,
        redshift=z,
        geocent_time=1126259462.4,
        rng=rng,
    )

    assert REQUIRED_INJECTION_KEYS.issubset(inj.keys())
    for k in REQUIRED_INJECTION_KEYS:
        assert np.isfinite(inj[k]), f"{k} not finite: {inj[k]}"

    assert inj["mass_1"] >= inj["mass_2"] > 0.0
    assert 0.0 <= inj["a_1"] <= 1.0
    assert 0.0 <= inj["a_2"] <= 1.0
    assert inj["luminosity_distance"] > 0.0

    # angle ranges
    assert 0.0 <= inj["tilt_1"] <= np.pi
    assert 0.0 <= inj["tilt_2"] <= np.pi
    assert 0.0 <= inj["phi_12"] < 2*np.pi
    assert 0.0 <= inj["phi_jl"] < 2*np.pi


def test_build_injection_source_to_detector_mass_conversion():
    rng = np.random.default_rng(2)
    z = 1.0
    m1_src = 20.0
    q = 0.5
    m2_src = m1_src * q

    inj = build_injection_from_mc(
        mass_1_source=m1_src,
        mass_ratio=q,
        spin_magnitude_1=0.1,
        spin_magnitude_2=0.2,
        redshift=z,
        rng=rng,
    )

    assert np.isclose(inj["mass_1"], m1_src * (1.0 + z))
    assert np.isclose(inj["mass_2"], m2_src * (1.0 + z))


def test_build_injection_distance_matches_astropy_cosmology():
    rng = np.random.default_rng(3)
    z = 0.3

    inj = build_injection_from_mc(
        mass_1_source=35.0,
        mass_ratio=0.7,
        spin_magnitude_1=0.3,
        spin_magnitude_2=0.6,
        redshift=z,
        rng=rng,
    )

    expected = cosmo.luminosity_distance(z).to(u.Mpc).value # type: ignore
    assert np.isclose(inj["luminosity_distance"], expected)


def test_build_injection_reproducible_given_seed():
    seed = 999
    args = dict(
        mass_1_source=35.0,
        mass_ratio=0.7,
        spin_magnitude_1=0.3,
        spin_magnitude_2=0.6,
        redshift=0.3,
        geocent_time=1126259462.4,
    )

    inj1 = build_injection_from_mc(**args, rng=np.random.default_rng(seed))
    inj2 = build_injection_from_mc(**args, rng=np.random.default_rng(seed))

    # compare numerically (don’t rely on dict equality semantics)
    assert inj1.keys() == inj2.keys()
    for k in inj1.keys():
        assert np.isclose(inj1[k], inj2[k]), f"Mismatch in key {k}: {inj1[k]} vs {inj2[k]}"


# def test_build_injection_raises_on_invalid_inputs():
#     rng = np.random.default_rng(4)

#     with pytest.raises(ValueError):
#         build_injection_from_mc(30.0, 1.2, 0.2, 0.2, 0.1, rng=rng)  # q>1

#     with pytest.raises(ValueError):
#         build_injection_from_mc(30.0, 0.8, 1.2, 0.2, 0.1, rng=rng)  # a1>1

#     with pytest.raises(ValueError):
#         build_injection_from_mc(30.0, 0.8, 0.2, 0.2, -0.1, rng=rng)  # z<0

#     with pytest.raises(ValueError):
#         build_injection_from_mc(1.0, 0.8, 0.2, 0.2, 0.1, rng=rng)    # m1 too small

#     with pytest.raises(ValueError):
#         build_injection_from_mc(10.0, 0.1, 0.2, 0.2, 0.1, rng=rng)   # m2 < 2 (since m2=1)