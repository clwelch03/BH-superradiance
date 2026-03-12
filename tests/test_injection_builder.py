import numpy as np
import pytest

pytest.importorskip("astropy")

import astropy.units as u
from astropy.cosmology import Planck15 as cosmo

from mc_pipeline.injection_builder import draw_extrinsics, build_injection_from_mc


def test_draw_extrinsics_reproducible_and_in_ranges():
    seed = 123
    t0 = 1126259462.4

    e1 = draw_extrinsics(t0, rng=np.random.default_rng(seed))
    e2 = draw_extrinsics(t0, rng=np.random.default_rng(seed))

    # reproducible
    for k in e1:
        assert np.isclose(e1[k], e2[k])

    # required keys + fixed time
    assert set(e1.keys()) == {"ra", "dec", "theta_jn", "psi", "phase", "geocent_time"}
    assert e1["geocent_time"] == t0

    # ranges
    assert 0.0 <= e1["ra"] <= 2 * np.pi
    assert -0.5 * np.pi <= e1["dec"] <= 0.5 * np.pi
    assert 0.0 <= e1["theta_jn"] <= np.pi
    assert 0.0 <= e1["psi"] <= np.pi
    assert 0.0 <= e1["phase"] <= 2 * np.pi


def test_build_injection_from_mc_basic_sanity():
    rng = np.random.default_rng(0)
    m1_src = 30.0
    q = 0.5
    z = 0.2

    inj = build_injection_from_mc(
        mass_1_source=m1_src,
        mass_ratio=q,
        spin_magnitude_1=0.3,
        spin_magnitude_2=0.7,
        redshift=z,
        geocent_time=100.0,
        rng=rng,
    )

    # required keys (core + extrinsics)
    for k in [
        "mass_1", "mass_2", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
        "luminosity_distance", "ra", "dec", "theta_jn", "psi", "phase", "geocent_time",
    ]:
        assert k in inj

    # masses: source -> detector frame
    m2_src = q * m1_src
    assert np.isclose(inj["mass_1"], m1_src * (1 + z))
    assert np.isclose(inj["mass_2"], m2_src * (1 + z))

    # spins passed through
    assert np.isclose(inj["a_1"], 0.3)
    assert np.isclose(inj["a_2"], 0.7)

    # angle ranges
    assert 0.0 <= inj["tilt_1"] <= np.pi
    assert 0.0 <= inj["tilt_2"] <= np.pi
    assert 0.0 <= inj["phi_12"] <= 2 * np.pi
    assert 0.0 <= inj["phi_jl"] <= 2 * np.pi

    # distance matches astropy cosmology
    dL_expected = cosmo.luminosity_distance(z).to(u.Mpc).value  # type: ignore
    assert np.isclose(inj["luminosity_distance"], dL_expected)


def test_build_injection_from_mc_reproducible_given_seed():
    kwargs = dict(
        mass_1_source=35.0,
        mass_ratio=0.8,
        spin_magnitude_1=0.1,
        spin_magnitude_2=0.2,
        redshift=0.4,
        geocent_time=123.456,
    )

    inj1 = build_injection_from_mc(**kwargs, rng=np.random.default_rng(999))
    inj2 = build_injection_from_mc(**kwargs, rng=np.random.default_rng(999))

    assert inj1.keys() == inj2.keys()
    for k in inj1:
        assert np.isclose(inj1[k], inj2[k])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mass_1_source=30.0, mass_ratio=0.5, spin_magnitude_1=0.2, spin_magnitude_2=0.2, redshift=-0.1),
        dict(mass_1_source=30.0, mass_ratio=0.5, spin_magnitude_1=1.2, spin_magnitude_2=0.2, redshift=0.1),
        dict(mass_1_source=30.0, mass_ratio=0.5, spin_magnitude_1=0.2, spin_magnitude_2=-0.1, redshift=0.1),
        dict(mass_1_source=30.0, mass_ratio=0.0, spin_magnitude_1=0.2, spin_magnitude_2=0.2, redshift=0.1),
        dict(mass_1_source=30.0, mass_ratio=1.1, spin_magnitude_1=0.2, spin_magnitude_2=0.2, redshift=0.1),
        dict(mass_1_source=np.nan, mass_ratio=0.5, spin_magnitude_1=0.2, spin_magnitude_2=0.2, redshift=0.1),
    ],
)
def test_build_injection_from_mc_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        build_injection_from_mc(**kwargs, rng=np.random.default_rng(0))