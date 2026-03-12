import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for CI/servers
import matplotlib.pyplot as plt

from gwpopulation.models.redshift import PowerLawRedshift
from mc_pipeline.monte_carlo_pipeline import sample_redshifts  # thanks to pytest.ini pythonpath=src

def test_redshift_sampling_histogram_overlay(tmp_path):
    rng = np.random.default_rng(123)

    lamb = 2.0
    z_max = 2.0

    # draw samples
    z_samples = sample_redshifts(
        num_samples=200_000,
        rate_evolution_index=lamb,
        max_redshift=z_max,
        redshift_grid_size=50_000,
        rng=rng,
    )

    assert np.all(np.isfinite(z_samples))
    assert z_samples.min() >= 0.0
    assert z_samples.max() <= z_max

    # compute model pdf for overlay
    model = PowerLawRedshift(z_max=z_max)
    z_grid = np.linspace(0.0, z_max, 5000)
    z_grid[0] = 1e-8
    pdf = model.probability(dataset={"redshift": z_grid}, lamb=lamb)

    # plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(z_samples, bins=80, range=(0.0, z_max), density=True,
            histtype="step", linewidth=2, label="samples")
    ax.plot(z_grid, pdf, linewidth=2, label="model pdf")
    ax.set_xlabel("redshift z")
    ax.set_ylabel("p(z)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(tmp_path / "redshift_sampling_overlay.png", dpi=150)
    plt.close(fig)
    

def test_redshift_sampling_reproducible_given_seed():
    seed = 999
    lamb = 1.5
    z_max = 2.0

    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)

    z1 = sample_redshifts(1000, lamb, max_redshift=z_max, rng=rng1)
    z2 = sample_redshifts(1000, lamb, max_redshift=z_max, rng=rng2)

    assert np.allclose(z1, z2)