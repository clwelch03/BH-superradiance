import numpy as np
import pytest

from gwpopulation.models.redshift import PowerLawRedshift
from mc_pipeline.redshift import sample_redshifts


def test_redshift_sampling_histogram_overlay(tmp_path):
    # Import matplotlib only for this test (keeps rest of suite lighter)
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")  # must be set before importing pyplot
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(123)

    lamb = 2.0
    z_max = 2.0

    # draw samples (reduced from 200k/50k to be more CI-friendly)
    z_samples = sample_redshifts(
        num_samples=50_000,
        rate_evolution_index=lamb,
        max_redshift=z_max,
        redshift_grid_size=10_000,
        rng=rng,
    )

    assert np.all(np.isfinite(z_samples))
    assert z_samples.min() >= 0.0
    assert z_samples.max() <= z_max + 1e-12

    # compute model pdf for overlay
    model = PowerLawRedshift(z_max=z_max)
    z_grid = np.linspace(0.0, z_max, 5000)
    z_grid[0] = 1e-8
    pdf = np.asarray(model.probability(dataset={"redshift": z_grid}, lamb=lamb), dtype=float)

    # normalize pdf defensively for comparisons (in case model returns unnormalized density)
    area = np.trapezoid(pdf, z_grid)
    assert np.isfinite(area) and area > 0
    pdf_norm = pdf / area

    # Numerical “close to reality” check:
    # compare histogram density to model density using an L1 distance on bins
    bins = 80
    hist, edges = np.histogram(z_samples, bins=bins, range=(0.0, z_max), density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    pdf_at_centers = np.interp(centers, z_grid, pdf_norm)
    bin_widths = np.diff(edges)

    l1 = float(np.sum(np.abs(hist - pdf_at_centers) * bin_widths))
    assert l1 < 0.12  # loose threshold to avoid flakiness but still catch regressions

    # plot (kept, since you want it)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        z_samples,
        bins=bins,
        range=(0.0, z_max),
        density=True,
        histtype="step",
        linewidth=2,
        label="samples",
    )
    ax.plot(z_grid, pdf_norm, linewidth=2, label="model pdf")
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

    z1 = sample_redshifts(1000, lamb, max_redshift=z_max, redshift_grid_size=5000, rng=rng1)
    z2 = sample_redshifts(1000, lamb, max_redshift=z_max, redshift_grid_size=5000, rng=rng2)

    # should be exactly identical given same RNG + deterministic code path
    assert np.array_equal(z1, z2)