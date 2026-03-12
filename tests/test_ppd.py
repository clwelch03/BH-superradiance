import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from mc_pipeline.ppd import bounds_to_edges, sample_from_2D_ppd


def _write_ppd(h5_path, ppd_array: np.ndarray) -> None:
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("ppd", data=ppd_array)


def test_bounds_to_edges_known_answer():
    edges = bounds_to_edges(0.0, 10.0, 6)
    expected = np.array([0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 10.0])
    assert np.allclose(edges, expected)
    assert edges.shape == (7,)


def test_sample_from_2D_ppd_reproducible_and_in_bounds(tmp_path):
    ppd = np.ones((5, 4), dtype=float)  # (n_y, n_x)
    h5_path = tmp_path / "ppd.h5"
    _write_ppd(h5_path, ppd)

    x_bounds = (0.0, 3.0)
    y_bounds = (-1.0, 1.0)
    seed = 123

    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)

    x1, y1 = sample_from_2D_ppd(str(h5_path), 2000, x_bounds, y_bounds, rng=rng1)
    x2, y2 = sample_from_2D_ppd(str(h5_path), 2000, x_bounds, y_bounds, rng=rng2)

    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)

    assert x1.shape == (2000,)
    assert y1.shape == (2000,)
    assert np.all(np.isfinite(x1)) and np.all(np.isfinite(y1))
    assert x1.min() >= x_bounds[0] - 1e-12 and x1.max() <= x_bounds[1] + 1e-12
    assert y1.min() >= y_bounds[0] - 1e-12 and y1.max() <= y_bounds[1] + 1e-12


def test_sample_from_2D_ppd_hot_cell_matches_axis_convention(tmp_path):
    # shape is (n_y, n_x); axis-0 is y and axis-1 is x
    n_y, n_x = 3, 4
    ppd = np.zeros((n_y, n_x), dtype=float)
    ppd[2, 1] = 1.0  # one hot cell at y-index=2, x-index=1

    h5_path = tmp_path / "ppd_hot.h5"
    _write_ppd(h5_path, ppd)

    x_bounds = (0.0, 3.0)  # gives nice edges: [0, 0.5, 1.5, 2.5, 3]
    y_bounds = (0.0, 2.0)  # edges: [0, 0.5, 1.5, 2]

    rng = np.random.default_rng(0)
    x, y = sample_from_2D_ppd(str(h5_path), 5000, x_bounds, y_bounds, rng=rng)

    x_edges = bounds_to_edges(*x_bounds, n_x)
    y_edges = bounds_to_edges(*y_bounds, n_y)

    in_hot = (
        (x >= x_edges[1]) & (x <= x_edges[2]) &
        (y >= y_edges[2]) & (y <= y_edges[3])
    )
    assert in_hot.mean() > 0.97  # should be ~1.0, tolerate tiny numerical edge cases


def test_sample_from_2D_ppd_raises_on_nonpositive_mass(tmp_path):
    ppd = np.zeros((3, 4), dtype=float)
    h5_path = tmp_path / "ppd_zero.h5"
    _write_ppd(h5_path, ppd)

    with pytest.raises(ValueError, match="non-positive total probability mass"):
        sample_from_2D_ppd(str(h5_path), 10, (0.0, 1.0), (0.0, 1.0), rng=np.random.default_rng(0))