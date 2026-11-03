import numpy as np

from examples.psi_noise import measure


def test_noise_shrinks_with_window_size_and_is_seeded() -> None:
    small = measure(50, 20, np.random.default_rng(1))
    large = measure(10_000, 20, np.random.default_rng(1))
    assert large.mean() < small.mean()
    assert np.array_equal(small, measure(50, 20, np.random.default_rng(1)))
