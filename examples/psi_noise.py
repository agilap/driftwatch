"""Measure the PSI that sampling noise alone produces between samples of one distribution.

Run from the repository root: python3 -m examples.psi_noise
"""

import os

# app.config requires a database URL at import time; this script never connects.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://unused@localhost/unused")

import numpy as np  # noqa: E402

from app.config import settings  # noqa: E402
from app.services.drift_engine import compute_psi  # noqa: E402

SEED = 42
REFERENCE_SIZE = 50_000
BINS = 10
TRIALS = 500
WINDOW_SIZES = (50, 100, 500, 2_000, 10_000)


def measure(window_size: int, trials: int, rng: np.random.Generator) -> np.ndarray:
    """Return one PSI per trial, each from fresh standard normal samples."""
    return np.array(
        [
            compute_psi(
                rng.standard_normal(REFERENCE_SIZE),
                rng.standard_normal(window_size),
                bins=BINS,
            )
            for _ in range(trials)
        ]
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(
        f"Standard normal, {TRIALS} trials per size, "
        f"{REFERENCE_SIZE:,}-row reference, {BINS} bins"
    )
    print("| Window size | Mean PSI | Trials red |")
    print("|---:|---:|---:|")
    for window_size in WINDOW_SIZES:
        psi = measure(window_size, TRIALS, rng)
        red = np.mean(psi > settings.psi_red_threshold)
        print(f"| {window_size:,} | {psi.mean():.4f} | {red:.1%} |")


if __name__ == "__main__":
    main()
