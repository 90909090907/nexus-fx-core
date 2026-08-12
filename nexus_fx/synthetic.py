from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from .universe import CURRENCIES, split_pair


def generate_fx_matrix(
    pairs: Sequence[str],
    n: int = 1200,
    seed: int = 42,
    freq: str = "h",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate internally consistent synthetic FX closes plus true latent increments."""
    rng = np.random.default_rng(seed)
    currencies = list(CURRENCIES)
    idx = {c: i for i, c in enumerate(currencies)}

    # Correlated latent shocks with a weak common risk factor and idiosyncratic noise.
    risk = rng.normal(0, 1.0e-3, size=n)
    beta = np.array([-0.30, 0.20, 0.10, 0.40, 0.30, -0.10, -0.35, -0.40])
    shocks = risk[:, None] * beta[None, :] + rng.normal(0, 5.5e-4, size=(n, len(currencies)))
    shocks -= shocks.mean(axis=1, keepdims=True)

    latent_log = np.cumsum(shocks, axis=0)
    dates = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")
    latent = pd.DataFrame(shocks, index=dates, columns=currencies)

    close = pd.DataFrame(index=dates)
    for pair in pairs:
        base, quote = split_pair(pair)
        log_rate = 0.20 + latent_log[:, idx[base]] - latent_log[:, idx[quote]]
        micro_noise = rng.normal(0, 2.0e-5, size=n)
        close[pair] = np.exp(log_rate + micro_noise)
    return close, latent
