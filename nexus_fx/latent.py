from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .universe import CURRENCIES, normalize_pair, split_pair


@dataclass
class LatentStateResult:
    strength: pd.DataFrame
    velocity: pd.DataFrame
    acceleration: pd.DataFrame
    uncertainty: pd.DataFrame
    reconstructed_returns: pd.DataFrame
    residual_returns: pd.DataFrame
    residual_z: pd.DataFrame


class LatentCurrencyEngine:
    """Estimate latent relative currency strength with a linear state-space model.

    Observation equation for pair BASE/QUOTE:
        r_pair,t = s_BASE,t - s_QUOTE,t + epsilon_t

    The system is identified by a zero-sum gauge: sum(s_i) = 0.
    """

    def __init__(
        self,
        currencies: Sequence[str] = CURRENCIES,
        persistence: float = 0.985,
        process_var: float = 2.0e-7,
        base_measurement_var: float = 2.0e-6,
        gauge_var: float = 1.0e-10,
        residual_window: int = 80,
    ) -> None:
        self.currencies = tuple(currencies)
        self.persistence = float(persistence)
        self.process_var = float(process_var)
        self.base_measurement_var = float(base_measurement_var)
        self.gauge_var = float(gauge_var)
        self.residual_window = int(residual_window)
        self._idx = {c: i for i, c in enumerate(self.currencies)}

    def _incidence(self, pairs: Sequence[str]) -> np.ndarray:
        h = np.zeros((len(pairs), len(self.currencies)), dtype=float)
        for row, pair in enumerate(pairs):
            base, quote = split_pair(pair)
            if base not in self._idx or quote not in self._idx:
                raise ValueError(f"Pair {pair} contains currency outside configured universe")
            h[row, self._idx[base]] = 1.0
            h[row, self._idx[quote]] = -1.0
        return h

    @staticmethod
    def log_returns(close: pd.DataFrame) -> pd.DataFrame:
        close = close.astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0)
        return np.log(close).diff()

    def fit(
        self,
        close: pd.DataFrame,
        spreads: Optional[pd.DataFrame] = None,
        quality: Optional[pd.DataFrame] = None,
    ) -> LatentStateResult:
        if close.empty:
            raise ValueError("close matrix is empty")

        close = close.copy()
        close.columns = [normalize_pair(c) for c in close.columns]
        close = close.sort_index()
        returns = self.log_returns(close)
        pairs = list(returns.columns)
        H_all = self._incidence(pairs)

        n = len(self.currencies)
        x = np.zeros(n)
        P = np.eye(n) * 1e-3
        F = np.eye(n) * self.persistence
        Q = np.eye(n) * self.process_var
        I = np.eye(n)

        strengths = []
        uncertainties = []
        reconstructed = []
        residuals = []

        # Gauge observation sum(currency strengths) = 0.
        h_gauge = np.ones((1, n), dtype=float)

        rolling_vol = returns.rolling(40, min_periods=10).std()

        for ts, row in returns.iterrows():
            # Predict
            x = F @ x
            P = F @ P @ F.T + Q

            valid = row.notna().to_numpy()
            if valid.any():
                y = row.to_numpy(dtype=float)[valid]
                H = H_all[valid]
                used_pairs = np.array(pairs, dtype=object)[valid]

                # Measurement variance increases for volatile, wide-spread, or low-quality observations.
                rv = rolling_vol.loc[ts, used_pairs].to_numpy(dtype=float)
                rv = np.where(np.isfinite(rv), np.maximum(rv ** 2, self.base_measurement_var), self.base_measurement_var)

                if spreads is not None:
                    srow = spreads.reindex(index=[ts], columns=used_pairs)
                    if not srow.empty:
                        sval = srow.iloc[0].to_numpy(dtype=float)
                        sval = np.where(np.isfinite(sval), np.maximum(sval, 0.0), 0.0)
                        rv = rv + sval ** 2

                if quality is not None:
                    qrow = quality.reindex(index=[ts], columns=used_pairs)
                    if not qrow.empty:
                        qval = qrow.iloc[0].to_numpy(dtype=float)
                        qval = np.where(np.isfinite(qval), np.clip(qval, 0.05, 1.0), 0.5)
                        rv = rv / qval

                # Append gauge equation.
                H_aug = np.vstack([H, h_gauge])
                y_aug = np.concatenate([y, [0.0]])
                R = np.diag(np.concatenate([rv, [self.gauge_var]]))

                innovation = y_aug - H_aug @ x
                S = H_aug @ P @ H_aug.T + R
                # pinv is deliberate: robust to partial/redundant FX universes.
                K = P @ H_aug.T @ np.linalg.pinv(S)
                x = x + K @ innovation
                P = (I - K @ H_aug) @ P @ (I - K @ H_aug).T + K @ R @ K.T

                # Numerical gauge cleanup.
                x = x - x.mean()

            rec = H_all @ x
            obs = row.to_numpy(dtype=float)
            res = obs - rec

            strengths.append(x.copy())
            uncertainties.append(np.sqrt(np.maximum(np.diag(P), 0.0)))
            reconstructed.append(rec)
            residuals.append(res)

        strength = pd.DataFrame(strengths, index=returns.index, columns=self.currencies)
        uncertainty = pd.DataFrame(uncertainties, index=returns.index, columns=self.currencies)
        reconstructed_returns = pd.DataFrame(reconstructed, index=returns.index, columns=pairs)
        residual_returns = pd.DataFrame(residuals, index=returns.index, columns=pairs)

        velocity = strength.diff().ewm(span=8, adjust=False, min_periods=3).mean()
        acceleration = velocity.diff().ewm(span=5, adjust=False, min_periods=3).mean()

        resid_mean = residual_returns.rolling(self.residual_window, min_periods=20).mean()
        resid_std = residual_returns.rolling(self.residual_window, min_periods=20).std(ddof=0)
        residual_z = (residual_returns - resid_mean) / resid_std.replace(0.0, np.nan)

        return LatentStateResult(
            strength=strength,
            velocity=velocity,
            acceleration=acceleration,
            uncertainty=uncertainty,
            reconstructed_returns=reconstructed_returns,
            residual_returns=residual_returns,
            residual_z=residual_z,
        )

    @staticmethod
    def standardized_snapshot(df: pd.DataFrame, window: int = 120) -> pd.Series:
        """Cross-time z-score of the latest latent state for each currency."""
        tail = df.tail(window)
        mu = tail.mean()
        sigma = tail.std(ddof=0).replace(0.0, np.nan)
        return ((df.iloc[-1] - mu) / sigma).replace([np.inf, -np.inf], np.nan)
