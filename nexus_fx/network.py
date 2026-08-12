from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .universe import normalize_pair, split_pair


@dataclass
class LeadLagEdge:
    source: str
    target: str
    lag: int
    correlation: float
    stability: float
    score: float


class CrossPairNetwork:
    def __init__(self, max_lag: int = 6, min_obs: int = 80, stability_splits: int = 3) -> None:
        self.max_lag = int(max_lag)
        self.min_obs = int(min_obs)
        self.stability_splits = int(stability_splits)

    @staticmethod
    def log_returns(close: pd.DataFrame) -> pd.DataFrame:
        out = close.astype(float).copy()
        out.columns = [normalize_pair(c) for c in out.columns]
        return np.log(out.where(out > 0)).diff()

    @staticmethod
    def _corr_at_lag(source: pd.Series, target: pd.Series, lag: int) -> float:
        # Positive lag means source_t is compared with target_(t+lag): source leads target.
        aligned = pd.concat([source, target.shift(-lag)], axis=1).dropna()
        if len(aligned) < 20:
            return np.nan
        return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))

    def lead_lag_edges(self, close: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
        r = self.log_returns(close).dropna(how="all")
        if len(r) < self.min_obs:
            return pd.DataFrame(columns=["source", "target", "lag", "correlation", "stability", "score"])

        records: List[LeadLagEdge] = []
        columns = list(r.columns)
        for source, target in permutations(columns, 2):
            best_lag, best_corr = None, np.nan
            for lag in range(1, self.max_lag + 1):
                c = self._corr_at_lag(r[source], r[target], lag)
                if np.isfinite(c) and (not np.isfinite(best_corr) or abs(c) > abs(best_corr)):
                    best_lag, best_corr = lag, c
            if best_lag is None:
                continue

            # Stability = fraction of chronological blocks with same correlation sign.
            vals = []
            for block in np.array_split(r[[source, target]].dropna(), self.stability_splits):
                if len(block) < 20:
                    continue
                vals.append(self._corr_at_lag(block[source], block[target], best_lag))
            vals = [v for v in vals if np.isfinite(v)]
            if not vals:
                stability = 0.0
            else:
                expected_sign = np.sign(best_corr)
                stability = float(np.mean([np.sign(v) == expected_sign for v in vals]))

            # Penalize weak and unstable edges. This is descriptive, not causal proof.
            score = abs(best_corr) * stability * np.sqrt(min(len(r), 1000) / 1000.0)
            records.append(LeadLagEdge(source, target, best_lag, best_corr, stability, score))

        df = pd.DataFrame([r.__dict__ for r in records])
        if df.empty:
            return df
        return df.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

    @staticmethod
    def divergence_table(residual_z: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
        if residual_z.empty:
            return pd.DataFrame(columns=["pair", "residual_z", "direction", "magnitude"])
        latest = residual_z.iloc[-1].dropna()
        out = pd.DataFrame({"pair": latest.index, "residual_z": latest.values})
        out["direction"] = np.where(out["residual_z"] > 0, "OVERPERFORMING", "UNDERPERFORMING")
        out["magnitude"] = out["residual_z"].abs()
        out = out[out["magnitude"] >= threshold]
        return out.sort_values("magnitude", ascending=False).reset_index(drop=True)

    @staticmethod
    def triangular_residuals(close: pd.DataFrame) -> pd.DataFrame:
        """Compute log-price consistency residuals for all available currency triangles.

        residual = log(A/B) + log(B/C) - log(A/C), with inversions handled automatically.
        Values are descriptive because asynchronous/free feeds can contain timestamp noise.
        """
        close = close.copy()
        close.columns = [normalize_pair(c) for c in close.columns]
        logs = np.log(close.where(close > 0))

        currencies = sorted({ccy for p in logs.columns for ccy in split_pair(p)})

        def get_log(a: str, b: str) -> Optional[pd.Series]:
            direct = a + b
            inverse = b + a
            if direct in logs:
                return logs[direct]
            if inverse in logs:
                return -logs[inverse]
            return None

        out: Dict[str, pd.Series] = {}
        for i, a in enumerate(currencies):
            for j, b in enumerate(currencies):
                if j <= i:
                    continue
                for k, c in enumerate(currencies):
                    if k <= j:
                        continue
                    ab, bc, ac = get_log(a, b), get_log(b, c), get_log(a, c)
                    if ab is None or bc is None or ac is None:
                        continue
                    out[f"{a}-{b}-{c}"] = ab + bc - ac
        return pd.DataFrame(out, index=close.index)
