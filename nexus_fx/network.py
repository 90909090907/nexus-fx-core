from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Dict, List, Optional

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
    """Cross-pair diagnostics for the NEXUS research core.

    v0.1.2 deliberately performs lead/lag calculations on NumPy arrays after
    extracting each pair from pandas. This removes deployment-dependent
    DataFrame/ndarray indexing behaviour seen on Streamlit Cloud.
    """

    def __init__(self, max_lag: int = 6, min_obs: int = 80, stability_splits: int = 3) -> None:
        self.max_lag = max(1, int(max_lag))
        self.min_obs = max(20, int(min_obs))
        self.stability_splits = max(2, int(stability_splits))

    @staticmethod
    def log_returns(close: pd.DataFrame) -> pd.DataFrame:
        if close is None or not isinstance(close, pd.DataFrame) or close.empty:
            return pd.DataFrame()
        out = close.copy()
        out.columns = [normalize_pair(str(c)) for c in out.columns]
        out = out.loc[:, ~out.columns.duplicated(keep="first")]
        out = out.apply(pd.to_numeric, errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.where(out > 0)
        return np.log(out).diff()

    @staticmethod
    def _corr_arrays(source: np.ndarray, target: np.ndarray, lag: int) -> float:
        """Pearson correlation where source_t leads target_(t+lag)."""
        x = np.asarray(source, dtype=float).reshape(-1)
        y = np.asarray(target, dtype=float).reshape(-1)
        lag = int(lag)
        if lag < 0:
            raise ValueError("lag must be >= 0")
        if lag > 0:
            if len(x) <= lag or len(y) <= lag:
                return np.nan
            x = x[:-lag]
            y = y[lag:]
        n = min(len(x), len(y))
        if n < 20:
            return np.nan
        x = x[:n]
        y = y[:n]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 20:
            return np.nan
        x = x[mask]
        y = y[mask]
        if np.nanstd(x) <= 1e-15 or np.nanstd(y) <= 1e-15:
            return np.nan
        corr = np.corrcoef(x, y)[0, 1]
        return float(corr) if np.isfinite(corr) else np.nan

    def lead_lag_edges(self, close: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
        cols = ["source", "target", "lag", "correlation", "stability", "score"]
        r = self.log_returns(close)
        if r.empty or r.shape[1] < 2:
            return pd.DataFrame(columns=cols)

        records: List[LeadLagEdge] = []
        names = [str(c) for c in r.columns]

        for source, target in permutations(names, 2):
            # Extract once from pandas, then work only with numeric arrays.
            pair = r.loc[:, [source, target]].dropna(how="any")
            if len(pair) < self.min_obs:
                continue

            values = pair.to_numpy(dtype=float, copy=True)
            if values.ndim != 2 or values.shape[1] != 2:
                continue
            x = values[:, 0]
            y = values[:, 1]

            best_lag: Optional[int] = None
            best_corr = np.nan
            for lag in range(1, self.max_lag + 1):
                corr = self._corr_arrays(x, y, lag)
                if np.isfinite(corr) and (best_lag is None or abs(corr) > abs(best_corr)):
                    best_lag = lag
                    best_corr = corr

            if best_lag is None or not np.isfinite(best_corr):
                continue

            # Chronological stability, split by integer positions only.
            block_indices = np.array_split(np.arange(len(values), dtype=int), self.stability_splits)
            block_corrs: List[float] = []
            for idx in block_indices:
                if len(idx) < max(20, best_lag + 5):
                    continue
                block = values[idx, :]
                corr = self._corr_arrays(block[:, 0], block[:, 1], best_lag)
                if np.isfinite(corr):
                    block_corrs.append(float(corr))

            if block_corrs and best_corr != 0:
                expected = np.sign(best_corr)
                stability = float(np.mean([np.sign(c) == expected for c in block_corrs]))
            else:
                stability = 0.0

            sample_factor = float(np.sqrt(min(len(values), 1000) / 1000.0))
            score = float(abs(best_corr) * stability * sample_factor)
            records.append(
                LeadLagEdge(
                    source=source,
                    target=target,
                    lag=int(best_lag),
                    correlation=float(best_corr),
                    stability=stability,
                    score=score,
                )
            )

        if not records:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame([x.__dict__ for x in records], columns=cols)
        return (
            df.sort_values(["score", "correlation"], ascending=[False, False])
            .head(max(1, int(top_n)))
            .reset_index(drop=True)
        )

    @staticmethod
    def divergence_table(residual_z: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
        cols = ["pair", "residual_z", "direction", "magnitude"]
        if residual_z is None or not isinstance(residual_z, pd.DataFrame) or residual_z.empty:
            return pd.DataFrame(columns=cols)
        latest = residual_z.iloc[-1].dropna()
        if latest.empty:
            return pd.DataFrame(columns=cols)
        out = pd.DataFrame({"pair": latest.index.astype(str), "residual_z": latest.to_numpy(dtype=float)})
        out["direction"] = np.where(out["residual_z"] > 0, "OVERPERFORMING", "UNDERPERFORMING")
        out["magnitude"] = out["residual_z"].abs()
        out = out[out["magnitude"] >= float(threshold)]
        return out.sort_values("magnitude", ascending=False).reset_index(drop=True)

    @staticmethod
    def triangular_residuals(close: pd.DataFrame) -> pd.DataFrame:
        if close is None or not isinstance(close, pd.DataFrame) or close.empty:
            return pd.DataFrame()
        close = close.copy()
        close.columns = [normalize_pair(str(c)) for c in close.columns]
        close = close.loc[:, ~close.columns.duplicated(keep="first")]
        numeric = close.apply(pd.to_numeric, errors="coerce")
        logs = np.log(numeric.where(numeric > 0))

        currencies = sorted({ccy for pair in logs.columns for ccy in split_pair(str(pair))})

        def get_log(a: str, b: str) -> Optional[pd.Series]:
            direct = a + b
            inverse = b + a
            if direct in logs.columns:
                return logs[direct]
            if inverse in logs.columns:
                return -logs[inverse]
            return None

        out: Dict[str, pd.Series] = {}
        for i, a in enumerate(currencies):
            for j in range(i + 1, len(currencies)):
                b = currencies[j]
                for k in range(j + 1, len(currencies)):
                    c = currencies[k]
                    ab = get_log(a, b)
                    bc = get_log(b, c)
                    ac = get_log(a, c)
                    if ab is None or bc is None or ac is None:
                        continue
                    out[f"{a}-{b}-{c}"] = ab + bc - ac
        return pd.DataFrame(out, index=logs.index)
