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
    def __init__(self, max_lag: int = 6, min_obs: int = 80, stability_splits: int = 3) -> None:
        self.max_lag = int(max_lag)
        self.min_obs = int(min_obs)
        self.stability_splits = max(2, int(stability_splits))

    @staticmethod
    def log_returns(close: pd.DataFrame) -> pd.DataFrame:
        if close.empty:
            return pd.DataFrame(index=close.index)
        out = close.astype(float).copy()
        out.columns = [normalize_pair(str(c)) for c in out.columns]
        # Defensive: a provider should not return duplicate normalized pairs, but if it does,
        # keep the first one so selecting r["EURUSD"] always returns a Series, not a DataFrame.
        out = out.loc[:, ~out.columns.duplicated(keep="first")]
        out = out.replace([np.inf, -np.inf], np.nan)
        return np.log(out.where(out > 0)).diff()

    @staticmethod
    def _corr_at_lag(source: pd.Series, target: pd.Series, lag: int) -> float:
        """Correlation where positive lag means source_t leads target_(t+lag)."""
        source = pd.Series(source, copy=False).rename("source")
        target = pd.Series(target, copy=False).rename("target")
        aligned = pd.concat([source, target.shift(-int(lag))], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(aligned) < 20:
            return np.nan
        # Constant samples have undefined correlation.
        if aligned["source"].nunique(dropna=True) < 2 or aligned["target"].nunique(dropna=True) < 2:
            return np.nan
        corr = aligned["source"].corr(aligned["target"])
        return float(corr) if pd.notna(corr) else np.nan

    @staticmethod
    def _chronological_blocks(df: pd.DataFrame, n_splits: int) -> List[pd.DataFrame]:
        """Split a DataFrame chronologically while *guaranteeing* DataFrame output.

        We intentionally avoid np.array_split(df, ...). Depending on the NumPy/pandas
        combination used by a deployment environment, that can route through ndarray
        semantics. Splitting integer positions and applying .iloc is deterministic.
        """
        if df.empty:
            return []
        positions = np.array_split(np.arange(len(df), dtype=int), max(1, int(n_splits)))
        return [df.iloc[pos].copy() for pos in positions if len(pos) > 0]

    def lead_lag_edges(self, close: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
        r = self.log_returns(close).dropna(how="all")
        empty_cols = ["source", "target", "lag", "correlation", "stability", "score"]
        if len(r) < self.min_obs or r.shape[1] < 2:
            return pd.DataFrame(columns=empty_cols)

        records: List[LeadLagEdge] = []
        columns = list(r.columns)

        for source, target in permutations(columns, 2):
            pair_df = r.loc[:, [source, target]].dropna()
            if len(pair_df) < self.min_obs:
                continue

            best_lag: Optional[int] = None
            best_corr = np.nan
            for lag in range(1, self.max_lag + 1):
                c = self._corr_at_lag(pair_df[source], pair_df[target], lag)
                if np.isfinite(c) and (not np.isfinite(best_corr) or abs(c) > abs(best_corr)):
                    best_lag, best_corr = lag, c
            if best_lag is None:
                continue

            # Stability = fraction of chronological blocks preserving the full-sample sign.
            vals: List[float] = []
            for block in self._chronological_blocks(pair_df, self.stability_splits):
                if len(block) < 20:
                    continue
                value = self._corr_at_lag(block[source], block[target], best_lag)
                if np.isfinite(value):
                    vals.append(float(value))

            if not vals or best_corr == 0:
                stability = 0.0
            else:
                expected_sign = np.sign(best_corr)
                stability = float(np.mean([np.sign(v) == expected_sign for v in vals]))

            # Descriptive network score; it is not evidence of causal identification.
            sample_factor = np.sqrt(min(len(pair_df), 1000) / 1000.0)
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

        df = pd.DataFrame([record.__dict__ for record in records], columns=empty_cols)
        if df.empty:
            return df
        return df.sort_values(["score", "correlation"], ascending=[False, False]).head(int(top_n)).reset_index(drop=True)

    @staticmethod
    def divergence_table(residual_z: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
        if residual_z.empty:
            return pd.DataFrame(columns=["pair", "residual_z", "direction", "magnitude"])
        latest = residual_z.iloc[-1].dropna()
        out = pd.DataFrame({"pair": latest.index, "residual_z": latest.values})
        out["direction"] = np.where(out["residual_z"] > 0, "OVERPERFORMING", "UNDERPERFORMING")
        out["magnitude"] = out["residual_z"].abs()
        out = out[out["magnitude"] >= float(threshold)]
        return out.sort_values("magnitude", ascending=False).reset_index(drop=True)

    @staticmethod
    def triangular_residuals(close: pd.DataFrame) -> pd.DataFrame:
        """Compute log-price consistency residuals for all available currency triangles.

        residual = log(A/B) + log(B/C) - log(A/C), with inversions handled automatically.
        Values are descriptive because asynchronous/free feeds can contain timestamp noise.
        """
        if close.empty:
            return pd.DataFrame(index=close.index)
        close = close.copy()
        close.columns = [normalize_pair(str(c)) for c in close.columns]
        close = close.loc[:, ~close.columns.duplicated(keep="first")]
        logs = np.log(close.astype(float).where(close.astype(float) > 0))

        currencies = sorted({ccy for pair in logs.columns for ccy in split_pair(pair)})

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
