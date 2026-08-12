from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .universe import normalize_pair, split_pair


REGIMES: Tuple[str, ...] = (
    "TREND",
    "RANGE",
    "VOL_EXPANSION",
    "VOL_CONTRACTION",
    "EVENT_LIKE",
    "TRANSITION",
)


@dataclass
class RegimeResult:
    probabilities: pd.DataFrame
    dominant: pd.Series
    metrics: pd.DataFrame

    @property
    def current_probabilities(self) -> pd.Series:
        if self.probabilities.empty:
            return pd.Series(dtype=float)
        return self.probabilities.iloc[-1].sort_values(ascending=False)

    @property
    def current_regime(self) -> str:
        if self.dominant.empty:
            return "UNKNOWN"
        return str(self.dominant.iloc[-1])


class RegimeEngine:
    """Probabilistic FX regime classifier using only price-derived structure.

    This is deliberately transparent and research-oriented. It does not claim
    economic causality. The output is a soft probability vector rather than a
    hard label, so downstream modules can condition on uncertainty.
    """

    def __init__(
        self,
        efficiency_window: int = 12,
        vol_short: int = 12,
        vol_long: int = 72,
        transition_window: int = 8,
    ) -> None:
        self.efficiency_window = max(6, int(efficiency_window))
        self.vol_short = max(6, int(vol_short))
        self.vol_long = max(self.vol_short + 8, int(vol_long))
        self.transition_window = max(4, int(transition_window))

    @staticmethod
    def _log_prices(close: pd.DataFrame) -> pd.DataFrame:
        if close is None or not isinstance(close, pd.DataFrame) or close.empty:
            return pd.DataFrame()
        x = close.copy()
        x.columns = [normalize_pair(str(c)) for c in x.columns]
        x = x.loc[:, ~x.columns.duplicated(keep="first")]
        x = x.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return np.log(x.where(x > 0))

    @staticmethod
    def _sigmoid(x: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
        arr = np.asarray(x, dtype=float)
        arr = np.clip(arr, -30.0, 30.0)
        out = 1.0 / (1.0 + np.exp(-arr))
        if np.ndim(x) == 0:
            return float(out)
        if isinstance(x, pd.Series):
            return pd.Series(out, index=x.index)
        return out

    @staticmethod
    def _softmax_frame(scores: pd.DataFrame) -> pd.DataFrame:
        arr = scores.to_numpy(dtype=float)
        arr = np.where(np.isfinite(arr), arr, -20.0)
        arr = arr - np.nanmax(arr, axis=1, keepdims=True)
        exp = np.exp(np.clip(arr, -40.0, 40.0))
        denom = exp.sum(axis=1, keepdims=True)
        denom = np.where(denom > 0, denom, 1.0)
        probs = exp / denom
        return pd.DataFrame(probs, index=scores.index, columns=scores.columns)

    def fit(self, close: pd.DataFrame) -> RegimeResult:
        logp = self._log_prices(close)
        if logp.empty or logp.shape[1] < 3:
            empty = pd.DataFrame(columns=REGIMES)
            return RegimeResult(empty, pd.Series(dtype=object), pd.DataFrame())

        ret = logp.diff()
        abs_ret = ret.abs()

        # Directional efficiency: displacement / path length. Values near 1 imply
        # persistent direction, values near 0 imply choppy/ranging movement.
        displacement = (logp - logp.shift(self.efficiency_window)).abs()
        path = abs_ret.rolling(self.efficiency_window, min_periods=max(4, self.efficiency_window // 2)).sum()
        efficiency_by_pair = displacement / path.replace(0.0, np.nan)
        efficiency = efficiency_by_pair.median(axis=1, skipna=True).clip(0.0, 1.0)

        short_vol_pair = ret.rolling(self.vol_short, min_periods=max(4, self.vol_short // 2)).std(ddof=0)
        long_vol_pair = ret.rolling(self.vol_long, min_periods=max(12, self.vol_long // 3)).std(ddof=0)
        vol_ratio_pair = short_vol_pair / long_vol_pair.replace(0.0, np.nan)
        vol_ratio = vol_ratio_pair.replace([np.inf, -np.inf], np.nan).median(axis=1, skipna=True).clip(0.05, 8.0)

        # Cross-sectional shock: median absolute standardized return across pairs.
        shock_z_pair = abs_ret / long_vol_pair.replace(0.0, np.nan)
        shock = shock_z_pair.replace([np.inf, -np.inf], np.nan).median(axis=1, skipna=True).clip(0.0, 12.0)

        # Breadth / dispersion: how uneven the cross-pair return field is.
        cross_dispersion = ret.std(axis=1, ddof=0)
        disp_mu = cross_dispersion.rolling(self.vol_long, min_periods=20).median()
        disp_mad = (cross_dispersion - disp_mu).abs().rolling(self.vol_long, min_periods=20).median()
        dispersion_z = ((cross_dispersion - disp_mu) / (1.4826 * disp_mad).replace(0.0, np.nan)).clip(-8, 8)

        # Transition score captures rapid structural changes in efficiency or vol.
        eff_delta = efficiency.diff(self.transition_window).abs()
        vol_delta = np.log(vol_ratio.replace(0.0, np.nan)).diff(self.transition_window).abs()
        transition = (eff_delta / 0.20 + vol_delta / 0.35).clip(0.0, 8.0)

        # Transparent regime score functions. Softmax turns them into probabilities.
        trend_score = 3.2 * (efficiency - 0.38) + 0.35 * np.log(vol_ratio.clip(lower=0.2))
        range_score = 3.0 * (0.42 - efficiency) - 0.25 * (vol_ratio - 1.0).abs()
        vol_exp_score = 2.4 * np.log(vol_ratio.clip(lower=0.2)) + 0.35 * (dispersion_z.fillna(0.0))
        vol_con_score = -2.4 * np.log(vol_ratio.clip(lower=0.2)) - 0.20 * shock.fillna(0.0)
        event_score = 1.45 * (shock - 1.35) + 0.55 * np.maximum(vol_ratio - 1.0, 0.0)
        transition_score = 1.25 * (transition - 0.75)

        scores = pd.DataFrame(
            {
                "TREND": trend_score,
                "RANGE": range_score,
                "VOL_EXPANSION": vol_exp_score,
                "VOL_CONTRACTION": vol_con_score,
                "EVENT_LIKE": event_score,
                "TRANSITION": transition_score,
            },
            index=logp.index,
        ).replace([np.inf, -np.inf], np.nan)

        # Avoid overconfidence when not enough history is available.
        valid_history = long_vol_pair.notna().sum(axis=1) >= max(3, logp.shape[1] // 3)
        scores.loc[~valid_history, :] = 0.0

        probabilities = self._softmax_frame(scores.fillna(0.0))
        dominant = probabilities.idxmax(axis=1).rename("regime")
        metrics = pd.DataFrame(
            {
                "efficiency": efficiency,
                "vol_ratio": vol_ratio,
                "shock": shock,
                "dispersion_z": dispersion_z,
                "transition": transition,
            },
            index=logp.index,
        )
        return RegimeResult(probabilities=probabilities, dominant=dominant, metrics=metrics)


class CausalGraphEngine:
    """Candidate causal graph from conditional lead/lag relationships.

    It is intentionally called a *candidate* graph. A significant temporal edge
    is not proof of economic causality. The engine adds several defenses against
    false edges: shared-currency factor residualization, contiguous stability,
    circular-shift permutation tests, and incremental predictive R².
    """

    COLUMNS = [
        "source",
        "target",
        "lag",
        "shared_factor",
        "raw_corr",
        "conditional_corr",
        "regime_corr",
        "stability",
        "perm_p",
        "incremental_r2",
        "edge_score",
        "n_obs",
    ]

    def __init__(
        self,
        max_lag: int = 6,
        min_obs: int = 100,
        stability_splits: int = 4,
        permutations_n: int = 48,
        random_state: int = 42,
    ) -> None:
        self.max_lag = max(1, int(max_lag))
        self.min_obs = max(50, int(min_obs))
        self.stability_splits = max(3, int(stability_splits))
        self.permutations_n = max(16, int(permutations_n))
        self.rng = np.random.default_rng(random_state)

    @staticmethod
    def log_returns(close: pd.DataFrame) -> pd.DataFrame:
        x = close.copy()
        x.columns = [normalize_pair(str(c)) for c in x.columns]
        x = x.loc[:, ~x.columns.duplicated(keep="first")]
        x = x.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return np.log(x.where(x > 0)).diff()

    @staticmethod
    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        n = min(x.size, y.size)
        if n < 20:
            return np.nan
        x = x[:n]
        y = y[:n]
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 20:
            return np.nan
        x = x[mask]
        y = y[mask]
        if np.std(x) <= 1e-15 or np.std(y) <= 1e-15:
            return np.nan
        c = np.corrcoef(x, y)[0, 1]
        return float(c) if np.isfinite(c) else np.nan

    @staticmethod
    def _residualize(y: pd.Series, factors: pd.DataFrame) -> pd.Series:
        if factors is None or factors.empty:
            return y.astype(float)
        frame = pd.concat([y.rename("y"), factors], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        out = pd.Series(index=y.index, dtype=float)
        if len(frame) < max(30, factors.shape[1] * 10):
            return y.astype(float)
        yy = frame.pop("y").to_numpy(dtype=float)
        X = frame.to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.pinv(X) @ yy
        residual = yy - X @ beta
        out.loc[frame.index] = residual
        return out

    @staticmethod
    def _align_lead(source: pd.Series, target: pd.Series, lag: int) -> pd.DataFrame:
        # source_t predicts target_(t+lag). Negative shift aligns the future target
        # value on the current source timestamp.
        return pd.concat(
            [source.rename("x"), target.shift(-int(lag)).rename("y_future"), target.rename("y_now")],
            axis=1,
        ).replace([np.inf, -np.inf], np.nan).dropna()

    def _stability(self, aligned: pd.DataFrame) -> float:
        n = len(aligned)
        if n < self.min_obs:
            return 0.0
        bounds = np.linspace(0, n, self.stability_splits + 1, dtype=int)
        vals: List[float] = []
        for i in range(self.stability_splits):
            block = aligned.iloc[int(bounds[i]): int(bounds[i + 1])]
            if len(block) < 20:
                continue
            c = self._corr(block["x"].to_numpy(), block["y_future"].to_numpy())
            if np.isfinite(c):
                vals.append(float(c))
        if len(vals) < 2:
            return 0.0
        global_corr = self._corr(aligned["x"].to_numpy(), aligned["y_future"].to_numpy())
        if not np.isfinite(global_corr) or abs(global_corr) < 1e-12:
            return 0.0
        sign_consistency = float(np.mean(np.sign(vals) == np.sign(global_corr)))
        magnitude_consistency = float(np.clip(np.median(np.abs(vals)) / abs(global_corr), 0.0, 1.0))
        variability_penalty = float(np.exp(-np.std(vals) / (abs(global_corr) + 1e-6)))
        return float(np.clip(sign_consistency * magnitude_consistency * variability_penalty, 0.0, 1.0))

    @staticmethod
    def _r2(y: np.ndarray, X: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).ravel()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        X = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.pinv(X) @ y
        pred = X @ beta
        sse = float(np.sum((y - pred) ** 2))
        sst = float(np.sum((y - np.mean(y)) ** 2))
        if sst <= 1e-18:
            return 0.0
        return float(1.0 - sse / sst)

    def _incremental_r2(self, aligned: pd.DataFrame) -> float:
        if len(aligned) < 30:
            return 0.0
        y = aligned["y_future"].to_numpy(dtype=float)
        baseline = aligned[["y_now"]].to_numpy(dtype=float)
        full = aligned[["y_now", "x"]].to_numpy(dtype=float)
        return float(max(0.0, self._r2(y, full) - self._r2(y, baseline)))

    def _permutation_p(self, aligned: pd.DataFrame, observed: float) -> float:
        if len(aligned) < self.min_obs or not np.isfinite(observed):
            return 1.0
        x = aligned["x"].to_numpy(dtype=float)
        y = aligned["y_future"].to_numpy(dtype=float)
        n = len(x)
        if n < 40:
            return 1.0
        hits = 0
        valid = 0
        min_shift = max(7, n // 10)
        max_shift = max(min_shift + 1, n - min_shift)
        for _ in range(self.permutations_n):
            if max_shift <= min_shift:
                break
            shift = int(self.rng.integers(min_shift, max_shift))
            xp = np.roll(x, shift)
            c = self._corr(xp, y)
            if np.isfinite(c):
                valid += 1
                if abs(c) >= abs(observed):
                    hits += 1
        if valid == 0:
            return 1.0
        return float((hits + 1) / (valid + 1))

    @staticmethod
    def _shared_factor(source: str, target: str) -> List[str]:
        sb, sq = split_pair(source)
        tb, tq = split_pair(target)
        return sorted(set((sb, sq)).intersection((tb, tq)))

    def build(
        self,
        close: pd.DataFrame,
        latent_strength: Optional[pd.DataFrame] = None,
        regime: Optional[RegimeResult] = None,
        top_n: int = 30,
    ) -> pd.DataFrame:
        returns = self.log_returns(close)
        if returns.empty or returns.shape[1] < 2:
            return pd.DataFrame(columns=self.COLUMNS)

        candidates: List[Dict[str, object]] = []
        names = list(map(str, returns.columns))

        # First pass: find each directed pair's best raw lag cheaply.
        for source, target in permutations(names, 2):
            base = pd.concat([returns[source].rename("x"), returns[target].rename("y")], axis=1).dropna()
            if len(base) < self.min_obs:
                continue
            best_lag = None
            best_raw = None
            for lag in range(1, self.max_lag + 1):
                aligned = self._align_lead(base["x"], base["y"], lag)
                c = self._corr(aligned["x"].to_numpy(), aligned["y_future"].to_numpy())
                if not np.isfinite(c):
                    continue
                if best_raw is None or abs(c) > abs(best_raw):
                    best_lag, best_raw = lag, float(c)
            if best_lag is not None and best_raw is not None:
                candidates.append({"source": source, "target": target, "lag": best_lag, "raw_corr": best_raw})

        # Evaluate strongest raw candidates with the expensive safeguards.
        candidates = sorted(candidates, key=lambda d: abs(float(d["raw_corr"])), reverse=True)
        candidates = candidates[: max(top_n * 4, 60)]
        records: List[Dict[str, object]] = []

        for cand in candidates:
            source = str(cand["source"])
            target = str(cand["target"])
            lag = int(cand["lag"])
            raw_corr = float(cand["raw_corr"])
            shared = self._shared_factor(source, target)

            sx = returns[source].astype(float)
            ty = returns[target].astype(float)
            if shared and latent_strength is not None and not latent_strength.empty:
                factor_cols = [c for c in shared if c in latent_strength.columns]
                factors = latent_strength.reindex(returns.index)[factor_cols] if factor_cols else pd.DataFrame(index=returns.index)
                sx_cond = self._residualize(sx, factors)
                ty_cond = self._residualize(ty, factors)
            else:
                sx_cond, ty_cond = sx, ty

            aligned = self._align_lead(sx_cond, ty_cond, lag)
            if len(aligned) < self.min_obs:
                continue
            cond_corr = self._corr(aligned["x"].to_numpy(), aligned["y_future"].to_numpy())
            if not np.isfinite(cond_corr):
                continue

            stability = self._stability(aligned)
            p = self._permutation_p(aligned, cond_corr)
            inc_r2 = self._incremental_r2(aligned)

            regime_corr = np.nan
            if regime is not None and not regime.dominant.empty:
                current = regime.current_regime
                labels = regime.dominant.reindex(aligned.index)
                subset = aligned.loc[labels == current]
                if len(subset) >= 30:
                    regime_corr = self._corr(subset["x"].to_numpy(), subset["y_future"].to_numpy())

            significance = float(np.clip(1.0 - p, 0.0, 1.0))
            r2_component = float(np.clip(np.sqrt(max(inc_r2, 0.0)) * 4.0, 0.0, 1.0))
            regime_component = float(np.clip(abs(regime_corr), 0.0, 1.0)) if np.isfinite(regime_corr) else 0.0
            sample_component = float(np.clip(np.sqrt(len(aligned) / 500.0), 0.25, 1.0))
            edge_score = sample_component * (
                0.34 * min(abs(cond_corr), 1.0)
                + 0.24 * stability
                + 0.20 * significance
                + 0.14 * r2_component
                + 0.08 * regime_component
            )

            records.append(
                {
                    "source": source,
                    "target": target,
                    "lag": lag,
                    "shared_factor": ",".join(shared) if shared else "—",
                    "raw_corr": raw_corr,
                    "conditional_corr": float(cond_corr),
                    "regime_corr": float(regime_corr) if np.isfinite(regime_corr) else np.nan,
                    "stability": float(stability),
                    "perm_p": float(p),
                    "incremental_r2": float(inc_r2),
                    "edge_score": float(edge_score),
                    "n_obs": int(len(aligned)),
                }
            )

        if not records:
            return pd.DataFrame(columns=self.COLUMNS)
        return (
            pd.DataFrame(records, columns=self.COLUMNS)
            .sort_values(["edge_score", "conditional_corr"], ascending=[False, False])
            .head(max(1, int(top_n)))
            .reset_index(drop=True)
        )


@dataclass
class StressResult:
    current_stress: float
    stress_percentile: float
    coherence: float
    series: pd.Series
    triangle_z: pd.DataFrame


class NetworkStressEngine:
    """Aggregate standardized triangle inconsistencies into an FX network stress index."""

    def __init__(self, window: int = 96, min_periods: int = 30) -> None:
        self.window = max(30, int(window))
        self.min_periods = max(15, int(min_periods))

    def fit(self, triangles: pd.DataFrame) -> StressResult:
        if triangles is None or triangles.empty:
            return StressResult(np.nan, np.nan, np.nan, pd.Series(dtype=float), pd.DataFrame())
        tri = triangles.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        mu = tri.rolling(self.window, min_periods=self.min_periods).median()
        mad = (tri - mu).abs().rolling(self.window, min_periods=self.min_periods).median()
        robust_sigma = (1.4826 * mad).clip(lower=1e-8)
        z = ((tri - mu) / robust_sigma).clip(-20, 20)
        stress = z.abs().median(axis=1, skipna=True).rename("network_stress")
        valid = stress.dropna()
        if valid.empty:
            return StressResult(np.nan, np.nan, np.nan, stress, z)
        current = float(valid.iloc[-1])
        hist = valid.iloc[:-1]
        percentile = float((hist <= current).mean() * 100.0) if len(hist) >= 10 else np.nan
        coherence = float(np.exp(-min(current, 10.0)))
        return StressResult(current, percentile, coherence, stress, z)
