from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from itertools import permutations
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from nexus_fx.data_yahoo import download_close_matrix
from nexus_fx.latent import LatentCurrencyEngine
from nexus_fx.universe import normalize_pair, split_pair
APP_VERSION = '0.2.2.4'
INTELLIGENCE_VERSION = 'STAT-INLINE-0.2.2.4'
REGIMES: Tuple[str, ...] = ('TREND', 'RANGE', 'VOL_EXPANSION', 'VOL_CONTRACTION', 'EVENT_LIKE', 'TRANSITION')

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
            return 'UNKNOWN'
        return str(self.dominant.iloc[-1])

    @property
    def current_reliability(self) -> float:
        if self.metrics.empty or 'regime_reliability' not in self.metrics:
            return np.nan
        value = self.metrics['regime_reliability'].iloc[-1]
        return float(value) if np.isfinite(value) else np.nan

    @property
    def current_entropy(self) -> float:
        if self.metrics.empty or 'posterior_entropy' not in self.metrics:
            return np.nan
        value = self.metrics['posterior_entropy'].iloc[-1]
        return float(value) if np.isfinite(value) else np.nan

class RegimeEngine:
    """Transparent probabilistic FX regime classifier.

    v0.2.1 adds uncertainty diagnostics to avoid interpreting a concentrated
    softmax posterior as a literally calibrated probability. The field
    ``regime_reliability`` is a self-consistency diagnostic, not an externally
    validated forecast probability.
    """

    def __init__(self, efficiency_window: int=12, vol_short: int=12, vol_long: int=72, transition_window: int=8, persistence_window: int=12) -> None:
        self.efficiency_window = max(6, int(efficiency_window))
        self.vol_short = max(6, int(vol_short))
        self.vol_long = max(self.vol_short + 8, int(vol_long))
        self.transition_window = max(4, int(transition_window))
        self.persistence_window = max(6, int(persistence_window))

    @staticmethod
    def _log_prices(close: pd.DataFrame) -> pd.DataFrame:
        if close is None or not isinstance(close, pd.DataFrame) or close.empty:
            return pd.DataFrame()
        x = close.copy()
        x.columns = [normalize_pair(str(c)) for c in x.columns]
        x = x.loc[:, ~x.columns.duplicated(keep='first')]
        x = x.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        return np.log(x.where(x > 0))

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
        displacement = (logp - logp.shift(self.efficiency_window)).abs()
        path = abs_ret.rolling(self.efficiency_window, min_periods=max(4, self.efficiency_window // 2)).sum()
        efficiency_by_pair = displacement / path.replace(0.0, np.nan)
        efficiency = efficiency_by_pair.median(axis=1, skipna=True).clip(0.0, 1.0)
        short_vol_pair = ret.rolling(self.vol_short, min_periods=max(4, self.vol_short // 2)).std(ddof=0)
        long_vol_pair = ret.rolling(self.vol_long, min_periods=max(12, self.vol_long // 3)).std(ddof=0)
        vol_ratio_pair = short_vol_pair / long_vol_pair.replace(0.0, np.nan)
        vol_ratio = vol_ratio_pair.replace([np.inf, -np.inf], np.nan).median(axis=1, skipna=True).clip(0.05, 8.0)
        shock_z_pair = abs_ret / long_vol_pair.replace(0.0, np.nan)
        shock = shock_z_pair.replace([np.inf, -np.inf], np.nan).median(axis=1, skipna=True).clip(0.0, 12.0)
        cross_dispersion = ret.std(axis=1, ddof=0)
        disp_mu = cross_dispersion.rolling(self.vol_long, min_periods=20).median()
        disp_mad = (cross_dispersion - disp_mu).abs().rolling(self.vol_long, min_periods=20).median()
        dispersion_z = ((cross_dispersion - disp_mu) / (1.4826 * disp_mad).replace(0.0, np.nan)).clip(-8, 8)
        eff_delta = efficiency.diff(self.transition_window).abs()
        vol_delta = np.log(vol_ratio.replace(0.0, np.nan)).diff(self.transition_window).abs()
        transition = (eff_delta / 0.2 + vol_delta / 0.35).clip(0.0, 8.0)
        trend_score = 3.2 * (efficiency - 0.38) + 0.35 * np.log(vol_ratio.clip(lower=0.2))
        range_score = 3.0 * (0.42 - efficiency) - 0.25 * (vol_ratio - 1.0).abs()
        vol_exp_score = 2.4 * np.log(vol_ratio.clip(lower=0.2)) + 0.35 * dispersion_z.fillna(0.0)
        vol_con_score = -2.4 * np.log(vol_ratio.clip(lower=0.2)) - 0.2 * shock.fillna(0.0)
        event_score = 1.45 * (shock - 1.35) + 0.55 * np.maximum(vol_ratio - 1.0, 0.0)
        transition_score = 1.25 * (transition - 0.75)
        scores = pd.DataFrame({'TREND': trend_score, 'RANGE': range_score, 'VOL_EXPANSION': vol_exp_score, 'VOL_CONTRACTION': vol_con_score, 'EVENT_LIKE': event_score, 'TRANSITION': transition_score}, index=logp.index).replace([np.inf, -np.inf], np.nan)
        valid_history = long_vol_pair.notna().sum(axis=1) >= max(3, logp.shape[1] // 3)
        scores.loc[~valid_history, :] = 0.0
        probabilities = self._softmax_frame(scores.fillna(0.0))
        dominant = probabilities.idxmax(axis=1).rename('regime')
        p = probabilities.clip(lower=1e-12)
        entropy = -(p * np.log(p)).sum(axis=1) / np.log(len(REGIMES))
        sorted_probs = np.sort(probabilities.to_numpy(dtype=float), axis=1)
        gap = pd.Series(sorted_probs[:, -1] - sorted_probs[:, -2], index=probabilities.index)
        current_match = dominant.eq(dominant.iloc[-1] if len(dominant) else 'UNKNOWN').astype(float)
        persistence = current_match.rolling(self.persistence_window, min_periods=3).mean()
        change = dominant.ne(dominant.shift(1)).astype(float)
        transition_rate = change.rolling(self.persistence_window, min_periods=3).mean()
        top_prob = probabilities.max(axis=1)
        reliability = (top_prob * (1.0 - entropy.clip(0.0, 1.0)) * (0.45 + 0.55 * persistence.fillna(0.5)) * (0.55 + 0.45 * gap.clip(0.0, 1.0))).clip(0.0, 1.0)
        metrics = pd.DataFrame({'efficiency': efficiency, 'vol_ratio': vol_ratio, 'shock': shock, 'dispersion_z': dispersion_z, 'transition': transition, 'posterior_entropy': entropy, 'confidence_gap': gap, 'dominant_persistence': persistence, 'transition_rate': transition_rate, 'regime_reliability': reliability}, index=logp.index)
        return RegimeResult(probabilities=probabilities, dominant=dominant, metrics=metrics)

class CausalGraphEngine:
    """Statistically hardened candidate lead/lag graph.

    Safeguards in v0.2.1:
      * lag selection on a chronologically earlier discovery sample,
      * shared-currency factor residualization,
      * chronological stability,
      * circular-shift permutation p-values,
      * Benjamini-Hochberg FDR q-values,
      * moving-block bootstrap confidence intervals,
      * expanding walk-forward validation,
      * recent-vs-history decay diagnostics.

    Surviving edges remain predictive hypotheses, not proof of causality.
    """
    COLUMNS = ['source', 'target', 'lag', 'shared_factor', 'raw_corr', 'conditional_corr', 'regime_corr', 'stability', 'analytic_p', 'perm_p', 'fdr_q', 'bootstrap_low', 'bootstrap_high', 'walkforward_ic', 'walkforward_sign_rate', 'walkforward_p', 'walkforward_n', 'incremental_r2', 'decay_score', 'decay_state', 'evidence_score', 'edge_score', 'survives', 'n_obs']

    def __init__(self, max_lag: int=6, min_obs: int=100, stability_splits: int=4, permutations_n: int=199, bootstrap_n: int=199, walkforward_folds: int=3, fdr_alpha: float=0.1, random_state: int=42) -> None:
        self.max_lag = max(1, int(max_lag))
        self.min_obs = max(50, int(min_obs))
        self.stability_splits = max(3, int(stability_splits))
        self.permutations_n = max(16, int(permutations_n))
        self.bootstrap_n = max(16, int(bootstrap_n))
        self.walkforward_folds = max(2, int(walkforward_folds))
        self.fdr_alpha = float(np.clip(fdr_alpha, 0.01, 0.25))
        self.rng = np.random.default_rng(random_state)

    @staticmethod
    def log_returns(close: pd.DataFrame) -> pd.DataFrame:
        x = close.copy()
        x.columns = [normalize_pair(str(c)) for c in x.columns]
        x = x.loc[:, ~x.columns.duplicated(keep='first')]
        x = x.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
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
    def _fit_factor_beta(y: pd.Series, factors: pd.DataFrame, index: pd.Index) -> Optional[np.ndarray]:
        if factors is None or factors.empty:
            return None
        frame = pd.concat([y.rename('y'), factors], axis=1).reindex(index)
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < max(30, factors.shape[1] * 10):
            return None
        yy = frame.pop('y').to_numpy(dtype=float)
        X = frame.to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(X)), X])
        return np.linalg.pinv(X) @ yy

    @staticmethod
    def _apply_factor_beta(y: pd.Series, factors: pd.DataFrame, beta: Optional[np.ndarray]) -> pd.Series:
        if beta is None or factors is None or factors.empty:
            return y.astype(float)
        frame = pd.concat([y.rename('y'), factors], axis=1).replace([np.inf, -np.inf], np.nan)
        valid = frame.notna().all(axis=1)
        out = pd.Series(index=y.index, dtype=float)
        if not valid.any():
            return out
        yy = frame.loc[valid, 'y'].to_numpy(dtype=float)
        X = frame.loc[valid].drop(columns='y').to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(X)), X])
        out.loc[valid] = yy - X @ beta
        return out

    def _residualize(self, y: pd.Series, factors: pd.DataFrame) -> pd.Series:
        beta = self._fit_factor_beta(y, factors, y.index)
        return self._apply_factor_beta(y, factors, beta)

    @staticmethod
    def _align_lead(source: pd.Series, target: pd.Series, lag: int) -> pd.DataFrame:
        return pd.concat([source.rename('x'), target.shift(-int(lag)).rename('y_future'), target.rename('y_now')], axis=1).replace([np.inf, -np.inf], np.nan).dropna()

    def _stability(self, aligned: pd.DataFrame) -> float:
        n = len(aligned)
        if n < self.min_obs:
            return 0.0
        bounds = np.linspace(0, n, self.stability_splits + 1, dtype=int)
        vals: List[float] = []
        for i in range(self.stability_splits):
            block = aligned.iloc[int(bounds[i]):int(bounds[i + 1])]
            if len(block) < 20:
                continue
            c = self._corr(block['x'].to_numpy(), block['y_future'].to_numpy())
            if np.isfinite(c):
                vals.append(float(c))
        if len(vals) < 2:
            return 0.0
        global_corr = self._corr(aligned['x'].to_numpy(), aligned['y_future'].to_numpy())
        if not np.isfinite(global_corr) or abs(global_corr) < 1e-12:
            return 0.0
        sign_consistency = float(np.mean(np.sign(vals) == np.sign(global_corr)))
        magnitude_consistency = float(np.clip(np.median(np.abs(vals)) / abs(global_corr), 0.0, 1.0))
        variability_penalty = float(np.exp(-np.std(vals) / (abs(global_corr) + 1e-06)))
        return float(np.clip(sign_consistency * magnitude_consistency * variability_penalty, 0.0, 1.0))

    @staticmethod
    def _r2(y: np.ndarray, X: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).ravel()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if len(y) < 10 or len(X) != len(y):
            return 0.0
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
        y = aligned['y_future'].to_numpy(dtype=float)
        baseline = aligned[['y_now']].to_numpy(dtype=float)
        full = aligned[['y_now', 'x']].to_numpy(dtype=float)
        return float(max(0.0, self._r2(y, full) - self._r2(y, baseline)))

    def _permutation_p(self, aligned: pd.DataFrame, observed: float) -> float:
        if len(aligned) < self.min_obs or not np.isfinite(observed):
            return 1.0
        x = aligned['x'].to_numpy(dtype=float)
        y = aligned['y_future'].to_numpy(dtype=float)
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
            c = self._corr(np.roll(x, shift), y)
            if np.isfinite(c):
                valid += 1
                if abs(c) >= abs(observed):
                    hits += 1
        if valid == 0:
            return 1.0
        return float((hits + 1) / (valid + 1))

    @staticmethod
    def _corr_pvalue(corr: float, n_obs: int) -> float:
        """Two-sided Fisher-z normal approximation for correlation.

        Used as a descriptive analytic p-value. FDR is intentionally applied
        to pooled out-of-sample walk-forward p-values, not to this in-sample
        approximation.
        """
        if not np.isfinite(corr) or int(n_obs) <= 3:
            return 1.0
        r = float(np.clip(corr, -0.999999, 0.999999))
        z = abs(np.arctanh(r)) * math.sqrt(max(int(n_obs) - 3, 1))
        return float(np.clip(math.erfc(z / math.sqrt(2.0)), 0.0, 1.0))

    @staticmethod
    def _bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
        p = np.asarray(pvalues, dtype=float)
        q = np.ones_like(p, dtype=float)
        valid = np.isfinite(p)
        if not valid.any():
            return q
        pv = np.clip(p[valid], 0.0, 1.0)
        order = np.argsort(pv)
        ranked = pv[order]
        m = len(ranked)
        raw = ranked * m / np.arange(1, m + 1)
        adj = np.minimum.accumulate(raw[::-1])[::-1]
        adj = np.clip(adj, 0.0, 1.0)
        restored = np.empty_like(adj)
        restored[order] = adj
        q[valid] = restored
        return q

    def _bootstrap_ci(self, aligned: pd.DataFrame, observed: float) -> Tuple[float, float]:
        n = len(aligned)
        if n < self.min_obs or not np.isfinite(observed):
            return (np.nan, np.nan)
        x = aligned['x'].to_numpy(dtype=float)
        y = aligned['y_future'].to_numpy(dtype=float)
        block = max(5, int(round(np.sqrt(n))))
        values: List[float] = []
        for _ in range(self.bootstrap_n):
            idx_parts: List[np.ndarray] = []
            while sum((len(a) for a in idx_parts)) < n:
                start_max = max(1, n - block + 1)
                start = int(self.rng.integers(0, start_max))
                idx_parts.append(np.arange(start, min(start + block, n)))
            idx = np.concatenate(idx_parts)[:n]
            c = self._corr(x[idx], y[idx])
            if np.isfinite(c):
                values.append(float(c))
        if len(values) < max(10, self.bootstrap_n // 3):
            return (np.nan, np.nan)
        low, high = np.quantile(values, [0.05, 0.95])
        return (float(low), float(high))

    @staticmethod
    def _shared_factor(source: str, target: str) -> List[str]:
        sb, sq = split_pair(source)
        tb, tq = split_pair(target)
        return sorted(set((sb, sq)).intersection((tb, tq)))

    def _conditional_series(self, source: str, target: str, returns: pd.DataFrame, latent_strength: Optional[pd.DataFrame], fit_index: Optional[pd.Index]=None) -> Tuple[pd.Series, pd.Series, List[str]]:
        sx = returns[source].astype(float)
        ty = returns[target].astype(float)
        shared = self._shared_factor(source, target)
        if not shared or latent_strength is None or latent_strength.empty:
            return (sx, ty, shared)
        factor_cols = [c for c in shared if c in latent_strength.columns]
        if not factor_cols:
            return (sx, ty, shared)
        factors = latent_strength.reindex(returns.index)[factor_cols]
        if fit_index is None:
            fit_index = returns.index
        beta_s = self._fit_factor_beta(sx, factors, fit_index)
        beta_t = self._fit_factor_beta(ty, factors, fit_index)
        return (self._apply_factor_beta(sx, factors, beta_s), self._apply_factor_beta(ty, factors, beta_t), shared)

    def _walkforward(self, source: str, target: str, returns: pd.DataFrame, latent_strength: Optional[pd.DataFrame]) -> Tuple[float, float, int, float, int]:
        base = pd.concat([returns[source], returns[target]], axis=1).dropna()
        n = len(base)
        if n < max(self.min_obs, 120):
            return (np.nan, 0.0, 0, 1.0, 0)
        test_size = max(30, n // 8)
        first_train = max(self.min_obs, n - self.walkforward_folds * test_size)
        fold_corrs: List[float] = []
        train_signs: List[float] = []
        pooled_x: List[np.ndarray] = []
        pooled_y: List[np.ndarray] = []
        for fold in range(self.walkforward_folds):
            train_end = first_train + fold * test_size
            test_end = min(n, train_end + test_size)
            if test_end - train_end < 20:
                continue
            train_idx = base.index[:max(0, train_end - self.max_lag)]
            purged_test_end = max(train_end, test_end - self.max_lag)
            test_idx = base.index[train_end:purged_test_end]
            if len(train_idx) < self.min_obs or len(test_idx) < 20:
                continue
            sx, ty, _ = self._conditional_series(source, target, returns, latent_strength, fit_index=train_idx)
            best_lag = None
            best_corr = None
            for lag in range(1, self.max_lag + 1):
                aligned = self._align_lead(sx, ty, lag)
                train_aligned = aligned.loc[aligned.index.intersection(train_idx)]
                c = self._corr(train_aligned['x'].to_numpy(), train_aligned['y_future'].to_numpy())
                if not np.isfinite(c):
                    continue
                if best_corr is None or abs(c) > abs(best_corr):
                    best_corr = float(c)
                    best_lag = lag
            if best_lag is None or best_corr is None:
                continue
            aligned = self._align_lead(sx, ty, best_lag)
            test_aligned = aligned.loc[aligned.index.intersection(test_idx)]
            x = test_aligned['x'].to_numpy(dtype=float)
            y = test_aligned['y_future'].to_numpy(dtype=float)
            c_test = self._corr(x, y)
            if np.isfinite(c_test):
                fold_corrs.append(float(c_test))
                train_signs.append(float(np.sign(best_corr)))
                if len(x) >= 20 and np.std(x) > 1e-15 and (np.std(y) > 1e-15):
                    pooled_x.append((x - np.mean(x)) / np.std(x))
                    pooled_y.append((y - np.mean(y)) / np.std(y))
        if not fold_corrs:
            return (np.nan, 0.0, 0, 1.0, 0)
        median_ic = float(np.median(fold_corrs))
        sign_hits = [np.sign(c) == train_signs[i] for i, c in enumerate(fold_corrs)]
        sign_rate = float(np.mean(sign_hits)) if sign_hits else 0.0
        if pooled_x:
            px = np.concatenate(pooled_x)
            py = np.concatenate(pooled_y)
            pooled_corr = self._corr(px, py)
            oos_n = int(min(len(px), len(py)))
            oos_p = self._corr_pvalue(pooled_corr, oos_n) if np.isfinite(pooled_corr) else 1.0
        else:
            oos_n = 0
            oos_p = 1.0
        return (median_ic, sign_rate, len(fold_corrs), float(oos_p), oos_n)

    def _decay(self, aligned: pd.DataFrame) -> Tuple[float, str]:
        n = len(aligned)
        if n < 80:
            return (np.nan, 'UNKNOWN')
        split = max(40, int(round(n * 0.67)))
        hist = aligned.iloc[:split]
        recent = aligned.iloc[split:]
        if len(recent) < 20:
            return (np.nan, 'UNKNOWN')
        c_hist = self._corr(hist['x'].to_numpy(), hist['y_future'].to_numpy())
        c_recent = self._corr(recent['x'].to_numpy(), recent['y_future'].to_numpy())
        if not np.isfinite(c_hist) or not np.isfinite(c_recent):
            return (np.nan, 'UNKNOWN')
        ratio = abs(c_recent) / (abs(c_hist) + 1e-06)
        sign_penalty = 1.0 if np.sign(c_recent) == np.sign(c_hist) else 0.25
        score = float(np.clip(ratio, 0.0, 1.0) * sign_penalty)
        if score >= 0.8:
            state = 'LOW'
        elif score >= 0.5:
            state = 'MEDIUM'
        else:
            state = 'HIGH'
        return (score, state)

    def build(self, close: pd.DataFrame, latent_strength: Optional[pd.DataFrame]=None, regime: Optional[RegimeResult]=None, top_n: int=30) -> pd.DataFrame:
        returns = self.log_returns(close)
        if returns.empty or returns.shape[1] < 2:
            out = pd.DataFrame(columns=self.COLUMNS)
            out.attrs['diagnostics'] = {}
            return out
        candidates: List[Dict[str, object]] = []
        names = list(map(str, returns.columns))
        for source, target in permutations(names, 2):
            base = pd.concat([returns[source].rename('x'), returns[target].rename('y')], axis=1).dropna()
            if len(base) < self.min_obs:
                continue
            discovery_n = max(self.min_obs, int(len(base) * 0.7))
            discovery_idx = base.index[:max(0, discovery_n - self.max_lag)]
            if len(discovery_idx) < self.min_obs:
                continue
            sx, ty, _ = self._conditional_series(source, target, returns, latent_strength, fit_index=discovery_idx)
            best_lag = None
            best_raw = None
            for lag in range(1, self.max_lag + 1):
                aligned = self._align_lead(sx, ty, lag)
                train = aligned.loc[aligned.index.intersection(discovery_idx)]
                c = self._corr(train['x'].to_numpy(), train['y_future'].to_numpy())
                if not np.isfinite(c):
                    continue
                if best_raw is None or abs(c) > abs(best_raw):
                    best_lag, best_raw = (lag, float(c))
            if best_lag is not None and best_raw is not None:
                candidates.append({'source': source, 'target': target, 'lag': best_lag, 'raw_corr': best_raw})
        candidates = sorted(candidates, key=lambda d: abs(float(d['raw_corr'])), reverse=True)
        candidates = candidates[:max(top_n * 5, 80)]
        records: List[Dict[str, object]] = []
        for cand in candidates:
            source = str(cand['source'])
            target = str(cand['target'])
            lag = int(cand['lag'])
            raw_corr = float(cand['raw_corr'])
            pair_base = pd.concat([returns[source].rename('x'), returns[target].rename('y')], axis=1).dropna()
            pair_discovery_n = max(self.min_obs, int(len(pair_base) * 0.7))
            pair_discovery_idx = pair_base.index[:max(0, pair_discovery_n - self.max_lag)]
            sx_cond, ty_cond, shared = self._conditional_series(source, target, returns, latent_strength, fit_index=pair_discovery_idx)
            aligned = self._align_lead(sx_cond, ty_cond, lag)
            if len(aligned) < self.min_obs:
                continue
            cond_corr = self._corr(aligned['x'].to_numpy(), aligned['y_future'].to_numpy())
            if not np.isfinite(cond_corr):
                continue
            stability = self._stability(aligned)
            analytic_p = self._corr_pvalue(cond_corr, len(aligned))
            p = self._permutation_p(aligned, cond_corr)
            inc_r2 = self._incremental_r2(aligned)
            boot_low, boot_high = self._bootstrap_ci(aligned, cond_corr)
            wf_ic, wf_sign, wf_folds, wf_p, wf_n = self._walkforward(source, target, returns, latent_strength)
            decay_score, decay_state = self._decay(aligned)
            regime_corr = np.nan
            if regime is not None and (not regime.dominant.empty):
                current = regime.current_regime
                labels = regime.dominant.reindex(aligned.index)
                subset = aligned.loc[labels == current]
                if len(subset) >= 30:
                    regime_corr = self._corr(subset['x'].to_numpy(), subset['y_future'].to_numpy())
            records.append({'source': source, 'target': target, 'lag': lag, 'shared_factor': ','.join(shared) if shared else '—', 'raw_corr': raw_corr, 'conditional_corr': float(cond_corr), 'regime_corr': float(regime_corr) if np.isfinite(regime_corr) else np.nan, 'stability': float(stability), 'analytic_p': float(analytic_p), 'perm_p': float(p), 'fdr_q': np.nan, 'bootstrap_low': float(boot_low) if np.isfinite(boot_low) else np.nan, 'bootstrap_high': float(boot_high) if np.isfinite(boot_high) else np.nan, 'walkforward_ic': float(wf_ic) if np.isfinite(wf_ic) else np.nan, 'walkforward_sign_rate': float(wf_sign), 'walkforward_p': float(wf_p), 'walkforward_n': int(wf_n), 'incremental_r2': float(inc_r2), 'decay_score': float(decay_score) if np.isfinite(decay_score) else np.nan, 'decay_state': decay_state, 'evidence_score': 0.0, 'edge_score': 0.0, 'survives': False, 'n_obs': int(len(aligned)), '_wf_folds': int(wf_folds)})
        if not records:
            out = pd.DataFrame(columns=self.COLUMNS)
            out.attrs['diagnostics'] = {'evaluated': 0, 'fdr_pass': 0, 'bootstrap_pass': 0, 'walkforward_pass': 0, 'survivors': 0, 'survival_rate': 0.0}
            return out
        df = pd.DataFrame(records)
        df['fdr_q'] = self._bh_qvalues(df['walkforward_p'].to_numpy(dtype=float))
        bootstrap_pass = (df['bootstrap_low'] > 0) & (df['bootstrap_high'] > 0) | (df['bootstrap_low'] < 0) & (df['bootstrap_high'] < 0)
        fdr_pass = df['fdr_q'] <= self.fdr_alpha
        perm_pass = df['perm_p'] <= 0.1
        wf_pass = (df['walkforward_ic'].abs().fillna(0.0) >= 0.07) & (df['walkforward_sign_rate'] >= 2.0 / 3.0) & (df['_wf_folds'] >= 2)
        stability_pass = df['stability'] >= 0.3
        decay_pass = df['decay_score'].fillna(0.0) >= 0.35
        significance_component = 0.55 * (1.0 - df['fdr_q']).clip(0.0, 1.0) + 0.45 * (1.0 - df['perm_p']).clip(0.0, 1.0)
        bootstrap_component = bootstrap_pass.astype(float)
        wf_component = (df['walkforward_ic'].abs().fillna(0.0) / 0.25).clip(0.0, 1.0) * df['walkforward_sign_rate'].clip(0.0, 1.0)
        r2_component = (np.sqrt(df['incremental_r2'].clip(lower=0.0)) * 4.0).clip(0.0, 1.0)
        regime_component = (df['regime_corr'].abs().fillna(0.0) / 0.35).clip(0.0, 1.0)
        decay_component = df['decay_score'].fillna(0.0).clip(0.0, 1.0)
        sample_component = np.sqrt(df['n_obs'] / 500.0).clip(0.35, 1.0)
        df['evidence_score'] = sample_component * (0.18 * (df['conditional_corr'].abs() / 0.35).clip(0.0, 1.0) + 0.14 * df['stability'].clip(0.0, 1.0) + 0.15 * significance_component + 0.12 * bootstrap_component + 0.2 * wf_component + 0.08 * r2_component + 0.05 * regime_component + 0.08 * decay_component)
        df['edge_score'] = df['evidence_score']
        df['survives'] = fdr_pass & perm_pass & bootstrap_pass & wf_pass & stability_pass & decay_pass & (df['evidence_score'] >= 0.45)
        seq_fdr = fdr_pass
        seq_perm = seq_fdr & perm_pass
        seq_boot = seq_perm & bootstrap_pass
        seq_wf = seq_boot & wf_pass
        seq_stability = seq_wf & stability_pass
        seq_decay = seq_stability & decay_pass
        seq_score = seq_decay & (df['evidence_score'] >= 0.45)
        finite_wfp = df.loc[np.isfinite(df['walkforward_p']), 'walkforward_p']
        finite_q = df.loc[np.isfinite(df['fdr_q']), 'fdr_q']
        diagnostics = {'evaluated': int(len(df)), 'fdr_pass': int(fdr_pass.sum()), 'permutation_pass': int(perm_pass.sum()), 'bootstrap_pass': int(bootstrap_pass.sum()), 'walkforward_pass': int(wf_pass.sum()), 'stability_pass': int(stability_pass.sum()), 'decay_pass': int(decay_pass.sum()), 'survivors': int(df['survives'].sum()), 'survival_rate': float(df['survives'].mean()) if len(df) else 0.0, 'fdr_alpha': self.fdr_alpha, 'min_walkforward_p': float(finite_wfp.min()) if len(finite_wfp) else np.nan, 'min_fdr_q': float(finite_q.min()) if len(finite_q) else np.nan, 'sequential': {'fdr': int(seq_fdr.sum()), 'permutation': int(seq_perm.sum()), 'bootstrap': int(seq_boot.sum()), 'walkforward': int(seq_wf.sum()), 'stability': int(seq_stability.sum()), 'decay': int(seq_decay.sum()), 'evidence_score': int(seq_score.sum())}}
        df = df.sort_values(['survives', 'evidence_score', 'walkforward_ic'], ascending=[False, False, False]).head(max(1, int(top_n))).reset_index(drop=True)
        df = df.drop(columns=['_wf_folds'], errors='ignore')
        df = df.reindex(columns=self.COLUMNS)
        df.attrs['diagnostics'] = diagnostics
        return df

@dataclass
class StressResult:
    current_stress: float
    stress_percentile: float
    coherence: float
    series: pd.Series
    triangle_z: pd.DataFrame

class NetworkStressEngine:
    """Aggregate standardized triangle inconsistencies into an FX network stress index."""

    def __init__(self, window: int=96, min_periods: int=30) -> None:
        self.window = max(30, int(window))
        self.min_periods = max(15, int(min_periods))

    def fit(self, triangles: pd.DataFrame) -> StressResult:
        if triangles is None or triangles.empty:
            return StressResult(np.nan, np.nan, np.nan, pd.Series(dtype=float), pd.DataFrame())
        tri = triangles.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        mu = tri.rolling(self.window, min_periods=self.min_periods).median()
        mad = (tri - mu).abs().rolling(self.window, min_periods=self.min_periods).median()
        robust_sigma = (1.4826 * mad).clip(lower=1e-08)
        z = ((tri - mu) / robust_sigma).clip(-20, 20)
        stress = z.abs().median(axis=1, skipna=True).rename('network_stress')
        valid = stress.dropna()
        if valid.empty:
            return StressResult(np.nan, np.nan, np.nan, stress, z)
        current = float(valid.iloc[-1])
        hist = valid.iloc[:-1]
        percentile = float((hist <= current).mean() * 100.0) if len(hist) >= 10 else np.nan
        coherence = float(np.exp(-min(current, 10.0)))
        return StressResult(current, percentile, coherence, stress, z)

class StructuralDiagnostics:
    """Deployment-safe divergence and triangle diagnostics."""

    @staticmethod
    def divergence_table(residual_z: pd.DataFrame, threshold: float=1.25) -> pd.DataFrame:
        cols = ['pair', 'residual_z', 'direction', 'magnitude']
        if residual_z is None or not isinstance(residual_z, pd.DataFrame) or residual_z.empty:
            return pd.DataFrame(columns=cols)
        latest = residual_z.iloc[-1].dropna()
        if latest.empty:
            return pd.DataFrame(columns=cols)
        out = pd.DataFrame({'pair': latest.index.astype(str), 'residual_z': latest.to_numpy(dtype=float)})
        out['direction'] = np.where(out['residual_z'] > 0, 'OVERPERFORMING', 'UNDERPERFORMING')
        out['magnitude'] = out['residual_z'].abs()
        return out[out['magnitude'] >= float(threshold)].sort_values('magnitude', ascending=False).reset_index(drop=True)

    @staticmethod
    def triangular_residuals(close: pd.DataFrame) -> pd.DataFrame:
        if close is None or not isinstance(close, pd.DataFrame) or close.empty:
            return pd.DataFrame()
        work = close.copy()
        work.columns = [normalize_pair(str(c)) for c in work.columns]
        work = work.loc[:, ~work.columns.duplicated(keep='first')]
        numeric = work.apply(pd.to_numeric, errors='coerce')
        logs = np.log(numeric.where(numeric > 0))
        currencies = sorted({ccy for pair in logs.columns for ccy in split_pair(str(pair))})

        def get_log(a: str, b: str):
            direct, inverse = (a + b, b + a)
            if direct in logs.columns:
                return logs.loc[:, direct]
            if inverse in logs.columns:
                return -logs.loc[:, inverse]
            return None
        result: Dict[str, pd.Series] = {}
        for i, a in enumerate(currencies):
            for j in range(i + 1, len(currencies)):
                b = currencies[j]
                for k in range(j + 1, len(currencies)):
                    c = currencies[k]
                    ab, bc, ac = (get_log(a, b), get_log(b, c), get_log(a, c))
                    if ab is None or bc is None or ac is None:
                        continue
                    result[f'{a}-{b}-{c}'] = ab + bc - ac
        return pd.DataFrame(result, index=logs.index)
st.set_page_config(page_title='NEXUS FX CORE v0.2.2.4', page_icon='◈', layout='wide')
st.markdown('\n<style>\n.stApp { background:#080b12; color:#eef3ff; }\n.block-container { padding-top: 1.1rem; max-width: 1500px; }\n.nexus-title {font-size:clamp(2.1rem,8vw,4.5rem);font-weight:800;letter-spacing:.08em;\n background:linear-gradient(90deg,#42e8ff,#8c68ff,#ff4fd8);-webkit-background-clip:text;color:transparent;}\n.nexus-sub {color:#93a4c7;letter-spacing:.18em;text-transform:uppercase;font-size:.76rem;}\n.card {background:linear-gradient(145deg,#111724,#0b101a);border:1px solid #202c43;border-radius:18px;padding:16px;}\n.metric-label {color:#8fa1c4;font-size:.72rem;text-transform:uppercase;letter-spacing:.12em;}\n.big {font-size:1.7rem;font-weight:750;}\n</style>\n', unsafe_allow_html=True)
st.markdown('<div class="nexus-title">NEXUS FX CORE</div>', unsafe_allow_html=True)
st.markdown('<div class="nexus-sub">Latent State · Regime Reliability · FDR · Bootstrap · Walk-Forward · Network Stress</div>', unsafe_allow_html=True)
st.caption(f'NEXUS v{APP_VERSION} · Intelligence {INTELLIGENCE_VERSION} · research prototype')
DEFAULT_PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD']
with st.expander('⚙️ Configuración', expanded=True):
    c1, c2, c3 = st.columns(3)
    interval = c1.selectbox('Temporalidad', ['1h', '15m', '5m', '1d'], index=0)
    period_options = {'1h': ['5d', '1mo', '3mo'], '15m': ['5d', '1mo'], '5m': ['1d', '5d', '1mo'], '1d': ['6mo', '1y', '2y', '5y']}
    period = c2.selectbox('Histórico', period_options[interval], index=min(1, len(period_options[interval]) - 1))
    max_lag = c3.slider('Lead/Lag máximo (velas)', 1, 12, 5)
    pairs = st.multiselect('Pares', DEFAULT_PAIRS, default=DEFAULT_PAIRS)
    with st.expander('Ajustes de investigación avanzada', expanded=False):
        a1, a2 = st.columns(2)
        top_edges = a1.slider('Aristas candidatas a mostrar', 10, 40, 20, step=5)
        permutations_n = a2.slider('Permutaciones por arista', 48, 499, 199, step=25)
        a3, a4 = st.columns(2)
        bootstrap_n = a3.slider('Bootstrap por arista', 48, 499, 199, step=25)
        fdr_alpha = a4.select_slider('FDR máximo (q)', options=[0.05, 0.075, 0.1, 0.15], value=0.1)

@st.cache_data(ttl=55, show_spinner=False)
def get_data(pairs_tuple, period, interval):
    return download_close_matrix(list(pairs_tuple), period=period, interval=interval)

def analyze(close: pd.DataFrame):
    latent_engine = LatentCurrencyEngine()
    latent = latent_engine.fit(close)
    regime_engine = RegimeEngine()
    regime = regime_engine.fit(close)
    diagnostics = StructuralDiagnostics()
    divergences = diagnostics.divergence_table(latent.residual_z, threshold=1.25)
    triangles = diagnostics.triangular_residuals(close)
    stress = NetworkStressEngine().fit(triangles)
    graph_warning = None
    try:
        graph = CausalGraphEngine(max_lag=max_lag, min_obs=80 if interval != '1d' else 60, permutations_n=permutations_n, bootstrap_n=bootstrap_n, walkforward_folds=3, fdr_alpha=fdr_alpha).build(close, latent_strength=latent.strength, regime=regime, top_n=top_edges)
    except Exception as exc:
        graph = pd.DataFrame(columns=CausalGraphEngine.COLUMNS)
        graph_warning = f'Grafo causal candidato desactivado por seguridad: {type(exc).__name__}: {exc}'
    return (latent, regime, graph, divergences, triangles, stress, graph_warning)
if st.button('⚡ Construir estado del mercado', use_container_width=True, type='primary'):
    if len(pairs) < 7:
        st.error('Selecciona al menos 7 pares conectados para estimar la red monetaria con estabilidad.')
        st.stop()
    try:
        with st.spinner('Sincronizando FX, resolviendo estado latente, régimen y grafo condicional...'):
            close = get_data(tuple(pairs), period, interval)
            result = analyze(close)
            st.session_state['nexus_v0224_result'] = (close,) + result
    except Exception as exc:
        st.exception(exc)
if 'nexus_v0224_result' not in st.session_state:
    st.info('Pulsa “Construir estado del mercado”. v0.2.2.4 endurece la evidencia estadística; sigue sin emitir órdenes BUY/SELL.')
    st.stop()
close, latent, regime, graph, divergences, triangles, stress, graph_warning = st.session_state['nexus_v0224_result']
if graph_warning:
    st.warning(graph_warning)
st.subheader('Estado global')
reg_probs = regime.current_probabilities
regime_name = regime.current_regime
regime_conf = float(reg_probs.iloc[0]) if not reg_probs.empty else np.nan
regime_reliability = regime.current_reliability
graph_diag = graph.attrs.get('diagnostics', {}) if isinstance(graph, pd.DataFrame) else {}
survivors = int(graph_diag.get('survivors', int(graph['survives'].sum()) if not graph.empty and 'survives' in graph else 0))
survival_rate = float(graph_diag.get('survival_rate', 0.0))
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(f'<div class="card"><div class="metric-label">Régimen dominante</div><div class="big">{regime_name}</div><small>posterior {regime_conf:.1%}</small></div>', unsafe_allow_html=True)
with m2:
    rtxt = f'{regime_reliability:.1%}' if np.isfinite(regime_reliability) else 'N/D'
    st.markdown(f'<div class="card"><div class="metric-label">Fiabilidad régimen</div><div class="big">{rtxt}</div><small>autoconsistencia, no prob. calibrada</small></div>', unsafe_allow_html=True)
with m3:
    pctl = stress.stress_percentile
    ptxt = f'{pctl:.0f}º' if np.isfinite(pctl) else 'N/D'
    st.markdown(f'<div class="card"><div class="metric-label">Network stress</div><div class="big">{stress.current_stress:.2f}</div><small>percentil {ptxt}</small></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="card"><div class="metric-label">Coherencia triangular</div><div class="big">{stress.coherence:.1%}</div><small>1 = máxima coherencia</small></div>', unsafe_allow_html=True)
with m5:
    st.markdown(f'<div class="card"><div class="metric-label">Aristas sobrevivientes</div><div class="big">{survivors}</div><small>survival rate {survival_rate:.1%}</small></div>', unsafe_allow_html=True)
if not reg_probs.empty:
    fig_reg = go.Figure(go.Bar(x=reg_probs.index, y=reg_probs.values))
    fig_reg.update_layout(template='plotly_dark', height=300, margin=dict(l=20, r=20, t=20, b=30), yaxis_tickformat='.0%', title='Posterior de régimen')
    st.plotly_chart(fig_reg, use_container_width=True)
st.subheader('Estado latente de monedas')
strength_z = LatentCurrencyEngine.standardized_snapshot(latent.strength).sort_values(ascending=False)
vel_z = LatentCurrencyEngine.standardized_snapshot(latent.velocity).reindex(strength_z.index)
acc_z = LatentCurrencyEngine.standardized_snapshot(latent.acceleration).reindex(strength_z.index)
unc = latent.uncertainty.iloc[-1].reindex(strength_z.index)
cols = st.columns(4)
for i, ccy in enumerate(strength_z.index):
    s = strength_z[ccy]
    v = vel_z.get(ccy, np.nan)
    a = acc_z.get(ccy, np.nan)
    with cols[i % 4]:
        st.markdown(f'<div class="card"><b>{ccy}</b><br><span style="font-size:2rem">{s:+.2f}σ</span><br><small>vel {v:+.2f}σ · acc {a:+.2f}σ · unc {unc[ccy]:.5f}</small></div>', unsafe_allow_html=True)
strength_hist = latent.strength.copy().replace([np.inf, -np.inf], np.nan)
cs_mean = strength_hist.mean(axis=1)
cs_std = strength_hist.std(axis=1, ddof=0).replace(0.0, np.nan)
strength_cs_z = strength_hist.sub(cs_mean, axis=0).div(cs_std, axis=0).clip(-5.0, 5.0).tail(350)
fig = go.Figure()
for ccy in strength_cs_z.columns:
    fig.add_trace(go.Scatter(x=strength_cs_z.index, y=strength_cs_z[ccy], mode='lines', name=ccy))
fig.add_hline(y=0.0, line_width=1, opacity=0.35)
fig.update_layout(template='plotly_dark', height=430, margin=dict(l=20, r=20, t=35, b=20), title='Fuerza relativa estandarizada (cross-section)', yaxis_title='σ relativo')
st.plotly_chart(fig, use_container_width=True)
t1, t2, t3, t4, t5 = st.tabs(['Grafo / Evidencia', 'Régimen', 'Divergencias', 'Stress / Triángulos', 'Diagnóstico'])
with t1:
    st.caption('v0.2.2.4 exige FDR, bootstrap, walk-forward, estabilidad y decay. Una arista sobreviviente sigue siendo una hipótesis predictiva, no causalidad económica demostrada.')
    if graph.empty:
        st.info('No hay aristas candidatas suficientes para esta muestra.')
    else:
        only_survivors = st.toggle('Mostrar solo aristas sobrevivientes', value=False)
        shown = graph.loc[graph['survives']].copy() if only_survivors else graph.copy()
        display_cols = ['source', 'target', 'lag', 'shared_factor', 'conditional_corr', 'fdr_q', 'bootstrap_low', 'bootstrap_high', 'walkforward_ic', 'walkforward_sign_rate', 'walkforward_p', 'walkforward_n', 'stability', 'decay_state', 'incremental_r2', 'evidence_score', 'survives', 'n_obs']
        st.dataframe(shown[display_cols], use_container_width=True, hide_index=True)
        top = shown.head(12).copy()
        if not top.empty:
            top['edge'] = top['source'] + ' → ' + top['target'] + ' (L' + top['lag'].astype(str) + ')'
            fig_edges = go.Figure(go.Bar(x=top['evidence_score'], y=top['edge'], orientation='h'))
            fig_edges.update_layout(template='plotly_dark', height=max(350, 32 * len(top)), margin=dict(l=20, r=20, t=35, b=20), title='NEXUS Evidence Score', yaxis={'autorange': 'reversed'}, xaxis_range=[0, 1])
            st.plotly_chart(fig_edges, use_container_width=True)
with t2:
    st.caption('El posterior de régimen no está externamente calibrado. v0.2.1 añade entropía, persistencia y una métrica separada de fiabilidad/autoconsistencia.')
    if regime.probabilities.empty:
        st.info('No hay histórico suficiente para régimen.')
    else:
        fig_hist = go.Figure()
        for name in regime.probabilities.columns:
            fig_hist.add_trace(go.Scatter(x=regime.probabilities.index, y=regime.probabilities[name], mode='lines', stackgroup='one', name=name))
        fig_hist.update_layout(template='plotly_dark', height=420, margin=dict(l=20, r=20, t=35, b=20), yaxis_tickformat='.0%', title='Evolución de probabilidades de régimen')
        st.plotly_chart(fig_hist, use_container_width=True)
        latest_metrics = regime.metrics.iloc[-1].to_frame('valor')
        st.dataframe(latest_metrics, use_container_width=True)
with t3:
    st.caption('Residual observado − reconstruido por el estado latente. Una anomalía relativa no es una orden de trading.')
    st.dataframe(divergences, use_container_width=True, hide_index=True)
with t4:
    st.caption('Network Stress estandariza inconsistencias triangulares contra su propia historia. Un aumento coordinado pesa más que un residual aislado.')
    if stress.series.empty:
        st.info('No hay suficientes triángulos para calcular stress.')
    else:
        fig_stress = go.Figure(go.Scatter(x=stress.series.index, y=stress.series, mode='lines', name='Network Stress'))
        fig_stress.update_layout(template='plotly_dark', height=340, margin=dict(l=20, r=20, t=35, b=20), title='FX Network Stress')
        st.plotly_chart(fig_stress, use_container_width=True)
        if not triangles.empty:
            latest_tri = triangles.iloc[-1].abs().sort_values(ascending=False).rename('abs_log_residual').to_frame().head(20)
            st.dataframe(latest_tri, use_container_width=True)
with t5:
    st.markdown('**Pipeline de evidencia v0.2.1:**')
    st.markdown('1. Descubre el lag en el 70% inicial (con purga temporal).  \n2. Residualiza la moneda compartida cuando existe.  \n3. Mide estabilidad cronológica.  \n4. Calcula `perm_p` con desplazamientos circulares.  \n5. Corrige pruebas múltiples con Benjamini–Hochberg sobre p-values OOS (`fdr_q`).  \n6. Exige intervalo bootstrap que no cruce cero.  \n7. Valida en walk-forward fuera de la ventana de descubrimiento.  \n8. Penaliza relaciones que muestran decay reciente.  \n9. Combina las pruebas en `evidence_score`.')
    d = graph_diag
    if d:
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric('Evaluadas', d.get('evaluated', 0))
        dc2.metric('Pasan FDR', d.get('fdr_pass', 0))
        dc3.metric('Pasan bootstrap', d.get('bootstrap_pass', 0))
        dc4, dc5, dc6 = st.columns(3)
        dc4.metric('Pasan walk-forward', d.get('walkforward_pass', 0))
        dc5.metric('Sobreviven', d.get('survivors', 0))
        dc6.metric('Edge Survival Rate', f"{d.get('survival_rate', 0.0):.1%}")
        pmin = d.get('min_walkforward_p', np.nan)
        qmin = d.get('min_fdr_q', np.nan)
        dx1, dx2 = st.columns(2)
        dx1.metric('Mejor p OOS', f'{pmin:.5f}' if np.isfinite(pmin) else 'N/D')
        dx2.metric('Mejor FDR q', f'{qmin:.5f}' if np.isfinite(qmin) else 'N/D')
        seq = d.get('sequential', {})
        if seq:
            st.markdown('**Embudo secuencial real:**')
            st.caption('FDR → permutación → bootstrap → walk-forward → estabilidad → decay → score')
            s1, s2, s3, s4 = st.columns(4)
            s1.metric('Tras FDR', seq.get('fdr', 0))
            s2.metric('+ Perm.', seq.get('permutation', 0))
            s3.metric('+ Bootstrap', seq.get('bootstrap', 0))
            s4.metric('+ Walk-forward', seq.get('walkforward', 0))
            s5, s6, s7 = st.columns(3)
            s5.metric('+ Estabilidad', seq.get('stability', 0))
            s6.metric('+ Decay', seq.get('decay', 0))
            s7.metric('+ Score', seq.get('evidence_score', 0))
    st.warning('FDR y walk-forward reducen falsos positivos, pero no convierten correlación temporal en causalidad. Antes de dinero real todavía faltan costes, spreads, slippage, macro point-in-time y backtest purgado completo.')
st.divider()
st.caption(f'Filas sincronizadas: {len(close):,} · Pares: {len(close.columns)} · Último dato: {close.index[-1]}')
st.warning('NEXUS v0.2.2.4 es investigación estadística estructural. No ejecuta operaciones ni recomienda apalancamiento.')