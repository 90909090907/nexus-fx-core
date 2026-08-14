from __future__ import annotations

from dataclasses import dataclass
import math
from itertools import permutations
from typing import Dict, List, Optional, Tuple

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

    @property
    def current_reliability(self) -> float:
        if self.metrics.empty or "regime_reliability" not in self.metrics:
            return np.nan
        value = self.metrics["regime_reliability"].iloc[-1]
        return float(value) if np.isfinite(value) else np.nan

    @property
    def current_entropy(self) -> float:
        if self.metrics.empty or "posterior_entropy" not in self.metrics:
            return np.nan
        value = self.metrics["posterior_entropy"].iloc[-1]
        return float(value) if np.isfinite(value) else np.nan


class RegimeEngine:
    """Transparent probabilistic FX regime classifier."""

    def __init__(
        self,
        efficiency_window: int = 12,
        vol_short: int = 12,
        vol_long: int = 72,
        transition_window: int = 8,
        persistence_window: int = 12,
    ) -> None:
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
        x = x.loc[:, ~x.columns.duplicated(keep="first")]
        x = x.apply(
            pd.to_numeric,
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

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

        return pd.DataFrame(
            probs,
            index=scores.index,
            columns=scores.columns,
        )

    def fit(self, close: pd.DataFrame) -> RegimeResult:
        logp = self._log_prices(close)

        if logp.empty or logp.shape[1] < 3:
            empty = pd.DataFrame(columns=REGIMES)
            return RegimeResult(
                empty,
                pd.Series(dtype=object),
                pd.DataFrame(),
            )

        ret = logp.diff()
        abs_ret = ret.abs()

        displacement = (
            logp - logp.shift(self.efficiency_window)
        ).abs()

        path = abs_ret.rolling(
            self.efficiency_window,
            min_periods=max(
                4,
                self.efficiency_window // 2,
            ),
        ).sum()

        efficiency_by_pair = (
            displacement
            / path.replace(0.0, np.nan)
        )

        efficiency = (
            efficiency_by_pair
            .median(axis=1, skipna=True)
            .clip(0.0, 1.0)
        )

        short_vol_pair = ret.rolling(
            self.vol_short,
            min_periods=max(
                4,
                self.vol_short // 2,
            ),
        ).std(ddof=0)

        long_vol_pair = ret.rolling(
            self.vol_long,
            min_periods=max(
                12,
                self.vol_long // 3,
            ),
        ).std(ddof=0)

        vol_ratio_pair = (
            short_vol_pair
            / long_vol_pair.replace(0.0, np.nan)
        )

        vol_ratio = (
            vol_ratio_pair
            .replace([np.inf, -np.inf], np.nan)
            .median(axis=1, skipna=True)
            .clip(0.05, 8.0)
        )

        shock_z_pair = (
            abs_ret
            / long_vol_pair.replace(0.0, np.nan)
        )

        shock = (
            shock_z_pair
            .replace([np.inf, -np.inf], np.nan)
            .median(axis=1, skipna=True)
            .clip(0.0, 12.0)
        )

        cross_dispersion = ret.std(
            axis=1,
            ddof=0,
        )

        disp_mu = cross_dispersion.rolling(
            self.vol_long,
            min_periods=20,
        ).median()

        disp_mad = (
            (cross_dispersion - disp_mu)
            .abs()
            .rolling(
                self.vol_long,
                min_periods=20,
            )
            .median()
        )

        dispersion_z = (
            (cross_dispersion - disp_mu)
            / (1.4826 * disp_mad)
            .replace(0.0, np.nan)
        ).clip(-8, 8)

        eff_delta = efficiency.diff(
            self.transition_window
        ).abs()

        vol_delta = (
            np.log(
                vol_ratio.replace(
                    0.0,
                    np.nan,
                )
            )
            .diff(self.transition_window)
            .abs()
        )

        transition = (
            eff_delta / 0.20
            + vol_delta / 0.35
        ).clip(0.0, 8.0)

        trend_score = (
            3.2 * (efficiency - 0.38)
            + 0.35
            * np.log(
                vol_ratio.clip(lower=0.2)
            )
        )

        range_score = (
            3.0 * (0.42 - efficiency)
            - 0.25
            * (vol_ratio - 1.0).abs()
        )

        vol_exp_score = (
            2.4
            * np.log(
                vol_ratio.clip(lower=0.2)
            )
            + 0.35
            * dispersion_z.fillna(0.0)
        )

        vol_con_score = (
            -2.4
            * np.log(
                vol_ratio.clip(lower=0.2)
            )
            - 0.20
            * shock.fillna(0.0)
        )

        event_score = (
            1.45
            * (shock - 1.35)
            + 0.55
            * np.maximum(
                vol_ratio - 1.0,
                0.0,
            )
        )

        transition_score = (
            1.25
            * (transition - 0.75)
        )

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
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        )

        valid_history = (
            long_vol_pair
            .notna()
            .sum(axis=1)
            >= max(
                3,
                logp.shape[1] // 3,
            )
        )

        scores.loc[
            ~valid_history,
            :,
        ] = 0.0

        probabilities = self._softmax_frame(
            scores.fillna(0.0)
        )

        dominant = probabilities.idxmax(
            axis=1
        ).rename("regime")

        p = probabilities.clip(
            lower=1e-12
        )

        entropy = (
            -(p * np.log(p)).sum(axis=1)
            / np.log(len(REGIMES))
        )

        sorted_probs = np.sort(
            probabilities.to_numpy(dtype=float),
            axis=1,
        )

        gap = pd.Series(
            sorted_probs[:, -1]
            - sorted_probs[:, -2],
            index=probabilities.index,
        )

        current_match = dominant.eq(
            dominant.iloc[-1]
            if len(dominant)
            else "UNKNOWN"
        ).astype(float)

        persistence = current_match.rolling(
            self.persistence_window,
            min_periods=3,
        ).mean()

        change = dominant.ne(
            dominant.shift(1)
        ).astype(float)

        transition_rate = change.rolling(
            self.persistence_window,
            min_periods=3,
        ).mean()

        top_prob = probabilities.max(axis=1)

        reliability = (
            top_prob
            * (
                1.0
                - entropy.clip(
                    0.0,
                    1.0,
                )
            )
            * (
                0.45
                + 0.55
                * persistence.fillna(0.5)
            )
            * (
                0.55
                + 0.45
                * gap.clip(
                    0.0,
                    1.0,
                )
            )
        ).clip(0.0, 1.0)

        metrics = pd.DataFrame(
            {
                "efficiency": efficiency,
                "vol_ratio": vol_ratio,
                "shock": shock,
                "dispersion_z": dispersion_z,
                "transition": transition,
                "posterior_entropy": entropy,
                "confidence_gap": gap,
                "dominant_persistence": persistence,
                "transition_rate": transition_rate,
                "regime_reliability": reliability,
            },
            index=logp.index,
        )

        return RegimeResult(
            probabilities=probabilities,
            dominant=dominant,
            metrics=metrics,
        )


ENGINE_VERSION = "MULTISTAGE-0.3.2"

CORE_PAIRS: Tuple[str, ...] = (
    "AUDUSD",
    "EURCHF",
    "EURGBP",
    "EURJPY",
    "EURUSD",
    "GBPCHF",
    "GBPJPY",
    "GBPUSD",
)


@dataclass
class MemoryStats:
    trials: int = 0
    hits: int = 0
    hit_rate: float = np.nan
    recent_trials: int = 0
    recent_hits: int = 0
    recent_hit_rate: float = np.nan


class EvidenceMemoryEngine:
    """Persistent experiment and signal memory.

    Supabase is used when SUPABASE_URL and SUPABASE_KEY
    are available. SQLite remains as local fallback.
    """

    VALID_KINDS = (
        "EXPERIMENT",
        "SIGNAL",
    )

    REMOTE_TABLES = {
        "SIGNAL": "nexus_signals",
        "EXPERIMENT": "nexus_experiments",
    }

    def __init__(
        self,
        path: Optional[str] = None,
    ) -> None:
        import os
        import sqlite3
        from pathlib import Path

        self.sqlite3 = sqlite3

        default_path = os.environ.get(
            "NEXUS_MEMORY_PATH",
            ".nexus_state/nexus_memory.sqlite3",
        )

        self.path = Path(
            path or default_path
        )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._init_db()

        self.supabase = None
        self.remote_enabled = False
        self.remote_error: Optional[str] = None

        url = os.environ.get(
            "SUPABASE_URL",
            "",
        ).strip()

        key = os.environ.get(
            "SUPABASE_KEY",
            "",
        ).strip()

        if not url or not key:
            try:
                import streamlit as st

                url = str(
                    st.secrets.get(
                        "SUPABASE_URL",
                        "",
                    )
                ).strip()

                key = str(
                    st.secrets.get(
                        "SUPABASE_KEY",
                        "",
                    )
                ).strip()

            except Exception:
                pass

        if url and key:
            try:
                from supabase import create_client

                self.supabase = create_client(
                    url,
                    key,
                )

                self.remote_enabled = True

            except Exception as exc:
                self.remote_error = str(exc)
                self.remote_enabled = False

    def _connect(self):
        con = self.sqlite3.connect(
            str(self.path),
            timeout=20,
        )

        con.execute(
            "PRAGMA journal_mode=WAL"
        )

        con.execute(
            "PRAGMA synchronous=NORMAL"
        )

        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_time TEXT NOT NULL,
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    lag INTEGER NOT NULL,
                    interval TEXT NOT NULL,
                    regime TEXT,
                    direction TEXT NOT NULL,
                    probability REAL,
                    evidence_score REAL,
                    reference_price REAL,
                    horizon_bars INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    outcome_time TEXT,
                    outcome_price REAL,
                    signed_return REAL,
                    hit INTEGER,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    kind TEXT NOT NULL DEFAULT 'SIGNAL',
                    UNIQUE(
                        signal_time,
                        source,
                        target,
                        lag,
                        interval,
                        direction
                    )
                )
                """
            )

            columns = {
                str(row[1])
                for row in con.execute(
                    "PRAGMA table_info(signals)"
                ).fetchall()
            }

            if "kind" not in columns:
                con.execute(
                    "ALTER TABLE signals "
                    "ADD COLUMN kind TEXT "
                    "NOT NULL DEFAULT 'SIGNAL'"
                )

            con.execute(
                "CREATE INDEX IF NOT EXISTS "
                "idx_memory_key "
                "ON signals("
                "source,target,lag,"
                "interval,regime,status)"
            )

            con.execute(
                "CREATE INDEX IF NOT EXISTS "
                "idx_memory_kind "
                "ON signals("
                "kind,status,source,target,"
                "lag,interval)"
            )

    @staticmethod
    def _time_key(value) -> str:
        return pd.Timestamp(
            value
        ).isoformat()

    @classmethod
    def _normalize_kind(
        cls,
        kind: str,
    ) -> str:
        value = str(kind).upper().strip()

        return (
            value
            if value in cls.VALID_KINDS
            else "EXPERIMENT"
        )

    @staticmethod
    def _finite_or_none(value):
        try:
            value = float(value)
        except Exception:
            return None

        return (
            value
            if np.isfinite(value)
            else None
        )

    def _remote_table(
        self,
        kind: str,
    ) -> str:
        return self.REMOTE_TABLES[
            self._normalize_kind(kind)
        ]

    def _remote_payload(
        self,
        signal_time,
        source: str,
        target: str,
        lag: int,
        interval: str,
        regime: str,
        direction: str,
        probability: float,
        evidence_score: float,
        reference_price: float,
        horizon_bars: int,
        kind: str,
    ) -> Dict[str, object]:

        return {
            "signal_time": self._time_key(
                signal_time
            ),
            "source": str(source),
            "target": str(target),
            "lag": int(lag),
            "interval": str(interval),
            "regime": str(regime),
            "direction": str(direction),
            "probability": self._finite_or_none(
                probability
            ),
            "evidence_score": self._finite_or_none(
                evidence_score
            ),
            "reference_price": self._finite_or_none(
                reference_price
            ),
            "horizon_bars": int(
                horizon_bars
            ),
            "status": "OPEN",
            "kind": self._normalize_kind(
                kind
            ),
        }

    def _record_local(
        self,
        signal_time,
        source: str,
        target: str,
        lag: int,
        interval: str,
        regime: str,
        direction: str,
        probability: float,
        evidence_score: float,
        reference_price: float,
        horizon_bars: int,
        kind: str,
    ) -> None:

        kind = self._normalize_kind(kind)

        key = (
            self._time_key(signal_time),
            str(source),
            str(target),
            int(lag),
            str(interval),
            str(direction),
        )

        with self._connect() as con:
            con.execute(
                """
                INSERT OR IGNORE INTO signals(
                    signal_time,
                    source,
                    target,
                    lag,
                    interval,
                    regime,
                    direction,
                    probability,
                    evidence_score,
                    reference_price,
                    horizon_bars,
                    status,
                    kind
                )
                VALUES(
                    ?,?,?,?,?,?,?,?,?,?,?,
                    'OPEN',?
                )
                """,
                (
                    key[0],
                    key[1],
                    key[2],
                    key[3],
                    key[4],
                    str(regime),
                    key[5],
                    float(probability),
                    float(evidence_score),
                    float(reference_price),
                    int(horizon_bars),
                    kind,
                ),
            )

            if kind == "SIGNAL":
                con.execute(
                    """
                    UPDATE signals
                    SET
                        kind='SIGNAL',
                        regime=?,
                        probability=?,
                        evidence_score=?,
                        reference_price=?,
                        horizon_bars=?
                    WHERE
                        signal_time=?
                        AND source=?
                        AND target=?
                        AND lag=?
                        AND interval=?
                        AND direction=?
                    """,
                    (
                        str(regime),
                        float(probability),
                        float(evidence_score),
                        float(reference_price),
                        int(horizon_bars),
                        key[0],
                        key[1],
                        key[2],
                        key[3],
                        key[4],
                        key[5],
                    ),
                )

    def record_signal(
        self,
        signal_time,
        source: str,
        target: str,
        lag: int,
        interval: str,
        regime: str,
        direction: str,
        probability: float,
        evidence_score: float,
        reference_price: float,
        horizon_bars: int,
        kind: str = "SIGNAL",
    ) -> None:

        kind = self._normalize_kind(kind)

        self._record_local(
            signal_time,
            source,
            target,
            lag,
            interval,
            regime,
            direction,
            probability,
            evidence_score,
            reference_price,
            horizon_bars,
            kind,
        )

        if (
            not self.remote_enabled
            or self.supabase is None
        ):
            return

        try:
            table = self._remote_table(kind)

            payload = self._remote_payload(
                signal_time,
                source,
                target,
                lag,
                interval,
                regime,
                direction,
                probability,
                evidence_score,
                reference_price,
                horizon_bars,
                kind,
            )

            self.supabase.table(
                table
            ).upsert(
                payload,
                on_conflict=(
                    "signal_time,"
                    "source,"
                    "target,"
                    "lag,"
                    "interval,"
                    "direction"
                ),
            ).execute()

        except Exception as exc:
            self.remote_error = str(exc)
                def _settle_local(
        self,
        close: pd.DataFrame,
        interval: str,
    ) -> int:
        work = close.copy().sort_index()
        idx = pd.DatetimeIndex(work.index)
        settled = 0

        with self._connect() as con:
            rows = con.execute(
                """
                SELECT
                    id,
                    signal_time,
                    target,
                    direction,
                    reference_price,
                    horizon_bars
                FROM signals
                WHERE status='OPEN'
                  AND interval=?
                ORDER BY id
                """,
                (str(interval),),
            ).fetchall()

            for (
                row_id,
                signal_time,
                target,
                direction,
                ref_price,
                horizon_bars,
            ) in rows:

                if target not in work.columns:
                    continue

                ts = pd.Timestamp(signal_time)

                if (
                    idx.tz is not None
                    and ts.tzinfo is None
                ):
                    ts = ts.tz_localize(idx.tz)

                elif (
                    idx.tz is None
                    and ts.tzinfo is not None
                ):
                    ts = ts.tz_convert(None)

                pos = int(
                    idx.searchsorted(
                        ts,
                        side="left",
                    )
                )

                if pos >= len(idx):
                    continue

                outcome_pos = (
                    pos + int(horizon_bars)
                )

                if outcome_pos >= len(idx):
                    continue

                outcome_price = pd.to_numeric(
                    work[target],
                    errors="coerce",
                ).iloc[outcome_pos]

                if (
                    not np.isfinite(outcome_price)
                    or not np.isfinite(ref_price)
                    or float(ref_price) <= 0
                ):
                    continue

                raw_ret = float(
                    outcome_price
                    / float(ref_price)
                    - 1.0
                )

                signed_ret = (
                    raw_ret
                    if str(direction).upper()
                    == "BUY"
                    else -raw_ret
                )

                hit = int(
                    signed_ret > 0.0
                )

                con.execute(
                    """
                    UPDATE signals
                    SET
                        status='CLOSED',
                        outcome_time=?,
                        outcome_price=?,
                        signed_return=?,
                        hit=?
                    WHERE id=?
                    """,
                    (
                        self._time_key(
                            idx[outcome_pos]
                        ),
                        float(outcome_price),
                        signed_ret,
                        hit,
                        row_id,
                    ),
                )

                settled += 1

        return settled

    def _settle_remote_table(
        self,
        table: str,
        close: pd.DataFrame,
        interval: str,
    ) -> int:

        if (
            not self.remote_enabled
            or self.supabase is None
        ):
            return 0

        work = close.copy().sort_index()
        idx = pd.DatetimeIndex(work.index)
        settled = 0

        try:
            response = (
                self.supabase
                .table(table)
                .select(
                    "id,"
                    "signal_time,"
                    "target,"
                    "direction,"
                    "reference_price,"
                    "horizon_bars"
                )
                .eq(
                    "status",
                    "OPEN",
                )
                .eq(
                    "interval",
                    str(interval),
                )
                .execute()
            )

            rows = (
                getattr(
                    response,
                    "data",
                    None,
                )
                or []
            )

            for row in rows:

                target = str(
                    row.get(
                        "target",
                        "",
                    )
                )

                if target not in work.columns:
                    continue

                ts = pd.Timestamp(
                    row.get("signal_time")
                )

                if (
                    idx.tz is not None
                    and ts.tzinfo is None
                ):
                    ts = ts.tz_localize(
                        idx.tz
                    )

                elif (
                    idx.tz is None
                    and ts.tzinfo is not None
                ):
                    ts = ts.tz_convert(None)

                pos = int(
                    idx.searchsorted(
                        ts,
                        side="left",
                    )
                )

                horizon_bars = int(
                    row.get(
                        "horizon_bars"
                    )
                    or 1
                )

                outcome_pos = (
                    pos + horizon_bars
                )

                if (
                    pos >= len(idx)
                    or outcome_pos >= len(idx)
                ):
                    continue

                outcome_price = pd.to_numeric(
                    work[target],
                    errors="coerce",
                ).iloc[outcome_pos]

                ref_price = (
                    self._finite_or_none(
                        row.get(
                            "reference_price"
                        )
                    )
                )

                if (
                    ref_price is None
                    or not np.isfinite(
                        outcome_price
                    )
                    or ref_price <= 0
                ):
                    continue

                raw_ret = float(
                    outcome_price
                    / ref_price
                    - 1.0
                )

                signed_ret = (
                    raw_ret
                    if str(
                        row.get(
                            "direction",
                            "",
                        )
                    ).upper() == "BUY"
                    else -raw_ret
                )

                hit = int(
                    signed_ret > 0.0
                )

                (
                    self.supabase
                    .table(table)
                    .update(
                        {
                            "status": "CLOSED",
                            "outcome_time":
                                self._time_key(
                                    idx[
                                        outcome_pos
                                    ]
                                ),
                            "outcome_price":
                                float(
                                    outcome_price
                                ),
                            "signed_return":
                                signed_ret,
                            "hit": hit,
                        }
                    )
                    .eq(
                        "id",
                        row["id"],
                    )
                    .execute()
                )

                settled += 1

        except Exception as exc:
            self.remote_error = (
                f"Supabase settle failed "
                f"({table}): {exc}"
            )

        return settled

    def settle(
        self,
        close: pd.DataFrame,
        interval: str,
    ) -> int:

        if (
            close is None
            or close.empty
        ):
            return 0

        local_count = (
            self._settle_local(
                close,
                interval,
            )
        )

        remote_count = 0

        if self.remote_enabled:

            remote_count += (
                self._settle_remote_table(
                    "nexus_signals",
                    close,
                    interval,
                )
            )

            remote_count += (
                self._settle_remote_table(
                    "nexus_experiments",
                    close,
                    interval,
                )
            )

        return max(
            local_count,
            remote_count,
        )

    def _remote_stats(
        self,
        source: str,
        target: str,
        lag: int,
        interval: str,
        regime: Optional[str],
        recent_n: int,
        kind: str,
    ) -> Optional[MemoryStats]:

        if (
            not self.remote_enabled
            or self.supabase is None
        ):
            return None

        table = self._remote_table(
            kind
        )

        try:
            query = (
                self.supabase
                .table(table)
                .select("hit,id")
                .eq(
                    "source",
                    str(source),
                )
                .eq(
                    "target",
                    str(target),
                )
                .eq(
                    "lag",
                    int(lag),
                )
                .eq(
                    "interval",
                    str(interval),
                )
                .eq(
                    "status",
                    "CLOSED",
                )
            )

            if (
                regime
                and regime != "UNKNOWN"
            ):
                query = query.eq(
                    "regime",
                    str(regime),
                )

            response = (
                query
                .order(
                    "id",
                    desc=True,
                )
                .execute()
            )

            rows = (
                getattr(
                    response,
                    "data",
                    None,
                )
                or []
            )

            trials = len(rows)

            hits = int(
                sum(
                    int(
                        bool(
                            r.get("hit")
                        )
                    )
                    for r in rows
                )
            )

            recent = rows[
                : int(recent_n)
            ]

            recent_trials = len(
                recent
            )

            recent_hits = int(
                sum(
                    int(
                        bool(
                            r.get("hit")
                        )
                    )
                    for r in recent
                )
            )

            return MemoryStats(
                trials=trials,
                hits=hits,
                hit_rate=(
                    hits / trials
                    if trials
                    else np.nan
                ),
                recent_trials=
                    recent_trials,
                recent_hits=
                    recent_hits,
                recent_hit_rate=(
                    recent_hits
                    / recent_trials
                    if recent_trials
                    else np.nan
                ),
            )

        except Exception as exc:
            self.remote_error = (
                f"Supabase stats failed "
                f"({table}): {exc}"
            )

            return None

    def _local_stats(
        self,
        source: str,
        target: str,
        lag: int,
        interval: str,
        regime: Optional[str],
        recent_n: int,
        kind: Optional[str],
    ) -> MemoryStats:

        where = (
            "source=? "
            "AND target=? "
            "AND lag=? "
            "AND interval=? "
            "AND status='CLOSED'"
        )

        params: List[object] = [
            str(source),
            str(target),
            int(lag),
            str(interval),
        ]

        if (
            regime
            and regime != "UNKNOWN"
        ):
            where += " AND regime=?"
            params.append(
                str(regime)
            )

        if kind:
            where += " AND kind=?"
            params.append(
                self._normalize_kind(
                    kind
                )
            )

        with self._connect() as con:

            total = con.execute(
                f"""
                SELECT
                    COUNT(*),
                    COALESCE(
                        SUM(hit),
                        0
                    )
                FROM signals
                WHERE {where}
                """,
                params,
            ).fetchone()

            recent = con.execute(
                f"""
                SELECT hit
                FROM signals
                WHERE {where}
                ORDER BY id DESC
                LIMIT ?
                """,
                params
                + [int(recent_n)],
            ).fetchall()

        trials = int(total[0])
        hits = int(total[1])

        recent_trials = len(
            recent
        )

        recent_hits = (
            int(
                sum(
                    int(r[0])
                    for r in recent
                )
            )
            if recent
            else 0
        )

        return MemoryStats(
            trials=trials,
            hits=hits,
            hit_rate=(
                hits / trials
                if trials
                else np.nan
            ),
            recent_trials=
                recent_trials,
            recent_hits=
                recent_hits,
            recent_hit_rate=(
                recent_hits
                / recent_trials
                if recent_trials
                else np.nan
            ),
        )

    def stats(
        self,
        source: str,
        target: str,
        lag: int,
        interval: str,
        regime: Optional[str] = None,
        recent_n: int = 100,
        kind: Optional[str] = None,
    ) -> MemoryStats:

        if kind:
            remote = (
                self._remote_stats(
                    source,
                    target,
                    lag,
                    interval,
                    regime,
                    recent_n,
                    self._normalize_kind(
                        kind
                    ),
                )
            )

            if remote is not None:
                return remote

        return self._local_stats(
            source,
            target,
            lag,
            interval,
            regime,
            recent_n,
            kind,
        )

    def _remote_export_table(
        self,
        kind: str,
    ) -> List[Dict[str, object]]:

        if (
            not self.remote_enabled
            or self.supabase is None
        ):
            return []

        table = self._remote_table(
            kind
        )

        try:
            response = (
                self.supabase
                .table(table)
                .select("*")
                .order("id")
                .execute()
            )

            rows = list(
                getattr(
                    response,
                    "data",
                    None,
                )
                or []
            )

            for row in rows:
                row["kind"] = kind

            return rows

        except Exception as exc:
            self.remote_error = (
                f"Supabase export failed "
                f"({table}): {exc}"
            )

            return []

    def export_frame(
        self,
    ) -> pd.DataFrame:

        if self.remote_enabled:

            rows = (
                self._remote_export_table(
                    "EXPERIMENT"
                )
                + self._remote_export_table(
                    "SIGNAL"
                )
            )

            if rows:
                frame = pd.DataFrame(
                    rows
                )

                if "id" in frame.columns:
                    frame = (
                        frame.sort_values(
                            "id",
                            kind="stable",
                        )
                    )

                return frame.reset_index(
                    drop=True
                )

        with self._connect() as con:
            return pd.read_sql_query(
                """
                SELECT *
                FROM signals
                ORDER BY id
                """,
                con,
            )


class CausalGraphEngine:
    """Multi-stage predictive hypothesis engine."""

    COLUMNS = [
        "source",
        "target",
        "lag",
        "shared_factor",
        "raw_corr",
        "conditional_corr",
        "regime_corr",
        "stability",
        "analytic_p",
        "perm_p",
        "fdr_q",
        "bootstrap_low",
        "bootstrap_high",
        "walkforward_ic",
        "walkforward_sign_rate",
        "walkforward_p",
        "walkforward_n",
        "incremental_r2",
        "decay_score",
        "decay_state",
        "evidence_score",
        "edge_score",
        "survives",
        "n_obs",
        "oos_accuracy",
        "oos_hits",
        "oos_trials",
        "probability_raw",
        "memory_probability",
        "signal_memory_trials",
        "research_probability",
        "research_trials",
        "predicted_probability",
        "probability_lower_95",
        "trigger_z",
        "signal_direction",
        "decision",
        "horizon_bars",
        "stage",
    ]

    def __init__(
        self,
        max_lag: int = 3,
        min_obs: int = 100,
        stability_splits: int = 4,
        permutations_n: int = 399,
        bootstrap_n: int = 399,
        walkforward_folds: int = 3,
        fdr_alpha: float = 0.05,
        probability_threshold: float = 0.70,
        trigger_z_min: float = 0.50,
        random_state: int = 42,
        memory: Optional[
            EvidenceMemoryEngine
        ] = None,
    ) -> None:

        self.max_lag = max(
            1,
            int(max_lag),
        )

        self.min_obs = max(
            50,
            int(min_obs),
        )

        self.stability_splits = max(
            3,
            int(stability_splits),
        )

        self.permutations_n = max(
            32,
            int(permutations_n),
        )

        self.bootstrap_n = max(
            32,
            int(bootstrap_n),
        )

        self.walkforward_folds = max(
            2,
            int(walkforward_folds),
        )

        self.fdr_alpha = float(
            np.clip(
                fdr_alpha,
                0.01,
                0.25,
            )
        )

        self.probability_threshold = float(
            np.clip(
                probability_threshold,
                0.50,
                0.95,
            )
        )

        self.trigger_z_min = float(
            np.clip(
                trigger_z_min,
                0.0,
                5.0,
            )
        )

        self.rng = (
            np.random.default_rng(
                random_state
            )
        )

        self.memory = memory

    @staticmethod
    def log_returns(
        close: pd.DataFrame,
    ) -> pd.DataFrame:

        x = close.copy()

        x.columns = [
            normalize_pair(str(c))
            for c in x.columns
        ]

        x = x.loc[
            :,
            ~x.columns.duplicated(
                keep="first"
            ),
        ]

        x = (
            x.apply(
                pd.to_numeric,
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

        return np.log(
            x.where(x > 0)
        ).diff()

    @staticmethod
    def _corr(
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:

        x = np.asarray(
            x,
            dtype=float,
        ).ravel()

        y = np.asarray(
            y,
            dtype=float,
        ).ravel()

        n = min(
            x.size,
            y.size,
        )

        if n < 20:
            return np.nan

        x = x[:n]
        y = y[:n]

        mask = (
            np.isfinite(x)
            & np.isfinite(y)
        )

        if int(mask.sum()) < 20:
            return np.nan

        x = x[mask]
        y = y[mask]

        if (
            np.std(x) <= 1e-15
            or np.std(y) <= 1e-15
        ):
            return np.nan

        c = np.corrcoef(
            x,
            y,
        )[0, 1]

        return (
            float(c)
            if np.isfinite(c)
            else np.nan
        )

    @staticmethod
    def _corr_pvalue(
        corr: float,
        n_obs: int,
    ) -> float:

        if (
            not np.isfinite(corr)
            or int(n_obs) <= 3
        ):
            return 1.0

        r = float(
            np.clip(
                corr,
                -0.999999,
                0.999999,
            )
        )

        z = (
            abs(np.arctanh(r))
            * math.sqrt(
                max(
                    int(n_obs) - 3,
                    1,
                )
            )
        )

        return float(
            np.clip(
                math.erfc(
                    z / math.sqrt(2.0)
                ),
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _bh_qvalues(
        pvalues: np.ndarray,
    ) -> np.ndarray:

        p = np.asarray(
            pvalues,
            dtype=float,
        )

        q = np.ones_like(
            p,
            dtype=float,
        )

        valid = np.isfinite(p)

        if not valid.any():
            return q

        pv = np.clip(
            p[valid],
            0.0,
            1.0,
        )

        order = np.argsort(pv)
        ranked = pv[order]
        m = len(ranked)

        raw = (
            ranked
            * m
            / np.arange(
                1,
                m + 1,
            )
        )

        adj = (
            np.minimum.accumulate(
                raw[::-1]
            )[::-1]
        )

        restored = np.empty_like(
            adj
        )

        restored[order] = np.clip(
            adj,
            0.0,
            1.0,
        )

        q[valid] = restored

        return q
                records: List[Dict[str, object]] = []
        oos_cache: Dict[int, pd.DataFrame] = {}

        for i, row in cand.iterrows():
            val = self._oos_validation(
                str(row.source),
                str(row.target),
                int(row.lag),
                returns,
                latent_strength,
                int(row.discovery_n),
            )

            if not val:
                continue

            rec = row.to_dict()
            rec.update(
                {
                    k: v
                    for k, v in val.items()
                    if k != "oos"
                }
            )

            rec["walkforward_p"] = float(
                val["walkforward_p"]
            )

            rec["analytic_p"] = (
                self._corr_pvalue(
                    float(
                        row.conditional_corr
                    ),
                    int(row.n_obs),
                )
            )

            records.append(rec)

            oos_cache[
                len(records) - 1
            ] = val["oos"]

        if not records:
            out = pd.DataFrame(
                columns=self.COLUMNS
            )

            out.attrs["diagnostics"] = {
                "stage0_universe":
                    stage0_tests,
                "stage1_discovery":
                    int(len(cand)),
                "stage2_oos": 0,
            }

            return out

        df = pd.DataFrame(records)

        df["fdr_q"] = (
            self._bh_qvalues(
                df[
                    "walkforward_p"
                ].to_numpy(
                    dtype=float
                )
            )
        )

        fdr_pass = (
            df["fdr_q"]
            <= self.fdr_alpha
        )

        wf_pass = (
            (
                df[
                    "walkforward_ic"
                ].abs()
                >= 0.05
            )
            & (
                df[
                    "walkforward_sign_rate"
                ]
                >= (2.0 / 3.0)
            )
            & (
                df["wf_folds"]
                >= 2
            )
        )

        finalists = (
            fdr_pass
            & wf_pass
        )

        df["perm_p"] = 1.0
        df["bootstrap_low"] = np.nan
        df["bootstrap_high"] = np.nan

        for idx in df.index[
            finalists
        ]:
            oos = oos_cache.get(
                int(idx)
            )

            obs = (
                self._corr(
                    oos[
                        "x"
                    ].to_numpy(),
                    oos[
                        "y_future"
                    ].to_numpy(),
                )
                if oos is not None
                else np.nan
            )

            df.loc[
                idx,
                "perm_p",
            ] = (
                self._permutation_p_oos(
                    oos,
                    obs,
                )
            )

            lo, hi = (
                self._bootstrap_ci_oos(
                    oos
                )
            )

            df.loc[
                idx,
                "bootstrap_low",
            ] = lo

            df.loc[
                idx,
                "bootstrap_high",
            ] = hi

        bootstrap_pass = (
            (
                (
                    df[
                        "bootstrap_low"
                    ] > 0
                )
                & (
                    df[
                        "bootstrap_high"
                    ] > 0
                )
            )
            |
            (
                (
                    df[
                        "bootstrap_low"
                    ] < 0
                )
                & (
                    df[
                        "bootstrap_high"
                    ] < 0
                )
            )
        )

        perm_pass = (
            df["perm_p"]
            <= 0.10
        )

        stability_pass = (
            df["stability"]
            >= 0.30
        )

        decay_pass = (
            df[
                "decay_score"
            ]
            .fillna(0.0)
            >= 0.35
        )

        current_regime = (
            regime.current_regime
            if regime is not None
            else "UNKNOWN"
        )

        df["regime_corr"] = np.nan
        df["probability_raw"] = np.nan
        df["memory_probability"] = np.nan
        df["signal_memory_trials"] = 0
        df["research_probability"] = np.nan
        df["research_trials"] = 0
        df["predicted_probability"] = np.nan
        df["probability_lower_95"] = np.nan
        df["trigger_z"] = np.nan
        df["signal_direction"] = "NONE"
        df["decision"] = "SIN SEÑAL"

        df["horizon_bars"] = (
            df["lag"].astype(int)
        )

        sig_comp = (
            0.55
            * (
                1
                - df["fdr_q"]
            ).clip(
                0,
                1,
            )
            + 0.45
            * (
                1
                - df["perm_p"]
            ).clip(
                0,
                1,
            )
        )

        boot_comp = (
            bootstrap_pass
            .astype(float)
        )

        wf_comp = (
            (
                df[
                    "walkforward_ic"
                ].abs()
                / 0.25
            ).clip(
                0,
                1,
            )
            * df[
                "walkforward_sign_rate"
            ].clip(
                0,
                1,
            )
        )

        r2_comp = (
            np.sqrt(
                df[
                    "incremental_r2"
                ].clip(
                    lower=0
                )
            )
            * 4
        ).clip(
            0,
            1,
        )

        decay_comp = (
            df[
                "decay_score"
            ]
            .fillna(0)
            .clip(
                0,
                1,
            )
        )

        sample_comp = (
            np.sqrt(
                df["n_obs"]
                / 500.0
            )
            .clip(
                0.35,
                1.0,
            )
        )

        df["evidence_score"] = (
            sample_comp
            * (
                0.22
                * (
                    df[
                        "conditional_corr"
                    ].abs()
                    / 0.35
                ).clip(
                    0,
                    1,
                )
                + 0.16
                * df[
                    "stability"
                ].clip(
                    0,
                    1,
                )
                + 0.18
                * sig_comp
                + 0.12
                * boot_comp
                + 0.20
                * wf_comp
                + 0.06
                * r2_comp
                + 0.06
                * decay_comp
            )
        )

        df["edge_score"] = (
            df["evidence_score"]
        )

        for idx, row in df.iterrows():

            signal_mem = (
                self.memory.stats(
                    str(row.source),
                    str(row.target),
                    int(row.lag),
                    interval,
                    current_regime,
                    kind="SIGNAL",
                )
                if self.memory
                is not None
                else MemoryStats()
            )

            research_mem = (
                self.memory.stats(
                    str(row.source),
                    str(row.target),
                    int(row.lag),
                    interval,
                    current_regime,
                    kind="EXPERIMENT",
                )
                if self.memory
                is not None
                else MemoryStats()
            )

            (
                hist_prob,
                mem_prob,
                pred_prob,
                lower,
            ) = (
                self._calibrate_probability(
                    float(
                        row.oos_accuracy
                    ),
                    int(
                        row.oos_hits
                    ),
                    int(
                        row.oos_trials
                    ),
                    signal_mem,
                )
            )

            research_prob = (
                research_mem.hit_rate
                if (
                    research_mem.trials
                    > 0
                    and np.isfinite(
                        research_mem.hit_rate
                    )
                )
                else np.nan
            )

            trigger_z = (
                self._latest_trigger_z(
                    returns,
                    str(row.source),
                )
            )

            df.loc[
                idx,
                "probability_raw",
            ] = hist_prob

            df.loc[
                idx,
                "memory_probability",
            ] = mem_prob

            df.loc[
                idx,
                "signal_memory_trials",
            ] = int(
                signal_mem.trials
            )

            df.loc[
                idx,
                "research_probability",
            ] = research_prob

            df.loc[
                idx,
                "research_trials",
            ] = int(
                research_mem.trials
            )

            df.loc[
                idx,
                "predicted_probability",
            ] = pred_prob

            df.loc[
                idx,
                "probability_lower_95",
            ] = lower

            df.loc[
                idx,
                "trigger_z",
            ] = trigger_z

            direction = "NONE"

            if (
                np.isfinite(trigger_z)
                and abs(trigger_z)
                >= self.trigger_z_min
            ):
                impulse = (
                    np.sign(
                        float(
                            row.conditional_corr
                        )
                    )
                    * np.sign(
                        trigger_z
                    )
                )

                direction = (
                    "BUY"
                    if impulse > 0
                    else (
                        "SELL"
                        if impulse < 0
                        else "NONE"
                    )
                )

            df.loc[
                idx,
                "signal_direction",
            ] = direction

        hard_survive = (
            fdr_pass
            & wf_pass
            & bootstrap_pass
            & perm_pass
            & stability_pass
            & decay_pass
            & (
                df[
                    "evidence_score"
                ]
                >= 0.45
            )
        )

        df["survives"] = (
            hard_survive
        )

        decision_mask = (
            hard_survive
            & (
                df[
                    "predicted_probability"
                ]
                >= self.probability_threshold
            )
            & (
                df[
                    "signal_direction"
                ]
                != "NONE"
            )
        )

        df.loc[
            decision_mask,
            "decision",
        ] = df.loc[
            decision_mask,
            "signal_direction",
        ]

        df["stage"] = "OOS"

        df.loc[
            fdr_pass,
            "stage",
        ] = "FDR"

        df.loc[
            finalists,
            "stage",
        ] = "FINALIST"

        df.loc[
            hard_survive,
            "stage",
        ] = "SURVIVOR"

        df.loc[
            decision_mask,
            "stage",
        ] = "SIGNAL"

        if (
            self.memory
            is not None
            and len(close.index)
        ):
            signal_time = (
                close.index[-1]
            )

            research_mask = (
                df[
                    "evidence_score"
                ]
                >= 0.35
            )

            for _, row in df.loc[
                research_mask
            ].iterrows():

                ref = pd.to_numeric(
                    close[
                        str(
                            row.target
                        )
                    ],
                    errors="coerce",
                ).dropna()

                if ref.empty:
                    continue

                research_direction = (
                    str(
                        row.signal_direction
                    )
                )

                if (
                    research_direction
                    == "NONE"
                ):
                    corr_sign = (
                        np.sign(
                            float(
                                row.conditional_corr
                            )
                        )
                    )

                    research_direction = (
                        "BUY"
                        if corr_sign >= 0
                        else "SELL"
                    )

                self.memory.record_signal(
                    signal_time=
                        signal_time,
                    source=
                        str(row.source),
                    target=
                        str(row.target),
                    lag=
                        int(row.lag),
                    interval=
                        interval,
                    regime=
                        current_regime,
                    direction=
                        research_direction,
                    probability=
                        float(
                            row[
                                "predicted_probability"
                            ]
                        )
                        if np.isfinite(
                            row[
                                "predicted_probability"
                            ]
                        )
                        else 0.5,
                    evidence_score=
                        float(
                            row[
                                "evidence_score"
                            ]
                        ),
                    reference_price=
                        float(
                            ref.iloc[-1]
                        ),
                    horizon_bars=
                        int(
                            row[
                                "horizon_bars"
                            ]
                        ),
                    kind="EXPERIMENT",
                )

            for _, row in df.loc[
                decision_mask
            ].iterrows():

                ref = pd.to_numeric(
                    close[
                        str(
                            row.target
                        )
                    ],
                    errors="coerce",
                ).dropna()

                if ref.empty:
                    continue

                self.memory.record_signal(
                    signal_time=
                        signal_time,
                    source=
                        str(row.source),
                    target=
                        str(row.target),
                    lag=
                        int(row.lag),
                    interval=
                        interval,
                    regime=
                        current_regime,
                    direction=
                        str(
                            row.decision
                        ),
                    probability=
                        float(
                            row[
                                "predicted_probability"
                            ]
                        ),
                    evidence_score=
                        float(
                            row[
                                "evidence_score"
                            ]
                        ),
                    reference_price=
                        float(
                            ref.iloc[-1]
                        ),
                    horizon_bars=
                        int(
                            row[
                                "horizon_bars"
                            ]
                        ),
                    kind="SIGNAL",
                )

        seq_fdr = (
            fdr_pass
        )

        seq_wf = (
            seq_fdr
            & wf_pass
        )

        seq_boot = (
            seq_wf
            & bootstrap_pass
        )

        seq_perm = (
            seq_boot
            & perm_pass
        )

        seq_stab = (
            seq_perm
            & stability_pass
        )

        seq_decay = (
            seq_stab
            & decay_pass
        )

        seq_score = (
            seq_decay
            & (
                df[
                    "evidence_score"
                ]
                >= 0.45
            )
        )

        seq_prob = (
            seq_score
            & (
                df[
                    "predicted_probability"
                ]
                >= self.probability_threshold
            )
        )

        seq_trigger = (
            seq_prob
            & (
                df[
                    "signal_direction"
                ]
                != "NONE"
            )
        )

        diagnostics = {
            "engine_version":
                ENGINE_VERSION,
            "pairs":
                int(len(names)),
            "directed_pairs":
                int(
                    len(names)
                    * max(
                        0,
                        len(names) - 1,
                    )
                ),
            "stage0_universe":
                int(stage0_tests),
            "stage1_discovery":
                int(len(cand)),
            "stage2_oos":
                int(len(df)),
            "stage3_fdr":
                int(
                    fdr_pass.sum()
                ),
            "stage4_heavy":
                int(
                    finalists.sum()
                ),
            "survivors":
                int(
                    hard_survive.sum()
                ),
            "signals":
                int(
                    decision_mask.sum()
                ),
            "research_candidates":
                int(
                    (
                        df[
                            "evidence_score"
                        ]
                        >= 0.35
                    ).sum()
                ),
            "probability_threshold":
                self.probability_threshold,
            "fdr_alpha":
                self.fdr_alpha,
            "best_probability":
                float(
                    df[
                        "predicted_probability"
                    ].max()
                )
                if len(df)
                else np.nan,
            "best_fdr_q":
                float(
                    df[
                        "fdr_q"
                    ].min()
                )
                if len(df)
                else np.nan,
            "sequential": {
                "fdr":
                    int(
                        seq_fdr.sum()
                    ),
                "walkforward":
                    int(
                        seq_wf.sum()
                    ),
                "bootstrap":
                    int(
                        seq_boot.sum()
                    ),
                "permutation":
                    int(
                        seq_perm.sum()
                    ),
                "stability":
                    int(
                        seq_stab.sum()
                    ),
                "decay":
                    int(
                        seq_decay.sum()
                    ),
                "evidence_score":
                    int(
                        seq_score.sum()
                    ),
                "probability":
                    int(
                        seq_prob.sum()
                    ),
                "trigger":
                    int(
                        seq_trigger.sum()
                    ),
            },
        }

        sort_prob = (
            df[
                "predicted_probability"
            ]
            .fillna(-1)
        )

        df = (
            df.assign(
                _sort_prob=
                    sort_prob
            )
            .sort_values(
                [
                    "decision",
                    "survives",
                    "_sort_prob",
                    "evidence_score",
                ],
                ascending=[
                    True,
                    False,
                    False,
                    False,
                ],
            )
            .drop(
                columns=[
                    "_sort_prob"
                ]
            )
        )

        df = (
            df.head(
                max(
                    1,
                    int(top_n),
                )
            )
            .reset_index(
                drop=True
            )
        )

        df = df.drop(
            columns=[
                "wf_folds",
                "discovery_n",
                "discovery_score",
            ],
            errors="ignore",
        )

        df = df.reindex(
            columns=self.COLUMNS
        )

        df.attrs[
            "diagnostics"
        ] = diagnostics

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

    def __init__(
        self,
        window: int = 96,
        min_periods: int = 30,
    ) -> None:
        self.window = max(
            30,
            int(window),
        )

        self.min_periods = max(
            15,
            int(min_periods),
        )

    def fit(
        self,
        triangles: pd.DataFrame,
    ) -> StressResult:

        if (
            triangles is None
            or triangles.empty
        ):
            return StressResult(
                np.nan,
                np.nan,
                np.nan,
                pd.Series(
                    dtype=float
                ),
                pd.DataFrame(),
            )

        tri = (
            triangles
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

        mu = tri.rolling(
            self.window,
            min_periods=
                self.min_periods,
        ).median()

        mad = (
            (tri - mu)
            .abs()
            .rolling(
                self.window,
                min_periods=
                    self.min_periods,
            )
            .median()
        )

        robust_sigma = (
            1.4826 * mad
        ).clip(
            lower=1e-8
        )

        z = (
            (tri - mu)
            / robust_sigma
        ).clip(
            -20,
            20,
        )

        stress = (
            z.abs()
            .median(
                axis=1,
                skipna=True,
            )
            .rename(
                "network_stress"
            )
        )

        valid = (
            stress.dropna()
        )

        if valid.empty:
            return StressResult(
                np.nan,
                np.nan,
                np.nan,
                stress,
                z,
            )

        current = float(
            valid.iloc[-1]
        )

        hist = valid.iloc[:-1]

        percentile = (
            float(
                (
                    hist <= current
                ).mean()
                * 100.0
            )
            if len(hist) >= 10
            else np.nan
        )

        coherence = float(
            np.exp(
                -min(
                    current,
                    10.0,
                )
            )
        )

        return StressResult(
            current_stress=
                current,
            stress_percentile=
                percentile,
            coherence=
                coherence,
            series=
                stress,
            triangle_z=
                z,
    )
