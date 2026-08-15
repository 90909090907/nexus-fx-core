from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


VALIDATION_VERSION = "SCIENTIFIC-0.4.0"


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for the shadow scientific-validation layer."""

    n_groups: int = 6
    n_test_groups: int = 2
    embargo_bars: int = 2
    max_paths: int = 15
    min_train_obs: int = 80
    min_test_obs: int = 24
    random_state: int = 42

    def normalized(self) -> "ValidationConfig":
        groups = max(4, int(self.n_groups))
        return ValidationConfig(
            n_groups=groups,
            n_test_groups=min(max(1, int(self.n_test_groups)), groups - 1),
            embargo_bars=max(0, int(self.embargo_bars)),
            max_paths=max(1, int(self.max_paths)),
            min_train_obs=max(20, int(self.min_train_obs)),
            min_test_obs=max(8, int(self.min_test_obs)),
            random_state=int(self.random_state),
        )


class ScientificValidationEngine:
    """
    Purged CPCV-style validation for already-discovered NEXUS edges.

    SHADOW MODE:
    this class adds diagnostics only. It never changes `decision`, `survives`,
    the current FDR funnel, or the existing BUY/SELL rules.
    """

    OUTPUT_COLUMNS: Tuple[str, ...] = (
        "cpcv_paths",
        "cpcv_median_ic",
        "cpcv_q10_ic",
        "cpcv_sign_consistency",
        "cpcv_median_accuracy",
        "cpcv_worst_accuracy",
        "cpcv_train_median_ic",
        "cpcv_overfit_gap",
        "cpcv_pseudo_sharpe",
        "cpcv_dsr",
        "validation_state",
    )

    def __init__(self, config: Optional[ValidationConfig] = None) -> None:
        self.config = (config or ValidationConfig()).normalized()

    @staticmethod
    def _pair_name(value: Any) -> str:
        return (
            str(value)
            .upper()
            .replace("/", "")
            .replace("=", "")
            .replace("X", "")
        )

    @classmethod
    def clean_returns(cls, returns: pd.DataFrame) -> pd.DataFrame:
        if (
            returns is None
            or not isinstance(returns, pd.DataFrame)
            or returns.empty
        ):
            return pd.DataFrame()

        out = returns.copy()
        out.columns = [cls._pair_name(c) for c in out.columns]
        out = out.loc[:, ~out.columns.duplicated(keep="first")]

        return (
            out.apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )

    @staticmethod
    def _pearson(x: pd.Series, y: pd.Series) -> float:
        z = pd.concat([x, y], axis=1).dropna()

        if len(z) < 3:
            return np.nan

        a = z.iloc[:, 0].to_numpy(dtype=float)
        b = z.iloc[:, 1].to_numpy(dtype=float)

        if np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
            return np.nan

        return float(np.corrcoef(a, b)[0, 1])

    @staticmethod
    def _accuracy(
        source: pd.Series,
        future_target: pd.Series,
        relation_sign: float,
    ) -> float:
        z = pd.concat([source, future_target], axis=1).dropna()

        if (
            z.empty
            or not np.isfinite(relation_sign)
            or relation_sign == 0
        ):
            return np.nan

        prediction = (
            np.sign(z.iloc[:, 0].to_numpy(dtype=float))
            * np.sign(relation_sign)
        )

        actual = np.sign(
            z.iloc[:, 1].to_numpy(dtype=float)
        )

        valid = (prediction != 0) & (actual != 0)

        if not np.any(valid):
            return np.nan

        return float(
            np.mean(prediction[valid] == actual[valid])
        )

    @staticmethod
    def _pseudo_returns(
        source: pd.Series,
        future_target: pd.Series,
        relation_sign: float,
    ) -> np.ndarray:
        z = pd.concat(
            [source, future_target],
            axis=1,
        ).dropna()

        if (
            z.empty
            or not np.isfinite(relation_sign)
            or relation_sign == 0
        ):
            return np.array([], dtype=float)

        position = (
            np.sign(z.iloc[:, 0].to_numpy(dtype=float))
            * np.sign(relation_sign)
        )

        pnl = (
            position
            * z.iloc[:, 1].to_numpy(dtype=float)
        )

        return pnl[np.isfinite(pnl)]

    @staticmethod
    def _sharpe(values: Sequence[float]) -> float:
        x = np.asarray(values, dtype=float)
        x = x[np.isfinite(x)]

        if x.size < 3:
            return np.nan

        sd = float(np.std(x, ddof=1))

        if sd <= 1e-15:
            return np.nan

        return float(
            np.mean(x) / sd
        )

    @staticmethod
    def _normal_cdf(z: float) -> float:
        if not np.isfinite(z):
            return np.nan

        return 0.5 * (
            1.0
            + math.erf(
                float(z) / math.sqrt(2.0)
            )
        )

    @staticmethod
    def _normal_ppf(p: float) -> float:
        """
        Acklam inverse-normal approximation.

        This avoids adding SciPy as a dependency.
        """

        p = float(p)

        if not 0.0 < p < 1.0:
            if p == 0.0:
                return -np.inf

            if p == 1.0:
                return np.inf

            return np.nan

        a = (
            -39.69683028665376,
            220.9460984245205,
            -275.9285104469687,
            138.3577518672690,
            -30.66479806614716,
            2.506628277459239,
        )

        b = (
            -54.47609879822406,
            161.5858368580409,
            -155.6989798598866,
            66.80131188771972,
            -13.28068155288572,
        )

        c = (
            -0.007784894002430293,
            -0.3223964580411365,
            -2.400758277161838,
            -2.549732539343734,
            4.374664141464968,
            2.938163982698783,
        )

        d = (
            0.007784695709041462,
            0.3224671290700398,
            2.445134137142996,
            3.754408661907416,
        )

        low = 0.02425
        high = 0.97575

        if p < low:
            q = math.sqrt(
                -2.0 * math.log(p)
            )

            return (
                (
                    (
                        (
                            (
                                c[0] * q + c[1]
                            ) * q + c[2]
                        ) * q + c[3]
                    ) * q + c[4]
                ) * q + c[5]
            ) / (
                (
                    (
                        (
                            d[0] * q + d[1]
                        ) * q + d[2]
                    ) * q + d[3]
                ) * q + 1.0
            )

        if p > high:
            q = math.sqrt(
                -2.0 * math.log(1.0 - p)
            )

            return -(
                (
                    (
                        (
                            (
                                c[0] * q + c[1]
                            ) * q + c[2]
                        ) * q + c[3]
                    ) * q + c[4]
                ) * q + c[5]
            ) / (
                (
                    (
                        (
                            d[0] * q + d[1]
                        ) * q + d[2]
                    ) * q + d[3]
                ) * q + 1.0
            )

        q = p - 0.5
        r = q * q

        return (
            (
                (
                    (
                        (
                            (
                                a[0] * r + a[1]
                            ) * r + a[2]
                        ) * r + a[3]
                    ) * r + a[4]
                ) * r + a[5]
            ) * q
        ) / (
            (
                (
                    (
                        (
                            b[0] * r + b[1]
                        ) * r + b[2]
                    ) * r + b[3]
                ) * r + b[4]
            ) * r + 1.0
        )

    @classmethod
    def deflated_sharpe_ratio(
        cls,
        observed_sharpe: float,
        n_obs: int,
        n_trials: int,
    ) -> float:
        """
        Diagnostic Deflated Sharpe Ratio.

        This is shadow information only.

        The pseudo-Sharpe used here is NOT yet an execution
        P&L metric because spread, slippage and transaction
        costs are not modelled in NEXUS v0.4.0.
        """

        sr = float(observed_sharpe)
        n = int(n_obs)
        trials = max(
            1,
            int(n_trials),
        )

        if (
            not np.isfinite(sr)
            or n < 3
        ):
            return np.nan

        if trials == 1:
            expected_max = 0.0

        else:
            gamma = 0.5772156649015329

            z1 = cls._normal_ppf(
                1.0 - 1.0 / trials
            )

            z2 = cls._normal_ppf(
                1.0
                - 1.0 / (
                    trials * math.e
                )
            )

            expected_max = (
                (1.0 - gamma) * z1
                + gamma * z2
            )

        denom = (
            1.0
            + 0.5 * sr * sr
        )

        z = (
            (sr - expected_max)
            * math.sqrt(
                max(n - 1, 1)
            )
            / math.sqrt(denom)
        )

        return float(
            cls._normal_cdf(z)
        )

    def _groups(
        self,
        n: int,
    ) -> List[np.ndarray]:

        return [
            np.asarray(x, dtype=int)
            for x in np.array_split(
                np.arange(n),
                min(
                    self.config.n_groups,
                    n,
                ),
            )
            if len(x)
        ]

    def _paths(
        self,
        group_count: int,
    ) -> List[Tuple[int, ...]]:

        paths = list(
            combinations(
                range(group_count),
                self.config.n_test_groups,
            )
        )

        if len(paths) <= self.config.max_paths:
            return paths

        rng = np.random.default_rng(
            self.config.random_state
        )

        pick = np.sort(
            rng.choice(
                len(paths),
                self.config.max_paths,
                replace=False,
            )
        )

        return [
            paths[int(i)]
            for i in pick
        ]

    @staticmethod
    def _blocks(
        positions: np.ndarray,
    ) -> List[Tuple[int, int]]:

        x = np.sort(
            np.unique(
                positions.astype(int)
            )
        )

        if x.size == 0:
            return []

        breaks = np.where(
            np.diff(x) > 1
        )[0]

        starts = np.r_[
            0,
            breaks + 1,
        ]

        ends = np.r_[
            breaks,
            len(x) - 1,
        ]

        return [
            (
                int(x[start]),
                int(x[end]),
            )
            for start, end
            in zip(starts, ends)
        ]

    def _purged_train(
        self,
        n: int,
        test_positions: np.ndarray,
        lag: int,
    ) -> np.ndarray:

        keep = np.ones(
            n,
            dtype=bool,
        )

        keep[test_positions] = False

        for start, end in self._blocks(
            test_positions
        ):
            left = max(
                0,
                start - max(
                    1,
                    int(lag),
                ),
            )

            right = min(
                n,
                end
                + self.config.embargo_bars
                + 1,
            )

            keep[left:right] = False

        return np.flatnonzero(keep)

    def _edge_paths(
        self,
        returns: pd.DataFrame,
        source: str,
        target: str,
        lag: int,
    ) -> pd.DataFrame:

        if (
            source not in returns.columns
            or target not in returns.columns
        ):
            return pd.DataFrame()

        lag = max(
            1,
            int(lag),
        )

        sample = pd.DataFrame(
            {
                "source": returns[source],
                "future_target": (
                    returns[target]
                    .shift(-lag)
                ),
            }
        ).dropna()

        if (
            len(sample)
            < self.config.min_train_obs
            + self.config.min_test_obs
        ):
            return pd.DataFrame()

        groups = self._groups(
            len(sample)
        )

        if (
            len(groups) < 4
            or self.config.n_test_groups
            >= len(groups)
        ):
            return pd.DataFrame()

        rows: List[
            Dict[str, Any]
        ] = []

        for (
            path_id,
            test_group_ids,
        ) in enumerate(
            self._paths(
                len(groups)
            )
        ):

            test_pos = np.sort(
                np.concatenate(
                    [
                        groups[i]
                        for i
                        in test_group_ids
                    ]
                )
            )

            train_pos = (
                self._purged_train(
                    len(sample),
                    test_pos,
                    lag,
                )
            )

            if (
                len(train_pos)
                < self.config.min_train_obs
                or len(test_pos)
                < self.config.min_test_obs
            ):
                continue

            train = sample.iloc[
                train_pos
            ]

            test = sample.iloc[
                test_pos
            ]

            train_ic = self._pearson(
                train["source"],
                train[
                    "future_target"
                ],
            )

            test_ic = self._pearson(
                test["source"],
                test[
                    "future_target"
                ],
            )

            if (
                not np.isfinite(train_ic)
                or not np.isfinite(test_ic)
            ):
                continue

            relation_sign = float(
                np.sign(train_ic)
            )

            accuracy = self._accuracy(
                test["source"],
                test["future_target"],
                relation_sign,
            )

            pseudo = (
                self._pseudo_returns(
                    test["source"],
                    test[
                        "future_target"
                    ],
                    relation_sign,
                )
            )

            rows.append(
                {
                    "path_id": int(
                        path_id
                    ),
                    "train_n": int(
                        len(train)
                    ),
                    "test_n": int(
                        len(test)
                    ),
                    "train_ic": float(
                        train_ic
                    ),
                    "test_ic": float(
                        test_ic
                    ),
                    "signed_test_ic": float(
                        np.sign(train_ic)
                        * test_ic
                    ),
                    "test_accuracy": (
                        accuracy
                    ),
                    "pseudo_sharpe": (
                        self._sharpe(
                            pseudo
                        )
                    ),
                    "pseudo_n": int(
                        len(pseudo)
                    ),
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _state(
        paths: int,
        median_ic: float,
        q10_ic: float,
        sign_consistency: float,
        median_accuracy: float,
    ) -> str:

        values = (
            median_ic,
            q10_ic,
            sign_consistency,
            median_accuracy,
        )

        if (
            paths < 4
            or not all(
                np.isfinite(v)
                for v in values
            )
        ):
            return "INSUFFICIENT"

        if (
            median_ic > 0.0
            and q10_ic > 0.0
            and sign_consistency >= 0.75
            and median_accuracy >= 0.52
        ):
            return "ROBUST"

        if (
            median_ic > 0.0
            and sign_consistency >= 0.60
            and median_accuracy >= 0.50
        ):
            return "FRAGILE"

        return "UNSTABLE"

    @staticmethod
    def estimate_family_pbo(
        paths: pd.DataFrame,
    ) -> float:
        """
        Approximate Probability of Backtest Overfitting.

        On each common CPCV path:
        1. Select the edge with highest |train IC|.
        2. Check its OOS percentile.
        3. Count how often the IS winner falls
           in the lower half OOS.
        """

        required = {
            "path_id",
            "edge_id",
            "train_ic",
            "signed_test_ic",
        }

        if (
            paths is None
            or paths.empty
            or not required.issubset(
                paths.columns
            )
        ):
            return np.nan

        failed: List[
            float
        ] = []

        for _, block in paths.groupby(
            "path_id"
        ):
            b = block.dropna(
                subset=[
                    "train_ic",
                    "signed_test_ic",
                ]
            ).copy()

            if len(b) < 2:
                continue

            winner = (
                b["train_ic"]
                .abs()
                .idxmax()
            )

            winner_oos = float(
                b.loc[
                    winner,
                    "signed_test_ic",
                ]
            )

            percentile = float(
                np.mean(
                    b[
                        "signed_test_ic"
                    ].to_numpy(
                        dtype=float
                    )
                    <= winner_oos
                )
            )

            failed.append(
                float(
                    percentile <= 0.50
                )
            )

        if not failed:
            return np.nan

        return float(
            np.mean(failed)
        )

    def validate_graph(
        self,
        returns: pd.DataFrame,
        graph: pd.DataFrame,
        family_trials: Optional[
            int
        ] = None,
    ) -> pd.DataFrame:
        """
        Return a copy of graph enriched with
        scientific-validation diagnostics.

        Required graph columns:
        source, target, lag.

        Existing signal columns remain untouched.
        """

        if isinstance(
            graph,
            pd.DataFrame,
        ):
            out = graph.copy()
        else:
            out = pd.DataFrame()

        for column in self.OUTPUT_COLUMNS:
            if (
                column
                == "validation_state"
            ):
                out[column] = (
                    "INSUFFICIENT"
                )
            else:
                out[column] = np.nan

        diagnostics: Dict[
            str,
            Any,
        ] = {
            "version": (
                VALIDATION_VERSION
            ),
            "mode": "SHADOW",
            "family_pbo": np.nan,
            "validated_edges": 0,
            "robust_edges": 0,
            "fragile_edges": 0,
            "unstable_edges": 0,
            "paths_total": 0,
            "config": asdict(
                self.config
            ),
        }

        if out.empty:
            out.attrs[
                "scientific_validation"
            ] = diagnostics
            return out

        returns_clean = (
            self.clean_returns(
                returns
            )
        )

        required = {
            "source",
            "target",
            "lag",
        }

        if (
            returns_clean.empty
            or not required.issubset(
                out.columns
            )
        ):
            diagnostics[
                "warning"
            ] = (
                "Missing returns "
                "or source/target/lag "
                "columns."
            )

            out.attrs[
                "scientific_validation"
            ] = diagnostics

            return out

        all_paths: List[
            pd.DataFrame
        ] = []

        trial_count = max(
            1,
            int(
                family_trials
                or len(out)
            ),
        )

        for idx, row in out.iterrows():
            source = self._pair_name(
                row["source"]
            )

            target = self._pair_name(
                row["target"]
            )

            try:
                lag = max(
                    1,
                    int(
                        row["lag"]
                    ),
                )
            except Exception:
                continue

            paths = self._edge_paths(
                returns_clean,
                source,
                target,
                lag,
            )

            if paths.empty:
                continue

            paths = paths.copy()

            paths["edge_id"] = (
                f"{source}"
                f"->{target}"
                f":L{lag}"
            )

            paths[
                "graph_index"
            ] = idx

            all_paths.append(
                paths
            )

            signed_ic = (
                pd.to_numeric(
                    paths[
                        "signed_test_ic"
                    ],
                    errors="coerce",
                )
            )

            raw_ic = pd.to_numeric(
                paths["test_ic"],
                errors="coerce",
            )

            train_ic = (
                pd.to_numeric(
                    paths[
                        "train_ic"
                    ],
                    errors="coerce",
                )
            )

            accuracy = (
                pd.to_numeric(
                    paths[
                        "test_accuracy"
                    ],
                    errors="coerce",
                )
            )

            pseudo_sr = (
                pd.to_numeric(
                    paths[
                        "pseudo_sharpe"
                    ],
                    errors="coerce",
                )
            )

            pseudo_n = (
                pd.to_numeric(
                    paths[
                        "pseudo_n"
                    ],
                    errors="coerce",
                )
            )

            reference = (
                pd.to_numeric(
                    pd.Series(
                        [
                            row.get(
                                "conditional_corr",
                                np.nan,
                            )
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]
            )

            if not np.isfinite(
                reference
            ):
                reference = float(
                    np.nanmedian(
                        train_ic
                    )
                )

            median_ic = float(
                np.nanmedian(
                    signed_ic
                )
            )

            q10_ic = float(
                np.nanquantile(
                    signed_ic,
                    0.10,
                )
            )

            sign_consistency = float(
                np.mean(
                    np.sign(raw_ic)
                    == np.sign(
                        reference
                    )
                )
            )

            median_accuracy = float(
                np.nanmedian(
                    accuracy
                )
            )

            worst_accuracy = float(
                np.nanmin(
                    accuracy
                )
            )

            train_median = float(
                np.nanmedian(
                    np.abs(
                        train_ic
                    )
                )
            )

            overfit_gap = float(
                train_median
                - np.nanmedian(
                    np.abs(
                        raw_ic
                    )
                )
            )

            median_sr = float(
                np.nanmedian(
                    pseudo_sr
                )
            )

            median_pseudo_n = (
                np.nanmedian(
                    pseudo_n
                )
            )

            if np.isfinite(
                median_pseudo_n
            ):
                median_n = int(
                    median_pseudo_n
                )
            else:
                median_n = 0

            dsr = (
                self.deflated_sharpe_ratio(
                    median_sr,
                    median_n,
                    trial_count,
                )
            )

            out.at[
                idx,
                "cpcv_paths",
            ] = int(
                len(paths)
            )

            out.at[
                idx,
                "cpcv_median_ic",
            ] = median_ic

            out.at[
                idx,
                "cpcv_q10_ic",
            ] = q10_ic

            out.at[
                idx,
                "cpcv_sign_consistency",
            ] = sign_consistency

            out.at[
                idx,
                "cpcv_median_accuracy",
            ] = median_accuracy

            out.at[
                idx,
                "cpcv_worst_accuracy",
            ] = worst_accuracy

            out.at[
                idx,
                "cpcv_train_median_ic",
            ] = train_median

            out.at[
                idx,
                "cpcv_overfit_gap",
            ] = overfit_gap

            out.at[
                idx,
                "cpcv_pseudo_sharpe",
            ] = median_sr

            out.at[
                idx,
                "cpcv_dsr",
            ] = dsr

            out.at[
                idx,
                "validation_state",
            ] = self._state(
                len(paths),
                median_ic,
                q10_ic,
                sign_consistency,
                median_accuracy,
            )

        if all_paths:
            paths_df = pd.concat(
                all_paths,
                ignore_index=True,
            )
        else:
            paths_df = pd.DataFrame()

        diagnostics[
            "family_pbo"
        ] = self.estimate_family_pbo(
            paths_df
        )

        diagnostics[
            "validated_edges"
        ] = int(
            out[
                "cpcv_paths"
            ].notna().sum()
        )

        diagnostics[
            "robust_edges"
        ] = int(
            (
                out[
                    "validation_state"
                ]
                == "ROBUST"
            ).sum()
        )

        diagnostics[
            "fragile_edges"
        ] = int(
            (
                out[
                    "validation_state"
                ]
                == "FRAGILE"
            ).sum()
        )

        diagnostics[
            "unstable_edges"
        ] = int(
            (
                out[
                    "validation_state"
                ]
                == "UNSTABLE"
            ).sum()
        )

        diagnostics[
            "paths_total"
        ] = int(
            len(paths_df)
        )

        out.attrs[
            "scientific_validation"
        ] = diagnostics

        out.attrs[
            "scientific_validation_paths"
        ] = paths_df

        return out


__all__ = [
    "VALIDATION_VERSION",
    "ValidationConfig",
    "ScientificValidationEngine",
      ]
