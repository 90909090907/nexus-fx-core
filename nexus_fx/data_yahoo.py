from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import pandas as pd
import yfinance as yf

from .universe import YAHOO_TICKERS, normalize_pair


def download_close_matrix(
    pairs: Sequence[str],
    period: str = "1mo",
    interval: str = "1h",
) -> pd.DataFrame:
    """Prototype research downloader using Yahoo Finance via yfinance.

    Important: this is intentionally not the production execution feed. The Nexus core accepts
    any synchronized close matrix and will later receive a broker-grade adapter.
    """
    normalized = [normalize_pair(p) for p in pairs]
    mapping: Dict[str, str] = {}
    missing = []
    for pair in normalized:
        ticker = YAHOO_TICKERS.get(pair)
        if ticker is None:
            missing.append(pair)
        else:
            mapping[pair] = ticker
    if missing:
        raise ValueError(f"No Yahoo prototype mapping for: {', '.join(missing)}")

    raw = yf.download(
        list(mapping.values()),
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
        multi_level_index=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("No FX data returned by Yahoo Finance")

    close = pd.DataFrame(index=raw.index)
    for pair, ticker in mapping.items():
        series = None
        if isinstance(raw.columns, pd.MultiIndex):
            # group_by='ticker': first level is ticker, second is field.
            if ticker in raw.columns.get_level_values(0):
                sub = raw[ticker]
                if "Close" in sub.columns:
                    series = sub["Close"]
            # Defensive fallback for alternative MultiIndex orientation.
            if series is None and "Close" in raw.columns.get_level_values(0):
                sub = raw["Close"]
                if ticker in sub.columns:
                    series = sub[ticker]
        else:
            if len(mapping) == 1 and "Close" in raw.columns:
                series = raw["Close"]
        if series is not None:
            close[pair] = pd.to_numeric(series, errors="coerce")

    close = close.sort_index().dropna(how="all")
    # Keep rows with enough simultaneous observations to identify the 8-currency system.
    min_pairs = max(4, min(7, len(close.columns)))
    close = close[close.notna().sum(axis=1) >= min_pairs]
    if close.empty:
        raise RuntimeError("Downloaded data could not be aligned into a usable FX matrix")
    return close
