from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

CURRENCIES: Tuple[str, ...] = ("USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD")

# Research universe. Yahoo symbols are only a prototype data source; the core is provider-agnostic.
YAHOO_TICKERS: Dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDJPY": "JPY=X",
    "USDCHF": "CHF=X",
    "USDCAD": "CAD=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "EURCHF": "EURCHF=X",
    "EURCAD": "EURCAD=X",
    "EURAUD": "EURAUD=X",
    "EURNZD": "EURNZD=X",
    "GBPJPY": "GBPJPY=X",
    "GBPCHF": "GBPCHF=X",
    "GBPCAD": "GBPCAD=X",
    "GBPAUD": "GBPAUD=X",
    "GBPNZD": "GBPNZD=X",
    "AUDJPY": "AUDJPY=X",
    "AUDCHF": "AUDCHF=X",
    "AUDCAD": "AUDCAD=X",
    "AUDNZD": "AUDNZD=X",
    "NZDJPY": "NZDJPY=X",
    "NZDCHF": "NZDCHF=X",
    "NZDCAD": "NZDCAD=X",
    "CADJPY": "CADJPY=X",
    "CADCHF": "CADCHF=X",
    "CHFJPY": "CHFJPY=X",
}


def split_pair(pair: str) -> Tuple[str, str]:
    pair = pair.replace("/", "").upper().strip()
    if len(pair) != 6:
        raise ValueError(f"Invalid FX pair: {pair!r}")
    base, quote = pair[:3], pair[3:]
    if base == quote:
        raise ValueError("Base and quote currency cannot be identical")
    return base, quote


def normalize_pair(pair: str) -> str:
    base, quote = split_pair(pair)
    return base + quote


def all_possible_pairs(currencies: Sequence[str] = CURRENCIES) -> List[str]:
    """Return canonical unordered combinations, not necessarily market conventions."""
    return [a + b for a, b in combinations(currencies, 2)]


@dataclass(frozen=True)
class PairSpec:
    symbol: str
    base: str
    quote: str

    @classmethod
    def from_symbol(cls, symbol: str) -> "PairSpec":
        base, quote = split_pair(symbol)
        return cls(symbol=base + quote, base=base, quote=quote)
