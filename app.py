from __future__ import annotations

from itertools import permutations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nexus_fx.data_yahoo import download_close_matrix
from nexus_fx.latent import LatentCurrencyEngine
from nexus_fx.universe import normalize_pair, split_pair

APP_VERSION = "0.1.3"
NETWORK_VERSION = "SAFE-INLINE-0.1.3"


class CrossPairNetworkSafe:
    """Deployment-safe cross-pair diagnostics.

    This implementation is intentionally embedded in app.py so Streamlit does
    not import the older nexus_fx/network.py file. All lead/lag calculations
    operate on plain 1-D numeric arrays and chronological integer slices.
    """

    COLUMNS = ["source", "target", "lag", "correlation", "stability", "score"]

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
    def _corr_arrays(source, target, lag: int) -> float:
        x = np.asarray(source, dtype=np.float64).ravel()
        y = np.asarray(target, dtype=np.float64).ravel()
        lag = int(lag)
        if lag < 0:
            return np.nan
        if lag:
            if x.size <= lag or y.size <= lag:
                return np.nan
            x = x[:-lag]
            y = y[lag:]
        n = int(min(x.size, y.size))
        if n < 20:
            return np.nan
        # Positional slices only; no string/label indexing can reach NumPy.
        x = x[0:n]
        y = y[0:n]
        mask = np.logical_and(np.isfinite(x), np.isfinite(y))
        if int(mask.sum()) < 20:
            return np.nan
        x = x[mask]
        y = y[mask]
        if x.size < 20 or np.std(x) <= 1e-15 or np.std(y) <= 1e-15:
            return np.nan
        value = np.corrcoef(x, y)[0, 1]
        return float(value) if np.isfinite(value) else np.nan

    def lead_lag_edges(self, close: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
        r = self.log_returns(close)
        if r.empty or r.shape[1] < 2:
            return pd.DataFrame(columns=self.COLUMNS)

        names = [str(c) for c in r.columns]
        records: List[Dict[str, float]] = []

        for source, target in permutations(names, 2):
            # Extract columns as Series first, then immediately convert to a 2-column ndarray.
            pair_df = pd.concat(
                [r.loc[:, source].rename("source"), r.loc[:, target].rename("target")],
                axis=1,
            ).dropna(how="any")
            if len(pair_df) < self.min_obs:
                continue

            values = pair_df.to_numpy(dtype=np.float64, copy=True)
            if values.ndim != 2 or values.shape[1] != 2:
                continue
            x = values[:, 0].copy()
            y = values[:, 1].copy()

            best_lag: Optional[int] = None
            best_corr: Optional[float] = None
            for lag in range(1, self.max_lag + 1):
                corr = self._corr_arrays(x, y, lag)
                if not np.isfinite(corr):
                    continue
                if best_corr is None or abs(corr) > abs(best_corr):
                    best_lag = lag
                    best_corr = float(corr)

            if best_lag is None or best_corr is None:
                continue

            # Manual chronological splitting using integer start/end positions only.
            n_rows = int(values.shape[0])
            boundaries = np.linspace(0, n_rows, self.stability_splits + 1, dtype=int)
            block_corrs: List[float] = []
            for block_i in range(self.stability_splits):
                start = int(boundaries[block_i])
                end = int(boundaries[block_i + 1])
                if end - start < max(20, best_lag + 5):
                    continue
                bx = values[start:end, 0]
                by = values[start:end, 1]
                corr = self._corr_arrays(bx, by, best_lag)
                if np.isfinite(corr):
                    block_corrs.append(float(corr))

            if block_corrs and best_corr != 0:
                expected_sign = np.sign(best_corr)
                stability = float(np.mean([np.sign(v) == expected_sign for v in block_corrs]))
            else:
                stability = 0.0

            sample_factor = float(np.sqrt(min(n_rows, 1000) / 1000.0))
            score = float(abs(best_corr) * stability * sample_factor)
            records.append(
                {
                    "source": source,
                    "target": target,
                    "lag": int(best_lag),
                    "correlation": float(best_corr),
                    "stability": stability,
                    "score": score,
                }
            )

        if not records:
            return pd.DataFrame(columns=self.COLUMNS)
        return (
            pd.DataFrame(records, columns=self.COLUMNS)
            .sort_values(["score", "correlation"], ascending=[False, False])
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
        return out[out["magnitude"] >= float(threshold)].sort_values("magnitude", ascending=False).reset_index(drop=True)

    @staticmethod
    def triangular_residuals(close: pd.DataFrame) -> pd.DataFrame:
        if close is None or not isinstance(close, pd.DataFrame) or close.empty:
            return pd.DataFrame()
        work = close.copy()
        work.columns = [normalize_pair(str(c)) for c in work.columns]
        work = work.loc[:, ~work.columns.duplicated(keep="first")]
        numeric = work.apply(pd.to_numeric, errors="coerce")
        logs = np.log(numeric.where(numeric > 0))
        currencies = sorted({ccy for pair in logs.columns for ccy in split_pair(str(pair))})

        def get_log(a: str, b: str):
            direct, inverse = a + b, b + a
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
                    ab, bc, ac = get_log(a, b), get_log(b, c), get_log(a, c)
                    if ab is None or bc is None or ac is None:
                        continue
                    result[f"{a}-{b}-{c}"] = ab + bc - ac
        return pd.DataFrame(result, index=logs.index)


st.set_page_config(page_title="NEXUS FX CORE", page_icon="◈", layout="wide")

st.markdown("""
<style>
.stApp { background:#080b12; color:#eef3ff; }
.block-container { padding-top: 1.1rem; max-width: 1500px; }
.nexus-title {font-size:clamp(2.1rem,8vw,4.5rem);font-weight:800;letter-spacing:.08em;
 background:linear-gradient(90deg,#42e8ff,#8c68ff,#ff4fd8);-webkit-background-clip:text;color:transparent;}
.nexus-sub {color:#93a4c7;letter-spacing:.22em;text-transform:uppercase;font-size:.78rem;}
.card {background:linear-gradient(145deg,#111724,#0b101a);border:1px solid #202c43;border-radius:18px;padding:16px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="nexus-title">NEXUS FX CORE</div>', unsafe_allow_html=True)
st.markdown('<div class="nexus-sub">Latent Currency State · Cross-Pair Matrix · Research Prototype</div>', unsafe_allow_html=True)
st.caption(f"NEXUS v{APP_VERSION} · Network {NETWORK_VERSION} · proyecto independiente")

DEFAULT_PAIRS = [
    "EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD",
    "EURGBP","EURJPY","GBPJPY","AUDJPY","CADJPY","CHFJPY","AUDNZD",
]

with st.expander("⚙️ Configuración", expanded=True):
    c1, c2, c3 = st.columns(3)
    interval = c1.selectbox("Temporalidad", ["1h", "15m", "5m", "1d"], index=0)
    period_options = {"1h": ["5d", "1mo", "3mo"], "15m": ["5d", "1mo"], "5m": ["1d", "5d", "1mo"], "1d": ["6mo", "1y", "2y", "5y"]}
    period = c2.selectbox("Histórico", period_options[interval], index=min(1, len(period_options[interval])-1))
    max_lag = c3.slider("Lead/Lag máximo (velas)", 1, 12, 5)
    pairs = st.multiselect("Pares", DEFAULT_PAIRS, default=DEFAULT_PAIRS)


@st.cache_data(ttl=55, show_spinner=False)
def get_data(pairs_tuple, period, interval):
    return download_close_matrix(list(pairs_tuple), period=period, interval=interval)


def analyze(close: pd.DataFrame):
    engine = LatentCurrencyEngine()
    latent = engine.fit(close)
    network = CrossPairNetworkSafe(max_lag=max_lag)

    network_warning = None
    try:
        edges = network.lead_lag_edges(close, top_n=25)
    except Exception as exc:
        # Lead/Lag must never prevent the rest of NEXUS from loading.
        edges = pd.DataFrame(columns=CrossPairNetworkSafe.COLUMNS)
        network_warning = f"Lead/Lag desactivado por seguridad: {type(exc).__name__}: {exc}"

    divergences = network.divergence_table(latent.residual_z, threshold=1.25)
    try:
        triangles = network.triangular_residuals(close)
    except Exception as exc:
        triangles = pd.DataFrame()
        extra = f"Triangulación desactivada: {type(exc).__name__}: {exc}"
        network_warning = f"{network_warning} | {extra}" if network_warning else extra

    return latent, edges, divergences, triangles, network_warning


if st.button("⚡ Construir estado del mercado", use_container_width=True, type="primary"):
    if len(pairs) < 7:
        st.error("Selecciona al menos 7 pares bien conectados para estimar las 8 monedas con estabilidad.")
        st.stop()
    try:
        with st.spinner("Sincronizando matriz FX y resolviendo el estado latente..."):
            close = get_data(tuple(pairs), period, interval)
            latent, edges, divergences, triangles, network_warning = analyze(close)
            st.session_state["nexus_result"] = (close, latent, edges, divergences, triangles, network_warning)
    except Exception as exc:
        st.exception(exc)

if "nexus_result" not in st.session_state:
    st.info("Pulsa “Construir estado del mercado”. En esta fase solo medimos estructura relativa; todavía no emitimos señales BUY/SELL.")
    st.stop()

close, latent, edges, divergences, triangles, network_warning = st.session_state["nexus_result"]
if network_warning:
    st.warning(network_warning)

strength_z = LatentCurrencyEngine.standardized_snapshot(latent.strength).sort_values(ascending=False)
vel_z = LatentCurrencyEngine.standardized_snapshot(latent.velocity).reindex(strength_z.index)
acc_z = LatentCurrencyEngine.standardized_snapshot(latent.acceleration).reindex(strength_z.index)
unc = latent.uncertainty.iloc[-1].reindex(strength_z.index)

st.subheader("Estado latente de monedas")
cols = st.columns(4)
for i, ccy in enumerate(strength_z.index):
    s = strength_z[ccy]
    v = vel_z.get(ccy, np.nan)
    a = acc_z.get(ccy, np.nan)
    with cols[i % 4]:
        st.markdown(
            f'<div class="card"><b>{ccy}</b><br><span style="font-size:2rem">{s:+.2f}σ</span><br>'
            f'<small>vel {v:+.2f}σ · acc {a:+.2f}σ · unc {unc[ccy]:.5f}</small></div>',
            unsafe_allow_html=True,
        )

fig = go.Figure()
for ccy in latent.strength.columns:
    rolling = latent.strength[ccy].rolling(120, min_periods=30)
    z = (latent.strength[ccy] - rolling.mean()) / rolling.std(ddof=0).replace(0.0, np.nan)
    fig.add_trace(go.Scatter(x=z.index, y=z, mode="lines", name=ccy))
fig.update_layout(template="plotly_dark", height=460, margin=dict(l=20,r=20,t=35,b=20), title="Fuerza relativa estandarizada")
st.plotly_chart(fig, use_container_width=True)

t1, t2, t3 = st.tabs(["Divergencias", "Lead / Lag", "Triangulación"])
with t1:
    st.caption("Residual observado − reconstruido por el estado latente. Un extremo es una anomalía relativa, no una orden de trading.")
    st.dataframe(divergences, use_container_width=True, hide_index=True)
with t2:
    st.caption("Relaciones predictivas descriptivas. Correlación adelantada ≠ causalidad demostrada.")
    if edges.empty:
        st.info("No hay relaciones Lead/Lag robustas disponibles para esta muestra.")
    else:
        st.dataframe(edges, use_container_width=True, hide_index=True)
with t3:
    if triangles.empty:
        st.info("No hay suficientes triángulos completos con los pares seleccionados.")
    else:
        latest_tri = triangles.iloc[-1].abs().sort_values(ascending=False).rename("abs_log_residual").to_frame().head(20)
        st.dataframe(latest_tri, use_container_width=True)

st.divider()
st.caption(f"Filas sincronizadas: {len(close):,} · Pares: {len(close.columns)} · Último dato: {close.index[-1]}")
st.warning("v0.1.3 es un laboratorio de estructura relativa. No usa todavía calendario macro, order flow institucional, opciones, Hawkes, causalidad condicional ni ejecución real.")
