from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nexus_fx.data_yahoo import download_close_matrix
from nexus_fx.latent import LatentCurrencyEngine
from nexus_fx.network import CrossPairNetwork

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
st.markdown('<div class="nexus-sub">Latent Currency State · Cross-Pair Matrix · Research Prototype v0.1</div>', unsafe_allow_html=True)
st.caption("Proyecto independiente. No reutiliza ni importa ningún módulo de la aplicación anterior.")

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
    network = CrossPairNetwork(max_lag=max_lag)
    edges = network.lead_lag_edges(close, top_n=25)
    divergences = network.divergence_table(latent.residual_z, threshold=1.25)
    triangles = network.triangular_residuals(close)
    return latent, edges, divergences, triangles

if st.button("⚡ Construir estado del mercado", use_container_width=True, type="primary"):
    if len(pairs) < 7:
        st.error("Selecciona al menos 7 pares bien conectados para estimar las 8 monedas con estabilidad.")
        st.stop()
    try:
        with st.spinner("Sincronizando matriz FX y resolviendo el estado latente..."):
            close = get_data(tuple(pairs), period, interval)
            latent, edges, divergences, triangles = analyze(close)
            st.session_state["nexus_result"] = (close, latent, edges, divergences, triangles)
    except Exception as exc:
        st.exception(exc)

if "nexus_result" not in st.session_state:
    st.info("Pulsa “Construir estado del mercado”. En esta fase solo medimos estructura relativa; todavía no emitimos señales BUY/SELL.")
    st.stop()

close, latent, edges, divergences, triangles = st.session_state["nexus_result"]

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
    z = (latent.strength[ccy] - latent.strength[ccy].rolling(120, min_periods=30).mean()) / latent.strength[ccy].rolling(120, min_periods=30).std(ddof=0)
    fig.add_trace(go.Scatter(x=z.index, y=z, mode="lines", name=ccy))
fig.update_layout(template="plotly_dark", height=460, margin=dict(l=20,r=20,t=35,b=20), title="Fuerza relativa estandarizada")
st.plotly_chart(fig, use_container_width=True)

t1, t2, t3 = st.tabs(["Divergencias", "Lead / Lag", "Triangulación"])
with t1:
    st.caption("Residual observado − reconstruido por el estado latente. Un extremo es una anomalía relativa, no una orden de trading.")
    st.dataframe(divergences, use_container_width=True, hide_index=True)
with t2:
    st.caption("Relaciones predictivas descriptivas. Correlación adelantada ≠ causalidad demostrada.")
    st.dataframe(edges, use_container_width=True, hide_index=True)
with t3:
    if triangles.empty:
        st.info("No hay suficientes triángulos completos con los pares seleccionados.")
    else:
        latest_tri = triangles.iloc[-1].abs().sort_values(ascending=False).rename("abs_log_residual").to_frame().head(20)
        st.dataframe(latest_tri, use_container_width=True)

st.divider()
st.caption(f"Filas sincronizadas: {len(close):,} · Pares: {len(close.columns)} · Último dato: {close.index[-1]}")
st.warning("v0.1 es un laboratorio de estructura relativa. No usa todavía calendario macro, order flow institucional, opciones, Hawkes, causalidad condicional ni ejecución real.")
