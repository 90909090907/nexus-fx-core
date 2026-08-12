from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nexus_fx.data_yahoo import download_close_matrix
from nexus_fx.intelligence import CausalGraphEngine, NetworkStressEngine, RegimeEngine
from nexus_fx.latent import LatentCurrencyEngine
from nexus_fx.universe import normalize_pair, split_pair

APP_VERSION = "0.2.1"
INTELLIGENCE_VERSION = "STAT-HARDENED-0.2.1"


class StructuralDiagnostics:
    """Deployment-safe divergence and triangle diagnostics."""

    @staticmethod
    def divergence_table(residual_z: pd.DataFrame, threshold: float = 1.25) -> pd.DataFrame:
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


st.set_page_config(page_title="NEXUS FX CORE v0.2.1", page_icon="◈", layout="wide")

st.markdown(
    """
<style>
.stApp { background:#080b12; color:#eef3ff; }
.block-container { padding-top: 1.1rem; max-width: 1500px; }
.nexus-title {font-size:clamp(2.1rem,8vw,4.5rem);font-weight:800;letter-spacing:.08em;
 background:linear-gradient(90deg,#42e8ff,#8c68ff,#ff4fd8);-webkit-background-clip:text;color:transparent;}
.nexus-sub {color:#93a4c7;letter-spacing:.18em;text-transform:uppercase;font-size:.76rem;}
.card {background:linear-gradient(145deg,#111724,#0b101a);border:1px solid #202c43;border-radius:18px;padding:16px;}
.metric-label {color:#8fa1c4;font-size:.72rem;text-transform:uppercase;letter-spacing:.12em;}
.big {font-size:1.7rem;font-weight:750;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="nexus-title">NEXUS FX CORE</div>', unsafe_allow_html=True)
st.markdown('<div class="nexus-sub">Latent State · Regime Reliability · FDR · Bootstrap · Walk-Forward · Network Stress</div>', unsafe_allow_html=True)
st.caption(f"NEXUS v{APP_VERSION} · Intelligence {INTELLIGENCE_VERSION} · research prototype")

DEFAULT_PAIRS = [
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "AUDNZD",
]

with st.expander("⚙️ Configuración", expanded=True):
    c1, c2, c3 = st.columns(3)
    interval = c1.selectbox("Temporalidad", ["1h", "15m", "5m", "1d"], index=0)
    period_options = {
        "1h": ["5d", "1mo", "3mo"],
        "15m": ["5d", "1mo"],
        "5m": ["1d", "5d", "1mo"],
        "1d": ["6mo", "1y", "2y", "5y"],
    }
    period = c2.selectbox("Histórico", period_options[interval], index=min(1, len(period_options[interval]) - 1))
    max_lag = c3.slider("Lead/Lag máximo (velas)", 1, 12, 5)
    pairs = st.multiselect("Pares", DEFAULT_PAIRS, default=DEFAULT_PAIRS)

    with st.expander("Ajustes de investigación avanzada", expanded=False):
        a1, a2 = st.columns(2)
        top_edges = a1.slider("Aristas candidatas a mostrar", 10, 40, 20, step=5)
        permutations_n = a2.slider("Permutaciones por arista", 16, 96, 48, step=16)
        a3, a4 = st.columns(2)
        bootstrap_n = a3.slider("Bootstrap por arista", 16, 96, 48, step=16)
        fdr_alpha = a4.select_slider("FDR máximo (q)", options=[0.05, 0.075, 0.10, 0.15], value=0.10)


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
        graph = CausalGraphEngine(
            max_lag=max_lag,
            min_obs=80 if interval != "1d" else 60,
            permutations_n=permutations_n,
            bootstrap_n=bootstrap_n,
            walkforward_folds=3,
            fdr_alpha=fdr_alpha,
        ).build(
            close,
            latent_strength=latent.strength,
            regime=regime,
            top_n=top_edges,
        )
    except Exception as exc:
        graph = pd.DataFrame(columns=CausalGraphEngine.COLUMNS)
        graph_warning = f"Grafo causal candidato desactivado por seguridad: {type(exc).__name__}: {exc}"

    return latent, regime, graph, divergences, triangles, stress, graph_warning


if st.button("⚡ Construir estado del mercado", use_container_width=True, type="primary"):
    if len(pairs) < 7:
        st.error("Selecciona al menos 7 pares conectados para estimar la red monetaria con estabilidad.")
        st.stop()
    try:
        with st.spinner("Sincronizando FX, resolviendo estado latente, régimen y grafo condicional..."):
            close = get_data(tuple(pairs), period, interval)
            result = analyze(close)
            st.session_state["nexus_v021_result"] = (close,) + result
    except Exception as exc:
        st.exception(exc)

if "nexus_v021_result" not in st.session_state:
    st.info("Pulsa “Construir estado del mercado”. v0.2.1 endurece la evidencia estadística; sigue sin emitir órdenes BUY/SELL.")
    st.stop()

close, latent, regime, graph, divergences, triangles, stress, graph_warning = st.session_state["nexus_v021_result"]
if graph_warning:
    st.warning(graph_warning)

# -----------------------------------------------------------------------------
# MARKET STATE OVERVIEW
# -----------------------------------------------------------------------------
st.subheader("Estado global")
reg_probs = regime.current_probabilities
regime_name = regime.current_regime
regime_conf = float(reg_probs.iloc[0]) if not reg_probs.empty else np.nan
regime_reliability = regime.current_reliability
graph_diag = graph.attrs.get("diagnostics", {}) if isinstance(graph, pd.DataFrame) else {}
survivors = int(graph_diag.get("survivors", int(graph["survives"].sum()) if (not graph.empty and "survives" in graph) else 0))
survival_rate = float(graph_diag.get("survival_rate", 0.0))

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(f'<div class="card"><div class="metric-label">Régimen dominante</div><div class="big">{regime_name}</div><small>posterior {regime_conf:.1%}</small></div>', unsafe_allow_html=True)
with m2:
    rtxt = f"{regime_reliability:.1%}" if np.isfinite(regime_reliability) else "N/D"
    st.markdown(f'<div class="card"><div class="metric-label">Fiabilidad régimen</div><div class="big">{rtxt}</div><small>autoconsistencia, no prob. calibrada</small></div>', unsafe_allow_html=True)
with m3:
    pctl = stress.stress_percentile
    ptxt = f"{pctl:.0f}º" if np.isfinite(pctl) else "N/D"
    st.markdown(f'<div class="card"><div class="metric-label">Network stress</div><div class="big">{stress.current_stress:.2f}</div><small>percentil {ptxt}</small></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="card"><div class="metric-label">Coherencia triangular</div><div class="big">{stress.coherence:.1%}</div><small>1 = máxima coherencia</small></div>', unsafe_allow_html=True)
with m5:
    st.markdown(f'<div class="card"><div class="metric-label">Aristas sobrevivientes</div><div class="big">{survivors}</div><small>survival rate {survival_rate:.1%}</small></div>', unsafe_allow_html=True)

if not reg_probs.empty:
    fig_reg = go.Figure(go.Bar(x=reg_probs.index, y=reg_probs.values))
    fig_reg.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=20, b=30), yaxis_tickformat=".0%", title="Posterior de régimen")
    st.plotly_chart(fig_reg, use_container_width=True)

# -----------------------------------------------------------------------------
# LATENT CURRENCY STATE
# -----------------------------------------------------------------------------
st.subheader("Estado latente de monedas")
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
        st.markdown(
            f'<div class="card"><b>{ccy}</b><br><span style="font-size:2rem">{s:+.2f}σ</span><br>'
            f'<small>vel {v:+.2f}σ · acc {a:+.2f}σ · unc {unc[ccy]:.5f}</small></div>',
            unsafe_allow_html=True,
        )

# Cross-sectional standardization is numerically stable and directly answers
# "which currency is strong relative to the rest right now?".
strength_hist = latent.strength.copy().replace([np.inf, -np.inf], np.nan)
cs_mean = strength_hist.mean(axis=1)
cs_std = strength_hist.std(axis=1, ddof=0).replace(0.0, np.nan)
strength_cs_z = strength_hist.sub(cs_mean, axis=0).div(cs_std, axis=0).clip(-5.0, 5.0).tail(350)
fig = go.Figure()
for ccy in strength_cs_z.columns:
    fig.add_trace(go.Scatter(x=strength_cs_z.index, y=strength_cs_z[ccy], mode="lines", name=ccy))
fig.add_hline(y=0.0, line_width=1, opacity=0.35)
fig.update_layout(template="plotly_dark", height=430, margin=dict(l=20, r=20, t=35, b=20), title="Fuerza relativa estandarizada (cross-section)", yaxis_title="σ relativo")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# INTELLIGENCE TABS
# -----------------------------------------------------------------------------
t1, t2, t3, t4, t5 = st.tabs(["Grafo / Evidencia", "Régimen", "Divergencias", "Stress / Triángulos", "Diagnóstico"])

with t1:
    st.caption("v0.2.1 exige FDR, bootstrap, walk-forward, estabilidad y decay. Una arista sobreviviente sigue siendo una hipótesis predictiva, no causalidad económica demostrada.")
    if graph.empty:
        st.info("No hay aristas candidatas suficientes para esta muestra.")
    else:
        only_survivors = st.toggle("Mostrar solo aristas sobrevivientes", value=False)
        shown = graph.loc[graph["survives"]].copy() if only_survivors else graph.copy()
        display_cols = [
            "source", "target", "lag", "shared_factor", "conditional_corr",
            "fdr_q", "bootstrap_low", "bootstrap_high", "walkforward_ic",
            "walkforward_sign_rate", "walkforward_p", "walkforward_n", "stability", "decay_state",
            "incremental_r2", "evidence_score", "survives", "n_obs",
        ]
        st.dataframe(shown[display_cols], use_container_width=True, hide_index=True)
        top = shown.head(12).copy()
        if not top.empty:
            top["edge"] = top["source"] + " → " + top["target"] + " (L" + top["lag"].astype(str) + ")"
            fig_edges = go.Figure(go.Bar(x=top["evidence_score"], y=top["edge"], orientation="h"))
            fig_edges.update_layout(template="plotly_dark", height=max(350, 32 * len(top)), margin=dict(l=20, r=20, t=35, b=20), title="NEXUS Evidence Score", yaxis={"autorange": "reversed"}, xaxis_range=[0, 1])
            st.plotly_chart(fig_edges, use_container_width=True)

with t2:
    st.caption("El posterior de régimen no está externamente calibrado. v0.2.1 añade entropía, persistencia y una métrica separada de fiabilidad/autoconsistencia.")
    if regime.probabilities.empty:
        st.info("No hay histórico suficiente para régimen.")
    else:
        fig_hist = go.Figure()
        for name in regime.probabilities.columns:
            fig_hist.add_trace(go.Scatter(x=regime.probabilities.index, y=regime.probabilities[name], mode="lines", stackgroup="one", name=name))
        fig_hist.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=35, b=20), yaxis_tickformat=".0%", title="Evolución de probabilidades de régimen")
        st.plotly_chart(fig_hist, use_container_width=True)
        latest_metrics = regime.metrics.iloc[-1].to_frame("valor")
        st.dataframe(latest_metrics, use_container_width=True)

with t3:
    st.caption("Residual observado − reconstruido por el estado latente. Una anomalía relativa no es una orden de trading.")
    st.dataframe(divergences, use_container_width=True, hide_index=True)

with t4:
    st.caption("Network Stress estandariza inconsistencias triangulares contra su propia historia. Un aumento coordinado pesa más que un residual aislado.")
    if stress.series.empty:
        st.info("No hay suficientes triángulos para calcular stress.")
    else:
        fig_stress = go.Figure(go.Scatter(x=stress.series.index, y=stress.series, mode="lines", name="Network Stress"))
        fig_stress.update_layout(template="plotly_dark", height=340, margin=dict(l=20, r=20, t=35, b=20), title="FX Network Stress")
        st.plotly_chart(fig_stress, use_container_width=True)
        if not triangles.empty:
            latest_tri = triangles.iloc[-1].abs().sort_values(ascending=False).rename("abs_log_residual").to_frame().head(20)
            st.dataframe(latest_tri, use_container_width=True)

with t5:
    st.markdown("**Pipeline de evidencia v0.2.1:**")
    st.markdown(
        "1. Descubre el lag en el 70% inicial (con purga temporal).  \n"
        "2. Residualiza la moneda compartida cuando existe.  \n"
        "3. Mide estabilidad cronológica.  \n"
        "4. Calcula `perm_p` con desplazamientos circulares.  \n"
        "5. Corrige pruebas múltiples con Benjamini–Hochberg sobre p-values OOS (`fdr_q`).  \n"
        "6. Exige intervalo bootstrap que no cruce cero.  \n"
        "7. Valida en walk-forward fuera de la ventana de descubrimiento.  \n"
        "8. Penaliza relaciones que muestran decay reciente.  \n"
        "9. Combina las pruebas en `evidence_score`."
    )
    d = graph_diag
    if d:
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Evaluadas", d.get("evaluated", 0))
        dc2.metric("Pasan FDR", d.get("fdr_pass", 0))
        dc3.metric("Pasan bootstrap", d.get("bootstrap_pass", 0))
        dc4, dc5, dc6 = st.columns(3)
        dc4.metric("Pasan walk-forward", d.get("walkforward_pass", 0))
        dc5.metric("Sobreviven", d.get("survivors", 0))
        dc6.metric("Edge Survival Rate", f"{d.get('survival_rate', 0.0):.1%}")
    st.warning("FDR y walk-forward reducen falsos positivos, pero no convierten correlación temporal en causalidad. Antes de dinero real todavía faltan costes, spreads, slippage, macro point-in-time y backtest purgado completo.")

st.divider()
st.caption(f"Filas sincronizadas: {len(close):,} · Pares: {len(close.columns)} · Último dato: {close.index[-1]}")
st.warning("NEXUS v0.2.1 es investigación estadística estructural. No ejecuta operaciones ni recomienda apalancamiento.")
