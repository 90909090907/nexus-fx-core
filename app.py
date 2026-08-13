from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nexus_fx.data_yahoo import download_close_matrix
from nexus_fx.latent import LatentCurrencyEngine
from nexus_fx.universe import normalize_pair, split_pair
from nexus_fx.intelligence import (
    CORE_PAIRS,
    ENGINE_VERSION,
    CausalGraphEngine,
    EvidenceMemoryEngine,
    NetworkStressEngine,
    RegimeEngine,
)

APP_VERSION = "0.3.0"
INTELLIGENCE_VERSION = ENGINE_VERSION


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
        return (
            out[out["magnitude"] >= float(threshold)]
            .sort_values("magnitude", ascending=False)
            .reset_index(drop=True)
        )

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


st.set_page_config(page_title="NEXUS FX CORE v0.3.0", page_icon="◈", layout="wide")

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
.signal-buy {border:1px solid #2c7f68;border-radius:18px;padding:18px;background:#0c1717;}
.signal-sell {border:1px solid #874c61;border-radius:18px;padding:18px;background:#171014;}
.signal-none {border:1px solid #39445a;border-radius:18px;padding:18px;background:#0d111a;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="nexus-title">NEXUS FX CORE</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="nexus-sub">8 pares · multi-stage · OOS · FDR · bootstrap · memoria · probabilidad calibrada</div>',
    unsafe_allow_html=True,
)
st.caption(f"NEXUS v{APP_VERSION} · Intelligence {INTELLIGENCE_VERSION} · research prototype")

FIXED_PAIRS = list(CORE_PAIRS)

with st.expander("⚙️ Configuración", expanded=True):
    c1, c2, c3 = st.columns(3)
    interval = c1.selectbox("Temporalidad", ["5m", "15m", "1h", "1d"], index=0)
    period_options = {
        "5m": ["1d", "5d", "1mo"],
        "15m": ["5d", "1mo"],
        "1h": ["5d", "1mo", "3mo"],
        "1d": ["6mo", "1y", "2y", "5y"],
    }
    default_period_index = 1 if len(period_options[interval]) > 1 else 0
    period = c2.selectbox("Histórico", period_options[interval], index=default_period_index)
    max_lag = c3.slider("Lead/Lag máximo (velas)", 1, 3, 1)

    st.markdown("**Universo cerrado de 8 pares:**")
    st.code(" · ".join(FIXED_PAIRS), language=None)
    st.caption("Los demás pares quedan fuera de discovery, validación, ranking y señal final.")

    with st.expander("Ajustes de investigación avanzada", expanded=False):
        a1, a2 = st.columns(2)
        top_edges = a1.slider("Hipótesis a mostrar", 10, 60, 30, step=5)
        probability_threshold = a2.select_slider(
            "Probabilidad mínima para señal",
            options=[0.70, 0.75, 0.80, 0.85, 0.90],
            value=0.70,
            format_func=lambda x: f"{x:.0%}",
        )
        a3, a4 = st.columns(2)
        permutations_n = a3.slider("Permutaciones finalistas", 99, 999, 399, step=100)
        bootstrap_n = a4.slider("Bootstrap finalistas", 99, 999, 399, step=100)
        a5, a6 = st.columns(2)
        fdr_alpha = a5.select_slider(
            "FDR máximo (q)", options=[0.01, 0.025, 0.05, 0.075, 0.10], value=0.05
        )
        trigger_z_min = a6.slider("Impulso mínimo de activación (z)", 0.0, 2.0, 0.50, step=0.10)


@st.cache_data(ttl=55, show_spinner=False)
def get_data(period: str, interval: str):
    return download_close_matrix(FIXED_PAIRS, period=period, interval=interval)


@st.cache_resource
def get_memory() -> EvidenceMemoryEngine:
    return EvidenceMemoryEngine()


def analyze(close: pd.DataFrame):
    latent = LatentCurrencyEngine().fit(close)
    regime = RegimeEngine().fit(close)

    diagnostics = StructuralDiagnostics()
    divergences = diagnostics.divergence_table(latent.residual_z, threshold=1.25)
    triangles = diagnostics.triangular_residuals(close)
    stress = NetworkStressEngine().fit(triangles)

    graph_warning = None
    memory = get_memory()
    try:
        graph = CausalGraphEngine(
            max_lag=max_lag,
            min_obs=80 if interval != "1d" else 60,
            permutations_n=permutations_n,
            bootstrap_n=bootstrap_n,
            walkforward_folds=3,
            fdr_alpha=fdr_alpha,
            probability_threshold=probability_threshold,
            trigger_z_min=trigger_z_min,
            memory=memory,
        ).build(
            close,
            latent_strength=latent.strength,
            regime=regime,
            top_n=top_edges,
            interval=interval,
            allowed_pairs=tuple(FIXED_PAIRS),
        )
    except Exception as exc:
        raise
        graph = pd.DataFrame(columns=CausalGraphEngine.COLUMNS)
        graph_warning = f"Motor multietapa desactivado por seguridad: {type(exc).__name__}: {exc}"

    return latent, regime, graph, divergences, triangles, stress, graph_warning


if st.button("⚡ Construir estado del mercado", use_container_width=True, type="primary"):
    try:
        with st.spinner("Sincronizando 8 pares, ejecutando embudo estadístico y consultando memoria..."):
            close = get_data(period, interval)
            missing = [p for p in FIXED_PAIRS if p not in close.columns]
            if missing:
                raise RuntimeError(f"Faltan pares del universo cerrado: {', '.join(missing)}")
            result = analyze(close)
            st.session_state["nexus_v030_result"] = (close,) + result
    except Exception as exc:
        st.exception(exc)

if "nexus_v030_result" not in st.session_state:
    st.info("Pulsa “Construir estado del mercado”. El motor solo emitirá BUY/SELL si supera todos los filtros y alcanza el umbral de probabilidad.")
    st.stop()

close, latent, regime, graph, divergences, triangles, stress, graph_warning = st.session_state["nexus_v030_result"]
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
signals_n = int(graph_diag.get("signals", int((graph.get("decision", pd.Series(dtype=str)).isin(["BUY", "SELL"])).sum()) if not graph.empty else 0))
best_prob = graph_diag.get("best_probability", np.nan)

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(f'<div class="card"><div class="metric-label">Régimen dominante</div><div class="big">{regime_name}</div><small>posterior {regime_conf:.1%}</small></div>', unsafe_allow_html=True)
with m2:
    rtxt = f"{regime_reliability:.1%}" if np.isfinite(regime_reliability) else "N/D"
    st.markdown(f'<div class="card"><div class="metric-label">Fiabilidad régimen</div><div class="big">{rtxt}</div><small>autoconsistencia</small></div>', unsafe_allow_html=True)
with m3:
    pctl = stress.stress_percentile
    ptxt = f"{pctl:.0f}º" if np.isfinite(pctl) else "N/D"
    stress_txt = f"{stress.current_stress:.2f}" if np.isfinite(stress.current_stress) else "N/D"
    st.markdown(f'<div class="card"><div class="metric-label">Network stress</div><div class="big">{stress_txt}</div><small>percentil {ptxt}</small></div>', unsafe_allow_html=True)
with m4:
    ctxt = f"{stress.coherence:.1%}" if np.isfinite(stress.coherence) else "N/D"
    st.markdown(f'<div class="card"><div class="metric-label">Supervivientes</div><div class="big">{survivors}</div><small>tras todos los filtros</small></div>', unsafe_allow_html=True)
with m5:
    btxt = f"{float(best_prob):.1%}" if np.isfinite(best_prob) else "N/D"
    st.markdown(f'<div class="card"><div class="metric-label">Señales activas</div><div class="big">{signals_n}</div><small>mejor prob. {btxt}</small></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FINAL SIGNAL PANEL
# -----------------------------------------------------------------------------
st.subheader("Señal final NEXUS")
if graph.empty or "decision" not in graph.columns:
    st.markdown('<div class="signal-none"><b>SIN SEÑAL</b><br><small>No hay hipótesis suficientes.</small></div>', unsafe_allow_html=True)
else:
    active = graph.loc[graph["decision"].isin(["BUY", "SELL"])].copy()
    active = active.sort_values(["predicted_probability", "evidence_score"], ascending=False)
    if active.empty:
        best_seen = pd.to_numeric(graph.get("predicted_probability"), errors="coerce").max()
        best_seen_txt = f"{best_seen:.1%}" if np.isfinite(best_seen) else "N/D"
        st.markdown(
            f'<div class="signal-none"><b>SIN SEÑAL</b><br><small>Ninguna hipótesis supera simultáneamente los filtros duros, el disparador y el umbral {probability_threshold:.0%}. Mejor probabilidad observada: {best_seen_txt}.</small></div>',
            unsafe_allow_html=True,
        )
    else:
        best = active.iloc[0]
        decision = str(best["decision"])
        css = "signal-buy" if decision == "BUY" else "signal-sell"
        horizon_minutes = None
        if interval.endswith("m"):
            try:
                horizon_minutes = int(interval[:-1]) * int(best["horizon_bars"])
            except Exception:
                horizon_minutes = None
        horizon_text = f"{horizon_minutes} min" if horizon_minutes is not None else f"{int(best['horizon_bars'])} vela(s)"
        st.markdown(
            f'<div class="{css}"><b style="font-size:1.7rem">{best["target"]} — {decision}</b><br>'
            f'<span style="font-size:1.35rem">Probabilidad estimada: {float(best["predicted_probability"]):.1%}</span><br>'
            f'<small>Fuente: {best["source"]} · lag {int(best["lag"])} · horizonte {horizon_text} · evidence {float(best["evidence_score"]):.3f} · FDR q {float(best["fdr_q"]):.4f}</small></div>',
            unsafe_allow_html=True,
        )
        if len(active) > 1:
            st.caption(f"Hay {len(active)} señales que superan el umbral. Se muestra primero la de mayor probabilidad calibrada.")

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
    u = unc.get(ccy, np.nan)
    with cols[i % 4]:
        st.markdown(
            f'<div class="card"><b>{ccy}</b><br><span style="font-size:2rem">{s:+.2f}σ</span><br>'
            f'<small>vel {v:+.2f}σ · acc {a:+.2f}σ · unc {u:.5f}</small></div>',
            unsafe_allow_html=True,
        )

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
t1, t2, t3, t4, t5, t6 = st.tabs([
    "Señales / Evidencia", "Régimen", "Divergencias", "Stress / Triángulos", "Memoria", "Diagnóstico"
])

with t1:
    st.caption("Una señal solo aparece cuando supera FDR, walk-forward, bootstrap, permutación, estabilidad, decay, evidence score, probabilidad mínima y disparador actual.")
    if graph.empty:
        st.info("No hay hipótesis candidatas suficientes para esta muestra.")
    else:
        only_survivors = st.toggle("Mostrar solo supervivientes", value=False)
        shown = graph.loc[graph["survives"]].copy() if only_survivors else graph.copy()
        display_cols = [
            "source", "target", "lag", "decision", "predicted_probability",
            "probability_lower_95", "oos_accuracy", "oos_trials", "memory_probability",
            "conditional_corr", "fdr_q", "bootstrap_low", "bootstrap_high",
            "walkforward_ic", "walkforward_sign_rate", "stability", "decay_state",
            "evidence_score", "trigger_z", "stage", "survives",
        ]
        display_cols = [c for c in display_cols if c in shown.columns]
        st.dataframe(shown[display_cols], use_container_width=True, hide_index=True)

        top = shown.head(12).copy()
        if not top.empty and "predicted_probability" in top:
            top["edge"] = top["source"] + " → " + top["target"] + " (L" + top["lag"].astype(str) + ")"
            fig_edges = go.Figure(go.Bar(x=top["predicted_probability"], y=top["edge"], orientation="h"))
            fig_edges.update_layout(template="plotly_dark", height=max(350, 32 * len(top)), margin=dict(l=20, r=20, t=35, b=20), title="Probabilidad direccional calibrada", yaxis={"autorange": "reversed"}, xaxis_tickformat=".0%", xaxis_range=[0, 1])
            st.plotly_chart(fig_edges, use_container_width=True)

with t2:
    st.caption("El posterior de régimen y la probabilidad de señal son métricas distintas. El posterior de régimen no se interpreta como probabilidad de trading.")
    if regime.probabilities.empty:
        st.info("No hay histórico suficiente para régimen.")
    else:
        fig_hist = go.Figure()
        for name in regime.probabilities.columns:
            fig_hist.add_trace(go.Scatter(x=regime.probabilities.index, y=regime.probabilities[name], mode="lines", stackgroup="one", name=name))
        fig_hist.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=35, b=20), yaxis_tickformat=".0%", title="Evolución de probabilidades de régimen")
        st.plotly_chart(fig_hist, use_container_width=True)
        if not regime.metrics.empty:
            st.dataframe(regime.metrics.iloc[-1].to_frame("valor"), use_container_width=True)

with t3:
    st.caption("Residual observado − reconstruido por el estado latente. Una anomalía relativa no es por sí sola una orden de trading.")
    st.dataframe(divergences, use_container_width=True, hide_index=True)

with t4:
    st.caption("Network Stress estandariza inconsistencias triangulares contra su propia historia.")
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
    st.caption("La memoria registra señales antes de conocer el resultado y las cierra cuando ya existe la vela futura correspondiente.")
    try:
        memory_df = get_memory().export_frame()
        if memory_df.empty:
            st.info("La memoria todavía no contiene señales cerradas o abiertas. Se irá llenando cuando aparezcan señales válidas.")
        else:
            closed = memory_df.loc[memory_df["status"] == "CLOSED"].copy()
            mm1, mm2, mm3 = st.columns(3)
            mm1.metric("Experimentos guardados", len(memory_df))
            mm2.metric("Cerrados", len(closed))
            hit_rate = closed["hit"].mean() if len(closed) else np.nan
            mm3.metric("Acierto memoria", f"{hit_rate:.1%}" if np.isfinite(hit_rate) else "N/D")
            st.dataframe(memory_df.tail(200).iloc[::-1], use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"No se pudo leer la memoria: {type(exc).__name__}: {exc}")

with t6:
    st.markdown("**Embudo de evidencia v0.3.0:**")
    st.markdown(
        "1. Universo fijo de 8 pares y relaciones dirigidas.  \n"
        "2. Discovery en el 70% inicial con filtros baratos.  \n"
        "3. Validación OOS en el 30% final sin reoptimizar el lag.  \n"
        "4. FDR Benjamini–Hochberg sobre la familia OOS.  \n"
        "5. Bootstrap y permutación solo para finalistas.  \n"
        "6. Estabilidad, decay y `evidence_score`.  \n"
        "7. Calibración de probabilidad direccional con muestra OOS + memoria.  \n"
        "8. Disparador actual y umbral mínimo antes de BUY/SELL.  \n"
        "9. Registro de la señal antes de conocer el resultado."
    )
    d = graph_diag
    if d:
        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("Pruebas discovery", d.get("stage0_universe", 0))
        dc2.metric("Pasan pre-filtro", d.get("stage1_discovery", 0))
        dc3.metric("Validadas OOS", d.get("stage2_oos", 0))
        dc4.metric("Pasan FDR", d.get("stage3_fdr", 0))
        de1, de2, de3, de4 = st.columns(4)
        de1.metric("Finalistas pesados", d.get("stage4_heavy", 0))
        de2.metric("Supervivientes", d.get("survivors", 0))
        de3.metric("Señales", d.get("signals", 0))
        bp = d.get("best_probability", np.nan)
        de4.metric("Mejor probabilidad", f"{bp:.1%}" if np.isfinite(bp) else "N/D")

        seq = d.get("sequential", {})
        if seq:
            st.markdown("**Embudo secuencial:**")
            st.caption("FDR → walk-forward → bootstrap → permutación → estabilidad → decay → score → probabilidad → trigger")
            st.json(seq)

            st.markdown("### 🔎 Diagnóstico de candidatos")
            st.json(graph_diag)

    st.warning("La probabilidad mostrada es una estimación empírica calibrada con los datos disponibles, no una garantía. Antes de dinero real siguen faltando costes, spread, slippage y validación independiente adicional.")

st.divider()
st.caption(f"Filas sincronizadas: {len(close):,} · Pares: {len(close.columns)} · Último dato: {close.index[-1]}")
st.warning("NEXUS v0.3.0 es un sistema de investigación estadística. BUY/SELL indica una hipótesis que superó los filtros configurados; no garantiza beneficio ni ejecuta operaciones.")
