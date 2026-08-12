from __future__ import annotations

# NEXUS FX CORE v0.2.2.1 — compatibility fix.
# Root Streamlit entry point. Keeps nexus_fx/app.py UI, removes its old
# inline RegimeEngine/CausalGraphEngine classes, and imports v0.2.2 from
# nexus_fx/intelligence.py. Also expands the diagnostics panel.

import ast
from pathlib import Path

UI_FILE = Path(__file__).resolve().parent / "nexus_fx" / "app.py"

source = UI_FILE.read_text(encoding="utf-8")

source = source.replace("STAT-INLINE-0.2.1.1", "STAT-MODULE-0.2.2.1")
source = source.replace("0.2.1.1", "0.2.2.1")
source = source.replace("v0.2.1", "v0.2.2.1")

source = source.replace(
    "regime.current_reliability",
    'getattr(regime, "current_reliability", np.nan)',
)
source = source.replace(
    "regime.current_entropy",
    'getattr(regime, "current_entropy", np.nan)',
)


needle = '        dc6.metric("Edge Survival Rate", f"{d.get(\'survival_rate\', 0.0):.1%}")'
extra = """        dc6.metric("Edge Survival Rate", f"{d.get('survival_rate', 0.0):.1%}")

        pmin = d.get("min_walkforward_p", np.nan)
        qmin = d.get("min_fdr_q", np.nan)
        dx1, dx2 = st.columns(2)
        dx1.metric("Mejor p OOS", f"{pmin:.5f}" if np.isfinite(pmin) else "N/D")
        dx2.metric("Mejor FDR q", f"{qmin:.5f}" if np.isfinite(qmin) else "N/D")

        seq = d.get("sequential", {})
        if seq:
            st.markdown("**Embudo secuencial real:**")
            st.caption("FDR → permutación → bootstrap → walk-forward → estabilidad → decay → score")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Tras FDR", seq.get("fdr", 0))
            s2.metric("+ Perm.", seq.get("permutation", 0))
            s3.metric("+ Bootstrap", seq.get("bootstrap", 0))
            s4.metric("+ Walk-forward", seq.get("walkforward", 0))
            s5, s6, s7 = st.columns(3)
            s5.metric("+ Estabilidad", seq.get("stability", 0))
            s6.metric("+ Decay", seq.get("decay", 0))
            s7.metric("+ Score", seq.get("evidence_score", 0))"""

if needle in source and "Mejor p OOS" not in source:
    source = source.replace(needle, extra, 1)

tree = ast.parse(source, filename=str(UI_FILE))

class RemoveOldEngines(ast.NodeTransformer):
    def visit_ClassDef(self, node):
        if node.name in {"RegimeEngine", "CausalGraphEngine"}:
            return None
        return self.generic_visit(node)

tree = RemoveOldEngines().visit(tree)
ast.fix_missing_locations(tree)

compat_code = """
from nexus_fx.intelligence import (
    RegimeEngine as _V022RegimeEngine,
    CausalGraphEngine as _V022CausalGraphEngine,
)

class RegimeEngine(_V022RegimeEngine):
    def fit(self, close, *args, **kwargs):
        return super().fit(close)

class CausalGraphEngine(_V022CausalGraphEngine):
    def __init__(self, *args, **kwargs):
        if "permutation_n" in kwargs and "permutations_n" not in kwargs:
            kwargs["permutations_n"] = kwargs.pop("permutation_n")
        super().__init__(*args, **kwargs)

    def build(
        self,
        close,
        latent_strength=None,
        regime=None,
        top_n=30,
        permutations_n=None,
        permutation_n=None,
        bootstrap_n=None,
        fdr_alpha=None,
        *args,
        **kwargs,
    ):
        old_perm = self.permutations_n
        old_boot = self.bootstrap_n
        old_fdr = self.fdr_alpha
        try:
            p = permutations_n if permutations_n is not None else permutation_n
            if p is not None:
                self.permutations_n = max(16, int(p))
            if bootstrap_n is not None:
                self.bootstrap_n = max(16, int(bootstrap_n))
            if fdr_alpha is not None:
                self.fdr_alpha = float(np.clip(fdr_alpha, 0.01, 0.25))
            return super().build(
                close,
                latent_strength=latent_strength,
                regime=regime,
                top_n=top_n,
            )
        finally:
            self.permutations_n = old_perm
            self.bootstrap_n = old_boot
            self.fdr_alpha = old_fdr
"""

compat_nodes = ast.parse(compat_code).body

insert_at = 0
for i, node in enumerate(tree.body):
    if isinstance(node, ast.ImportFrom) and node.module == "__future__":
        insert_at = i + 1

for node in reversed(compat_nodes):
    tree.body.insert(insert_at, node)

ast.fix_missing_locations(tree)

compiled = compile(tree, str(UI_FILE), "exec")
exec(compiled, globals(), globals())
