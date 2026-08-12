from __future__ import annotations

# NEXUS FX CORE v0.2.2 — mobile-safe launcher.
# Root Streamlit entry point. Keeps nexus_fx/app.py UI, removes its old
# inline RegimeEngine/CausalGraphEngine classes, and imports v0.2.2 from
# nexus_fx/intelligence.py. Also expands the diagnostics panel.

import ast
from pathlib import Path

UI_FILE = Path(__file__).resolve().parent / "nexus_fx" / "app.py"

source = UI_FILE.read_text(encoding="utf-8")

source = source.replace("STAT-INLINE-0.2.1.1", "STAT-MODULE-0.2.2")
source = source.replace("0.2.1.1", "0.2.2")
source = source.replace("v0.2.1", "v0.2.2")

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

engine_import = ast.ImportFrom(
    module="nexus_fx.intelligence",
    names=[
        ast.alias(name="RegimeEngine", asname=None),
        ast.alias(name="CausalGraphEngine", asname=None),
    ],
    level=0,
)

insert_at = 0
for i, node in enumerate(tree.body):
    if isinstance(node, ast.ImportFrom) and node.module == "__future__":
        insert_at = i + 1

tree.body.insert(insert_at, engine_import)
ast.fix_missing_locations(tree)

compiled = compile(tree, str(UI_FILE), "exec")
exec(compiled, globals(), globals())
