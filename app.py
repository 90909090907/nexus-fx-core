from __future__ import annotations

# NEXUS FX CORE v0.2.2.2 — graph compatibility fix.
# Keeps the existing nexus_fx/app.py UI and its inline classes/API intact.
# Only the statistical graph methods are patched from nexus_fx/intelligence.py.

import ast
from pathlib import Path

UI_FILE = Path(__file__).resolve().parent / "nexus_fx" / "app.py"
source = UI_FILE.read_text(encoding="utf-8")

source = source.replace("STAT-INLINE-0.2.1.1", "STAT-PATCHED-0.2.2.2")
source = source.replace("0.2.1.1", "0.2.2.2")
source = source.replace("v0.2.1", "v0.2.2.2")

tree = ast.parse(source, filename=str(UI_FILE))

patch_code = """
from nexus_fx.intelligence import CausalGraphEngine as _NEXUS_V022_GRAPH

def _nexus_v022_build(
    self,
    close,
    latent_strength=None,
    regime=None,
    top_n=30,
    *args,
    **kwargs,
):
    self.permutations_n = max(199, int(getattr(self, "permutations_n", 199)))
    self.bootstrap_n = max(199, int(getattr(self, "bootstrap_n", 199)))
    result = _NEXUS_V022_GRAPH.build(
        self,
        close,
        latent_strength=latent_strength,
        regime=regime,
        top_n=top_n,
    )
    return result

CausalGraphEngine._walkforward = _NEXUS_V022_GRAPH._walkforward
CausalGraphEngine.build = _nexus_v022_build
"""

patch_nodes = ast.parse(patch_code).body

insert_at = None
for i, node in enumerate(tree.body):
    if isinstance(node, ast.ClassDef) and node.name == "CausalGraphEngine":
        insert_at = i + 1
        break

if insert_at is None:
    raise RuntimeError("No se encontró CausalGraphEngine en nexus_fx/app.py")

for node in reversed(patch_nodes):
    tree.body.insert(insert_at, node)

ast.fix_missing_locations(tree)
compiled = compile(tree, str(UI_FILE), "exec")
exec(compiled, globals(), globals())
