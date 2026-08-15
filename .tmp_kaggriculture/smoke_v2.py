from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out_v2_smoke"

spec = importlib.util.spec_from_file_location("benchmark_v2", ROOT / "benchmark_v2.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot import benchmark_v2.py")
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
payload = b.exact_target_payload()
paths = b.build_agents(payload)
seeds = [990001, 990002]
rows = []
summaries = []
for variant in b.VARIANTS:
    result = b.evaluate_parallel(
        variant.name,
        paths[variant.name],
        "target_exact",
        paths["target_exact"],
        "smoke",
        seeds,
    )
    rows.extend(result)
    summaries.append(b.summarize(result))
b.write_rows(rows, OUT / "smoke_results.csv")
(OUT / "smoke_report.json").write_text(
    json.dumps({"seeds": seeds, "summaries": summaries}, indent=2, allow_nan=True),
    encoding="utf-8",
)
print(json.dumps(summaries, indent=2, allow_nan=True))
