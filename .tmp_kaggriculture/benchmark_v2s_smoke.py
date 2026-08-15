from __future__ import annotations

import json
import shutil
from pathlib import Path

import benchmark_v2s as bench


def main() -> None:
    smoke_out = bench.ROOT / "out_v2s_smoke"
    if smoke_out.exists():
        shutil.rmtree(smoke_out)
    smoke_out.mkdir(parents=True)
    bench.OUT = smoke_out
    bench.AGENTS = smoke_out / "agents"
    paths = bench.build_agents(bench.exact_target())
    seeds = [410001, 410002]
    summaries = []
    all_rows = []
    for variant in bench.VARIANTS:
        rows = bench.evaluate(variant.name, paths[variant.name], "fast_smoke", seeds)
        all_rows.extend(rows)
        summaries.append(bench.summarize(rows))
    bench.write_rows(all_rows, smoke_out / "results.csv")
    report = {"seeds": seeds, "summaries": summaries}
    (smoke_out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
