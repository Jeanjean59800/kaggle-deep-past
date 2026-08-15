from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import statistics
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out_v2s"
AGENTS = OUT / "agents"
OVERLAY_PATH = ROOT / "v2s_overlay.py"
TARGET_NOTEBOOK_URL = (
    "https://raw.githubusercontent.com/mohanprasath-dev/kaggriculture/"
    "0047a7010be39d478e3c3348def2ebae584db736/kaggriculture-t.ipynb"
)
TARGET_SHA256 = "d39dba50793d9777c990347443bf0c481c78adaea86055f6f6b0600dcfcd9f2e"
ENV_COMMIT = "28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c"

Agent = Callable[..., dict]


@dataclass(frozen=True)
class Variant:
    name: str
    overlay: bool
    replacements: tuple[tuple[str, str], ...]
    description: str


@dataclass
class SeedResult:
    variant: str
    split: str
    seed: int
    challenger_seat0: float
    baseline_seat1: float
    margin_seat0: float
    baseline_seat0: float
    challenger_seat1: float
    margin_seat1: float
    paired_margin: float
    elapsed_seconds: float


PREEMPT = (
    ("_PREEMPT_ENABLED = False", "_PREEMPT_ENABLED = True"),
    ("_PREEMPT_FRACTION = 2.0", "_PREEMPT_FRACTION = 1.0"),
)
DISABLE_FLOOR = (("_V2S_ENABLE_FLOOR_HOLD = True", "_V2S_ENABLE_FLOOR_HOLD = False"),)
DISABLE_CASH = (("_V2S_ENABLE_CASH_RESCUE = True", "_V2S_ENABLE_CASH_RESCUE = False"),)
DISABLE_HIRE = (("_V2S_ENABLE_MISSED_HIRE = True", "_V2S_ENABLE_MISSED_HIRE = False"),)
DISABLE_TERMINAL = (("_V2S_ENABLE_TERMINAL_HARVEST = True", "_V2S_ENABLE_TERMINAL_HARVEST = False"),)
FEED_GUARD = (("_V17_FEED_GUARD = False", "_V17_FEED_GUARD = True"),)
SELECTOR2 = ((
    '"high" if "YARN_STORE" in shops and not dominated else "low"',
    '"high" if shops.count("YARN_STORE") >= 2 and not dominated else "low"',
),)

VARIANTS = [
    Variant("control_preemption", False, PREEMPT, "Previously validated preemption control."),
    Variant(
        "v2s_core",
        True,
        PREEMPT + DISABLE_FLOOR + DISABLE_CASH + DISABLE_HIRE + DISABLE_TERMINAL,
        "Official 1.32.7 curves, exact seed cap, impossible SELL clamp.",
    ),
    Variant(
        "v2s_recovery",
        True,
        PREEMPT + DISABLE_FLOOR + DISABLE_CASH + FEED_GUARD,
        "Core plus feed guard, missed-hire repair and terminal harvest.",
    ),
    Variant(
        "v2s_market",
        True,
        PREEMPT + DISABLE_HIRE + DISABLE_TERMINAL,
        "Core plus price-floor hold and cash rescue.",
    ),
    Variant(
        "v2s_full",
        True,
        PREEMPT + FEED_GUARD,
        "All surgical corrections with original route selector.",
    ),
    Variant(
        "v2s_selector2",
        True,
        PREEMPT + DISABLE_FLOOR + DISABLE_CASH + FEED_GUARD + SELECTOR2,
        "Recovery layer plus repeated-Yarn high-route threshold.",
    ),
]


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def exact_target() -> bytes:
    notebook = fetch_json(TARGET_NOTEBOOK_URL)
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and "AGENT_B64" in "".join(cell.get("source", []))
    )
    namespace: dict[str, object] = {}
    exec(compile(source, "target_payload.py", "exec"), namespace)
    payload = base64.b64decode(str(namespace["AGENT_B64"]))
    digest = hashlib.sha256(payload).hexdigest()
    if digest != TARGET_SHA256:
        raise RuntimeError(f"Target SHA mismatch: {digest}")
    return payload


def replace_exact(text: str, replacements: tuple[tuple[str, str], ...], label: str) -> str:
    for old, new in replacements:
        count = text.count(old)
        if count != 1:
            raise RuntimeError(f"{label}: expected one occurrence of {old!r}, found {count}")
        text = text.replace(old, new, 1)
    return text


def build_agents(payload: bytes) -> dict[str, Path]:
    AGENTS.mkdir(parents=True, exist_ok=True)
    base = payload.decode("utf-8")
    overlay = OVERLAY_PATH.read_text(encoding="utf-8")
    paths: dict[str, Path] = {}
    for variant in VARIANTS:
        text = base + ("\n" + overlay if variant.overlay else "")
        text = replace_exact(text, variant.replacements, variant.name)
        text += f"\n# benchmark_variant={variant.name!r}\n"
        path = AGENTS / f"{variant.name}.py"
        path.write_text(text, encoding="utf-8")
        compile(text, str(path), "exec")
        paths[variant.name] = path
    return paths


def load_agent(path: Path, label: str) -> Agent:
    token = hashlib.sha1(f"{path}:{label}:{os.getpid()}:{time.time_ns()}".encode()).hexdigest()[:16]
    name = f"agent_{token}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise RuntimeError(f"No agent in {path}")
    return agent


def play(path0: Path, path1: Path, seed: int, label: str) -> tuple[float, float]:
    from kaggle_environments import make

    env = make(
        "kaggriculture",
        configuration={"seed": int(seed), "actTimeout": 3, "runTimeout": 1200},
        debug=False,
    )
    env.run([load_agent(path0, label + "a"), load_agent(path1, label + "b")])
    rewards = [state.reward for state in env.state]
    statuses = [str(state.status) for state in env.state]
    if any(reward is None or not math.isfinite(float(reward)) for reward in rewards):
        raise RuntimeError(f"Bad reward seed={seed}: {rewards}, {statuses}")
    if any(status != "DONE" for status in statuses):
        raise RuntimeError(f"Incomplete seed={seed}: {rewards}, {statuses}")
    return float(rewards[0]), float(rewards[1])


def worker(task: tuple[str, str, str, int]) -> SeedResult:
    variant, challenger_raw, split, seed = task
    challenger = Path(challenger_raw)
    baseline = challenger.parent / "control_preemption.py"
    started = time.perf_counter()
    c0, b1 = play(challenger, baseline, seed, f"{variant}_{split}_{seed}_0")
    b0, c1 = play(baseline, challenger, seed, f"{variant}_{split}_{seed}_1")
    m0, m1 = c0 - b1, c1 - b0
    return SeedResult(
        variant, split, seed,
        c0, b1, m0, b0, c1, m1,
        (m0 + m1) / 2.0,
        time.perf_counter() - started,
    )


def evaluate(name: str, path: Path, split: str, seeds: list[int]) -> list[SeedResult]:
    workers = max(1, min(int(os.environ.get("BENCH_WORKERS", "4")), len(seeds)))
    tasks = [(name, str(path), split, int(seed)) for seed in seeds]
    rows: list[SeedResult] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, task): task[-1] for task in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            seed = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"{name}/{split} failed seed={seed}: {exc}") from exc
            if done == 1 or done % 4 == 0 or done == len(tasks):
                mean = statistics.fmean(row.paired_margin for row in rows)
                print(f"[{split}] {name}: {done}/{len(tasks)} mean={mean:.1f}", flush=True)
    return sorted(rows, key=lambda row: row.seed)


def ci(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2 or np.all(values == values[0]):
        value = float(values[0]) if len(values) else float("nan")
        return value, value
    result = stats.bootstrap(
        (values,), np.mean, confidence_level=0.95, method="BCa",
        n_resamples=20000, random_state=np.random.default_rng(20260815), vectorized=False,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def summarize(rows: list[SeedResult]) -> dict[str, object]:
    values = np.asarray([row.paired_margin for row in rows], dtype=float)
    low, high = ci(values)
    wins = int(np.sum(values > 0)); losses = int(np.sum(values < 0)); ties = int(np.sum(values == 0))
    if len(values) > 1 and float(np.std(values, ddof=1)) > 0:
        test = stats.ttest_1samp(values, 0.0, alternative="greater")
        t_p = float(test.pvalue)
    else:
        t_p = float("nan")
    return {
        "variant": rows[0].variant,
        "split": rows[0].split,
        "n_seed_blocks": len(rows),
        "n_games": 2 * len(rows),
        "wins": wins, "ties": ties, "losses": losses,
        "mean_margin": float(np.mean(values)),
        "median_margin": float(np.median(values)),
        "sd_margin": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "ci95_low": low, "ci95_high": high,
        "one_sided_t_p": t_p,
        "mean_challenger_reward": float(np.mean([(r.challenger_seat0 + r.challenger_seat1) / 2 for r in rows])),
        "mean_baseline_reward": float(np.mean([(r.baseline_seat0 + r.baseline_seat1) / 2 for r in rows])),
        "passes": bool(low > 0 and math.isfinite(t_p) and t_p < 0.05),
    }


def write_rows(rows: list[SeedResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader(); writer.writerows(asdict(row) for row in rows)


def package(main_path: Path, archive_path: Path) -> str:
    payload = main_path.read_bytes()
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        info = tarfile.TarInfo("main.py")
        info.size = len(payload); info.mtime = 0; info.mode = 0o644
        info.uid = info.gid = 0; info.uname = info.gname = ""
        archive.addfile(info, io.BytesIO(payload))
    with archive_path.open("wb") as handle:
        with gzip.GzipFile(filename="", fileobj=handle, mode="wb", mtime=0) as gz:
            gz.write(raw.getvalue())
    return hashlib.sha256(archive_path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    paths = build_agents(exact_target())

    control_rows = evaluate("control_preemption", paths["control_preemption"], "identity", [300001, 300002])
    if any(row.paired_margin != 0 for row in control_rows):
        raise RuntimeError("Control identity failed")

    smoke_seeds = [410001, 410002]
    smoke_rows: list[SeedResult] = []
    smoke_summaries = []
    survivors = []
    for variant in VARIANTS[1:]:
        rows = evaluate(variant.name, paths[variant.name], "smoke", smoke_seeds)
        smoke_rows.extend(rows)
        summary = summarize(rows); smoke_summaries.append(summary)
        if summary["mean_challenger_reward"] >= 50000 and summary["mean_margin"] > -5000:
            survivors.append(variant.name)
    write_rows(smoke_rows, OUT / "smoke_results.csv")
    print(f"Smoke survivors: {survivors}", flush=True)

    dev_rows: list[SeedResult] = []
    dev_summaries = []
    dev_seeds = list(range(520000, 520008))
    for name in survivors:
        rows = evaluate(name, paths[name], "development", dev_seeds)
        dev_rows.extend(rows); dev_summaries.append(summarize(rows))
    if dev_rows: write_rows(dev_rows, OUT / "development_results.csv")

    ranked = sorted(dev_summaries, key=lambda s: (s["ci95_low"], s["mean_margin"]), reverse=True)
    finalists = [str(s["variant"]) for s in ranked[:3] if float(s["mean_margin"]) > 0]

    val_rows: list[SeedResult] = []
    val_summaries = []
    val_seeds = list(range(630000, 630016))
    for name in finalists:
        rows = evaluate(name, paths[name], "validation", val_seeds)
        val_rows.extend(rows); val_summaries.append(summarize(rows))
    if val_rows: write_rows(val_rows, OUT / "validation_results.csv")

    eligible = [s for s in val_summaries if float(s["mean_margin"]) > 0 and float(s["ci95_low"]) >= -250]
    selected = max(eligible, key=lambda s: (s["ci95_low"], s["mean_margin"])) if eligible else None
    selected_name = str(selected["variant"]) if selected else "control_preemption"
    print(f"Locked selection: {selected_name}", flush=True)

    holdout_summary = None
    if selected_name != "control_preemption":
        holdout_seeds = list(range(740000, 740040))
        holdout_rows = evaluate(selected_name, paths[selected_name], "locked_holdout", holdout_seeds)
        write_rows(holdout_rows, OUT / "holdout_results.csv")
        holdout_summary = summarize(holdout_rows)
        if not bool(holdout_summary["passes"]):
            selected_name = "control_preemption"

    final_main = OUT / "main.py"
    shutil.copy2(paths[selected_name], final_main)
    submission_sha = package(final_main, OUT / "submission.tar.gz")
    report = {
        "environment_commit": ENV_COMMIT,
        "target_sha256": TARGET_SHA256,
        "baseline": "control_preemption",
        "smoke_summaries": smoke_summaries,
        "survivors": survivors,
        "development_summaries": dev_summaries,
        "finalists": finalists,
        "validation_summaries": val_summaries,
        "provisional_selected": str(selected["variant"]) if selected else None,
        "locked_holdout_summary": holdout_summary,
        "final_selected": selected_name,
        "v2_increment_validated": bool(selected_name != "control_preemption" and holdout_summary and holdout_summary["passes"]),
        "main_sha256": hashlib.sha256(final_main.read_bytes()).hexdigest(),
        "submission_sha256": submission_sha,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
