from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import statistics
import sys
import tarfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
AGENTS = OUT / "agents"
TARGET_NOTEBOOK_URL = (
    "https://raw.githubusercontent.com/mohanprasath-dev/kaggriculture/"
    "0047a7010be39d478e3c3348def2ebae584db736/kaggriculture-t.ipynb"
)
TARGET_MAIN_SHA256 = "d39dba50793d9777c990347443bf0c481c78adaea86055f6f6b0600dcfcd9f2e"
ENV_COMMIT = "2f22627ff7def9fc23d57bebcab184476153c678"
BRANCH = "scratch/kaggriculture-eval-20260815"

Agent = Callable[..., dict]


@dataclass(frozen=True)
class Variant:
    name: str
    replacements: tuple[tuple[str, str], ...] = ()


@dataclass
class SeedResult:
    variant: str
    split: str
    seed: int
    challenger_seat0: float
    target_seat1: float
    margin_seat0: float
    target_seat0: float
    challenger_seat1: float
    margin_seat1: float
    paired_margin: float
    elapsed_seconds: float


VARIANTS = [
    Variant("target_exact"),
    Variant(
        "pre_f025_b10_m4",
        (
            ("_PREEMPT_ENABLED = False", "_PREEMPT_ENABLED = True"),
            ("_PREEMPT_FRACTION = 2.0", "_PREEMPT_FRACTION = 0.25"),
            ("_PREEMPT_MAX_BATCH = 30", "_PREEMPT_MAX_BATCH = 10"),
        ),
    ),
    Variant(
        "pre_f050_b10_m4",
        (
            ("_PREEMPT_ENABLED = False", "_PREEMPT_ENABLED = True"),
            ("_PREEMPT_FRACTION = 2.0", "_PREEMPT_FRACTION = 0.50"),
            ("_PREEMPT_MAX_BATCH = 30", "_PREEMPT_MAX_BATCH = 10"),
        ),
    ),
    Variant(
        "pre_f075_b10_m4",
        (
            ("_PREEMPT_ENABLED = False", "_PREEMPT_ENABLED = True"),
            ("_PREEMPT_FRACTION = 2.0", "_PREEMPT_FRACTION = 0.75"),
            ("_PREEMPT_MAX_BATCH = 30", "_PREEMPT_MAX_BATCH = 10"),
        ),
    ),
    Variant(
        "pre_f100_b10_m4",
        (
            ("_PREEMPT_ENABLED = False", "_PREEMPT_ENABLED = True"),
            ("_PREEMPT_FRACTION = 2.0", "_PREEMPT_FRACTION = 1.0"),
            ("_PREEMPT_MAX_BATCH = 30", "_PREEMPT_MAX_BATCH = 10"),
        ),
    ),
    Variant(
        "pre_f050_b20_m4",
        (
            ("_PREEMPT_ENABLED = False", "_PREEMPT_ENABLED = True"),
            ("_PREEMPT_FRACTION = 2.0", "_PREEMPT_FRACTION = 0.50"),
            ("_PREEMPT_MAX_BATCH = 30", "_PREEMPT_MAX_BATCH = 20"),
        ),
    ),
    Variant(
        "pre_f100_b20_m4",
        (
            ("_PREEMPT_ENABLED = False", "_PREEMPT_ENABLED = True"),
            ("_PREEMPT_FRACTION = 2.0", "_PREEMPT_FRACTION = 1.0"),
            ("_PREEMPT_MAX_BATCH = 30", "_PREEMPT_MAX_BATCH = 20"),
        ),
    ),
    Variant(
        "pre_f100_b30_m4",
        (
            ("_PREEMPT_ENABLED = False", "_PREEMPT_ENABLED = True"),
            ("_PREEMPT_FRACTION = 2.0", "_PREEMPT_FRACTION = 1.0"),
        ),
    ),
    Variant(
        "pre_f100_b30_m1",
        (
            ("_PREEMPT_ENABLED = False", "_PREEMPT_ENABLED = True"),
            ("_PREEMPT_FRACTION = 2.0", "_PREEMPT_FRACTION = 1.0"),
            ("_PREEMPT_MIN_FUTURE_QUANTITY = 4", "_PREEMPT_MIN_FUTURE_QUANTITY = 1"),
        ),
    ),
    Variant(
        "pre_f100_b30_m8",
        (
            ("_PREEMPT_ENABLED = False", "_PREEMPT_ENABLED = True"),
            ("_PREEMPT_FRACTION = 2.0", "_PREEMPT_FRACTION = 1.0"),
            ("_PREEMPT_MIN_FUTURE_QUANTITY = 4", "_PREEMPT_MIN_FUTURE_QUANTITY = 8"),
        ),
    ),
    Variant(
        "pre_f100_b30_m4_s168",
        (
            ("_PREEMPT_ENABLED = False", "_PREEMPT_ENABLED = True"),
            ("_PREEMPT_FRACTION = 2.0", "_PREEMPT_FRACTION = 1.0"),
            ("_PREEMPT_START = 120", "_PREEMPT_START = 168"),
        ),
    ),
    Variant(
        "pre_f100_b30_m4_feed",
        (
            ("_PREEMPT_ENABLED = False", "_PREEMPT_ENABLED = True"),
            ("_PREEMPT_FRACTION = 2.0", "_PREEMPT_FRACTION = 1.0"),
            ("_V17_FEED_GUARD = False", "_V17_FEED_GUARD = True"),
        ),
    ),
    Variant(
        "pre_f100_b30_m4_no_counters",
        (
            ("_PREEMPT_ENABLED = False", "_PREEMPT_ENABLED = True"),
            ("_PREEMPT_FRACTION = 2.0", "_PREEMPT_FRACTION = 1.0"),
            ("_V17_R5_FRACTION = 1.0", "_V17_R5_FRACTION = 0.0"),
            ("_V17_MD_FRACTION = 2.0", "_V17_MD_FRACTION = 0.0"),
        ),
    ),
]


def fetch_exact_target() -> bytes:
    print(f"Downloading exact target notebook: {TARGET_NOTEBOOK_URL}", flush=True)
    with urllib.request.urlopen(TARGET_NOTEBOOK_URL, timeout=60) as response:
        notebook = json.loads(response.read().decode("utf-8"))
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and "AGENT_B64" in "".join(cell.get("source", []))
    )
    namespace: dict[str, object] = {}
    exec(compile(source, "target_notebook_payload.py", "exec"), namespace)
    payload = base64.b64decode(str(namespace["AGENT_B64"]))
    digest = hashlib.sha256(payload).hexdigest()
    if digest != TARGET_MAIN_SHA256:
        raise RuntimeError(f"Target SHA mismatch: expected {TARGET_MAIN_SHA256}, got {digest}")
    print(f"Exact target decoded and SHA-verified: {digest}", flush=True)
    return payload


def build_agents(payload: bytes) -> dict[str, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    AGENTS.mkdir(parents=True, exist_ok=True)
    target_text = payload.decode("utf-8")
    paths: dict[str, Path] = {}
    for variant in VARIANTS:
        text = target_text
        for old, new in variant.replacements:
            occurrences = text.count(old)
            if occurrences != 1:
                raise RuntimeError(
                    f"Patch {variant.name!r}: expected one occurrence of {old!r}, found {occurrences}"
                )
            text = text.replace(old, new, 1)
        marker = f'\n# benchmark_variant = {variant.name!r}\n'
        path = AGENTS / f"{variant.name}.py"
        path.write_text(text + marker, encoding="utf-8")
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
        paths[variant.name] = path
    return paths


def load_agent(path: Path, label: str) -> Agent:
    module_name = f"kaggriculture_{label}_{hashlib.sha1(str(path).encode()).hexdigest()[:10]}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    function = getattr(module, "agent", None)
    if not callable(function):
        raise RuntimeError(f"No callable agent in {path}")
    return function


def run_game(agent0: Agent, agent1: Agent, seed: int) -> tuple[float, float]:
    from kaggle_environments import make

    env = make(
        "kaggriculture",
        configuration={"seed": int(seed), "actTimeout": 3},
        debug=False,
    )
    env.run([agent0, agent1])
    rewards = [state.reward for state in env.state]
    statuses = [str(state.status) for state in env.state]
    if any(reward is None or not math.isfinite(float(reward)) for reward in rewards):
        raise RuntimeError(f"Non-finite reward for seed {seed}: rewards={rewards}, statuses={statuses}")
    if any(status not in {"DONE", "ACTIVE"} for status in statuses):
        raise RuntimeError(f"Bad status for seed {seed}: rewards={rewards}, statuses={statuses}")
    return float(rewards[0]), float(rewards[1])


def evaluate_variant(
    variant_name: str,
    variant_path: Path,
    target_path: Path,
    split: str,
    seeds: Iterable[int],
) -> list[SeedResult]:
    challenger = load_agent(variant_path, f"{variant_name}_challenger")
    target_a = load_agent(target_path, f"{variant_name}_target_a")
    target_b = load_agent(target_path, f"{variant_name}_target_b")
    challenger_b = load_agent(variant_path, f"{variant_name}_challenger_b")
    rows: list[SeedResult] = []
    seeds = list(seeds)
    for index, seed in enumerate(seeds, 1):
        started = time.perf_counter()
        c0, t1 = run_game(challenger, target_a, seed)
        t0, c1 = run_game(target_b, challenger_b, seed)
        margin0 = c0 - t1
        margin1 = c1 - t0
        rows.append(
            SeedResult(
                variant=variant_name,
                split=split,
                seed=int(seed),
                challenger_seat0=c0,
                target_seat1=t1,
                margin_seat0=margin0,
                target_seat0=t0,
                challenger_seat1=c1,
                margin_seat1=margin1,
                paired_margin=(margin0 + margin1) / 2.0,
                elapsed_seconds=time.perf_counter() - started,
            )
        )
        if index == 1 or index % 4 == 0 or index == len(seeds):
            current = [row.paired_margin for row in rows]
            print(
                f"[{split}] {variant_name}: {index}/{len(seeds)} seeds, "
                f"mean paired margin={statistics.fmean(current):.2f}",
                flush=True,
            )
    return rows


def bootstrap_mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    if len(values) < 2 or np.all(values == values[0]):
        value = float(values[0]) if len(values) else float("nan")
        return value, value
    rng = np.random.default_rng(20260815)
    result = stats.bootstrap(
        (values,),
        np.mean,
        confidence_level=confidence,
        method="BCa",
        n_resamples=30000,
        random_state=rng,
        vectorized=False,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def summarize(rows: list[SeedResult]) -> dict[str, object]:
    values = np.asarray([row.paired_margin for row in rows], dtype=float)
    wins = int(np.sum(values > 0))
    losses = int(np.sum(values < 0))
    ties = int(np.sum(values == 0))
    mean = float(np.mean(values))
    median = float(np.median(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    se = sd / math.sqrt(len(values)) if values.size else float("nan")
    ci_low, ci_high = bootstrap_mean_ci(values)
    ttest = stats.ttest_1samp(values, popmean=0.0, alternative="greater") if len(values) > 1 else None
    nonzero = values[values != 0]
    if len(nonzero) >= 2:
        try:
            wilcoxon = stats.wilcoxon(nonzero, alternative="greater", zero_method="wilcox")
            wilcoxon_p = float(wilcoxon.pvalue)
            wilcoxon_stat = float(wilcoxon.statistic)
        except ValueError:
            wilcoxon_p = float("nan")
            wilcoxon_stat = float("nan")
    else:
        wilcoxon_p = float("nan")
        wilcoxon_stat = float("nan")
    sign = stats.binomtest(wins, wins + losses, p=0.5, alternative="greater") if wins + losses else None
    challenger_rewards = np.asarray(
        [(row.challenger_seat0 + row.challenger_seat1) / 2.0 for row in rows], dtype=float
    )
    target_rewards = np.asarray(
        [(row.target_seat0 + row.target_seat1) / 2.0 for row in rows], dtype=float
    )
    return {
        "variant": rows[0].variant if rows else None,
        "split": rows[0].split if rows else None,
        "n_seed_blocks": len(rows),
        "n_games": 2 * len(rows),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "mean_paired_margin": mean,
        "median_paired_margin": median,
        "sd_paired_margin": sd,
        "se_paired_margin": se,
        "bootstrap_bca_95_ci_low": ci_low,
        "bootstrap_bca_95_ci_high": ci_high,
        "paired_t_statistic": float(ttest.statistic) if ttest else float("nan"),
        "paired_t_one_sided_p": float(ttest.pvalue) if ttest else float("nan"),
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_one_sided_p": wilcoxon_p,
        "exact_sign_one_sided_p": float(sign.pvalue) if sign else float("nan"),
        "cohen_dz": mean / sd if sd > 0 else float("inf") if mean > 0 else 0.0,
        "mean_challenger_reward": float(np.mean(challenger_rewards)),
        "mean_target_reward": float(np.mean(target_rewards)),
        "mean_elapsed_seconds_per_seed_block": float(np.mean([row.elapsed_seconds for row in rows])),
        "significant_positive_at_5pct": bool(ci_low > 0 and ttest is not None and float(ttest.pvalue) < 0.05),
    }


def write_rows(rows: list[SeedResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def package_submission(agent_path: Path, archive_path: Path) -> str:
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(agent_path, arcname="main.py")
    return hashlib.sha256(archive_path.read_bytes()).hexdigest()


def run_smoke(paths: dict[str, Path]) -> dict[str, object]:
    seeds = [2026081501, 2026081502]
    names = ["target_exact", "pre_f050_b10_m4", "pre_f100_b30_m4"]
    all_rows: list[SeedResult] = []
    summaries = []
    for name in names:
        rows = evaluate_variant(name, paths[name], paths["target_exact"], "smoke", seeds)
        all_rows.extend(rows)
        summaries.append(summarize(rows))
    write_rows(all_rows, OUT / "smoke_results.csv")
    report = {
        "mode": "smoke",
        "target_sha256": TARGET_MAIN_SHA256,
        "environment_commit": ENV_COMMIT,
        "summaries": summaries,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_full(paths: dict[str, Path]) -> dict[str, object]:
    # Development is intentionally small and only ranks hyperparameters.
    dev_seeds = list(range(410000, 410032))
    val_seeds = list(range(520000, 520048))
    holdout_seeds = list(range(730000, 730096))

    dev_rows: list[SeedResult] = []
    dev_summaries: list[dict[str, object]] = []
    for variant in VARIANTS:
        rows = evaluate_variant(variant.name, paths[variant.name], paths["target_exact"], "development", dev_seeds)
        dev_rows.extend(rows)
        dev_summaries.append(summarize(rows))
    write_rows(dev_rows, OUT / "development_results.csv")

    ranked = sorted(
        (summary for summary in dev_summaries if summary["variant"] != "target_exact"),
        key=lambda summary: (
            float(summary["mean_paired_margin"]),
            float(summary["bootstrap_bca_95_ci_low"]),
            -float(summary["sd_paired_margin"]),
        ),
        reverse=True,
    )
    finalist_names = [str(summary["variant"]) for summary in ranked[:3]]
    print(f"Development finalists: {finalist_names}", flush=True)

    val_rows: list[SeedResult] = []
    val_summaries: list[dict[str, object]] = []
    for name in finalist_names:
        rows = evaluate_variant(name, paths[name], paths["target_exact"], "validation", val_seeds)
        val_rows.extend(rows)
        val_summaries.append(summarize(rows))
    write_rows(val_rows, OUT / "validation_results.csv")

    selected = max(
        val_summaries,
        key=lambda summary: (
            float(summary["mean_paired_margin"]),
            float(summary["bootstrap_bca_95_ci_low"]),
            -float(summary["sd_paired_margin"]),
        ),
    )
    selected_name = str(selected["variant"])
    print(f"Locked finalist before holdout: {selected_name}", flush=True)

    holdout_rows = evaluate_variant(
        selected_name,
        paths[selected_name],
        paths["target_exact"],
        "locked_holdout",
        holdout_seeds,
    )
    write_rows(holdout_rows, OUT / "holdout_results.csv")
    holdout_summary = summarize(holdout_rows)

    final_agent = OUT / "main.py"
    shutil.copy2(paths[selected_name], final_agent)
    agent_sha = hashlib.sha256(final_agent.read_bytes()).hexdigest()
    archive_sha = package_submission(final_agent, OUT / "submission.tar.gz")

    report = {
        "mode": "full",
        "method": {
            "primary_unit": "seed block averaged over both seat assignments",
            "selection": "variants ranked on development; top 3 reranked on validation; one finalist locked before holdout",
            "primary_claim_rule": "BCa 95% CI lower bound > 0 and one-sided paired t p < 0.05 on locked holdout",
            "development_seed_range": [dev_seeds[0], dev_seeds[-1]],
            "validation_seed_range": [val_seeds[0], val_seeds[-1]],
            "holdout_seed_range": [holdout_seeds[0], holdout_seeds[-1]],
        },
        "target": {
            "notebook_url": TARGET_NOTEBOOK_URL,
            "main_sha256": TARGET_MAIN_SHA256,
        },
        "environment_commit": ENV_COMMIT,
        "development_summaries": dev_summaries,
        "finalists": finalist_names,
        "validation_summaries": val_summaries,
        "selected_variant": selected_name,
        "locked_holdout_summary": holdout_summary,
        "final_agent_sha256": agent_sha,
        "submission_sha256": archive_sha,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary_rows = dev_summaries + val_summaries + [holdout_summary]
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in summary_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()

    if OUT.exists():
        shutil.rmtree(OUT)
    payload = fetch_exact_target()
    paths = build_agents(payload)
    report = run_smoke(paths) if args.mode == "smoke" else run_full(paths)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
