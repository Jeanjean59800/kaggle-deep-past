from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import stats

import benchmark as b
import benchmark_e300 as e

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out_league"
AGENTS = OUT / "agents"
OPPONENTS = {
    "mapleleaf42": (
        "https://raw.githubusercontent.com/5thDimension-Sean/kaggriculture/"
        "87976000a86ccc41841a59ee7c2a3524b87bd4d9/main.py"
    ),
    "v23_sparse": (
        "https://raw.githubusercontent.com/Chirantan02/Kaggriculture/"
        "16d2a2777cf2a7e5172e4f6d8272f6143a695243/main_v23.py"
    ),
    "amerob_planner": (
        "https://raw.githubusercontent.com/amerob/kaggriculture/"
        "3acd8d617971cddab133d368579392e1a575b03a/main.py"
    ),
}
POLICIES = ("target_exact", "pre_seed", "pre_seed_end696")
SEEDS = list(range(940000, 940016))


def fetch_opponents() -> dict[str, Path]:
    paths = {}
    for name, url in OPPONENTS.items():
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        path = AGENTS / f"opponent_{name}.py"
        path.write_bytes(payload)
        compile(payload.decode("utf-8"), str(path), "exec")
        paths[name] = path
        print(f"Fetched {name}: sha256={hashlib.sha256(payload).hexdigest()}", flush=True)
    return paths


def load_agent(path: Path, tag: str):
    module_name = f"league_{tag}_{os.getpid()}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attr in ("agent", "kaggriculture_agent", "_kaggle_submission_entrypoint"):
        fn = getattr(module, attr, None)
        if callable(fn):
            return fn
    raise RuntimeError(f"No supported agent callable in {path}")


def worker(args: tuple[str, str, str, str, int]) -> dict[str, object]:
    policy, policy_raw, opponent, opponent_raw, seed = args
    policy_path = Path(policy_raw)
    opponent_path = Path(opponent_raw)
    tag = f"{policy}_{opponent}_{seed}"
    p0 = load_agent(policy_path, tag + "_p0")
    p1 = load_agent(policy_path, tag + "_p1")
    o0 = load_agent(opponent_path, tag + "_o0")
    o1 = load_agent(opponent_path, tag + "_o1")
    started = time.perf_counter()
    policy_seat0, opponent_seat1 = b.run_game(p0, o1, seed)
    opponent_seat0, policy_seat1 = b.run_game(o0, p1, seed)
    return {
        "policy": policy,
        "opponent": opponent,
        "seed": seed,
        "policy_seat0": policy_seat0,
        "opponent_seat1": opponent_seat1,
        "opponent_seat0": opponent_seat0,
        "policy_seat1": policy_seat1,
        "policy_mean_reward": (policy_seat0 + policy_seat1) / 2.0,
        "opponent_mean_reward": (opponent_seat0 + opponent_seat1) / 2.0,
        "mean_margin": ((policy_seat0 - opponent_seat1) + (policy_seat1 - opponent_seat0)) / 2.0,
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarize_delta(values: list[float]) -> dict[str, object]:
    x = np.asarray(values, dtype=float)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    ci_low, ci_high = b.bootstrap_mean_ci(x)
    test = stats.ttest_1samp(x, 0.0, alternative="greater") if len(x) > 1 else None
    nonzero = x[x != 0]
    sign = stats.binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5, alternative="greater") if len(nonzero) else None
    return {
        "n_blocks": len(x),
        "mean": mean,
        "median": float(np.median(x)),
        "sd": sd,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "t_one_sided_p": float(test.pvalue) if test else float("nan"),
        "wins": int(np.sum(x > 0)),
        "ties": int(np.sum(x == 0)),
        "losses": int(np.sum(x < 0)),
        "sign_one_sided_p": float(sign.pvalue) if sign else float("nan"),
        "strict_positive": bool(ci_low > 0 and test is not None and float(test.pvalue) < 0.05),
    }


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    AGENTS.mkdir(parents=True)
    e.AGENTS = AGENTS
    exact_payload = b.fetch_exact_target()
    policy_paths = e.build_agents(exact_payload)
    opponent_paths = fetch_opponents()

    tasks = [
        (policy, str(policy_paths[policy]), opponent, str(opponent_paths[opponent]), seed)
        for policy in POLICIES
        for opponent in OPPONENTS
        for seed in SEEDS
    ]
    rows = []
    workers = max(1, min(int(os.environ.get("BENCH_WORKERS", "4")), len(tasks)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"League task failed {task}: {exc}") from exc
            if done == 1 or done % 8 == 0 or done == len(tasks):
                print(f"league {done}/{len(tasks)}", flush=True)
    rows.sort(key=lambda row: (str(row["opponent"]), int(row["seed"]), str(row["policy"])))

    fields = list(rows[0].keys())
    with (OUT / "league_raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    index = {(str(r["policy"]), str(r["opponent"]), int(r["seed"])): r for r in rows}
    report: dict[str, object] = {
        "environment_commit": b.ENV_COMMIT,
        "target_sha256": b.TARGET_MAIN_SHA256,
        "seed_range": [SEEDS[0], SEEDS[-1]],
        "opponents": OPPONENTS,
        "policies": list(POLICIES),
        "comparisons": {},
    }
    summary_rows = []
    for policy in POLICIES[1:]:
        aggregate_reward_delta = []
        aggregate_margin_delta = []
        policy_report = {}
        for opponent in OPPONENTS:
            reward_delta = []
            margin_delta = []
            own_margins = []
            for seed in SEEDS:
                candidate = index[(policy, opponent, seed)]
                exact = index[("target_exact", opponent, seed)]
                reward_delta.append(float(candidate["policy_mean_reward"]) - float(exact["policy_mean_reward"]))
                margin_delta.append(float(candidate["mean_margin"]) - float(exact["mean_margin"]))
                own_margins.append(float(candidate["mean_margin"]))
            aggregate_reward_delta.extend(reward_delta)
            aggregate_margin_delta.extend(margin_delta)
            policy_report[opponent] = {
                "reward_delta_vs_exact": summarize_delta(reward_delta),
                "margin_delta_vs_exact": summarize_delta(margin_delta),
                "candidate_mean_margin": statistics.fmean(own_margins),
                "candidate_seed_win_rate": sum(value > 0 for value in own_margins) / len(own_margins),
            }
            summary_rows.append({
                "policy": policy,
                "opponent": opponent,
                "metric": "reward_delta_vs_exact",
                **summarize_delta(reward_delta),
            })
            summary_rows.append({
                "policy": policy,
                "opponent": opponent,
                "metric": "margin_delta_vs_exact",
                **summarize_delta(margin_delta),
            })
        policy_report["ALL_OPPONENT_BLOCKS"] = {
            "reward_delta_vs_exact": summarize_delta(aggregate_reward_delta),
            "margin_delta_vs_exact": summarize_delta(aggregate_margin_delta),
        }
        report["comparisons"][policy] = policy_report

    fields = sorted({key for row in summary_rows for key in row})
    with (OUT / "league_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    (OUT / "league_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
