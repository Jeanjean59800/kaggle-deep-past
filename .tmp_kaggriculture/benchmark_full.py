from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import benchmark as b


EXTRA_VARIANTS = [
    b.Variant(
        "pre_f100_b30_m4_price050",
        (
            ("_PREEMPT_ENABLED = False", "_PREEMPT_ENABLED = True"),
            ("_PREEMPT_FRACTION = 2.0", "_PREEMPT_FRACTION = 1.0"),
            ("_PREEMPT_MIN_PRICE_RATIO = 0.0", "_PREEMPT_MIN_PRICE_RATIO = 0.50"),
        ),
    ),
    b.Variant(
        "pre_f100_b30_m4_price075",
        (
            ("_PREEMPT_ENABLED = False", "_PREEMPT_ENABLED = True"),
            ("_PREEMPT_FRACTION = 2.0", "_PREEMPT_FRACTION = 1.0"),
            ("_PREEMPT_MIN_PRICE_RATIO = 0.0", "_PREEMPT_MIN_PRICE_RATIO = 0.75"),
        ),
    ),
    b.Variant(
        "pre_f100_b30_m4_stop715",
        (
            ("_PREEMPT_ENABLED = False", "_PREEMPT_ENABLED = True"),
            ("_PREEMPT_FRACTION = 2.0", "_PREEMPT_FRACTION = 1.0"),
            ("_PREEMPT_STOP = 680", "_PREEMPT_STOP = 715"),
        ),
    ),
    b.Variant(
        "pre_f100_b30_m1_stop715",
        (
            ("_PREEMPT_ENABLED = False", "_PREEMPT_ENABLED = True"),
            ("_PREEMPT_FRACTION = 2.0", "_PREEMPT_FRACTION = 1.0"),
            ("_PREEMPT_MIN_FUTURE_QUANTITY = 4", "_PREEMPT_MIN_FUTURE_QUANTITY = 1"),
            ("_PREEMPT_STOP = 680", "_PREEMPT_STOP = 715"),
        ),
    ),
    b.Variant(
        "pre_f100_b30_m4_clone0",
        (
            ("_PREEMPT_ENABLED = False", "_PREEMPT_ENABLED = True"),
            ("_PREEMPT_FRACTION = 2.0", "_PREEMPT_FRACTION = 1.0"),
            ("_PREEMPT_MAX_CLONE_DISTANCE = 6", "_PREEMPT_MAX_CLONE_DISTANCE = 0"),
        ),
    ),
    b.Variant(
        "pre_f100_b30_m1_price050",
        (
            ("_PREEMPT_ENABLED = False", "_PREEMPT_ENABLED = True"),
            ("_PREEMPT_FRACTION = 2.0", "_PREEMPT_FRACTION = 1.0"),
            ("_PREEMPT_MIN_FUTURE_QUANTITY = 4", "_PREEMPT_MIN_FUTURE_QUANTITY = 1"),
            ("_PREEMPT_MIN_PRICE_RATIO = 0.0", "_PREEMPT_MIN_PRICE_RATIO = 0.50"),
        ),
    ),
]

# Keep target_exact as a negative-control identity check and add the expanded grid.
b.VARIANTS = b.VARIANTS + EXTRA_VARIANTS


def _paired_seed_worker(args: tuple[str, str, str, str, int]) -> b.SeedResult:
    variant_name, variant_path_raw, target_path_raw, split, seed = args
    variant_path = Path(variant_path_raw)
    target_path = Path(target_path_raw)
    unique = f"{variant_name}_{split}_{seed}_{os.getpid()}"
    challenger_a = b.load_agent(variant_path, unique + "_ca")
    challenger_b = b.load_agent(variant_path, unique + "_cb")
    target_a = b.load_agent(target_path, unique + "_ta")
    target_b = b.load_agent(target_path, unique + "_tb")
    started = time.perf_counter()
    c0, t1 = b.run_game(challenger_a, target_a, seed)
    t0, c1 = b.run_game(target_b, challenger_b, seed)
    margin0 = c0 - t1
    margin1 = c1 - t0
    return b.SeedResult(
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


def evaluate_parallel(
    variant_name: str,
    variant_path: Path,
    target_path: Path,
    split: str,
    seeds: list[int],
) -> list[b.SeedResult]:
    workers = max(1, min(int(os.environ.get("BENCH_WORKERS", "4")), len(seeds)))
    tasks = [
        (variant_name, str(variant_path), str(target_path), split, int(seed))
        for seed in seeds
    ]
    rows: list[b.SeedResult] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_paired_seed_worker, task): task[-1] for task in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            seed = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                raise RuntimeError(
                    f"{variant_name}/{split} failed on seed {seed}: {exc}"
                ) from exc
            if done == 1 or done % 4 == 0 or done == len(tasks):
                mean = sum(row.paired_margin for row in rows) / len(rows)
                print(
                    f"[{split}] {variant_name}: {done}/{len(tasks)} seed blocks; "
                    f"running mean={mean:.2f}",
                    flush=True,
                )
    rows.sort(key=lambda row: row.seed)
    return rows


def rank_key(summary: dict[str, object]) -> tuple[float, float, float, float]:
    return (
        float(summary["mean_paired_margin"]),
        float(summary["bootstrap_bca_95_ci_low"]),
        -float(summary["losses"]),
        -float(summary["sd_paired_margin"]),
    )


def package_submission(agent_path: Path, archive_path: Path) -> str:
    # Deterministic tar payload; gzip header itself is stable through mtime=0.
    import gzip
    import io

    data = agent_path.read_bytes()
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        info = tarfile.TarInfo("main.py")
        info.size = len(data)
        info.mtime = 0
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(data))
    with archive_path.open("wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as gz:
            gz.write(tar_buffer.getvalue())
    return hashlib.sha256(archive_path.read_bytes()).hexdigest()


def main() -> None:
    if b.OUT.exists():
        shutil.rmtree(b.OUT)
    payload = b.fetch_exact_target()
    paths = b.build_agents(payload)

    # Exact identity check: validates seat pairing and state reset before tuning.
    control_seeds = list(range(310000, 310004))
    control_rows = evaluate_parallel(
        "target_exact", paths["target_exact"], paths["target_exact"], "identity_control", control_seeds
    )
    control_summary = b.summarize(control_rows)
    if any(row.paired_margin != 0 for row in control_rows):
        raise RuntimeError(f"Identity control failed: {control_summary}")
    b.write_rows(control_rows, b.OUT / "identity_control.csv")

    # Stage 1: hyperparameter ranking only.
    development_seeds = list(range(410000, 410024))
    candidates = [variant for variant in b.VARIANTS if variant.name != "target_exact"]
    development_rows: list[b.SeedResult] = []
    development_summaries: list[dict[str, object]] = []
    for variant in candidates:
        rows = evaluate_parallel(
            variant.name,
            paths[variant.name],
            paths["target_exact"],
            "development",
            development_seeds,
        )
        development_rows.extend(rows)
        development_summaries.append(b.summarize(rows))
    b.write_rows(development_rows, b.OUT / "development_results.csv")
    development_ranked = sorted(development_summaries, key=rank_key, reverse=True)
    finalist_names = [str(summary["variant"]) for summary in development_ranked[:4]]
    print(f"Development finalists: {finalist_names}", flush=True)

    # Stage 2: independent validation chooses exactly one frozen policy.
    validation_seeds = list(range(520000, 520040))
    validation_rows: list[b.SeedResult] = []
    validation_summaries: list[dict[str, object]] = []
    for name in finalist_names:
        rows = evaluate_parallel(
            name,
            paths[name],
            paths["target_exact"],
            "validation",
            validation_seeds,
        )
        validation_rows.extend(rows)
        validation_summaries.append(b.summarize(rows))
    b.write_rows(validation_rows, b.OUT / "validation_results.csv")
    selected = max(validation_summaries, key=rank_key)
    selected_name = str(selected["variant"])
    print(f"Policy locked before holdout: {selected_name}", flush=True)

    # Stage 3: untouched holdout; no decisions are made after this starts.
    holdout_seeds = list(range(730000, 730128))
    holdout_rows = evaluate_parallel(
        selected_name,
        paths[selected_name],
        paths["target_exact"],
        "locked_holdout",
        holdout_seeds,
    )
    b.write_rows(holdout_rows, b.OUT / "holdout_results.csv")
    holdout_summary = b.summarize(holdout_rows)

    final_agent = b.OUT / "main.py"
    shutil.copy2(paths[selected_name], final_agent)
    agent_sha = hashlib.sha256(final_agent.read_bytes()).hexdigest()
    submission_sha = package_submission(final_agent, b.OUT / "submission.tar.gz")

    all_summaries = [control_summary] + development_summaries + validation_summaries + [holdout_summary]
    fields = sorted({key for row in all_summaries for key in row})
    with (b.OUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_summaries)

    report = {
        "mode": "parallel_full_locked_holdout",
        "environment": {
            "repository": "Kaggle/kaggle-environments",
            "commit": b.ENV_COMMIT,
            "episode_steps": 720,
        },
        "target": {
            "notebook_url": b.TARGET_NOTEBOOK_URL,
            "main_sha256": b.TARGET_MAIN_SHA256,
        },
        "protocol": {
            "primary_unit": "one random seed averaged over both seat assignments",
            "identity_control_seed_range": [control_seeds[0], control_seeds[-1]],
            "development_seed_range": [development_seeds[0], development_seeds[-1]],
            "validation_seed_range": [validation_seeds[0], validation_seeds[-1]],
            "locked_holdout_seed_range": [holdout_seeds[0], holdout_seeds[-1]],
            "candidate_count": len(candidates),
            "finalist_count": len(finalist_names),
            "selection_rule": "rank mean paired margin, then BCa lower bound, then fewer losses and lower variance",
            "primary_success_rule": "locked holdout BCa 95% CI lower bound > 0 and one-sided paired t p < 0.05",
            "multiplicity_control": "all model selection finished before the single locked holdout was evaluated",
        },
        "identity_control": control_summary,
        "development_summaries": development_summaries,
        "development_ranking": [str(row["variant"]) for row in development_ranked],
        "finalists": finalist_names,
        "validation_summaries": validation_summaries,
        "selected_variant": selected_name,
        "locked_holdout_summary": holdout_summary,
        "final_agent_sha256": agent_sha,
        "submission_sha256": submission_sha,
    }
    (b.OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

    if not bool(holdout_summary["significant_positive_at_5pct"]):
        raise SystemExit("Locked holdout did not meet the predeclared significance rule")


if __name__ == "__main__":
    main()
