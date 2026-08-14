from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import benchmark as b


LOCKED_VARIANT = "pre_f100_b30_m4"
HOLDOUT_SEEDS = list(range(840000, 840160))
IDENTITY_SEEDS = [839990, 839991, 839992, 839993]


def worker(args: tuple[str, str, str, str, int]) -> b.SeedResult:
    variant_name, variant_path_raw, target_path_raw, split, seed = args
    variant_path = Path(variant_path_raw)
    target_path = Path(target_path_raw)
    tag = f"{variant_name}_{split}_{seed}_{os.getpid()}"
    challenger0 = b.load_agent(variant_path, tag + "_c0")
    challenger1 = b.load_agent(variant_path, tag + "_c1")
    target0 = b.load_agent(target_path, tag + "_t0")
    target1 = b.load_agent(target_path, tag + "_t1")
    started = time.perf_counter()
    c0, t1 = b.run_game(challenger0, target1, seed)
    t0, c1 = b.run_game(target0, challenger1, seed)
    m0 = c0 - t1
    m1 = c1 - t0
    return b.SeedResult(
        variant=variant_name,
        split=split,
        seed=seed,
        challenger_seat0=c0,
        target_seat1=t1,
        margin_seat0=m0,
        target_seat0=t0,
        challenger_seat1=c1,
        margin_seat1=m1,
        paired_margin=(m0 + m1) / 2.0,
        elapsed_seconds=time.perf_counter() - started,
    )


def evaluate(name: str, variant_path: Path, target_path: Path, split: str, seeds: list[int]) -> list[b.SeedResult]:
    tasks = [(name, str(variant_path), str(target_path), split, seed) for seed in seeds]
    rows: list[b.SeedResult] = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(worker, task): task[-1] for task in tasks}
        for completed, future in enumerate(as_completed(futures), 1):
            seed = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"{name}/{split} failed at seed {seed}: {exc}") from exc
            if completed == 1 or completed % 8 == 0 or completed == len(tasks):
                mean = sum(row.paired_margin for row in rows) / len(rows)
                wins = sum(row.paired_margin > 0 for row in rows)
                losses = sum(row.paired_margin < 0 for row in rows)
                print(
                    f"[{split}] {completed}/{len(tasks)} blocks, mean={mean:.2f}, W-L={wins}-{losses}",
                    flush=True,
                )
    return sorted(rows, key=lambda row: row.seed)


def deterministic_tar(agent_path: Path, output_path: Path) -> str:
    import gzip
    import io

    payload = agent_path.read_bytes()
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
        info = tarfile.TarInfo("main.py")
        info.size = len(payload)
        info.mtime = 0
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(payload))
    with output_path.open("wb") as handle:
        with gzip.GzipFile(filename="", fileobj=handle, mode="wb", mtime=0) as gz:
            gz.write(raw.getvalue())
    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def main() -> None:
    if b.OUT.exists():
        shutil.rmtree(b.OUT)
    target_payload = b.fetch_exact_target()
    paths = b.build_agents(target_payload)

    identity = evaluate(
        "target_exact",
        paths["target_exact"],
        paths["target_exact"],
        "identity_control",
        IDENTITY_SEEDS,
    )
    if any(row.paired_margin != 0 for row in identity):
        raise RuntimeError("Identity control failed")
    b.write_rows(identity, b.OUT / "identity_control.csv")

    # This policy was frozen after the two-seed pilot. No model selection,
    # parameter changes, or early stopping uses the following holdout.
    holdout = evaluate(
        LOCKED_VARIANT,
        paths[LOCKED_VARIANT],
        paths["target_exact"],
        "locked_confirmation",
        HOLDOUT_SEEDS,
    )
    b.write_rows(holdout, b.OUT / "holdout_results.csv")
    summary = b.summarize(holdout)

    main_path = b.OUT / "main.py"
    shutil.copy2(paths[LOCKED_VARIANT], main_path)
    main_sha = hashlib.sha256(main_path.read_bytes()).hexdigest()
    submission_sha = deterministic_tar(main_path, b.OUT / "submission.tar.gz")

    with (b.OUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(summary))
        writer.writeheader()
        writer.writerow(summary)

    report = {
        "experiment": "pre-registered locked confirmation",
        "environment_commit": b.ENV_COMMIT,
        "target_notebook": b.TARGET_NOTEBOOK_URL,
        "target_payload_sha256": b.TARGET_MAIN_SHA256,
        "locked_variant": LOCKED_VARIANT,
        "policy_delta": {
            "_PREEMPT_ENABLED": True,
            "_PREEMPT_FRACTION": 1.0,
            "_PREEMPT_MAX_BATCH": 30,
            "all_other_target_code": "unchanged except an inert benchmark marker comment",
        },
        "protocol": {
            "pilot_seed_blocks_used_before_lock": 2,
            "identity_control_seed_blocks": len(IDENTITY_SEEDS),
            "locked_holdout_seed_range": [HOLDOUT_SEEDS[0], HOLDOUT_SEEDS[-1]],
            "locked_holdout_seed_blocks": len(HOLDOUT_SEEDS),
            "locked_holdout_games": 2 * len(HOLDOUT_SEEDS),
            "primary_unit": "one seed block averaged over both seat assignments",
            "primary_success_rule": "BCa 95% CI lower bound > 0 and one-sided paired t p < 0.05",
            "post_holdout_model_changes": 0,
        },
        "identity_control": b.summarize(identity),
        "locked_holdout": summary,
        "main_sha256": main_sha,
        "submission_sha256": submission_sha,
    }
    (b.OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

    if not bool(summary["significant_positive_at_5pct"]):
        raise SystemExit("Locked confirmation did not pass the predeclared significance rule")


if __name__ == "__main__":
    main()
