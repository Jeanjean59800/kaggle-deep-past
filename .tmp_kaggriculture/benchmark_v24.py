from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import statistics
import tarfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import benchmark as b

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out_v24"
AGENTS = OUT / "agents"
V23_URL = (
    "https://raw.githubusercontent.com/Chirantan02/Kaggriculture/"
    "16d2a2777cf2a7e5172e4f6d8272f6143a695243/main_v23.py"
)

PARAMS = {
    "v24_conservative": {
        "base_ratio": 0.45,
        "peak_ratio": 0.65,
        "recovery_base": 0.55,
        "recovery_peak": 0.75,
        "defer_capacity": 82,
        "force_capacity": 90,
        "max_batch": 20,
    },
    "v24_moderate": {
        "base_ratio": 0.65,
        "peak_ratio": 0.78,
        "recovery_base": 0.65,
        "recovery_peak": 0.82,
        "defer_capacity": 78,
        "force_capacity": 88,
        "max_batch": 25,
    },
    "v24_aggressive": {
        "base_ratio": 0.80,
        "peak_ratio": 0.86,
        "recovery_base": 0.75,
        "recovery_peak": 0.90,
        "defer_capacity": 74,
        "force_capacity": 85,
        "max_batch": 30,
    },
}

WRAPPER = r'''

# V24 centralized market-timing controller.
_V24_CORE_AGENT = agent
_V24_ITEMS = ("WOOL", "MELON", "STRAWBERRY", "MILK")
_V24_BASE_RATIO = {base_ratio!r}
_V24_PEAK_RATIO = {peak_ratio!r}
_V24_RECOVERY_BASE = {recovery_base!r}
_V24_RECOVERY_PEAK = {recovery_peak!r}
_V24_DEFER_CAPACITY = {defer_capacity!r}
_V24_FORCE_CAPACITY = {force_capacity!r}
_V24_MAX_BATCH = {max_batch!r}
_V24_START = 240
_V24_STOP = 672
_V24_HISTORY = 96
_V24_STATE = {{
    0: {{"last_step": -1, "pending": {{}}, "history": {{}}}},
    1: {{"last_step": -1, "pending": {{}}, "history": {{}}}},
}}


def _v24_step(obs):
    explicit = _get(obs, "step")
    if explicit is not None:
        return max(0, int(explicit or 0))
    return max(0, int(_get(obs, "day", 0) or 0) * 24 + int(_get(obs, "hour", 0) or 0))


def _v24_shed_tiles(board_size):
    half = board_size // 2
    return {{(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)}}


def _v24_projected_shed(obs, action):
    private = _get(obs, "private", {{}}) or {{}}
    projected = {{key: max(0, int(value or 0)) for key, value in dict(_get(private, "shed", {{}}) or {{}}).items()}}
    farm = _farm(obs, _seat(obs))
    board_size = len(list(_get(farm, "tiles", []) or [])) or 10
    access = _v24_shed_tiles(board_size)
    positions = [
        _get(farm, "farmer"),
        *list(_get(farm, "hands", []) or []),
    ]
    inventories = list(_get(private, "inventories", []) or [])
    unit_actions = [action.get("farmer", ["PASS"]), *list(action.get("hands") or [])]
    room = max(0, 100 - sum(projected.values()))
    for index, raw in enumerate(unit_actions):
        if room <= 0 or not isinstance(raw, list) or not raw:
            continue
        position = positions[index] if index < len(positions) else None
        try:
            standing_at_shed = tuple(position) in access
        except TypeError:
            standing_at_shed = False
        if not standing_at_shed:
            continue
        inventory = dict(inventories[index] or {{}}) if index < len(inventories) else {{}}
        if raw[0] == "DROP":
            for item, value in inventory.items():
                quantity = min(room, max(0, int(value or 0)))
                if quantity > 0:
                    projected[item] = projected.get(item, 0) + quantity
                    room -= quantity
                if room <= 0:
                    break
        elif raw[0] == "PLACE" and len(raw) >= 2:
            item = raw[1]
            requested = int(raw[2]) if len(raw) >= 3 else 1
            quantity = min(room, max(0, requested), max(0, int(inventory.get(item, 0) or 0)))
            if quantity > 0:
                projected[item] = projected.get(item, 0) + quantity
                room -= quantity
    return projected


def _v24_controller(obs, action, configuration, step):
    action = _copy_action(action)
    seat = _seat(obs)
    state = _V24_STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {{"last_step": step, "pending": {{}}, "history": {{}}}}
        _V24_STATE[seat] = state
    state["last_step"] = step
    prices = dict(_get(_get(obs, "market", {{}}) or {{}}, "prices", {{}}) or {{}})
    for item in _V24_ITEMS:
        history = list(state["history"].get(item, []))
        history.append(max(1, int(prices.get(item, _MARKET_PARAMS[item][0]) or 1)))
        state["history"][item] = history[-_V24_HISTORY:]

    projected = _v24_projected_shed(obs, action)
    total_shed = sum(max(0, int(value or 0)) for value in projected.values())
    pending = state["pending"]
    kept = []
    reserved = {{}}

    for raw in list(action.get("market") or []):
        order = list(raw)
        if not _is_sell(order):
            kept.append(order)
            continue
        item = order[1]
        quantity = max(0, int(order[2] or 0))
        available = max(0, int(projected.get(item, 0) or 0) - int(reserved.get(item, 0) or 0))
        base = float(_MARKET_PARAMS[item][0])
        price = float(prices.get(item, base) or 0)
        peak = float(max(state["history"].get(item, [price]) or [price]))
        crash = (
            item in _V24_ITEMS
            and _V24_START <= step < _V24_STOP
            and total_shed <= _V24_DEFER_CAPACITY
            and quantity > 0
            and quantity <= available
            and price < base * _V24_BASE_RATIO
            and price < peak * _V24_PEAK_RATIO
        )
        if crash:
            pending[item] = max(0, int(pending.get(item, 0) or 0)) + quantity
            continue
        kept.append(order)
        actual = min(quantity, available)
        reserved[item] = int(reserved.get(item, 0) or 0) + actual
        if actual > 0 and pending.get(item, 0):
            pending[item] = max(0, int(pending.get(item, 0) or 0) - actual)

    for item in _V24_ITEMS:
        owed = max(0, int(pending.get(item, 0) or 0))
        if owed <= 0 or len(kept) >= 10:
            continue
        available = max(0, int(projected.get(item, 0) or 0) - int(reserved.get(item, 0) or 0))
        if available <= 0:
            continue
        base = float(_MARKET_PARAMS[item][0])
        price = float(prices.get(item, base) or 0)
        peak = float(max(state["history"].get(item, [price]) or [price]))
        recovered = (
            price >= base * _V24_RECOVERY_BASE
            or price >= peak * _V24_RECOVERY_PEAK
        )
        forced = step >= _V24_STOP or total_shed >= _V24_FORCE_CAPACITY
        if not (recovered or forced):
            continue
        quantity = min(owed, available, _V24_MAX_BATCH)
        if quantity <= 0:
            continue
        kept.append(["SELL", item, quantity])
        reserved[item] = int(reserved.get(item, 0) or 0) + quantity
        pending[item] = owed - quantity

    # Terminal value is cash only. Liquidate every remaining shed product.
    if step >= 716:
        for item in _MARKET_PARAMS:
            if len(kept) >= 10:
                break
            available = max(0, int(projected.get(item, 0) or 0) - int(reserved.get(item, 0) or 0))
            if available <= 0:
                continue
            kept.append(["SELL", item, available])
            reserved[item] = int(reserved.get(item, 0) or 0) + available
            if item in pending:
                pending[item] = max(0, int(pending.get(item, 0) or 0) - available)

    action["market"] = kept[:10]
    return _align_hands(_rank_sell_slots(obs, action, configuration), obs)


def agent(obs, configuration=None):
    try:
        step = _v24_step(obs)
        return _v24_controller(obs, _V24_CORE_AGENT(obs, configuration), configuration, step)
    except Exception:
        return _V24_CORE_AGENT(obs, configuration)


__version__ = "V24-adaptive-market-{name}"
'''


def fetch_v23() -> bytes:
    with urllib.request.urlopen(V23_URL, timeout=60) as response:
        payload = response.read()
    print(f"v23 sha256={hashlib.sha256(payload).hexdigest()}", flush=True)
    compile(payload.decode("utf-8"), "v23.py", "exec")
    return payload


def build_agents(v23_payload: bytes, target_payload: bytes) -> dict[str, Path]:
    AGENTS.mkdir(parents=True, exist_ok=True)
    paths = {}
    exact_v23 = v23_payload.decode("utf-8")
    sources = {
        "v23_exact": exact_v23,
        "target_notebook": target_payload.decode("utf-8"),
    }
    for name, params in PARAMS.items():
        sources[name] = exact_v23 + WRAPPER.format(name=name, **params)
    for name, source in sources.items():
        path = AGENTS / f"{name}.py"
        path.write_text(source + f"\n# remote_variant={name!r}\n", encoding="utf-8")
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
        paths[name] = path
    return paths


def load_agent(path: Path, tag: str):
    module_name = f"v24_{tag}_{os.getpid()}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "agent", None)
    if not callable(fn):
        raise RuntimeError(f"No agent in {path}")
    return fn


def worker(args: tuple[str, str, str, str, int]) -> b.SeedResult:
    variant, challenger_raw, target_raw, split, seed = args
    challenger_path = Path(challenger_raw)
    target_path = Path(target_raw)
    tag = f"{variant}_{split}_{seed}"
    c0 = load_agent(challenger_path, tag + "_c0")
    c1 = load_agent(challenger_path, tag + "_c1")
    t0 = load_agent(target_path, tag + "_t0")
    t1 = load_agent(target_path, tag + "_t1")
    started = time.perf_counter()
    challenger_seat0, target_seat1 = b.run_game(c0, t1, seed)
    target_seat0, challenger_seat1 = b.run_game(t0, c1, seed)
    margin0 = challenger_seat0 - target_seat1
    margin1 = challenger_seat1 - target_seat0
    return b.SeedResult(
        variant=variant,
        split=split,
        seed=seed,
        challenger_seat0=challenger_seat0,
        target_seat1=target_seat1,
        margin_seat0=margin0,
        target_seat0=target_seat0,
        challenger_seat1=challenger_seat1,
        margin_seat1=margin1,
        paired_margin=(margin0 + margin1) / 2.0,
        elapsed_seconds=time.perf_counter() - started,
    )


def evaluate(name: str, challenger: Path, target: Path, split: str, seeds: list[int]) -> list[b.SeedResult]:
    workers = max(1, min(int(os.environ.get("BENCH_WORKERS", "4")), len(seeds)))
    tasks = [(name, str(challenger), str(target), split, int(seed)) for seed in seeds]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, task): task[-1] for task in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            seed = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"{name}/{split} failed at seed {seed}: {exc}") from exc
            if done == 1 or done % 4 == 0 or done == len(tasks):
                mean = statistics.fmean(row.paired_margin for row in rows)
                wins = sum(row.paired_margin > 0 for row in rows)
                losses = sum(row.paired_margin < 0 for row in rows)
                print(f"[{split}] {name} {done}/{len(tasks)} mean={mean:.2f} W-L={wins}-{losses}", flush=True)
    return sorted(rows, key=lambda row: row.seed)


def rank_key(summary: dict[str, object]):
    return (
        float(summary["mean_paired_margin"]),
        float(summary["bootstrap_bca_95_ci_low"]),
        -float(summary["losses"]),
        -float(summary["sd_paired_margin"]),
    )


def deterministic_submission(main_path: Path, archive_path: Path) -> str:
    import gzip
    payload = main_path.read_bytes()
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
        info = tarfile.TarInfo("main.py")
        info.size = len(payload)
        info.mtime = 0
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(payload))
    with archive_path.open("wb") as handle:
        with gzip.GzipFile(filename="", fileobj=handle, mode="wb", mtime=0) as gz:
            gz.write(raw.getvalue())
    return hashlib.sha256(archive_path.read_bytes()).hexdigest()


def write_summary(rows: list[dict[str, object]], path: Path):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    v23_payload = fetch_v23()
    target_payload = b.fetch_exact_target()
    paths = build_agents(v23_payload, target_payload)

    identity_seeds = list(range(950000, 950004))
    identity = evaluate("v23_identity", paths["v23_exact"], paths["v23_exact"], "identity", identity_seeds)
    if any(row.paired_margin != 0 for row in identity):
        raise RuntimeError("v23 identity control failed")
    b.write_rows(identity, OUT / "identity.csv")

    development_seeds = list(range(951000, 951024))
    development_rows = []
    development_summaries = []
    for name in PARAMS:
        rows = evaluate(name, paths[name], paths["v23_exact"], "development_vs_v23", development_seeds)
        development_rows.extend(rows)
        development_summaries.append(b.summarize(rows))
    b.write_rows(development_rows, OUT / "development.csv")
    ranked = sorted(development_summaries, key=rank_key, reverse=True)
    finalists = [str(row["variant"]) for row in ranked[:2]]
    print("FINALISTS", finalists, flush=True)

    validation_seeds = list(range(952000, 952032))
    validation_rows = []
    validation_summaries = []
    for name in finalists:
        rows = evaluate(name, paths[name], paths["v23_exact"], "validation_vs_v23", validation_seeds)
        validation_rows.extend(rows)
        validation_summaries.append(b.summarize(rows))
    b.write_rows(validation_rows, OUT / "validation.csv")
    selected = max(validation_summaries, key=rank_key)
    selected_name = str(selected["variant"])
    print("LOCKED", selected_name, flush=True)

    holdout_seeds = list(range(953000, 953096))
    holdout = evaluate(selected_name, paths[selected_name], paths["v23_exact"], "locked_holdout_vs_v23", holdout_seeds)
    b.write_rows(holdout, OUT / "holdout_vs_v23.csv")
    holdout_summary = b.summarize(holdout)

    notebook_seeds = list(range(954000, 954064))
    notebook_rows = evaluate(selected_name, paths[selected_name], paths["target_notebook"], "locked_vs_target_notebook", notebook_seeds)
    b.write_rows(notebook_rows, OUT / "holdout_vs_target_notebook.csv")
    notebook_summary = b.summarize(notebook_rows)

    final_main = OUT / "main.py"
    shutil.copy2(paths[selected_name], final_main)
    main_sha = hashlib.sha256(final_main.read_bytes()).hexdigest()
    submission_sha = deterministic_submission(final_main, OUT / "submission.tar.gz")

    summaries = [b.summarize(identity)] + development_summaries + validation_summaries + [holdout_summary, notebook_summary]
    write_summary(summaries, OUT / "summary.csv")
    report = {
        "environment_commit": b.ENV_COMMIT,
        "v23_source_url": V23_URL,
        "v23_sha256": hashlib.sha256(v23_payload).hexdigest(),
        "target_notebook_sha256": b.TARGET_MAIN_SHA256,
        "protocol": {
            "primary_unit": "seed block averaged across both seat assignments",
            "development_seed_range": [development_seeds[0], development_seeds[-1]],
            "validation_seed_range": [validation_seeds[0], validation_seeds[-1]],
            "locked_holdout_seed_range": [holdout_seeds[0], holdout_seeds[-1]],
            "target_notebook_seed_range": [notebook_seeds[0], notebook_seeds[-1]],
            "selection_completed_before_holdout": True,
        },
        "development_summaries": development_summaries,
        "development_ranking": [str(row["variant"]) for row in ranked],
        "finalists": finalists,
        "validation_summaries": validation_summaries,
        "selected_variant": selected_name,
        "locked_holdout_vs_v23": holdout_summary,
        "locked_vs_target_notebook": notebook_summary,
        "main_sha256": main_sha,
        "submission_sha256": submission_sha,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not bool(notebook_summary["significant_positive_at_5pct"]):
        raise SystemExit("Selected v24 failed to beat target notebook on locked seeds")


if __name__ == "__main__":
    main()
