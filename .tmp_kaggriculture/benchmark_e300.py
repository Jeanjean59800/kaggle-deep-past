from __future__ import annotations

import base64
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
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import benchmark as b

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out_e300"
AGENTS = OUT / "agents"
LAYER_B85 = 'c-qYyYjfI2@;kp`j(j*szyT+f-Sq`mIa%_k^@fz~TyBq|R6>kFXCbkA*k)7t@7Irc3fXv5xvfhjLCo~@bocao#3M2Kj(=K@#Q51@AR?KTd6>k1$c4zWc~z9DEDDi?TbTol5W}vJw_#L@%c`t$*<Z+Ym@Y&iVZppwEM*y|OVP>YI=hwyjTfR^g{6qXRHX7+=3*{IRmeq`5FT~`dr|fm;Z~GkaV0iMn5F>LS%o>EC`JER<PtVZ@oAA~8(#RVD>ixd9~qT$(Jxl8e-i&*#fz|vv$QATG)k%k5>te0*{AhCpG`i(+KZJG*T5uz!m16>8dq!ll<N&H7Mm=Iqpb+%zys{P4&xM=kU~aTnyuqVL|HBm0M&Org8+{pcs)5A2PbdGA1=g5giAOOHF0`!ei59!zIZ?R5S(9(Kf?46!ynzrpHF_%GcSI6p=Qp<r>7^cPRB6VX@T5C`}4c;$wj*-+OJQ}&L;G9G5Pi6VuDZS7iTA*UyaYse#39Sj!!2a+C5%3e*2cj-k<)0pPwg_Q~Z20KD&5-`u^YJGXU&rE<n`hu;mwb%OgR5I)^>+yeD4t#G5cFWLKeA)EpU@X9VWKC^{9)k3I1t!Cd5(0dW{O5}o0oC-7eli22g=5YVAV(brb1wUC!0kjPC@%1tMm7u_RTE^jxWAaM!&1)%Xb19Cv1c9#I+OA8YhBF#$iAxkB%#y?7-*WqnvfVx_C41AW0L6_0&LOI)k!GQD!U0xs_i^J#QNgZgFRk;Jz^|(SyD}yMj(z58};Y|Q`4eGC$<Z(R02yE_148$eiz-QP*IH&xTz^gPS=&40?fONDk!+b4~h-v%d$@zJE*6oSMNnDhjI^+sGNzo=!r?c*i(Bhtm8uz3vtPDZOaRHWDhG`@_IN1}3w=2*iB{HS4t~efvLmI-b)4>dkR<u8!o_x4y+u`91cVM3F7FG^Z1vL{-Mq=om4$u4QQe9rgw;kq9D-92#g!T%QCITGI!*nS-$csh>XFOE0$*VL)2lvn;!bw!pI}rE9uqXQTuWLo1G4{mo6{t&GZVC6U0x3LG++o%|h)cOHI$d9uisC6EBuQ$V28yM`NPw|L9D$zjD$gbu4y3N!-{f;}{^`w|_y0nhbPpjIAO7$Y&9mz~*B<KK`?v4zT|)q7k$1ij)dG%oY-h}SkEE+uZnxX7sx6qlSVOF^fiM+elBl49hnHtnDFvycLQ_zvgh*3VYX~xg?Gu`Sp3Y^FB@mhB@)A+6!~9C3-&n{TXjw@x2N6|y4xxkN9-R|z78gnt=Md#FaEi-3TT^HQJ|x6j2<2rKU1{7!{0FW>0JY7g%A-|S0AYyfO&BMzCM+o`ntc@U==q5A9<07%L@~0rAqdzHg7vjR1p=c_<`R!2$~If&Ae45o3T*Ni7E0gyszLBGU%rwH`NB%if;T4O_)<s#DyjUUqvkveI!?MrJ1E@f;}M&?hf#FiO^MwoRD*?deQ&U<jNsR_+F|h6RYoBpMv1!Ovx!qy256NXt>cu8oWCj?S27@1fZjy8lTB72;a{w?Xz(P~o>@;uEMRirEWbh5fN*jcq}dI?DbX}TJ`%r#%Vi!%RYDBR6F^>&sZjbxl=w!%ItLJ2lo%g#h$_3v^76o7My`~<87Z3kE@80N`Sw!7kcjD#4G{LpBFYyZ>TQ+1fWNnq3?d%Fx<QY@Fg(sw<g+3slLibrT6W#ohsR0)6ycsxD5LK>#J&0i+2DBmbK9BXFxPmVq`t_^$iRAYsU+qd17c<K4iq&ECGHKvm=PZzMc2b@0t;XVeRXzVLJnujN_#|vzZfK8*2C!ye1<rHqf@G+H(@Y0#31jy?}=4>bO+&PBh!UC3&({`_1VryZ1^}m8V+V|7-L8;xP(yzC07U2CKXCb@Uewf19C_>3Eqe%T-EVHWPnAHMu*x$-~>4w_yKJZt1K$)iMiTUQ#da=coZS16`mwM&+!>E5j#0_CgIsfpGq*yqSUm2g2WszdBEQgRe;BP;!18uNw}UbLeMHe4hemYV2g_Yy6O{X&`*OpHCUNC>&h?(kt!NCx3CNq5joK?h@$6@43(%$fG;&XW3tfdyM_f#-Q0?oux6(Yn6}4IzqMx$+;KD9=dlO3MSR-^xbw-Ui+3L2aWmkD$(aE(gPf@QIZv3#aGppGW^n4);LNjps|07h>!i&NaGdm$x3z2K29&%8##5(aA-{=Z)^X{I)<0I3HYs^eKylQ=JOR|m=3{5Pdq-z$R8OdY-@~LUxB-jLd$z9-5D1Zi_1l5YQ9z<t(>djb+7>wP?g$x|dNq>6JFeqEu#IJ+HuNeswVwrqL}~$5lC5IwhRS~zW$qxxL9El;T`ztREOFrl1fdG9qQRTi`Ju2D=U~|+Z(z$l5Ia4QrOS+K6lV|isRl*3#Eb<SK>eCbD1Cxs;93{{C#m44%L%T3X$jd+Gbx5ThoMtgv%>ULA~<AM=}66}2&|<>frC+L49t-9T>4PRS`HYB@%uc>5{FU33XGN+v&wCVuBlJqG^|v825&$evYP<(Fvo18qfUV`JMr|XIIM?DZ^6qEi*KY)Z#RC<v$R63J*0};JD*FL!x!upg5tuHI;cL24nL0hU@4s%?x1$Xrb53~9On87@7~=F>dsMEEXOtBtHbZK-<_QO6FsLfdRBqQ`fHQ%H<(_S#_KR?yXSK38VRgjR1sy-?YEQ3`M7;#+GejOpYfaSo}IrNk3Zr7wb8y3e&25OX1>{~E&OBJr_gFqlTQCBF_BuT$$jFfX}D39JP5C$YQ$J#75(VJw&}GS8~8E2yq=t%j$dB{KD=Vdurl-l6L|?QsK^}jGE4$&O&9kS`>(lkZkySvj}0X!AT5D$fz9tNY!yNQ4ctR1Jx}83sV-;3qJV6Y%Z>M?SY(nrhs8!hS%eT7cF=G1X<`Y1fLRJ9kZP6}y3B&1b%^OXq06drQ^E0P)ZeC>sSK|GKahs#F5{9Yx<p<D83H*gF2(7u?3eMHv7viJIlv(=t1w3wElra@&toveWtKrXhMAU_hfq$HnFs|n={dDyrSa;1EaI>teJL6`+B-2-CC+}sAWY(A8c-Y21bOV*szyLabvb@^%qV8kIEtTqyg2_2eFi-8QFD)$*g{YIoEA7=mG0?YkkR@(8vcD-vwu@zpwn0C)4K4sNBg`dp4$$s3&Y)67w62&{9+<yO39yTt)PrBfiq1Ni`F((91IQkZ&ttzRl5#c7t{oY3`dR!almwKrt?qkks+B6>rn6OPFva-^L=D7gr>-BrBs-VV$@Iqwv1COv?4kRX1WJP8CgXmS?qSgR@}XSqU!!W7y#^Rg8{00x?t@37J=Fk(|vqAAe;czWheJYyuq9e`<d8!6gL?#lVjHheB@$F&SZ;rLF<l`=~Ew*u2qukM&|uIF0K?hYOhjW!>LwQt{_1`*L^^Df9xu6Rx-sNWEf%3K;FVZ7yl3hdlN&N0uKtGYT?lBC-{XVDOiZyBBF$oydbtmnRvRNM7K&~sk|)((#)RD#64_kW~_&{H$-DSW1}c~nO2~fAo50q(!O?de}x8SFeThGKmAv$kLyW3*HH%RzS_N>U6$1*vGh`z!b4u2?PVWXwzfsla4^6glA}Sp$scS(iy*mFZ0usSCa1VZX}3`e+OtMrPidbR%sa4{%06^ERY&h%USE1&r=oQ?s=9`0Z$saV)s3}UsT4sWE)K*Ag%rAfn8QaBLdajR)ns_9K<#_0&}yBp$KX5i?8aAL?mYRIO4vEU%{)Nvbi#7>vNdgq)62{V=gxY+u936~==N`5@9z$79?~74cz>BSTTci0+xoOnKTdb9RK-(kXYq2~%py6_BX^B=4Njk&j&nQgwLyKL%20TBr?U;r0=v62+=)0+a@d_G!N5EVXm{R73}CGE8VJ<l1UU2D0Ov^$V7>1?{8yvuZ<A~e8SOHcazPn!w8GFZ-`ZC>)B%?m6=1x~N?4$B$_QK89MZzL(P7iU)U58vy?lsOa%w!dGIGxr_;>;+%;l}N^O^<mLDvidDvoQJ7?Wt|imDzYsX?vnjuA_YYE^kv$U9mb+P^D=_IPKzMnw~W+TgIse=EwLxS&rAa{sJnnDJtZ!eq+L{#Wg1F*ZWE9O5e>wab3G1GCiarK>I<!f<>asf`}zk@$$-tA8Wo<qCZZ=U<Bq%g<6mf<pINKsCmxLd+pWg0c#8+*V^pC13<2Lp$5+tWK{L2W2t}n2mq7n;kUofoWt8S;wMN+f6*S8fff%CZ5(esx>6Y+O$*a^lKaTNXR{%G5N-0_)#bsRROZ=@PLYHHV1>MyEbLxnw^8{9E!`eRPV&tOVXf+fA6OSZJ7x1-d-^dq()NBA-E8Vvich;HHGr=9fS*$0n{N({8;Ft<2@~;!?#UOv^$!*Yi6$STlKuwgf<I9zSE-LeP!e!?z17~{F|S0YUg*<`o@f_(R=O)ost>AgP%dD%VtBBbJcw&`_W2bR|IN=B!<d|*3M-Vf^baTLKojg1gtFUP*p6}>IuQalOnu1#ubeT(cTPJCOa+zu}z=N_96Owaxp&hFcteYC(MJ{tmR#*s%m|v@2LyC?*^(=ZBLzMk8Zt?3wUSU@W{LNy1RYU*b(I7@XnUtcW7tb{r-zhKm+R=F*v#8=<bqwuLqnK8kl}>q4`Sf|K!yLBxm=v1<gvWC_qN;cxS<6X8xs{ky*oqBNU%{pDE6K`@0He^S|VU1)u1LzpwBOZ&YCt#P9UfE^@EkF4FbGn@6@U^j9<#ZQlgR-*daowqowG<J2WjTiEYajF)$QBg2U4iP^D_-Hom`sR+VElF)T6_=#hDoBPH0>-NDg&`64k$q%>&qX)jp+tSH-+k!nsr+?Qi4BL#kec=K1a<yU5_!v#gQQZkJL*xa9HAA2TxAng<WX!z`!d)BOu|LP??K(hW50Rn{Is{J{8hyb1_P?{YZ?sc(jrssIqM3XDwbu{ap|JjN->mzy?=%H@m|g|gE)9w#D{HrAm~`lfO22O4Y2f=ZURF8%dSgUDESLq<VcsNnp7O2xCZgQvc1q*LNw5AIzmwjYXj}X7g}oy3!5;7e@Lj9+UMB8ccxTR0&nND`3X`{y+>~NWA24;)a?4QFji;A_^LR;Dg7|xvzeCj-mSrOQP><5P!=U+iyLYg&d!sB{L2wOe7wi`Vz*-xD`pyrI`sx=reUxq=w0#K)aJ%(iB!1Nt'
LAYER = zlib.decompress(base64.b85decode(LAYER_B85)).decode("utf-8")


def patch_preemption(text: str) -> str:
    patches = (
        ("_PREEMPT_ENABLED = False", "_PREEMPT_ENABLED = True"),
        ("_PREEMPT_FRACTION = 2.0", "_PREEMPT_FRACTION = 1.0"),
    )
    for old, new in patches:
        if text.count(old) != 1:
            raise RuntimeError(f"Expected one {old!r}, found {text.count(old)}")
        text = text.replace(old, new, 1)
    return text


def build_agents(payload: bytes) -> dict[str, Path]:
    AGENTS.mkdir(parents=True, exist_ok=True)
    exact = payload.decode("utf-8")
    pre = patch_preemption(exact)
    variants = {
        "target_exact": exact,
        "preemption_only": pre,
        "pre_seed": pre + LAYER.replace("_E300_LAST_DAY_STEP = 696", "_E300_LAST_DAY_STEP = 9999", 1),
        "pre_seed_end704": pre + LAYER.replace("_E300_LAST_DAY_STEP = 696", "_E300_LAST_DAY_STEP = 704", 1),
        "pre_seed_end696": pre + LAYER,
    }
    paths = {}
    for name, text in variants.items():
        path = AGENTS / f"{name}.py"
        path.write_text(text + f"\n# remote_variant={name!r}\n", encoding="utf-8")
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
        paths[name] = path
    return paths


def load_agent(path: Path, tag: str):
    module_name = f"e300_{tag}_{os.getpid()}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "agent", None)
    if not callable(fn):
        raise RuntimeError(f"No agent in {path}")
    return fn


def paired_worker(args: tuple[str, str, str, str, int]) -> b.SeedResult:
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
    workers = min(max(1, int(os.environ.get("BENCH_WORKERS", "4"))), len(seeds))
    tasks = [(name, str(challenger), str(target), split, int(seed)) for seed in seeds]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(paired_worker, task): task[-1] for task in tasks}
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
    payload = b.fetch_exact_target()
    paths = build_agents(payload)

    identity_seeds = list(range(930000, 930004))
    identity = evaluate("identity", paths["target_exact"], paths["target_exact"], "identity", identity_seeds)
    if any(row.paired_margin != 0 for row in identity):
        raise RuntimeError("Identity control failed")
    b.write_rows(identity, OUT / "identity.csv")

    candidate_names = ["pre_seed", "pre_seed_end704", "pre_seed_end696"]
    development_seeds = list(range(931000, 931024))
    development_rows = []
    development_summaries = []
    for name in candidate_names:
        rows = evaluate(name, paths[name], paths["target_exact"], "development_vs_exact", development_seeds)
        development_rows.extend(rows)
        development_summaries.append(b.summarize(rows))
    b.write_rows(development_rows, OUT / "development.csv")
    ranked = sorted(development_summaries, key=rank_key, reverse=True)
    finalists = [str(row["variant"]) for row in ranked[:2]]
    print("FINALISTS", finalists, flush=True)

    validation_seeds = list(range(932000, 932032))
    validation_rows = []
    validation_summaries = []
    for name in finalists:
        rows = evaluate(name, paths[name], paths["target_exact"], "validation_vs_exact", validation_seeds)
        validation_rows.extend(rows)
        validation_summaries.append(b.summarize(rows))
    b.write_rows(validation_rows, OUT / "validation.csv")
    selected = max(validation_summaries, key=rank_key)
    selected_name = str(selected["variant"])
    print("LOCKED", selected_name, flush=True)

    holdout_seeds = list(range(933000, 933096))
    holdout = evaluate(selected_name, paths[selected_name], paths["target_exact"], "locked_holdout_vs_exact", holdout_seeds)
    b.write_rows(holdout, OUT / "holdout_vs_exact.csv")
    holdout_summary = b.summarize(holdout)

    incremental_seeds = list(range(934000, 934064))
    incremental = evaluate(selected_name + "_vs_preemption", paths[selected_name], paths["preemption_only"], "locked_incremental_vs_preemption", incremental_seeds)
    b.write_rows(incremental, OUT / "holdout_vs_preemption.csv")
    incremental_summary = b.summarize(incremental)

    final_main = OUT / "main.py"
    shutil.copy2(paths[selected_name], final_main)
    main_sha = hashlib.sha256(final_main.read_bytes()).hexdigest()
    submission_sha = deterministic_submission(final_main, OUT / "submission.tar.gz")

    summaries = [b.summarize(identity)] + development_summaries + validation_summaries + [holdout_summary, incremental_summary]
    write_summary(summaries, OUT / "summary.csv")
    report = {
        "environment_commit": b.ENV_COMMIT,
        "target_sha256": b.TARGET_MAIN_SHA256,
        "protocol": {
            "primary_unit": "seed block averaged across both seat assignments",
            "development_seed_range": [development_seeds[0], development_seeds[-1]],
            "validation_seed_range": [validation_seeds[0], validation_seeds[-1]],
            "locked_holdout_seed_range": [holdout_seeds[0], holdout_seeds[-1]],
            "incremental_holdout_seed_range": [incremental_seeds[0], incremental_seeds[-1]],
            "selection_completed_before_holdout": True,
        },
        "development_summaries": development_summaries,
        "development_ranking": [str(row["variant"]) for row in ranked],
        "finalists": finalists,
        "validation_summaries": validation_summaries,
        "selected_variant": selected_name,
        "locked_holdout_vs_exact": holdout_summary,
        "locked_incremental_vs_preemption": incremental_summary,
        "main_sha256": main_sha,
        "submission_sha256": submission_sha,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not bool(holdout_summary["significant_positive_at_5pct"]):
        raise SystemExit("Selected policy failed locked holdout vs exact")


if __name__ == "__main__":
    main()
