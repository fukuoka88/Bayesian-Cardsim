#!/usr/bin/env python3
from __future__ import annotations
"""
stress_test.py — Threshold stability under changing base fraud rates and tactic mixes.
See header in earlier message for detailed usage.
"""
import argparse, os, subprocess, json
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util
from pathlib import Path

def import_train_module(repo_root: str = "."):
    train_path = Path(repo_root) / "train.py"
    if not train_path.exists():
        raise FileNotFoundError(f"train.py not found at {train_path}")
    spec = importlib.util.spec_from_file_location("train_module", str(train_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator_template", type=str, default=None)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--base_rates", type=str, default=None)
    ap.add_argument("--tactic_mixes", type=str, default=None)
    ap.add_argument("--n_train", type=int, default=200000)
    ap.add_argument("--n_test", type=int, default=80000)
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--outputs_dir", type=str, default="outputs/stress")
    ap.add_argument("--scenarios_json", type=str, default=None)
    ap.add_argument("--cost_fp", type=float, default=0.5)
    ap.add_argument("--cost_fn", type=float, default=6.0)
    return ap.parse_args()

def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def generate_dataset(cmd_template: str, config: str, out_path: str, n: int, base_rate: float, mix: str):
    cmd = cmd_template.format(
        config=config if config else "",
        out=out_path,
        out_train=out_path,
        out_test=out_path,
        n=n,
        n_train=n,
        n_test=n,
        base_rate=base_rate,
        mix=mix
    )
    print(f"[GEN] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    args = parse_args()
    ensure_dirs(args.data_dir, args.outputs_dir)
    train_mod = import_train_module(".")

    scenarios = []
    if args.generator_template and args.base_rates and args.tactic_mixes:
        base_rates = [float(x.strip()) for x in args.base_rates.split(",") if x.strip()]
        mixes = [x.strip() for x in args.tactic_mixes.split(",") if x.strip()]
        for br in base_rates:
            for mix in mixes:
                name = f"br{br:g}_{mix}"
                train_csv = os.path.join(args.data_dir, f"train_{name}.csv")
                test_csv  = os.path.join(args.data_dir, f"test_{name}.csv")
                generate_dataset(args.generator_template, args.config, train_csv, args.n_train, br, mix)
                generate_dataset(args.generator_template, args.config, test_csv,  args.n_test,  br, mix)
                scenarios.append({"name": name,"train_csv": train_csv,"test_csv": test_csv,"base_rate": br,"mix": mix})
    elif args.scenarios_json:
        with open(args.scenarios_json, "r") as f:
            scenarios = json.load(f)
    else:
        raise SystemExit("Provide either --generator_template with --base_rates/--tactic_mixes OR --scenarios_json.")

    rows = []
    for sc in scenarios:
        name = sc["name"]
        print(f"[EVAL] {name}")
        rpt = train_mod.train_eval(sc["train_csv"], sc["test_csv"], args.cost_fp, args.cost_fn, out_dir=os.path.join(args.outputs_dir, name))
        rows.append({
            "scenario": name,
            "base_rate": float(sc.get("base_rate", float("nan"))),
            "mix": sc.get("mix",""),
            "ROC_AUC": rpt["ROC_AUC"],
            "PR_AUC": rpt["PR_AUC"],
            "Brier": rpt["Brier"],
            "Threshold": rpt["Threshold"],
            "ExpectedCost_per_txn": rpt["ExpectedCost_per_txn"],
            "tn": rpt["Confusion"]["tn"],
            "fp": rpt["Confusion"]["fp"],
            "fn": rpt["Confusion"]["fn"],
            "tp": rpt["Confusion"]["tp"],
            "TrainRows": rpt["TrainRows"],
            "TestRows": rpt["TestRows"],
        })

    df = pd.DataFrame(rows).sort_values(["mix","base_rate","scenario"])
    summary_csv = os.path.join(args.outputs_dir, "summary.csv")
    df.to_csv(summary_csv, index=False)
    print(f"[DONE] Wrote {summary_csv}")

    # Plot threshold vs base fraud rate (single-plot rule)
    plt.figure()
    if df["mix"].nunique() > 1:
        for mix in sorted(df["mix"].unique()):
            sub = df[df["mix"] == mix].sort_values("base_rate")
            plt.plot(sub["base_rate"], sub["Threshold"], marker="o", label=mix)
        plt.legend()
    else:
        sub = df.sort_values("base_rate")
        plt.plot(sub["base_rate"], sub["Threshold"], marker="o")
    plt.xlabel("Base fraud rate")
    plt.ylabel("Chosen decision threshold")
    plt.title("Threshold stability across base rates / tactic mixes")
    fig_path = os.path.join(args.outputs_dir, "threshold_vs_base_rate.png")
    plt.savefig(fig_path, bbox_inches="tight")
    print(f"[DONE] Wrote {fig_path}")

if __name__ == "__main__":
    main()
