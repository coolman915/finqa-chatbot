#!/usr/bin/env python3
"""Query MongoDB for evaluation runs and predictions.

Usage:
    python scripts/query_results.py runs                      # list recent runs
    python scripts/query_results.py run <run_id>              # show run details
    python scripts/query_results.py failures <run_id>         # show failures for a run
    python scripts/query_results.py history <entry_id>        # show entry across runs
    python scripts/query_results.py best                      # show best run
    python scripts/query_results.py compare <id1> <id2>       # compare two runs
"""

import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from dotenv import load_dotenv
load_dotenv(_project_root / ".env", override=True)

from finqa_chatbot.storage import get_mongo_store


def _pp(obj: dict | list) -> None:
    """Pretty-print a dict or list."""
    print(json.dumps(obj, indent=2, default=str))


def cmd_runs(store, args):
    runs = store.get_recent_runs(limit=args.limit)
    if not runs:
        print("No runs found.")
        return
    print(f"{'Run ID':<40} {'Split':>5} {'N':>5} {'Exe':>7} {'Prog':>7} {'Time':>6} {'Started'}")
    print("-" * 100)
    for r in runs:
        off = r.get("results_official") or {}
        exe = off.get("exe_acc")
        prog = off.get("prog_acc")
        dur = r.get("duration_seconds")
        started = r.get("started_at", "")
        if hasattr(started, "strftime"):
            started = started.strftime("%Y-%m-%d %H:%M")
        exe_s = f"{exe:>6.1%}" if exe is not None else f"{'n/a':>6}"
        prog_s = f"{prog:>6.1%}" if prog is not None else f"{'n/a':>6}"
        dur_s = f"{dur:>5.0f}s" if dur is not None else f"{'n/a':>6}"
        print(
            f"{r['run_id']:<40} "
            f"{r.get('split', '?'):>5} "
            f"{r.get('num_examples', '?'):>5} "
            f"{exe_s:>7} "
            f"{prog_s:>7} "
            f"{dur_s:>6} "
            f"{started}"
        )


def cmd_run(store, args):
    run = store.get_run(args.run_id)
    if not run:
        print(f"Run '{args.run_id}' not found.")
        return
    _pp(run)


def cmd_failures(store, args):
    failures = store.get_failures(args.run_id)
    if not failures:
        print(f"No failures found for run '{args.run_id}'.")
        return
    print(f"{len(failures)} failures:")
    print()
    for f in failures[:args.limit]:
        print(f"  Entry: {f['entry_id']}")
        print(f"    Predicted: {f.get('raw_program', 'n/a')}")
        print(f"    Gold:      {f.get('gold_program', 'n/a')}")
        print(f"    Exe result: {f.get('exe_result')}  Gold answer: {f.get('gold_answer')}")
        if f.get("error"):
            print(f"    Error: {f['error']}")
        print()


def cmd_history(store, args):
    preds = store.get_entry_history(args.entry_id)
    if not preds:
        print(f"No predictions found for entry '{args.entry_id}'.")
        return
    print(f"History for {args.entry_id} ({len(preds)} runs):")
    print()
    for p in preds:
        exe_mark = "PASS" if p.get("exe_correct") else "FAIL"
        prog_mark = "PASS" if p.get("prog_correct") else "FAIL"
        print(f"  {p['run_id']}")
        print(f"    exe={exe_mark}  prog={prog_mark}  rounds={p.get('rounds_used')}")
        print(f"    program: {p.get('raw_program', 'n/a')}")
        print()


def cmd_best(store, args):
    run = store.get_best_run()
    if not run:
        print("No completed runs found.")
        return
    print("Best run by exe_acc:")
    _pp(run)


def cmd_compare(store, args):
    preds1 = store.get_predictions(args.run_id_1)
    preds2 = store.get_predictions(args.run_id_2)
    if not preds1 or not preds2:
        print("One or both runs not found.")
        return

    d1 = {p["entry_id"]: p for p in preds1}
    d2 = {p["entry_id"]: p for p in preds2}
    common = set(d1.keys()) & set(d2.keys())

    improved = []  # exe went from fail to pass
    regressed = []  # exe went from pass to fail

    for eid in common:
        e1 = d1[eid].get("exe_correct", False)
        e2 = d2[eid].get("exe_correct", False)
        if not e1 and e2:
            improved.append(eid)
        elif e1 and not e2:
            regressed.append(eid)

    run1 = store.get_run(args.run_id_1)
    run2 = store.get_run(args.run_id_2)
    off1 = (run1 or {}).get("results_official", {})
    off2 = (run2 or {}).get("results_official", {})

    print(f"Comparing {args.run_id_1} vs {args.run_id_2}")
    print(f"  Common entries: {len(common)}")
    ea1 = off1.get("exe_acc")
    ea2 = off2.get("exe_acc")
    if ea1 is not None and ea2 is not None:
        print(f"  exe_acc: {ea1:.1%} -> {ea2:.1%} ({ea2 - ea1:+.1%})")
    pa1 = off1.get("prog_acc")
    pa2 = off2.get("prog_acc")
    if pa1 is not None and pa2 is not None:
        print(f"  prog_acc: {pa1:.1%} -> {pa2:.1%} ({pa2 - pa1:+.1%})")
    print(f"  Improved (fail->pass): {len(improved)}")
    print(f"  Regressed (pass->fail): {len(regressed)}")

    if improved and args.verbose:
        print("\n  Improved entries:")
        for eid in improved[:20]:
            print(f"    {eid}")
    if regressed and args.verbose:
        print("\n  Regressed entries:")
        for eid in regressed[:20]:
            print(f"    {eid}")


def main():
    parser = argparse.ArgumentParser(description="Query FinQA evaluation results from MongoDB")
    sub = parser.add_subparsers(dest="command", required=True)

    p_runs = sub.add_parser("runs", help="List recent runs")
    p_runs.add_argument("--limit", type=int, default=20)

    p_run = sub.add_parser("run", help="Show run details")
    p_run.add_argument("run_id")

    p_fail = sub.add_parser("failures", help="Show failures for a run")
    p_fail.add_argument("run_id")
    p_fail.add_argument("--limit", type=int, default=50)

    p_hist = sub.add_parser("history", help="Show entry predictions across runs")
    p_hist.add_argument("entry_id")

    sub.add_parser("best", help="Show best run by exe_acc")

    p_cmp = sub.add_parser("compare", help="Compare two runs")
    p_cmp.add_argument("run_id_1")
    p_cmp.add_argument("run_id_2")
    p_cmp.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    store = get_mongo_store()
    if store is None:
        print("ERROR: MongoDB not available. Set MONGODB_URI in .env")
        sys.exit(1)

    commands = {
        "runs": cmd_runs,
        "run": cmd_run,
        "failures": cmd_failures,
        "history": cmd_history,
        "best": cmd_best,
        "compare": cmd_compare,
    }
    commands[args.command](store, args)
    store.close()


if __name__ == "__main__":
    main()
