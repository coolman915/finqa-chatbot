#!/usr/bin/env python3
"""Query MongoDB for evaluation runs and predictions.

Usage:
    python scripts/query_results.py runs                      # list recent runs
    python scripts/query_results.py run <run_id>              # show run details
    python scripts/query_results.py failures <run_id>         # show failures with LLM eval details
    python scripts/query_results.py history <entry_id>        # show entry across runs
    python scripts/query_results.py best                      # show best run
    python scripts/query_results.py compare <id1> <id2>       # compare two runs
    python scripts/query_results.py evaluate <run_id>         # run LLM evaluation on unevaluated failures
    python scripts/query_results.py trace <run_id> <entry_id> # show step-by-step trace
    python scripts/query_results.py node-stats <run_id>       # show per-node aggregate stats

LLM evaluation fields (stored per prediction in MongoDB):
    llm_correct:     bool | None   — LLM's verdict (None = not yet evaluated)
    failure_reason:  str | None    — Classification: correct_alternate, wrong_number,
                                     wrong_computation, sign_error, scale_error,
                                     missing_step, extra_step, wrong_approach,
                                     invalid_program, rounding_error
    llm_explanation: str | None    — Brief LLM explanation of why it failed
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
from finqa_chatbot.evaluation.llm_eval import llm_evaluate_prediction


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
        if f.get("failure_reason"):
            llm_tag = "CORRECT" if f.get("llm_correct") else f["failure_reason"]
            print(f"    LLM eval: {llm_tag}")
            if f.get("llm_explanation"):
                print(f"    Explanation: {f['llm_explanation']}")
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


def cmd_evaluate(store, args):
    """Run LLM evaluation on failures for a given run."""
    # Find predictions that failed and haven't been LLM-evaluated yet
    query = {
        "run_id": args.run_id,
        "llm_correct": None,
        "$or": [{"exe_correct": False}, {"prog_correct": False}],
    }
    preds = list(store.predictions.find(query))
    if not preds:
        print(f"No unevaluated failures found for run '{args.run_id}'.")
        return

    print(f"Found {len(preds)} unevaluated failures. Running LLM evaluation...")
    print()

    counts: dict[str, int] = {}
    llm_correct_count = 0

    for i, pred in enumerate(preds, 1):
        entry_id = pred["entry_id"]
        # Fetch gold entry from dataset collection
        gold_doc = store.dataset.find_one({"_id": entry_id})
        if not gold_doc:
            print(f"  [{i}/{len(preds)}] {entry_id} — SKIP (gold entry not found)")
            continue

        result = llm_evaluate_prediction(
            question=gold_doc.get("question", ""),
            gold_program=gold_doc.get("program", ""),
            pred_program=pred.get("raw_program", ""),
            gold_answer=gold_doc.get("exe_ans"),
            pred_answer=pred.get("exe_result"),
            text_answer=gold_doc.get("answer", ""),
            table=gold_doc.get("table"),
            pre_text=gold_doc.get("pre_text"),
            post_text=gold_doc.get("post_text"),
        )

        store.update_prediction_llm_eval(args.run_id, entry_id, result)

        reason = result.get("failure_reason", "unknown")
        counts[reason] = counts.get(reason, 0) + 1
        if result.get("llm_correct"):
            llm_correct_count += 1

        status = "CORRECT" if result.get("llm_correct") else reason
        print(f"  [{i}/{len(preds)}] {entry_id} — {status}")

    print()
    print(f"LLM evaluation complete: {len(preds)} predictions evaluated")
    print(f"  LLM judged correct: {llm_correct_count}")
    print(f"  Failure breakdown:")
    for reason, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")


def _print_list(items: list, indent: str = "         ", max_items: int = 15) -> None:
    """Print a list of strings, one per line, with truncation."""
    for item in items[:max_items]:
        text = str(item)
        if len(text) > 200:
            text = text[:200] + "..."
        print(f"{indent}{text}")
    if len(items) > max_items:
        print(f"{indent}... and {len(items) - max_items} more")


def cmd_trace(store, args):
    """Show step-by-step trace for a single entry in a run."""
    steps = store.get_trace(args.run_id, args.entry_id)
    if not steps:
        print(f"No trace found for run '{args.run_id}', entry '{args.entry_id}'.")
        return

    print(f"Trace for {args.entry_id} in {args.run_id} ({len(steps)} steps):")
    print()

    for s in steps:
        node = s.get("node_name", "?")
        dur = s.get("duration_ms")
        rnd = s.get("round_number", 0)
        dur_s = f"{dur:.0f}ms" if dur is not None else "n/a"
        outputs = s.get("outputs", {})

        print(f"  [{s.get('step_index', '?')}] {node}  (round {rnd}, {dur_s})")

        # Node-specific display
        if node == "init":
            print(f"       question: {outputs.get('question', '')}")

        elif node == "scheduler":
            print(f"       round: {outputs.get('round_number')}  agents: {outputs.get('active_agents', [])}")

        elif node == "table_agent":
            n = outputs.get("num_lookups", 0)
            print(f"       {n} lookups:")
            _print_list(outputs.get("lookups", []))

        elif node == "context_agent":
            n = outputs.get("num_quotes", 0)
            nums = outputs.get("numbers_found", [])
            print(f"       {n} quotes, numbers found: {nums}")
            _print_list(outputs.get("quotes", []))

        elif node == "summarizer":
            print(f"       program: {outputs.get('program', '')}")
            candidates = outputs.get("candidates", [])
            if len(candidates) > 1:
                print(f"       candidates ({len(candidates)}):")
                _print_list(candidates)
            reasoning = outputs.get("reasoning", [])
            if reasoning:
                print(f"       reasoning:")
                _print_list(reasoning)

        elif node == "executor":
            print(f"       result: {outputs.get('result')}  invalid: {outputs.get('invalid', False)}")
            detail = outputs.get("detail", "")
            if detail:
                print(f"       detail: {detail}")

        elif node == "verifier":
            status = outputs.get("status", "")
            print(f"       status: {status}")
            issues = outputs.get("issues", [])
            if issues:
                print(f"       issues:")
                _print_list(issues)
            targets = outputs.get("flag_targets", [])
            if targets:
                print(f"       re-engage: {targets}")
            detail = outputs.get("detail", [])
            if detail:
                _print_list(detail)

        elif node == "finalize":
            print(f"       final_answer: {outputs.get('final_answer')}")
            prog = outputs.get("selected_program", "")
            if prog:
                print(f"       program: {prog}")
            if outputs.get("used_fallback"):
                print(f"       (used fallback from earlier round)")

        else:
            for k, v in outputs.items():
                val = str(v)
                if len(val) > 120:
                    val = val[:120] + "..."
                print(f"       {k}: {val}")

        # LLM usage
        lu = s.get("llm_usage")
        if lu:
            print(
                f"       llm: {lu.get('num_calls', 0)} calls, "
                f"{lu.get('total_tokens', 0)} tokens "
                f"({lu.get('total_prompt_tokens', 0)}p + "
                f"{lu.get('total_completion_tokens', 0)}c) "
                f"[{lu.get('model', '')}]"
            )

        # Error
        err = s.get("error")
        if err:
            print(f"       ERROR: {err.get('type', '?')}: {err.get('message', '')}")

        print()


def cmd_node_stats(store, args):
    """Show per-node aggregate stats for a run."""
    stats = store.get_node_stats(args.run_id)
    if not stats:
        print(f"No trace data found for run '{args.run_id}'.")
        return

    print(f"Node stats for {args.run_id}:")
    print()
    print(
        f"  {'Node':<16} {'Count':>6} {'Avg ms':>8} {'Max ms':>8} "
        f"{'Tokens':>8} {'Errors':>7}"
    )
    print("  " + "-" * 55)

    for s in stats:
        print(
            f"  {s['node_name']:<16} {s['count']:>6} "
            f"{s.get('avg_duration_ms', 0):>8.1f} "
            f"{s.get('max_duration_ms', 0):>8.1f} "
            f"{s.get('total_tokens', 0):>8} "
            f"{s.get('error_count', 0):>7}"
        )


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

    p_eval = sub.add_parser("evaluate", help="Run LLM evaluation on failures for a run")
    p_eval.add_argument("run_id")

    p_trace = sub.add_parser("trace", help="Show step-by-step trace for an entry")
    p_trace.add_argument("run_id")
    p_trace.add_argument("entry_id")

    p_nstats = sub.add_parser("node-stats", help="Show per-node aggregate stats")
    p_nstats.add_argument("run_id")

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
        "evaluate": cmd_evaluate,
        "trace": cmd_trace,
        "node-stats": cmd_node_stats,
    }
    commands[args.command](store, args)
    store.close()


if __name__ == "__main__":
    main()
