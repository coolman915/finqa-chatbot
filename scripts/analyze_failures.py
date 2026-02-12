"""Analyze failure cases from predictions file."""
import sys, os
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from dotenv import load_dotenv
load_dotenv(_project_root / ".env", override=True)

import json
from finqa_chatbot.evaluation.official import _relaxed_equal, program_tokenization, relaxed_equal_program
from finqa_chatbot.dsl.executor import eval_program

with open('output/predictions_dev_100.json') as f:
    preds = json.load(f)
with open('../FinQA_paper/dataset/dev.json') as f:
    gold = json.load(f)

data_dict = {e['id']: e for e in gold}

failures = []
for pred in preds:
    entry = data_dict[pred['id']]
    gold_res = entry['qa']['exe_ans']
    gold_prog = entry['qa']['program']
    gold_tokens = program_tokenization(gold_prog)
    pred_tokens = pred['predicted']

    invalid_flag, exe_res = eval_program(pred_tokens, entry['table'])
    exe_ok = not invalid_flag and _relaxed_equal(exe_res, gold_res)
    prog_ok = relaxed_equal_program(gold_tokens, pred_tokens)

    if not exe_ok:
        failures.append({
            'id': pred['id'],
            'question': entry['qa']['question'],
            'gold_prog': gold_prog,
            'pred_prog': pred.get('program_str', ''),
            'gold_ans': gold_res,
            'pred_ans': exe_res if not invalid_flag else 'INVALID',
            'prog_match': prog_ok,
        })

print(f'Total failures: {len(failures)}')
print()
for f in failures:
    ratio = ''
    try:
        p, g = float(f['pred_ans']), float(f['gold_ans'])
        if g != 0:
            ratio = f'  ratio={p/g:.4f}'
    except:
        pass
    print(f"ID: {f['id']}")
    print(f"Q: {f['question']}")
    print(f"Gold: {f['gold_prog']}  => {f['gold_ans']}")
    print(f"Pred: {f['pred_prog']}  => {f['pred_ans']}{ratio}")
    print(f"ProgMatch: {f['prog_match']}")
    print()
