"""Orchestrate the streamlined Run 1 experiments (E1, E2, E3, E4) sequentially on GPU."""
import json
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'artefatos/enrichment_v2/versoes_pronunciamentos.jsonl'
ARTIFACTS = BASE / 'artefatos'


def run_train(name, extra_args):
    run_dir = ARTIFACTS / name
    best_json = run_dir / 'best.json'
    if best_json.exists() and (run_dir / 'best.pt').exists():
        data = json.loads(best_json.read_text(encoding='utf-8'))
        print(f"[SKIP-DONE] {name}: epoch={data['epoch']} macro_iou50={data['macro_f1']:.4f}", flush=True)
        return data
    cmd = [
        sys.executable, str(BASE / 'train.py'),
        '--mode', 'comprehensive',
        '--seed', '42',
        '--epochs', '16',
        '--patience', '5',
        '--selection', 'iou50',
        '--scheduler', 'linear',
        '--inside-weight', '2.0',
        '--include-partial',
        '--auxiliary',
        '--skip-test',
        '--source-jsonl', str(SOURCE),
        '--selection-source-jsonl', str(SOURCE),
        '--run-dir', str(run_dir),
    ] + extra_args
    t0 = time.time()
    print(f"[START] {name}: {' '.join(extra_args)}", flush=True)
    res = subprocess.run(cmd, cwd=str(BASE), check=True)
    dt = time.time() - t0
    data = json.loads(best_json.read_text(encoding='utf-8'))
    print(f"[DONE] {name} in {dt/60:.2f} min: best_epoch={data['epoch']} macro_iou50={data['macro_f1']:.4f}", flush=True)
    return data


def main():
    t_start = time.time()
    summary = {'e1': {}, 'e2': {}, 'e3': {}}

    # Stage 1: E1 Boundary Supervision Policies (Fold 0)
    e1_configs = [
        ('run1_e1a_border_mask_s42', ['--boundary-policy', 'border_mask']),
        ('run1_e1b_border_soft_s42', ['--boundary-policy', 'border_soft', '--border-soft-weight', '0.25']),
        ('run1_e1c_curated_dedup_s42', ['--boundary-policy', 'curated_dedup']),
    ]
    for name, args in e1_configs:
        summary['e1'][name] = run_train(name, args)

    # Pick best boundary policy from E1 (or 'none' if baseline 0.6759 was higher, breaking ties by exact F1)
    baseline_best = json.loads((ARTIFACTS / 'compare_v2_robertalexpt_base_s42/best.json').read_text(encoding='utf-8'))
    candidates = [('none', baseline_best)]
    policy_map = {
        'run1_e1a_border_mask_s42': 'border_mask',
        'run1_e1b_border_soft_s42': 'border_soft',
        'run1_e1c_curated_dedup_s42': 'curated_dedup',
    }
    for name, data in summary['e1'].items():
        candidates.append((policy_map[name], data))
    candidates.sort(key=lambda x: (round(x[1]['macro_f1'], 4), round(x[1]['dev']['overall']['f1'], 4)), reverse=True)
    best_policy = candidates[0][0]
    print(f"[WINNER E1] boundary_policy={best_policy} (macro_iou50={candidates[0][1]['macro_f1']:.4f})", flush=True)
    summary['best_e1_policy'] = best_policy

    # Stage 2: E2 Differential Head LR & Aux Loss Weight (Fold 0, using best E1 policy)
    e2_configs = [
        ('run1_e2a_difflr_1e4_aux025_s42', ['--boundary-policy', best_policy, '--head-learning-rate', '1e-4', '--aux-loss-weight', '0.25']),
        ('run1_e2b_difflr_1e4_aux050_s42', ['--boundary-policy', best_policy, '--head-learning-rate', '1e-4', '--aux-loss-weight', '0.50']),
    ]
    for name, args in e2_configs:
        summary['e2'][name] = run_train(name, args)

    # Determine overall best configuration across Baseline, E1, and E2 for Stage 3 Multi-Fold CV
    all_fold0 = [('baseline', [], baseline_best)]
    for name, args in e1_configs:
        all_fold0.append((name, args, summary['e1'][name]))
    for name, args in e2_configs:
        all_fold0.append((name, args, summary['e2'][name]))
    all_fold0.sort(key=lambda x: (round(x[2]['macro_f1'], 4), round(x[2]['dev']['overall']['f1'], 4)), reverse=True)
    winner_name, winner_args, winner_data = all_fold0[0]
    print(f"[WINNER E1+E2] {winner_name} (macro_iou50={winner_data['macro_f1']:.4f}, args={winner_args})", flush=True)
    summary['winner_fold0'] = {'name': winner_name, 'args': winner_args, 'macro_f1': winner_data['macro_f1']}

    # Stage 3: E3 Multi-Fold Validation across 3 additional balanced 3-family process groups (Folds 2, 3, 5)
    for fold in (2, 3, 5):
        fname = f"run1_e3_cv_fold{fold}_s42"
        summary['e3'][f'fold_{fold}'] = run_train(fname, winner_args + ['--fold', str(fold)])

    out_path = ARTIFACTS / 'enrichment_v2/resumo_run1_enxuta.json'
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[ALL DONE] Total time: {(time.time() - t_start)/60:.2f} min -> {out_path}", flush=True)


if __name__ == '__main__':
    main()
