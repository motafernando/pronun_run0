"""Audita varredura de learning rate e peso de continuação a partir do mesmo checkpoint."""
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent/'artefatos'
RUNS={name:BASE/f'sweep_v2_{name}_s42' for name in ('lr5e6_i2','lr2e5_i2','lr2e5_i1','lr2e5_i3')}


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    parent=BASE/'compare_v2_robertalexpt_base_s42'
    checkpoint_sha=hashlib.sha256((parent/'best.pt').read_bytes()).hexdigest()
    manifests={name:read(path/'manifest.json') for name,path in RUNS.items()}
    first=next(iter(manifests.values()))
    for name,m in manifests.items():
        diff=[key for key in first if key not in {'learning_rate','inside_weight'} and first[key]!=m[key]]
        if diff:raise ValueError(f'{name}: diferenças inesperadas: {diff}')
        if m['init_checkpoint_sha256']!=checkpoint_sha or not m['skip_test']:
            raise ValueError(f'{name}: checkpoint inicial ou isolamento do teste incorreto')
    result={'parent_checkpoint_sha256':checkpoint_sha,'same_assignments':True,
            'same_source_and_settings_except_rate_and_inside_weight':True,'independent_test_used':False,
            'runs':{}}
    for name,path in RUNS.items():
        if (path/'test.json').exists():raise ValueError(f'{name}: teste histórico consultado')
        best=read(path/'best.json');dev=best['dev'];log=read(path/'training_log.json')
        result['runs'][name]={
            'learning_rate':manifests[name]['learning_rate'],
            'inside_weight':manifests[name]['inside_weight'],
            'epochs_run':len(log),'best_epoch':best['epoch'],
            'macro_iou50_f1':best['macro_f1'],
            'micro_iou50_f1':dev['iou50']['overall']['f1'],
            'micro_exact_f1':dev['overall']['f1'],
            'predicted_spans':dev['overall']['pred'],
            'checkpoint_sha256':hashlib.sha256((path/'best.pt').read_bytes()).hexdigest(),
        }
    out=BASE/'enrichment_v2'/'comparacao_hiperparametros.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
