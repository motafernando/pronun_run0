"""Audita os três checkpoints iniciados de pesos pré-treinados no enrichment_v2."""
import hashlib
import json
from pathlib import Path


BASE=Path(__file__).resolve().parent/'artefatos'
RUNS={
    'RoBERTaLexPT-base':BASE/'compare_v2_robertalexpt_base_s42',
    'BERTimbau-base':BASE/'compare_v2_bertimbau_base_s42',
    'BERTimbau-large':BASE/'compare_v2_bertimbau_large_s42',
}


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    manifests={name:read(path/'manifest.json') for name,path in RUNS.items()}
    first=next(iter(manifests.values()))
    controlled=set(first)-{'model','model_revision'}
    for name,manifest in manifests.items():
        differing=[key for key in controlled if manifest[key]!=first[key]]
        if differing:raise ValueError(f'{name}: configurações diferentes: {differing}')
        if manifest['init_checkpoint_sha256'] is not None or not manifest['skip_test']:
            raise ValueError(f'{name}: inicialização ou teste indevido')
    result={'same_source_sha256':first['source_sha256'],'same_assignments':True,
            'same_settings_except_model':True,'selection':'macro F1 IoU >= 0.50 on one dev group',
            'independent_test_used':False,'models':{}}
    for name,path in RUNS.items():
        if (path/'test.json').exists():raise ValueError(f'{name}: teste histórico consultado')
        manifest=manifests[name];best=read(path/'best.json');dev=best['dev']
        result['models'][name]={
            'model':manifest['model'],'revision':manifest['model_revision'],
            'best_epoch':best['epoch'],'epochs_run':len(read(path/'training_log.json')),
            'dev_documents':dev['documents'],'gold_spans':dev['overall']['gold'],
            'predicted_spans':dev['overall']['pred'],
            'macro_iou50_f1':best['macro_f1'],
            'micro_iou50_f1':dev['iou50']['overall']['f1'],
            'micro_exact_f1':dev['overall']['f1'],
            'attributes_on_reference_spans':dev['attributes_on_reference_spans'],
            'checkpoint_sha256':hashlib.sha256((path/'best.pt').read_bytes()).hexdigest(),
        }
    out=BASE/'enrichment_v2'/'comparacao_modelos.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
