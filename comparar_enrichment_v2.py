"""Compara RoBERTaLexPT-base bruto e enrichment_v2 com seleção fixa."""
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent/'artefatos'
RUNS={'original':BASE/'compare_v2_raw_control_s42','enriquecido':BASE/'compare_v2_robertalexpt_base_s42'}


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    manifests={name:read(path/'manifest.json') for name,path in RUNS.items()}
    raw,enriched=manifests['original'],manifests['enriquecido']
    if raw['selection_source_sha256']!=enriched['source_sha256']:
        raise ValueError('A seleção original não usa a fonte enriquecida')
    varying={'source','source_sha256','selection_source_sha256'}
    differences=[key for key in set(raw)|set(enriched) if key not in varying and raw.get(key)!=enriched.get(key)]
    if differences:raise ValueError(f'Configurações diferentes: {differences}')
    if raw['init_checkpoint_sha256'] is not None or not raw['skip_test']:
        raise ValueError('O controle deve começar de pesos pré-treinados e manter o teste fechado')
    result={'same_assignments':True,'same_training_settings':True,
            'selection_source_sha256':raw['selection_source_sha256'],
            'note':'Mesmas 328 versões selecionadas; cinco janelas originais não fornecem rótulo positivo.',
            'independent_test_used':False,'runs':{}}
    for name,path in RUNS.items():
        if (path/'test.json').exists():raise ValueError(f'Teste histórico consultado em {name}')
        b=read(path/'best.json');d=b['dev']
        result['runs'][name]={
            'source_sha256':manifests[name]['source_sha256'],
            'checkpoint_sha256':hashlib.sha256((path/'best.pt').read_bytes()).hexdigest(),
            'epochs_run':len(read(path/'training_log.json')),'best_epoch':b['epoch'],
            'dev_documents':d['documents'],'gold_spans':d['overall']['gold'],
            'predicted_spans':d['overall']['pred'],
            'macro_iou50_f1':b['macro_f1'],'micro_iou50_f1':d['iou50']['overall']['f1'],
            'micro_exact_f1':d['overall']['f1'],
            'attributes_on_reference_spans':d['attributes_on_reference_spans']}
    out=BASE/'enrichment_v2'/'comparacao_enrichment.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
