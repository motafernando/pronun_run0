"""Compare supervised continuation-weight variants using validation only."""
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent/'artefatos'
PAIRS=[('fold0','comprehensive_iou50_s42','supervisor_continuacao_s42'),
       ('fold2','fold2_continuacao_1_s42','fold2_continuacao_2_s42')]


def main():
    result=[]
    for fold,control,supervised in PAIRS:
        a,b=BASE/control,BASE/supervised
        if not all((x/'best.json').exists() for x in (a,b)):
            print('pendente',fold);continue
        ma,mb=[json.loads((x/'manifest.json').read_text(encoding='utf-8')) for x in (a,b)]
        if ma['assignments']!=mb['assignments'] or ma['seed']!=mb['seed']:
            raise ValueError('Par não tem mesma divisão e seed: '+fold)
        row={'fold':fold,'control':control,'supervised':supervised}
        for key,folder in [('control',a),('supervised',b)]:
            best=json.loads((folder/'best.json').read_text(encoding='utf-8'))
            row[key+'_epoch']=best['epoch'];row[key+'_dev_macro_iou50']=best['macro_f1']
            row[key+'_dev_overall_exact']=best['dev']['overall'];row[key+'_dev_overall_iou50']=best['dev']['iou50']['overall']
            row[key+'_by_label_iou50']=best['dev']['iou50']['by_label']
        result.append(row)
    out={'status':'exploratório: seleção e comparação na mesma validação; sem teste independente para esta etapa',
         'supervision':'peso 2 na loss para tokens I; sem LLM externo e sem alteração do gold','comparisons':result}
    (BASE/'comparacao_supervisao.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
