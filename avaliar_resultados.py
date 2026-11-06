"""Compare saved predictions and lexical baseline on the identical frozen test."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'gold'))
from piloto_regras_pronunciamento import predict as rules
from poc import load_snapshot


def counts(gold,pred,overlap=False):
    candidates=[]
    for i,g in enumerate(gold):
        for j,p in enumerate(pred):
            if g['label']!=p['label']:continue
            a,b=g['begin'],g['end'];c,d=p['char_start_index'],p['char_end_index']
            inter=max(0,min(b,d)-max(a,c));union=max(b,d)-min(a,c)
            iou=inter/union if union else 0
            if (iou>=.5 if overlap else a==c and b==d):candidates.append((iou,i,j))
    gi=set();pi=set()
    for _,i,j in sorted(candidates,reverse=True):
        if i not in gi and j not in pi:gi.add(i);pi.add(j)
    return len(gi),len(pred),len(gold)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args()
    base=Path(__file__).resolve().parent
    docs={x['id']:x for x in load_snapshot(base.parent/'relatorios/pronunciamentos_2026-11-06')}
    preds=json.loads((args.run/'test_predictions.json').read_text(encoding='utf-8'))
    stats={};detail=[]
    for model in ['model','lexical']:
        per=[]
        for x in preds:
            gold=[s for s in docs[x['document']]['versions'][0]['spans'] if s['label'] in {'Pronunciamento','Encaminhamento','Prazo'}]
            pred=x['entities'] if model=='model' else rules(x['text'])['entities']
            per.append(dict(document=x['document'],exact=counts(gold,pred),iou50=counts(gold,pred,True)))
        totals={}
        for method in ['exact','iou50']:
            tp,pred,gold=[sum(x[method][i] for x in per) for i in range(3)]
            totals[method]={'tp':tp,'predictions':pred,'reference':gold,'precision':tp/pred if pred else 0,'recall':tp/gold if gold else 0,'f1':2*tp/(pred+gold) if pred+gold else 0}
        stats[model]=totals;detail.append(dict(method=model,per_document=per))
    out={'population':'mesmos documentos do teste congelado, um processo; pareamento IoU guloso por documento e família','methods':stats,'details':detail}
    (args.run/'comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
