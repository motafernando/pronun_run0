"""Create a private supervisor review queue from TRAIN only; never modifies reference."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from poc import load_snapshot,split_dataset
from train import MODEL,REVISION,TwoHead,predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int,default=12);args=p.parse_args()
    base=Path(__file__).resolve().parent;docs=load_snapshot(base.parent/'relatorios'/'pronunciamentos_2026-11-06')
    assign,_=split_dataset(docs);eligible=[]
    for d in docs:
        if assign[d['id']]!='train' or not d['curated']:continue
        labels={s['label'] for s in d['versions'][0]['spans']}
        if {'Pronunciamento','Encaminhamento','Prazo'}<=labels:eligible.append(d)
    # Fixed order, no cherry-picking by model success or test outcomes.
    eligible=sorted(eligible,key=lambda d:d['id'])[:args.limit]
    if not eligible:raise ValueError('Nenhum documento elegível')
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,use_fast=True)
    model=TwoHead(MODEL).to(device);model.load_state_dict(torch.load(args.checkpoint,map_location=device,weights_only=True));model.eval()
    queue=[]
    for d in eligible:
        v=d['versions'][0];pred=predict(model,tok,v['text'],device)
        gold={(s['label'],s['begin'],s['end']) for s in v['spans'] if s['label'] in {'Prazo','Pronunciamento','Encaminhamento'}}
        predicted={(s['label'],s['begin'],s['end']) for s in pred}
        for origin,diffs in [('reference_only',gold-predicted),('prediction_only',predicted-gold)]:
            for label,a,b in sorted(diffs):
                queue.append(dict(document=d['id'],source=v['export_path'],split='train',kind=origin,
                    label=label,start=a,end=b,span=v['text'][a:b],
                    context=v['text'][max(0,a-100):min(len(v['text']),b+100)],
                    verdict='pending_supervisor_review',explanation=None))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(queue,ensure_ascii=False,indent=2),encoding='utf-8')
    print('train documents',len(eligible),'mismatches',len(queue),'queue',args.out)


if __name__=='__main__':main()
