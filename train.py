"""Train an independent two-head pronouncement extractor on audited CAS snapshots."""
import argparse
import collections
import hashlib
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET','1')
import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from poc import (HEAD_A,HEAD_B,ATTR_HEADS,load_snapshot,split_dataset,encode_labels,encode_auxiliary,
                 bio_keys,decode_spans,class_weights,match_iou,continuation_multipliers,
                 boundary_mask_and_weights,postprocess_spans_v3,predict_relations_v3)

BASE=Path(__file__).resolve().parent
MODEL='eduagarcia/RoBERTaLexPT-base'
REVISION='9dea30875c93dc2d33b35c2338dfc93c86715b9e'


class TwoHead(nn.Module):
    def __init__(self,model_path,revision=REVISION):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(model_path,revision=revision,add_pooling_layer=False)
        hidden=self.encoder.config.hidden_size
        self.drop=nn.Dropout(.1);self.a=nn.Linear(hidden,len(HEAD_A));self.b=nn.Linear(hidden,len(HEAD_B))
    def forward(self,ids,mask):
        h=self.drop(self.encoder(input_ids=ids,attention_mask=mask).last_hidden_state)
        return self.a(h),self.b(h)


class AuxiliaryModel(TwoHead):
    def __init__(self,model_path,revision=REVISION):
        super().__init__(model_path,revision)
        self.aux=nn.ModuleDict({name:nn.Linear(self.encoder.config.hidden_size,len(labels)) for name,labels in ATTR_HEADS.items()})
    def forward_all(self,ids,mask):
        h=self.drop(self.encoder(input_ids=ids,attention_mask=mask).last_hidden_state)
        return self.a(h),self.b(h),{name:head(h) for name,head in self.aux.items()}


def windows(tokenizer,text,max_length=384,stride=96):
    return tokenizer(text,max_length=max_length,stride=stride,truncation=True,return_overflowing_tokens=True,
                     return_offsets_mapping=True,padding='max_length',return_attention_mask=True)


def dataset(docs,split,assignments,tokenizer,mode,seed,include_partial=False,auxiliary=False,boundary_policy='none',border_soft_weight=0.25):
    items=[];counts=collections.Counter()
    for d in docs:
        if assignments[d['id']]!=split:continue
        if split!='train' and not d['curated']:continue
        if not d['curated'] and mode=='curated':continue
        if split=='train' and d['curated'] and boundary_policy=='curated_dedup':
            boost = 1.5 if (mode=='comprehensive' and d['alternatives']) else 1.0
            versions=[(d['versions'][0],False,boost)]
        else:
            versions=[(v,not d['curated'],1/len(d['versions'])) for v in (d['versions'] if split=='train' else d['versions'][:1])]
            if split=='train' and d['curated'] and mode=='comprehensive' and d['alternatives']:
                versions.extend((v,True,.5/len(d['alternatives'])) for v in d['alternatives'])
            if split=='train' and include_partial and d.get('partial') and not d['curated']:
                versions.extend((v,True,.25/len(d['partial'])) for v in d['partial'])
                counts['partial_versions']+=len(d['partial'])
            elif split=='train' and include_partial and d.get('partial') and d['curated'] and boundary_policy!='curated_dedup':
                versions.extend((v,True,.25/len(d['partial'])) for v in d['partial'])
                counts['partial_versions']+=len(d['partial'])
        gold_spans=d['versions'][0]['spans'] if d['curated'] else None
        for v,is_weak,version_weight in versions:
            enc=windows(tokenizer,v['text'])
            for i,offsets in enumerate(enc['offset_mapping']):
                a,b=encode_labels(offsets,v['spans'])
                gold_a,gold_b=encode_labels(offsets,gold_spans) if (d['curated'] and is_weak and gold_spans is not None) else (None,None)
                if is_weak:
                    # Only annotated positive positions supervise weak material.
                    a=[-100 if x==0 else x for x in a];b=[-100 if x==0 else x for x in b]
                wa=[1.0]*len(a);wb=[1.0]*len(b)
                if d['curated'] and is_weak and boundary_policy in ('border_mask','border_soft'):
                    a,b,wa,wb=boundary_mask_and_weights(a,b,gold_a,gold_b,boundary_policy,border_soft_weight)
                if all(x==-100 for x in a+b):continue
                entry=dict(input_ids=enc['input_ids'][i],attention_mask=enc['attention_mask'][i],a=a,b=b,
                           wa=wa,wb=wb,weight=version_weight,document=d['id'])
                if auxiliary:entry['aux']=encode_auxiliary(offsets,v['spans'])
                items.append(entry)
                counts['windows']+=1
        counts['documents']+=1;counts['versions']+=len(versions)
    return items,dict(counts)


def score(gold,pred):
    tp=len(gold&pred);p=tp/len(pred) if pred else 0.;r=tp/len(gold) if gold else 0.
    return dict(gold=len(gold),pred=len(pred),tp=tp,precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0.)


def tolerant_score(gold,pred):
    tp=match_iou(gold,pred);p=tp/len(pred) if pred else 0.;r=tp/len(gold) if gold else 0.
    return dict(gold=len(gold),pred=len(pred),tp=tp,precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0.)


@torch.no_grad()
def predict(model,tokenizer,text,device,postproc_v3=False):
    enc=windows(tokenizer,text); token_scores={}; aux_scores={}
    has_aux=hasattr(model,'forward_all')
    for ids,mask,offs in zip(enc['input_ids'],enc['attention_mask'],enc['offset_mapping']):
        t_ids=torch.tensor([ids],device=device);t_mask=torch.tensor([mask],device=device)
        if has_aux:
            ai,bi,aux=model.forward_all(t_ids,t_mask)
        else:
            ai,bi=model(t_ids,t_mask);aux=None
        for idx,((a,b),x,y) in enumerate(zip(offs,ai[0].float().cpu().numpy(),bi[0].float().cpu().numpy())):
            if b<=a:continue
            key=(a,b)
            if key not in token_scores:token_scores[key]=[x.copy(),y.copy(),1]
            else:token_scores[key][0]+=x;token_scores[key][1]+=y;token_scores[key][2]+=1
            if aux is not None:
                for name,logits in aux.items():
                    akey=(name,a,b);arr=logits[0,idx].float().cpu().numpy()
                    if akey not in aux_scores:aux_scores[akey]=[arr.copy(),1]
                    else:aux_scores[akey][0]+=arr;aux_scores[akey][1]+=1
    offsets=sorted(token_scores)
    out=[]
    for head,labels in [(0,HEAD_A),(1,HEAD_B)]:
        ids=[int(np.argmax(token_scores[k][head]/token_scores[k][2])) for k in offsets]
        for label,a,b in decode_spans(offsets,ids,labels):
            while a<b and text[a].isspace():a+=1
            while b>a and text[b-1].isspace():b-=1
            if a<b:out.append(dict(label=label,span=text[a:b],char_start_index=a,char_end_index=b,begin=a,end=b))
    out=sorted(out,key=lambda x:(x['begin'],x['end']))
    if postproc_v3:
        out=postprocess_spans_v3(text,out)
    if has_aux and aux_scores:
        for s in out:
            props=[]
            heads_for_span=[s['label']]+(['Procedencia'] if s['label']=='Pronunciamento' else [])
            for hname in heads_for_span:
                if hname not in ATTR_HEADS:continue
                rows=[arr/n for (h,a,b),(arr,n) in aux_scores.items() if h==hname and a<s['end'] and b>s['begin']]
                if rows:
                    pred_prop=ATTR_HEADS[hname][int(np.argmax(np.mean(rows,axis=0)))]
                    props.append(pred_prop)
            s['properties']=props
    return out


@torch.no_grad()
def predict_documento(model,text,tokenizer=None,device=None,postproc_v3=True,include_relations=True):
    """Artemis-compatible document prediction contract for pronouncements."""
    if not text or not text.strip():
        return dict(text=text,entities=[],relations=[])
    if device is None:
        device=next(model.parameters()).device
    if tokenizer is None:
        tokenizer=getattr(model,'tokenizer',None) or AutoTokenizer.from_pretrained(MODEL,revision=REVISION,use_fast=True)
    entities=predict(model,tokenizer,text,device,postproc_v3=postproc_v3)
    relations=predict_relations_v3(text,entities) if include_relations else []
    return dict(text=text,entities=entities,relations=relations)


def evaluate(model,tokenizer,docs,split,assignments,device,postproc_v3=False):
    model.eval();gold=set();pred=set();details=[]
    for d in docs:
        if assignments[d['id']]!=split or not d['curated']:continue
        v=d['versions'][0]
        if any(s['label'] not in {'Prazo','Pronunciamento','Encaminhamento'} for s in v['spans']):continue
        output=predict(model,tokenizer,v['text'],device,postproc_v3=postproc_v3)
        rels=predict_relations_v3(v['text'],output)
        g={(d['id'],)+k for k in bio_keys(v['spans'])}
        p={(d['id'],s['label'],s['begin'],s['end']) for s in output}
        gold.update(g);pred.update(p)
        details.append(dict(document=d['id'],metrics=score(g,p),text=v['text'],entities=output,relations=rels))
    result={'overall':score(gold,pred),'by_label':{},'iou50':{'overall':tolerant_score(gold,pred),'by_label':{}},'documents':len(details)}
    for label in ['Pronunciamento','Encaminhamento','Prazo']:
        result['by_label'][label]=score({x for x in gold if x[1]==label},{x for x in pred if x[1]==label})
        result['iou50']['by_label'][label]=tolerant_score({x for x in gold if x[1]==label},{x for x in pred if x[1]==label})
    return result,details


@torch.no_grad()
def evaluate_attributes(model,tokenizer,docs,split,assignments,device):
    """Attribute accuracy on reference spans; not end-to-end extraction."""
    model.eval();stats={name:collections.Counter() for name in ATTR_HEADS}
    for d in docs:
        if assignments[d['id']]!=split or not d['curated']:continue
        v=d['versions'][0]
        if any(s['label'] not in {'Prazo','Pronunciamento','Encaminhamento'} for s in v['spans']):continue
        enc=windows(tokenizer,v['text']);scores={}
        for ids,mask,offs in zip(enc['input_ids'],enc['attention_mask'],enc['offset_mapping']):
            _,_,aux=model.forward_all(torch.tensor([ids],device=device),torch.tensor([mask],device=device))
            for i,(a,b) in enumerate(offs):
                if b<=a:continue
                for name,logits in aux.items():
                    key=(name,a,b);arr=logits[0,i].float().cpu().numpy()
                    if key not in scores:scores[key]=[arr.copy(),1]
                    else:scores[key][0]+=arr;scores[key][1]+=1
        for span in v['spans']:
            for name,labels in ATTR_HEADS.items():
                if span['label']!=('Pronunciamento' if name=='Procedencia' else name):continue
                choices=set(span.get('properties',[])) & set(labels)
                if len(choices)!=1:continue
                rows=[arr/n for (h,a,b),(arr,n) in scores.items() if h==name and a<span['end'] and b>span['begin']]
                if not rows:continue
                expected=next(iter(choices));pred=labels[int(np.argmax(np.mean(rows,axis=0)))]
                stats[name]['support']+=1;stats[name]['correct']+=pred==expected
                stats[name]['expected_'+expected]+=1
    return {name:dict(**counter,accuracy=counter['correct']/counter['support'] if counter['support'] else None) for name,counter in stats.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['curated','augmented','comprehensive'],default='curated')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=10);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--model',default=MODEL);p.add_argument('--revision',default=None)
    p.add_argument('--learning-rate',type=float,default=2e-5);p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--class-weight-alpha',type=float,default=0.)
    p.add_argument('--skip-test',action='store_true',help='Não consultar o teste já visto durante experimentos adicionais')
    p.add_argument('--selection',choices=['exact','iou50'],default='exact')
    p.add_argument('--scheduler',choices=['linear','plateau'],default='linear')
    p.add_argument('--patience',type=int,default=3)
    p.add_argument('--inside-weight',type=float,default=1.,help='Peso supervisionado para continuação de span; não é RL')
    p.add_argument('--fold',type=int,default=0,help='Troca o grupo de validação, mantendo o teste original reservado')
    p.add_argument('--source-jsonl',type=Path,help='Versões alternativas enriquecidas, com mesmos documentos')
    p.add_argument('--selection-source-jsonl',type=Path,help='Fonte fixa para elegibilidade e deduplicação das versões')
    p.add_argument('--include-partial',action='store_true')
    p.add_argument('--auxiliary',action='store_true',help='Treina propriedades em cabeças por família')
    p.add_argument('--init-checkpoint',type=Path,help='Pesos de extração previamente treinados')
    p.add_argument('--boundary-policy',choices=['none','border_mask','border_soft','curated_dedup'],default='none')
    p.add_argument('--border-soft-weight',type=float,default=0.25)
    p.add_argument('--head-learning-rate',type=float,default=None,help='Taxa diferenciada para cabeças lineares (a, b, aux)')
    p.add_argument('--aux-loss-weight',type=float,default=0.25)
    p.add_argument('--postproc-v3',action='store_true',help='Aplica pós-processamento v3 de fronteiras na avaliação')
    args=p.parse_args();args.run_dir.mkdir(parents=True,exist_ok=True)
    revision=args.revision or (REVISION if args.model==MODEL else None)
    if args.batch_size<1 or args.learning_rate<=0:raise ValueError('Batch e taxa de aprendizagem devem ser positivos')
    head_lr=args.head_learning_rate if args.head_learning_rate is not None else args.learning_rate
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    source=args.source_jsonl or BASE.parent/'relatorios'/'pronunciamentos_2026-11-06'/'versoes_pronunciamentos.jsonl'
    docs=load_snapshot(BASE.parent/'relatorios'/'pronunciamentos_2026-11-06',source,args.selection_source_jsonl)
    if args.fold and not args.skip_test:raise ValueError('Folds exploratórios exigem --skip-test')
    assignments,groups=split_dataset(docs,fold=args.fold)
    # Save split before loading pretrained weights or training.
    if args.inside_weight<1:raise ValueError('--inside-weight precisa ser >= 1')
    manifest={'assignments':assignments,'groups':groups,'split_seed':42,'fold':args.fold,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'selection_source_sha256':hashlib.sha256(args.selection_source_jsonl.read_bytes()).hexdigest() if args.selection_source_jsonl else None,'mode':args.mode,'seed':args.seed,'model':args.model,'model_revision':revision,'learning_rate':args.learning_rate,'head_learning_rate':head_lr,'aux_loss_weight':args.aux_loss_weight,'boundary_policy':args.boundary_policy,'border_soft_weight':args.border_soft_weight,'postproc_v3':args.postproc_v3,'batch_size':args.batch_size,'class_weight_alpha':args.class_weight_alpha,'inside_weight':args.inside_weight,'epochs':args.epochs,'selection':args.selection,'scheduler':args.scheduler,'patience':args.patience,'skip_test':args.skip_test,'include_partial':args.include_partial,'auxiliary':args.auxiliary,'init_checkpoint_sha256':hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest() if args.init_checkpoint else None}
    (args.run_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=revision,use_fast=True)
    if not tokenizer.is_fast:raise ValueError('Fast tokenizer required')
    items,counts=dataset(docs,'train',assignments,tokenizer,args.mode,args.seed,args.include_partial,args.auxiliary,args.boundary_policy,args.border_soft_weight)
    if not items:raise ValueError('Empty train')
    print('DATA',json.dumps(counts),flush=True)
    model=AuxiliaryModel(args.model,revision) if args.auxiliary else TwoHead(args.model,revision)
    if args.init_checkpoint:
        missing,unexpected=model.load_state_dict(torch.load(args.init_checkpoint,map_location='cpu',weights_only=True),strict=False)
        if unexpected or any(not key.startswith('aux.') for key in missing):raise ValueError(f'Checkpoint incompatível: missing={missing}, unexpected={unexpected}')
    device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
    weights=[]
    for key,labels in [('a',HEAD_A),('b',HEAD_B)]:
        counts_labels=collections.Counter(x for item in items for x in item[key] if x>=0)
        w=class_weights([x for item in items for x in item[key]],len(labels),args.class_weight_alpha)
        weights.append(torch.tensor(w,device=device))
        print('CLASS_WEIGHTS',key,json.dumps({'counts':dict(counts_labels),'weights':w,'alpha':args.class_weight_alpha}),flush=True)
    continuation_weights=[torch.tensor(continuation_multipliers(list(range(len(labels))),labels,args.inside_weight),device=device) for labels in (HEAD_A,HEAD_B)]
    if head_lr != args.learning_rate:
        enc_params=[p for n,p in model.named_parameters() if n.startswith('encoder.')]
        head_params=[p for n,p in model.named_parameters() if not n.startswith('encoder.')]
        optim=torch.optim.AdamW([
            {'params':enc_params,'lr':args.learning_rate,'base_lr':args.learning_rate},
            {'params':head_params,'lr':head_lr,'base_lr':head_lr},
        ],weight_decay=.01)
    else:
        optim=torch.optim.AdamW([{'params':list(model.parameters()),'lr':args.learning_rate,'base_lr':args.learning_rate}],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,mode='max',factor=.5,patience=2,min_lr=1e-6) if args.scheduler=='plateau' else None
    batches=(len(items)+args.batch_size-1)//args.batch_size
    total=max(1,batches*args.epochs);warm=max(1,int(.1*total));step=0
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda');best=-1;stale=0;logs=[]
    for epoch in range(1,args.epochs+1):
        model.train();random.shuffle(items);loss_sum=0
        for start in range(0,len(items),args.batch_size):
            batch=items[start:start+args.batch_size];ids=torch.tensor([x['input_ids'] for x in batch],device=device);mask=torch.tensor([x['attention_mask'] for x in batch],device=device)
            ya=torch.tensor([x['a'] for x in batch],device=device);yb=torch.tensor([x['b'] for x in batch],device=device)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
                if args.auxiliary:a,b,aux=model.forward_all(ids,mask)
                else:a,b=model(ids,mask)
                parts=[]
                for j,item in enumerate(batch):
                    t_wa=torch.tensor(item.get('wa',[1.0]*len(item['a'])),device=device,dtype=torch.float32)
                    t_wb=torch.tensor(item.get('wb',[1.0]*len(item['b'])),device=device,dtype=torch.float32)
                    for out,target,weight,lookup,tok_w in [(a[j],ya[j],weights[0],continuation_weights[0],t_wa),(b[j],yb[j],weights[1],continuation_weights[1],t_wb)]:
                        valid=target!=-100
                        if valid.any():
                            if args.inside_weight==1 and args.boundary_policy!='border_soft':
                                head_loss=nn.functional.cross_entropy(out,target,weight=weight,ignore_index=-100)
                            else:
                                mult=lookup[target.clamp(min=0)]*tok_w
                                raw=nn.functional.cross_entropy(out,target,weight=weight,ignore_index=-100,reduction='none')
                                denom=(mult[valid]*weight[target[valid]]).sum()
                                head_loss=(raw*mult).sum()/denom if denom>0 else raw.mean()
                            parts.append(head_loss*item['weight'])
                    if args.auxiliary:
                        for name,out in aux.items():
                            target=torch.tensor(item['aux'][name],device=device)
                            if (target!=-100).any():parts.append(args.aux_loss_weight*nn.functional.cross_entropy(out[j],target,ignore_index=-100)*item['weight'])
                loss=sum(parts)/len(parts)
            scaler.scale(loss).backward();scaler.unscale_(optim);nn.utils.clip_grad_norm_(model.parameters(),1.)
            step+=1;scale=step/warm if step<=warm else max(0.,(total-step)/(total-warm)) if total>warm else 1.
            if args.scheduler=='linear':
                    for group in optim.param_groups:group['lr']=group.get('base_lr',args.learning_rate)*scale
            scaler.step(optim);scaler.update();loss_sum+=float(loss.detach())
        dev,_=evaluate(model,tokenizer,docs,'dev',assignments,device,postproc_v3=args.postproc_v3)
        dev_post,_=evaluate(model,tokenizer,docs,'dev',assignments,device,postproc_v3=True)
        dev['postproc_v3']=dict(overall=dev_post['overall'],by_label=dev_post['by_label'],iou50=dev_post['iou50'])
        if args.auxiliary:dev['attributes_on_reference_spans']=evaluate_attributes(model,tokenizer,docs,'dev',assignments,device)
        family_scores=dev['by_label'] if args.selection=='exact' else dev['iou50']['by_label']
        macro=sum(x['f1'] for x in family_scores.values())/3
        log=dict(epoch=epoch,loss=loss_sum/max(1,batches),learning_rate=optim.param_groups[0]['lr'],head_learning_rate=optim.param_groups[-1]['lr'],selection_score=macro,dev=dev);logs.append(log)
        print('EPOCH',json.dumps(log,ensure_ascii=False),flush=True)
        (args.run_dir/'training_log.json').write_text(json.dumps(logs,ensure_ascii=False,indent=2),encoding='utf-8')
        # Macro across the three families; test is evaluated only once after selection.
        if scheduler:scheduler.step(macro)
        if macro>best:
            best=macro;stale=0
            torch.save(model.state_dict(),args.run_dir/'best.pt')
            (args.run_dir/'best.json').write_text(json.dumps(dict(epoch=epoch,dev=dev,macro_f1=macro),ensure_ascii=False,indent=2),encoding='utf-8')
        else:
            stale+=1
            if stale>=args.patience:break
    if args.skip_test:
        print('TEST SKIPPED: experimental run after observing test',flush=True);return
    model.load_state_dict(torch.load(args.run_dir/'best.pt',map_location=device,weights_only=True))
    test,detail=evaluate(model,tokenizer,docs,'test',assignments,device,postproc_v3=args.postproc_v3)
    (args.run_dir/'test.json').write_text(json.dumps(test,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.run_dir/'test_predictions.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding='utf-8')
    print('TEST',json.dumps(test,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
