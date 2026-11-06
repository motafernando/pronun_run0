"""Use completed curation as structural truth, preserving safe annotator boundaries.

No INCEpTION writes. Each derived version has exactly the curated families,
properties, and relations. Only boundaries of unambiguous paired spans may vary.
"""
import copy
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
SOURCE=BASE.parent/'relatorios'/'pronunciamentos_2026-11-06'/'versoes_pronunciamentos.jsonl'


def iou(a,b):
    inter=max(0,min(a['end'],b['end'])-max(a['begin'],b['begin']))
    union=max(a['end'],b['end'])-min(a['begin'],b['begin'])
    return inter/union if union else 0


def match_spans(gold,annotation):
    candidates={};reverse={}
    for i,a in enumerate(annotation):
        for j,g in enumerate(gold):
            if a['label']!=g['label'] or a['label'] not in {'Pronunciamento','Encaminhamento','Prazo'}:continue
            score=iou(a,g)
            if score>=.2:
                candidates.setdefault(i,[]).append((j,score));reverse.setdefault(j,[]).append((i,score))
    return {i:pairs[0][0] for i,pairs in candidates.items() if len(pairs)==1 and pairs[0][1]>=.5 and len(reverse[pairs[0][0]])==1}


def enrich_version(gold,annotation):
    if gold['text']!=annotation['text']:raise ValueError('textos diferentes')
    out=copy.deepcopy(annotation);mapping=match_spans(gold['spans'],annotation['spans'])
    inverse={gi:ai for ai,gi in mapping.items()};property_changes=[];projected=[]
    for gi,g in enumerate(gold['spans']):
        s=copy.deepcopy(g)
        if gi in inverse:
            a=annotation['spans'][inverse[gi]]
            if a.get('properties',[])!=g.get('properties',[]):
                property_changes.append(dict(annotation=inverse[gi],curation=gi,
                    before=a.get('properties',[]),after=g.get('properties',[])))
            s['begin'],s['end']=a['begin'],a['end']
            s['text']=out['text'][s['begin']:s['end']]
        projected.append(s)
    out['spans']=projected
    out['relations']=copy.deepcopy(gold.get('relations',[]))
    out['pron_spans']=len(projected)
    out['pron_relations']=len(out['relations'])
    out['all_relations']=annotation.get('all_relations',len(annotation.get('relations',[])))-annotation.get('pron_relations',len(annotation.get('relations',[])))+len(out['relations'])
    for key in ('families','properties','missing_properties','relation_labels','relation_patterns'):
        if key in gold:out[key]=copy.deepcopy(gold[key])
    out['nested_prazos']=sum(a['begin']<=b['begin'] and b['end']<=a['end'] and (a['begin'],a['end'])!=(b['begin'],b['end'])
        for a in projected for b in projected if a['label']!='Prazo' and b['label']=='Prazo')
    sig=sorted((s['label'],s['begin'],s['end'],tuple(s.get('properties',[]))) for s in projected)
    out['signature_sha256']=hashlib.sha256(json.dumps(sig,ensure_ascii=False).encode()).hexdigest()
    famsig=sorted((s['label'],s['begin'],s['end']) for s in projected)
    out['signature_family_sha256']=hashlib.sha256(json.dumps(famsig).encode()).hexdigest()
    out['enrichment']=dict(source=gold.get('export_path'),method='curation structure, annotator boundaries only for mutual unique same-family IoU >= 0.5',
                           matched=len(mapping),original_signature_sha256=annotation.get('signature_sha256'))
    report=dict(matched=len(mapping),unmapped_annotation_spans=len(annotation['spans'])-len(mapping),unmapped_gold_spans=len(gold['spans'])-len(inverse),
        property_changes=property_changes,relations_removed=copy.deepcopy(annotation.get('relations',[])),
        relations_copied=copy.deepcopy(out['relations']),boundary_fallback=len(gold['spans'])-len(inverse))
    return out,report


def main():
    records=[json.loads(x) for x in SOURCE.read_text(encoding='utf-8').splitlines()]
    curated={v['document']:v for v in records if v['curation'] and v['state']=='CURATION-COMPLETE'}
    result=[];reports=[]
    for version in records:
        gold=curated.get(version['document'])
        if version['curation'] or gold is None or version['state'] not in {'COMPLETE','IN-PROGRESS'}:
            result.append(version);continue
        new,rep=enrich_version(gold,version)
        if ([(s['label'],s.get('properties',[])) for s in new['spans']]
                !=[(s['label'],s.get('properties',[])) for s in gold['spans']]
                or new['relations']!=gold.get('relations',[])):
            raise AssertionError('Projeção estrutural incompleta: '+version['export_path'])
        rep.update(document=version['document'],user=version['user'],state=version['state'],curation_source=gold['export_path'],annotation_source=version['export_path'])
        reports.append(rep);result.append(new)
    out=BASE/'artefatos'/'enrichment_v2';out.mkdir(parents=True,exist_ok=True)
    (out/'versoes_pronunciamentos.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in result),encoding='utf-8')
    summary=dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),versions=len(records),enriched_versions=len(reports),
      matched_spans=sum(x['matched'] for x in reports),property_changes=sum(len(x['property_changes']) for x in reports),
      relations_copied=sum(len(x['relations_copied']) for x in reports),relations_removed=sum(len(x['relations_removed']) for x in reports),
      annotation_unmatched=sum(x['unmapped_annotation_spans'] for x in reports),curation_unmatched=sum(x['unmapped_gold_spans'] for x in reports),
      dropped_annotation_spans=sum(x['unmapped_annotation_spans'] for x in reports),gold_boundary_fallbacks=sum(x['boundary_fallback'] for x in reports),details=reports)
    (out/'relatorio.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='details'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
