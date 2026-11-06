"""Dataset and two-head token classifier for experimental pronouncements."""
import collections
import hashlib
import json
import random
from pathlib import Path

HEAD_A=['O','B-Pronunciamento','I-Pronunciamento','B-Encaminhamento','I-Encaminhamento']
HEAD_B=['O','B-Prazo','I-Prazo']
ATTR_HEADS={
    'Pronunciamento':['Despacho','Sentenca','DecisaoInterlocutoria','DecisaoDeInstanciaSuperior'],
    # Curadoria concluída também usa estes dois valores nesta família.
    'Encaminhamento':['Intimacao','Notificacao','Citacao','Despacho','DecisaoDeInstanciaSuperior'],
    'Prazo':['Explicito','Implicito','Condicionado'],
    'Procedencia':['ProcedentePoloAtivo','ParcialmenteProcedentePoloAtivo','ImprocedentePoloAtivo'],
}


def encode_auxiliary(offsets,spans):
    result={}
    for head,values in ATTR_HEADS.items():
        target_family='Pronunciamento' if head=='Procedencia' else head
        out=[]
        for a,b in offsets:
            matches=[s for s in spans if b>a and s['label']==target_family and s['begin']<b and s['end']>a]
            choices=set(p for s in matches for p in s.get('properties',[]) if p in values)
            out.append(values.index(next(iter(choices))) if len(matches)==1 and len(choices)==1 else -100)
        result[head]=out
    return result


def class_weights(targets,n_classes,alpha):
    counts=collections.Counter(x for x in targets if x>=0)
    raw=[counts[i]**(-alpha) if counts[i] else 0. for i in range(n_classes)]
    # Normalize weighted contribution across the observed training labels.
    denominator=sum(counts[i]*raw[i] for i in range(n_classes))
    scale=sum(counts.values())/denominator if denominator else 1.
    return [x*scale for x in raw]


def continuation_multipliers(targets,labels,factor):
    if factor<1:raise ValueError('Fator deve ser >= 1')
    return [factor if x>=0 and labels[x].startswith('I-') else 1 for x in targets]


def match_iou(gold,pred,threshold=.5):
    candidates=[]
    gs=sorted(gold);ps=sorted(pred)
    for i,(doc,lab,a,b) in enumerate(gs):
        for j,(doc2,lab2,c,d) in enumerate(ps):
            if doc!=doc2 or lab!=lab2:continue
            overlap=max(0,min(b,d)-max(a,c));union=max(b,d)-min(a,c)
            iou=overlap/union if union else 0.
            if iou>=threshold:candidates.append((iou,i,j))
    used_g=set();used_p=set()
    for _,i,j in sorted(candidates,reverse=True):
        if i not in used_g and j not in used_p:used_g.add(i);used_p.add(j)
    return len(used_g)


def group_documents(docs):
    parent={d['id']:d['id'] for d in docs}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(x,y):
        x,y=find(x),find(y)
        if x!=y:parent[max(x,y)]=min(x,y)
    seen={}
    for d in docs:
        if not d['cnj']:raise ValueError('documento sem número CNJ: '+d['id'])
        for key in [('hash',d['hash'])]+[('cnj',x) for x in d['cnj']]:
            if key in seen:union(d['id'],seen[key])
            else:seen[key]=d['id']
    return {d['id']:find(d['id']) for d in docs}


def encode_labels(offsets,spans):
    outputs=[]
    for labels in (HEAD_A,HEAD_B):
        y=[];previous=None
        for start,end in offsets:
            if end<=start:y.append(-100);previous=None;continue
            matches=[(i,s) for i,s in enumerate(spans) if s['begin']<end and s['end']>start]
            uncertain=any(s['label'] not in {'Pronunciamento','Encaminhamento','Prazo'} for _,s in matches)
            valid=[(i,s) for i,s in matches if 'B-'+s['label'] in labels]
            if uncertain or len(valid)>1:y.append(-100);previous=None;continue
            if not valid:y.append(0);previous=None;continue
            i,s=valid[0];tag=('I-' if previous==i else 'B-')+s['label']
            y.append(labels.index(tag));previous=i
        outputs.append(y)
    return outputs


def decode_spans(offsets, ids, labels):
    out=[];current=None
    for (start,end),idx in zip(offsets,ids):
        if end<=start:continue
        tag=labels[idx];kind=tag[2:] if tag!='O' else None
        if current and (kind!=current[0] or tag.startswith('B-')):
            out.append(tuple(current));current=None
        if kind:
            if current is None:current=[kind,start,end]
            else:current[2]=end
    if current:out.append(tuple(current))
    return out


def load_snapshot(base,source=None,selection_source=None):
    candidates=json.loads((base/'candidatos_treino.json').read_text(encoding='utf-8'))
    versions=[json.loads(x) for x in (source or base/'versoes_pronunciamentos.jsonl').read_text(encoding='utf-8').splitlines()]
    bypath={v['export_path']:v for v in versions}
    selection={v['export_path']:v for v in (json.loads(x) for x in selection_source.read_text(encoding='utf-8').splitlines())} if selection_source else bypath
    if bypath.keys()!=selection.keys() or any(v['text_sha256']!=selection[path]['text_sha256'] for path,v in bypath.items()):
        raise ValueError('Fonte de seleção não corresponde às mesmas versões e textos')
    def selected_signature(v):return selection[v['export_path']]['signature_sha256']
    bydoc=collections.defaultdict(list);partial_by_doc=collections.defaultdict(list)
    for v in versions:
        positive=selection[v['export_path']]['pron_spans']
        if not v['curation'] and v['state']=='COMPLETE' and positive:
            bydoc[v['document']].append(v)
        if ((v['curation'] and v['state']=='CURATION-IN-PROGRESS') or (not v['curation'] and v['state']=='IN-PROGRESS')) and positive:
            partial_by_doc[v['document']].append(v)
    docs=[]
    for c in candidates:
        selected=[bypath[p] for p in c['versions']]
        alternatives=[];seen=set()
        if c['source']=='curation_complete':
            seen={selected_signature(selected[0])}
            for v in sorted(bydoc[c['document']],key=lambda x:x['export_path']):
                if v['text_sha256']!=c['text_sha256']:raise ValueError('Texto divergente: '+c['document'])
                if selected_signature(v) not in seen:
                    alternatives.append(v);seen.add(selected_signature(v))
        seen.update(selected_signature(v) for v in selected+alternatives)
        partial=[]
        for v in sorted(partial_by_doc[c['document']],key=lambda x:x['export_path']):
            if v['text_sha256']!=c['text_sha256']:raise ValueError('Texto parcial divergente: '+c['document'])
            if selected_signature(v) not in seen:
                partial.append(v);seen.add(selected_signature(v))
        docs.append(dict(id=c['document'],hash=c['text_sha256'],cnj=c['cnj_candidates'],curated=c['source']=='curation_complete',versions=selected,alternatives=alternatives,partial=partial))
    return docs


def split_dataset(docs,seed=42,fold=0):
    groups=group_documents(docs);grouped=collections.defaultdict(list)
    for d in docs:grouped[groups[d['id']]].append(d)
    eligible=[g for g,ds in grouped.items() if any(d['curated'] and all(s['label'] in {'Prazo','Pronunciamento','Encaminhamento'} for s in d['versions'][0]['spans']) for d in ds)]
    random.Random(seed).shuffle(eligible)
    # Choose small held-out process groups, never the giant groups by default.
    eligible.sort(key=lambda g:abs(sum(d['curated'] for d in grouped[g])-6))
    if fold<0 or len(eligible)<3 or fold+1>=len(eligible):raise ValueError('Não há grupos elegíveis para o fold pedido')
    test,dev=eligible[0],eligible[1+fold]
    assignments={g:('test' if g==test else 'dev' if g==dev else 'train') for g in grouped}
    return {d['id']:assignments[groups[d['id']]] for d in docs},groups


def bio_keys(spans):
    return {(s['label'],s['begin'],s['end']) for s in spans if s['label'] in {'Prazo','Pronunciamento','Encaminhamento'}}


def boundary_mask_and_weights(a_ann, b_ann, a_gold, b_gold, policy='none', soft_weight=0.25):
    if policy == 'none' or a_gold is None or b_gold is None:
        return list(a_ann), list(b_ann), [1.0]*len(a_ann), [1.0]*len(b_ann)
    if policy == 'border_mask':
        a_out = [-100 if (x >= 0 and g >= 0 and x != g) else x for x, g in zip(a_ann, a_gold)]
        b_out = [-100 if (x >= 0 and g >= 0 and x != g) else x for x, g in zip(b_ann, b_gold)]
        return a_out, b_out, [1.0]*len(a_ann), [1.0]*len(b_ann)
    if policy == 'border_soft':
        wa = [soft_weight if (x >= 0 and g >= 0 and x != g) else 1.0 for x, g in zip(a_ann, a_gold)]
        wb = [soft_weight if (x >= 0 and g >= 0 and x != g) else 1.0 for x, g in zip(b_ann, b_gold)]
        return list(a_ann), list(b_ann), wa, wb
    raise ValueError(f'Política de fronteira desconhecida: {policy}')


def postprocess_spans_v3(text, spans):
    cleaned = []
    for s in sorted(spans, key=lambda x: (x['begin'], x['end'])):
        a, b = s['begin'], s['end']
        while a < b and text[a].isspace():
            a += 1
        while b > a and (text[b - 1].isspace() or text[b - 1] in ',;:'):
            b -= 1
        if a < b:
            item = dict(s)
            item.update(begin=a, end=b, char_start_index=a, char_end_index=b, span=text[a:b])
            cleaned.append(item)
    merged = []
    for s in cleaned:
        if merged and merged[-1]['label'] == s['label'] and s['begin'] == merged[-1]['end']:
            prev = merged[-1]
            prev['end'] = s['end']
            prev['char_end_index'] = s['end']
            prev['span'] = text[prev['begin']:prev['end']]
            continue
        merged.append(s)
    return merged


def predict_relations_v3(text, entities):
    relations = []
    seen = set()
    # 1. Detect interrupted spans separated by a Prazo -> 'continuado'
    for i, first in enumerate(entities):
        if first['label'] not in {'Encaminhamento', 'Pronunciamento'}:
            continue
        for j in range(i + 1, len(entities)):
            second = entities[j]
            if second['label'] != first['label'] or second['begin'] < first['end']:
                continue
            between = [
                (k, e) for k, e in enumerate(entities)
                if e['label'] == 'Prazo' and e['begin'] >= first['begin'] - 5 and e['end'] <= second['end'] + 5
                and not (e['end'] <= first['begin'] or e['begin'] >= second['end'])
            ]
            gap_len = second['begin'] - first['end']
            if between and gap_len <= 60:
                rel_key = ('continuado', i, j)
                if rel_key not in seen:
                    seen.add(rel_key)
                    relations.append(dict(label='continuado', source=i, target=j))
                break

    # 2. Link each Prazo to its governing Encaminhamento or Pronunciamento -> 'estabelecido'
    for p_idx, p in enumerate(entities):
        if p['label'] != 'Prazo':
            continue
        enclosing = []
        nearby = []
        for t_idx, t in enumerate(entities):
            if t['label'] not in {'Encaminhamento', 'Pronunciamento'}:
                continue
            overlap = max(0, min(p['end'], t['end']) - max(p['begin'], t['begin']))
            if overlap > 0:
                # Prefer Encaminhamento when nested inside both Pronunciamento and Encaminhamento
                prio = 0 if t['label'] == 'Encaminhamento' else 1
                enclosing.append((prio, t['end'] - t['begin'], t_idx))
            else:
                if t['end'] <= p['begin']:
                    dist = p['begin'] - t['end']
                    if dist <= 250:
                        prio = 0 if t['label'] == 'Encaminhamento' else 1
                        nearby.append((dist, prio, t_idx))
                elif t['begin'] >= p['end']:
                    dist = t['begin'] - p['end']
                    if dist <= 80:
                        prio = 0 if t['label'] == 'Encaminhamento' else 1
                        nearby.append((dist + 40, prio, t_idx))
        target_idx = None
        if enclosing:
            enclosing.sort()
            target_idx = enclosing[0][2]
        elif nearby:
            nearby.sort()
            target_idx = nearby[0][2]
        if target_idx is not None:
            rel_key = ('estabelecido', p_idx, target_idx)
            if rel_key not in seen:
                seen.add(rel_key)
                relations.append(dict(label='estabelecido', source=p_idx, target=target_idx))

    # 3. Link Encaminhamento to its preceding Pronunciamento -> 'vinculado'
    continuado_targets = {r['target'] for r in relations if r['label'] == 'continuado'}
    for e_idx, e in enumerate(entities):
        if e['label'] != 'Encaminhamento' or e_idx in continuado_targets:
            continue
        preceding_pron = [
            (e['begin'] - p['end'], p_idx)
            for p_idx, p in enumerate(entities)
            if p['label'] == 'Pronunciamento' and p['end'] <= e['begin'] and (e['begin'] - p['end']) <= 250
        ]
        if preceding_pron:
            preceding_pron.sort()
            p_idx = preceding_pron[0][1]
            rel_key = ('vinculado', e_idx, p_idx)
            if rel_key not in seen:
                seen.add(rel_key)
                relations.append(dict(label='vinculado', source=e_idx, target=p_idx))
    return relations


