"""Experimental CLI: pre-trained weights plus a saved local PoC checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from train import MODEL,REVISION,TwoHead,AuxiliaryModel,predict_documento


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('text',type=Path);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('-o','--output',type=Path,required=True)
    p.add_argument('--no-postproc',action='store_true')
    args=p.parse_args();torch.set_num_threads(4)
    text=args.text.read_text(encoding='utf-8')
    if not text.strip():raise ValueError('Documento vazio')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,use_fast=True)
    state=torch.load(args.checkpoint,map_location=device,weights_only=True)
    has_aux=any(k.startswith('aux.') for k in state)
    model=(AuxiliaryModel(MODEL) if has_aux else TwoHead(MODEL)).to(device)
    model.load_state_dict(state);model.eval()
    doc_pred=predict_documento(model,text,tokenizer=tokenizer,device=device,postproc_v3=not args.no_postproc,include_relations=True)
    payload=dict(name=args.text.stem,text=text,entities=doc_pred['entities'],relations=doc_pred['relations'],
                 method='modelo experimental independente para pronunciamentos',checkpoint=str(args.checkpoint))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"{len(doc_pred['entities'])} trechos e {len(doc_pred['relations'])} relações em {args.output}")


if __name__=='__main__':main()
