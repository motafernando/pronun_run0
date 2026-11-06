"""Experimental CLI: pre-trained weights plus a saved local PoC checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from train import MODEL,REVISION,TwoHead,predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('text',type=Path);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('-o','--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4)
    text=args.text.read_text(encoding='utf-8')
    if not text.strip():raise ValueError('Documento vazio')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,use_fast=True)
    model=TwoHead(MODEL).to(device)
    model.load_state_dict(torch.load(args.checkpoint,map_location=device,weights_only=True));model.eval()
    entities=predict(model,tokenizer,text,device)
    payload=dict(name=args.text.stem,text=text,entities=entities,relations=[],
                 method='modelo experimental independente para pronunciamentos',checkpoint=str(args.checkpoint))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{len(entities)} trechos em {args.output}')


if __name__=='__main__':main()
