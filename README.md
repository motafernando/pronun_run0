# Extra??o de Pronunciamentos Judiciais ? Run 0 (Consolidado)

Esta ?rvore (`main`) consolida o melhor modelo da **Run 0** para extra??o de trechos de **Pronunciamento**, **Encaminhamento** e **Prazo** em publica??es judiciais, estruturado para consumo e integra??o futura no pipeline **Artemis**.

> Para o hist?rico completo das 21 execu??es experimentais (varredura de hiperpar?metros, continua??o de 48 ?pocas, ciclos de supervis?o e controles intermedi?rios), consulte a branch [`experimentos-run0`](../../tree/experimentos-run0).

---

## 1. Modelo Selecionado (`compare_v2_robertalexpt_base_s42`)

- **Encoder base:** `eduagarcia/RoBERTaLexPT-base` (revis?o `9dea30875c93dc2d33b35c2338dfc93c86715b9e`)
- **Arquitetura:** Classificador de tokens de duas cabe?as BIO (`HEAD_A`: `O`, `B/I-Pronunciamento`, `B/I-Encaminhamento`; `HEAD_B`: `O`, `B/I-Prazo`) + cabe?as auxiliares por fam?lia (`AuxiliaryModel`)
- **Fonte de supervis?o:** `enrichment_v2` (curadoria conclu?da como fonte estrutural de verdade para fam?lias, propriedades e rela??es; 138 documentos, 328 vers?es de treino, 558 janelas de 384 subtokens com stride 96)
- **Configura??o de treino:** Seed 42, batch 2, learning rate `2e-5`, warmup linear de 10%, weight decay `0,01`, peso `2,0` para tokens `I-`, sele??o por Macro F1 (IoU ? 0,50) em valida??o agrupada por processo (?poca 5 de 10)
- **SHA-256 do checkpoint (`best.pt`):** `1f5571dbf03a1f443712ab306a9294774cb6801064a3351b39c548be873c0f16`

### Desempenho na Valida??o (Desenvolvimento)

| Modelo | Melhor ?poca | Macro F1 (IoU ? 0,50) | Micro F1 (IoU ? 0,50) | Micro F1 Exato | Spans Preditos |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RoBERTaLexPT-base (`enrichment_v2`)** | **5** | **0,676** | **0,647** | **0,176** | **20** |
| RoBERTaLexPT-base (Anota??es Originais) | 7 | 0,571 | 0,541 | 0,162 | 23 |
| BERTimbau Large (`enrichment_v2`) | 4 | 0,583 | 0,545 | 0,121 | 19 |
| BERTimbau Base (`enrichment_v2`) | 5 | 0,572 | 0,415 | 0,151 | 39 |

---

## 2. Onde Obter o Checkpoint (`best.pt`)

1. **GitHub Release (acesso direto via link):**
   - Dispon?vel nos assets da release [`v0.1.0-run0`](../../releases/tag/v0.1.0-run0) (`best.pt`, 473 MB).
2. **Hugging Face Hub (Organiza??o `CEIA-Ultimatum`):**
   - Reposit?rio: `CEIA-Ultimatum/ultimatum-ner-models`, branch `pronun-run0`.

---

## 3. Como Executar a Infer?ncia

```powershell
# 1. Executar testes unit?rios do pipeline
python -m unittest test_poc test_enrichment test_auxiliares -v

# 2. Gerar predi??es em JSON (offsets de caractere validados contra o texto original)
python inferir.py caminho/para/documento.txt --checkpoint artefatos/compare_v2_robertalexpt_base_s42/best.pt -o predicoes.json
```

### Contrato de Sa?da JSON (`inferir.py` / `predict`)
Cada entidade extra?da preserva o recorte literal e seus offsets exatos no documento original:
```json
{
  "name": "documento",
  "text": "<texto original>",
  "entities": [
    {
      "label": "Pronunciamento",
      "span": "<trecho literal>",
      "char_start_index": 0,
      "char_end_index": 120,
      "begin": 0,
      "end": 120
    }
  ],
  "relations": []
}
```

---

## 4. Estrutura do Reposit?rio (`main`)

- `poc.py`: Codifica??o BIO de duas cabe?as, cabe?as auxiliares de propriedades, agrupamento transitivo por hash/CNJ e c?lculo de IoU.
- `train.py`: Defini??o de `TwoHead` e `AuxiliaryModel`, janelas deslizantes (384/96), fun??o `predict` com agrega??o de logits entre janelas sobrepostas e loop de treinamento.
- `inferir.py`: CLI enxuta para infer?ncia em documentos avulsos.
- `enriquecimento.py`: Proje??o da estrutura curada sobre vers?es individuais (`enrichment_v2`).
- `artefatos/compare_v2_robertalexpt_base_s42/`: `manifest.json`, `best.json` e `training_log.json` do modelo selecionado.
- `COMPARACAO_MODELOS_V2.md` e `COMPARACAO_ENRICHMENT_V2.md`: Relat?rios t?cnicos da sele??o de arquitetura e do ganho do enriquecimento.
