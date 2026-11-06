# Relatório de Handoff e Integração no Artemis — Análise de Pronunciamento (Baseline PoC)

**Data:** 06/11/2026  
**Escopo:** Consolidação do baseline de extração de pronunciamentos, encaminhamentos (ordens) e prazos (`Run 0` e `Run 1`) e instruções de integração no pipeline do Artemis.

---

## 1. O que foi feito

Construímos e avaliamos o primeiro pipeline neural de análise de pronunciamento judicial sobre o projeto `Ultimatum NER Dataset` (INCEpTION), partindo do baseline exploratório por regras até um modelo multi-head baseado em `eduagarcia/RoBERTaLexPT-base` (revisão `9dea30875c93dc2d33b35c2338dfc93c86715b9e`).

### Etapas executadas
1. **Inventário e higienização de fronteiras (`enrichment_v2` / `Run 1`):**
   - Base: 152 documentos candidatos (150 textos únicos, agrupados em 17 processos distintos para evitar vazamento entre treino e validação).
   - 131 documentos com curadoria concluída contendo pronunciamentos (477 trechos: 281 `Encaminhamento`, 106 `Prazo`, 89 `Pronunciamento`, 1 `Despacho`; 268 relações) + 21 documentos com anotações individuais (44 versões, 168 trechos).
   - Como 31,7% dos trechos pareados entre anotadores individuais e curadoria divergem no token exato de início ou fim, aplicamos **projeção da estrutura curada** e **mascaramento de fronteiras divergentes (`border_mask`, `ignore_index = -100`)** nas versões individuais durante o treino.
2. **Arquitetura (`AuxiliaryModel` / `TwoHead` em [`train.py`](file:///C:/Users/Fernando/Documents/New%20project%203/inception/pronunciamento_poc/train.py#L25-L43) e [`poc.py`](file:///C:/Users/Fernando/Documents/New%20project%203/inception/pronunciamento_poc/poc.py)):**
   - **Duas cabeças BIO principais (`HEAD_A` para atos e `HEAD_B` para prazos):**
     - `HEAD_A`: `O`, `B/I-Pronunciamento`, `B/I-Encaminhamento`;
     - `HEAD_B`: `O`, `B/I-Prazo` (permite que um `Prazo` fique aninhado dentro de um `Encaminhamento` ou `Pronunciamento`, usando janelas de 384 subtokens com *stride* de 96).
   - **4 cabeças auxiliares por span (`ATTR_HEADS`, mean-pooling sobre os tokens do trecho):**
     - `Pronunciamento` (subtipo do ato: despacho, decisão, sentença, acórdão);
     - `Procedencia` (procedente, improcedente, parcial, etc.);
     - `Encaminhamento` (intimação, citação, publicação, redistribuição, etc.);
     - `Prazo` (normalizado: `5d`, `8d`, `10d`, `15d`, `30d`, etc.).
3. **Pós-processamento e extração de relações (`Run 1`):**
   - `postprocess_spans_v3`: poda pontuação residual isolada nas bordas (`,;:`) e funde apenas spans colados (`distância == 0`) da mesma família.
   - `predict_relations_v3`: reconstrói os vínculos `estabelecido` (`Prazo -> Encaminhamento/Pronunciamento`), `vinculado` (`Encaminhamento -> Pronunciamento`) e `continuado` (trechos partidos por um prazo intercalado).

---

## 2. Como está o desempenho (Resumo de Métricas)

| Abordagem / Modelo | Recorte de Avaliação | Macro F1 (IoU ≥ 0,50) | Micro F1 (IoU ≥ 0,50) | Micro F1 Exato | Observação principal |
|---|---|---:|---:|---:|---|
| **Baseline por regras léxicas** | 14 textos únicos (67 trechos) | — | 0,486 | 0,171 | Quebra em mudanças de linha e cabeçalhos (`Intimado(s)/Citado(s)`). |
| **Run 0 — `RoBERTaLexPT-base` (anotação bruta)** | Fold 0 (1 processo, 6 docs curados) | 0,571 | 0,541 | 0,162 | Controle sem projeção da curadoria. |
| **Run 0 — `RoBERTaLexPT-base` (`enrichment_v2`)** | Fold 0 (1 processo, 6 docs curados) | 0,676 | 0,647 | 0,176 | Superou `BERTimbau Large` (0,583) e `BERTimbau Base` (0,572). |
| **Run 1 — `run1_e1a_border_mask_s42` (Recomendado)** | Fold 0 (1 processo, 6 docs curados) | **0,772** | **0,733** | **0,200** | `Prazo`: 1,000; `Encaminhamento`: 0,769; `Pronunciamento`: 0,545. |
| **Run 1 — Validação Cruzada (4 processos)** | Folds 0, 2, 3 e 5 (24 docs curados, 77 trechos) | **0,721 ± 0,049** | **0,706 ± 0,044** | **0,271** (até 0,458 no Fold 5) | Média entre 4 processos independentes com `postproc_v3`. |
| **Relações (`predict_relations_v3`)** | 131 docs curados | — | — | **0,890** (`estabelecido`) | Vínculo `Prazo -> Ato`: P=0,873, R=0,908 (89/98). |

### O que já resolve vs. Limites atuais
- **Pronto para uso em protótipo/triagem:** localização da região onde há ordem judicial (`Encaminhamento`) e `Prazo` (F1 por sobreposição de `0,72` a `0,77`), normalização do valor do prazo (`100%` nos trechos de referência dos 4 folds) e vínculo entre o prazo e a ordem correspondente (`F1 = 0,890`).
- **Limitações conhecidas (ainda é um baseline PoC):**
  1. **Fronteira exata (`F1 exato` entre `0,20` e `0,46`):** períodos decisórios longos sofrem recorte parcial no início ou fim.
  2. **Subtipos finos (`Tipo de Pronunciamento` 35%, `Procedência` 50%):** o número de exemplos por subclasse no treino ainda é pequeno.

---

## 3. Onde estão o código e os pesos (`best.pt`)

### GitHub (Código, testes e checkpoints em Releases)
- **Repositório:** `https://github.com/motafernando/pronun_run0`
  - **Branch `main` (consolidada para integração no Artemis):** código limpo ([`poc.py`](file:///C:/Users/Fernando/Documents/New%20project%203/inception/pronunciamento_poc/poc.py), [`train.py`](file:///C:/Users/Fernando/Documents/New%20project%203/inception/pronunciamento_poc/train.py), [`inferir.py`](file:///C:/Users/Fernando/Documents/New%20project%203/inception/pronunciamento_poc/inferir.py), [`enriquecimento.py`](file:///C:/Users/Fernando/Documents/New%20project%203/inception/pronunciamento_poc/enriquecimento.py)), testes unitários e metadados dos dois modelos principais (`artefatos/run1_e1a_border_mask_s42/best.json` e `artefatos/compare_v2_robertalexpt_base_s42/best.json`).
  - **Branch `experimentos-run0`:** histórico completo das 29 runs (`Run 0` + `Run 1`), scripts de ablação e relatórios detalhados.
- **Releases com os arquivos `best.pt` (473 MB):**
  - **Run 1 (Recomendado — `v0.2.0-run1`):** `https://github.com/motafernando/pronun_run0/releases/tag/v0.2.0-run1`
    - Assets anexados na release: `best.pt` (SHA-256: `dba4a6d9b428d109c348090da5d38153a83269958eb2d93651c10f053ea9a3ea`), `best.json` e `manifest.json`.
  - **Run 0 (Baseline anterior — `v0.1.0-run0`):** `https://github.com/motafernando/pronun_run0/releases/tag/v0.1.0-run0`
    - Assets anexados na release: `best.pt` (SHA-256: `1f5571dbf03a1f443712ab306a9294774cb6801064a3351b39c548be873c0f16`), `best.json` e `manifest.json`.

### Hugging Face Hub (Mesmo repositório dos modelos NER)
- **Repositório:** `CEIA-Ultimatum/ultimatum-ner-models`
  - **Branch `pronun-run0`:** código consolidado + `artefatos/run1_e1a_border_mask_s42/best.pt` + `artefatos/compare_v2_robertalexpt_base_s42/best.pt`.
  - **Branch `pronun-run1`:** matriz completa de experimentos + `artefatos/run1_e1a_border_mask_s42/best.pt`.

---

## 4. Como integrar no Artemis (Passo a Passo)

O módulo [`train.py`](file:///C:/Users/Fernando/Documents/New%20project%203/inception/pronunciamento_poc/train.py#L147-L157) (e a CLI [`inferir.py`](file:///C:/Users/Fernando/Documents/New%20project%203/inception/pronunciamento_poc/inferir.py)) expõe a função `predict_documento`, que recebe o texto bruto da publicação e devolve `entities` com `char_start_index`, `char_end_index`, `begin`, `end`, `label`, `span`, `properties` e `relations`, seguindo o mesmo contrato de offsets do visualizador/adaptador do Artemis.

### 4.1. Via linha de comando (`inferir.py`)
```powershell
python inferir.py documento.txt --checkpoint artefatos/run1_e1a_border_mask_s42/best.pt -o saida_pronun.json
```

### 4.2. Via Python (baixando o checkpoint do Hugging Face Hub)
```python
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from train import MODEL, REVISION, AuxiliaryModel, TwoHead, predict_documento

# 1. Baixar o checkpoint recomendado (Run 1 - border_mask)
ckpt_path = hf_hub_download(
    repo_id="CEIA-Ultimatum/ultimatum-ner-models",
    filename="artefatos/run1_e1a_border_mask_s42/best.pt",
    revision="pronun-run0",
)

# 2. Carregar tokenizer e pesos
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, use_fast=True)
state = torch.load(ckpt_path, map_location=device, weights_only=True)

has_aux = any(k.startswith("aux.") for k in state)
model = (AuxiliaryModel(MODEL, revision=REVISION) if has_aux else TwoHead(MODEL, revision=REVISION)).to(device)
model.load_state_dict(state)
model.eval()

# 3. Rodar inferência no documento
texto = "Ante o exposto, julgo procedente o pedido. Intime-se a parte autora para manifestação no prazo de 15 dias."
resultado = predict_documento(
    model,
    texto,
    tokenizer=tokenizer,
    device=device,
    postproc_v3=True,
    include_relations=True,
)
```

### 4.3. Formato real de saída de `predict_documento`
```json
{
  "text": "Ante o exposto, julgo procedente o pedido. Intime-se a parte autora para manifestação no prazo de 15 dias.",
  "entities": [
    {
      "label": "Pronunciamento",
      "span": "Ante o exposto, julgo procedente o pedido",
      "char_start_index": 0,
      "char_end_index": 41,
      "begin": 0,
      "end": 41,
      "properties": [
        "DecisaoInterlocutoria",
        "ImprocedentePoloAtivo"
      ]
    },
    {
      "label": "Encaminhamento",
      "span": "Intime-se a parte autora para manifestação",
      "char_start_index": 43,
      "char_end_index": 85,
      "begin": 43,
      "end": 85,
      "properties": [
        "Intimacao"
      ]
    },
    {
      "label": "Prazo",
      "span": "prazo de 15 dias",
      "char_start_index": 89,
      "char_end_index": 105,
      "begin": 89,
      "end": 105,
      "properties": [
        "Explicito"
      ]
    }
  ],
  "relations": [
    {
      "label": "estabelecido",
      "source": 2,
      "target": 1
    },
    {
      "label": "vinculado",
      "source": 1,
      "target": 0
    }
  ]
}
```

### 4.4. Como acoplar à Ficha do Artemis
1. **Compatibilidade direta com `visualizar_predicoes.py`:** `entities` já usa `char_start_index`, `char_end_index`, `label` e `span`, e `relations` já usa índices inteiros `source` e `target` sobre a lista `entities` (exatamente o contrato descrito no `GUIA_APRESENTACAO.md`).
2. **Campos adicionais na Ficha Gerada:**
   - **Decisões (`Pronunciamento`):** exibir o trecho e subtipos previstos em `properties` (`despacho`, `decisao`, `sentenca`, `acordao` e `procedencia`).
   - **Ordens (`Encaminhamento`):** exibir o trecho da determinação judicial e o tipo (`intimacao`, `citacao`, etc.).
   - **Prazos (`Prazo`):** exibir a expressão textual, o valor normalizado em `properties` (ex.: `15d`) e o arco `estabelecido` apontando para a ordem correspondente.
