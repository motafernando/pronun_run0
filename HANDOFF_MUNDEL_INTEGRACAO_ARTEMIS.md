# Relatório de Handoff e Integração no Artemis — Análise de Pronunciamento (Baseline PoC)

**Data:** 06/11/2026  
**Escopo:** Consolidação do baseline de extração de pronunciamentos, encaminhamentos (ordens) e prazos (`Run 0` e `Run 1`) e instruções de integração no pipeline do Artemis.

---

## 1. O que foi feito

Construímos e avaliamos o primeiro pipeline neural de análise de pronunciamento judicial sobre o projeto `Ultimatum NER Dataset` (INCEpTION), partindo do baseline exploratório por regras até um modelo multi-head baseado em `eduagarcia/RoBERTaLexPT-base`.

### Etapas executadas
1. **Inventário e higienização de fronteiras (`enrichment_v2` / `Run 1`):**
   - Base: 152 documentos candidatos (150 textos únicos, agrupados em 17 processos distintos para evitar vazamento entre treino e validação).
   - 131 documentos com curadoria concluída contendo pronunciamentos (477 trechos: 281 `Encaminhamento`, 106 `Prazo`, 89 `Pronunciamento`, 1 `Despacho`; 268 relações) + 21 documentos com anotações individuais (44 versões, 168 trechos).
   - Como 31,7% dos trechos pareados entre anotadores individuais e curadoria divergem no token exato de início ou fim, aplicamos **projeção da estrutura curada** e **mascaramento de fronteiras divergentes (`border_mask`, `ignore_index = -100`)** nas versões individuais durante o treino.
2. **Arquitetura (`AuxiliaryModel` / `TwoHead` em [`poc.py`](file:///C:/Users/Fernando/Documents/New%20project%203/inception/pronunciamento_poc/poc.py)):**
   - **Cabeça principal BIO (7 classes):** `O`, `B/I-Pronunciamento`, `B/I-Encaminhamento`, `B/I-Prazo` (janelas de 384 subtokens com *stride* de 96).
   - **4 cabeças auxiliares por span (mean-pooling sobre os tokens do trecho):**
     - `Pronunciamento` (subtipo do ato: despacho, decisão, sentença, acórdão);
     - `Procedencia` (procedente, improcedente, parcial, etc.);
     - `Encaminhamento` (intimação, citação, publicação, redistribuição, etc.);
     - `Prazo` (normalizado: `5d`, `8d`, `10d`, `15d`, `30d`, etc.).
3. **Pós-processamento e extração de relações (`Run 1`):**
   - `postprocess_spans_v3`: poda pontuação residual isolada nas bordas (`,;:`) e funde apenas spans colados (` distância == 0 `) da mesma família.
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
  - **Branch `main` (consolidada para integração no Artemis):** código limpo (`poc.py`, `inferir.py`, `train.py`, `enriquecimento.py`), testes unitários e metadados dos dois modelos principais (`run1_e1a_border_mask_s42` e `compare_v2_robertalexpt_base_s42`).
  - **Branch `experimentos-run0`:** histórico completo das 29 runs (`Run 0` + `Run 1`), scripts de ablação e relatórios detalhados.
- **Releases com os arquivos `best.pt` (473 MB):**
  - **Run 1 (Recomendado — `v0.2.0-run1`):** `https://github.com/motafernando/pronun_run0/releases/tag/v0.2.0-run1`
    - Asset: `run1_e1a_border_mask_s42_best.pt` (SHA-256: `dba4a6d9b428d109c348090da5d38153a83269958eb2d93651c10f053ea9a3ea`)
  - **Run 0 (Baseline anterior — `v0.1.0-run0`):** `https://github.com/motafernando/pronun_run0/releases/tag/v0.1.0-run0`
    - Asset: `compare_v2_robertalexpt_base_s42_best.pt` (SHA-256: `1f5571dbf03a1f443712ab306a9294774cb6801064a3351b39c548be873c0f16`)

### Hugging Face Hub (Mesmo repositório dos modelos NER)
- **Repositório:** `CEIA-Ultimatum/ultimatum-ner-models`
  - **Branch `pronun-run0`:** código consolidado + `artefatos/run1_e1a_border_mask_s42/best.pt` + `artefatos/compare_v2_robertalexpt_base_s42/best.pt`.
  - **Branch `pronun-run1`:** matriz completa de experimentos + `artefatos/run1_e1a_border_mask_s42/best.pt`.

---

## 4. Como integrar no Artemis (Passo a Passo)

O módulo [`poc.py`](file:///C:/Users/Fernando/Documents/New%20project%203/inception/pronunciamento_poc/poc.py) já expõe a função `predict_documento`, que recebe o texto bruto da publicação e devolve os spans com offsets de caractere (`begin`, `end`), classe, propriedades previstas, confiança e relações entre trechos, no mesmo padrão de offsets usado pelo NER do Artemis.

### 4.1. Baixar o checkpoint via Hugging Face Hub
```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="CEIA-Ultimatum/ultimatum-ner-models",
    filename="artefatos/run1_e1a_border_mask_s42/best.pt",
    revision="pronun-run0",
)
```

### 4.2. Carregar o modelo e rodar inferência em um documento
```python
import torch
from transformers import AutoTokenizer
from poc import AuxiliaryModel, predict_documento

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

model_name = ckpt.get("model_name", "eduagarcia/RoBERTaLexPT-base")
revision = ckpt.get("hf_revision", None)

tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
model = AuxiliaryModel(
    model_name=model_name,
    label_list=ckpt["label_list"],
    prop_maps=ckpt["prop_maps"],
    revision=revision,
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

texto = "Intime-se a parte autora para manifestação no prazo de 15 dias."

resultado = predict_documento(
    model=model,
    text=texto,
    tokenizer=tokenizer,
    label_list=ckpt["label_list"],
    prop_maps=ckpt["prop_maps"],
    device=device,
    postproc_v3=True,
    include_relations=True,
)
```

### 4.3. Formato de saída de `predict_documento`
```json
{
  "spans": [
    {
      "id": "T1",
      "family": "Encaminhamento",
      "begin": 0,
      "end": 42,
      "text": "Intime-se a parte autora para manifestação",
      "score": 0.9412,
      "properties": {
        "Encaminhamento": "intimacao"
      }
    },
    {
      "id": "T2",
      "family": "Prazo",
      "begin": 55,
      "end": 62,
      "text": "15 dias",
      "score": 0.9885,
      "properties": {
        "Prazo": "15d"
      }
    }
  ],
  "relations": [
    {
      "source_id": "T2",
      "target_id": "T1",
      "label": "estabelecido",
      "source_family": "Prazo",
      "target_family": "Encaminhamento",
      "rule": "prazo_nearest_act"
    }
  ]
}
```

### 4.4. Como acoplar à Ficha do Artemis
1. **Execução em paralelo ou em sequência ao NER v1:** ambos os modelos consomem exatamente a mesma `string` de entrada (`text`) e devolvem offsets `(begin, end)` baseados em caracteres sobre esse texto.
2. **Campos adicionais sugeridos na Ficha Gerada:**
   - **Ordens / Encaminhamentos:** listar `span["text"]` (e subtipo `span["properties"].get("Encaminhamento")`) para cada span da família `Encaminhamento`.
   - **Prazos identificados:** listar `span["text"]` + valor normalizado (`span["properties"].get("Prazo")`) e exibir o vínculo `estabelecido` apontando para a ordem correspondente (`target_id`).
   - **Decisões / Pronunciamentos:** listar os trechos de `Pronunciamento` com destaque visual no documento, mantendo o clique para o trecho de origem.
