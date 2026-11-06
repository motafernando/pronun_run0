# Plano de Experimentos — Run 1 (Pós-Diagnóstico da Run 0)

Este plano estrutura a próxima bateria de experimentos (**Run 1**) a partir das evidências consolidadas na **Run 0**, cujo melhor ponto foi o checkpoint `compare_v2_robertalexpt_base_s42` (**Macro F1 IoU ≥ 0,50 = 0,676**, **Micro F1 IoU = 0,647**, **Micro F1 Exato = 0,176**).

---

## 1. Diagnóstico Técnico da Run 0 (Por que o modelo estacionou em 0,676?)

1. **Conflito de supervisão nas fronteiras (`enrichment_v2`):**
   - Entre os **964 pareamentos seguros** (IoU ≥ 0,50) entre versões de anotadores individuais e curadoria concluída, apenas **658 (68,3%)** possuem início e fim exatamente idênticos.
   - Nos outros **306 pareamentos (31,7%)**, o treino em modo `comprehensive` recebe o mesmo texto com rótulos `B-`, `I-` e `O` conflitantes nos tokens de borda.
   - Consequência observada: continuar treinando por 48 épocas (`continue_v2_robertalexpt_base_48e_lr1e5_s42`) ou varrer learning rate e peso de `I-` (`sweep_v2_*`) reduziu a loss de treino de `0,0225` para `0,0054`, mas degradou a validação (`0,546` a `0,600`).
2. **Subtreinamento das cabeças auxiliares de propriedades (`ATTR_HEADS`):**
   - Com taxa de aprendizagem uniforme (`2e-5`) e parada na época 5, as cabeças lineares recém-inicializadas para `Pronunciamento` (4 classes) e `Procedencia` (3 classes) tiveram baixa convergência (`0/5` em Tipo de Pronunciamento e `2/5` em Procedência sobre spans humanos).
3. **Baixo suporte estatístico da validação de grupo único (`fold=0`):**
   - O grupo de desenvolvimento padrão possui apenas **6 documentos curados e 14 spans** (dos quais todos os 6 Encaminhamentos são `Intimacao` e todos os 3 Prazos são `Implicito`), gerando alta variância e impedindo medir discriminação real entre subtipos.
4. **Ausência de predição ponta a ponta de propriedades e relações na inferência:**
   - `predict()` em `train.py` decodifica apenas as duas cabeças BIO (`HEAD_A` e `HEAD_B`), sem exportar os subtipos preditos por `AuxiliaryModel.aux` nem as relações curadas presentes em `enrichment_v2`.

---

## 2. Matriz de Experimentos da Run 1

### Experimento 1 (E1) — Supervisão de Fronteiras Discordantes
**Objetivo:** Eliminar o ruído de gradiente nas bordas onde os anotadores individuais divergem da curadoria concluída, mantendo o ganho de diversidade das versões adicionais.

- **Base fixa:** `eduagarcia/RoBERTaLexPT-base`, `enrichment_v2`, seed 42, batch 2, janela 384/96, `inside_weight=2.0`, seleção por Macro F1 IoU ≥ 0,50 (`--skip-test`).
- **Variantes controladas (`--boundary-policy`):**
  1. **`E1a_border_mask` (Máscara na diferença simétrica de borda):**
     - Nos documentos com curadoria concluída, para cada versão individual pareada, os subtokens que caem na diferença simétrica entre o span do anotador e o span curado (`span_annot Δ span_gold`) recebem `ignore_index = -100` nas cabeças BIO. O núcleo consensual (`span_annot ∩ span_gold`) e a versão curada principal permanecem com supervisão integral.
  2. **`E1b_border_soft_weight` (Atenuação de peso nos tokens de borda divergente):**
     - Em vez de mascarar totalmente, aplica multiplicador `0,25` na loss de token apenas nas posições de borda divergente das versões individuais.
  3. **`E1c_curated_bounds_dedup` (Fronteira canônica da curadoria com pesos agregados):**
     - Mantém a fronteira exata da curadoria nas cópias enriquecidas e consolida o peso das versões individuais concordantes sem duplicar janelas contraditórias.

**Métrica de sucesso:** Superar simultaneamente o Macro F1 IoU (`> 0,676`) e o Micro F1 Exato (`> 0,176`) sem inflar falsos positivos (`spans preditos <= 20`).

---

### Experimento 2 (E2) — Learning Rate Diferenciada e Decodificação Ponta a Ponta com Atributos
**Objetivo:** Acelerar a convergência das cabeças de propriedades (`Pronunciamento`, `Encaminhamento`, `Prazo`, `Procedencia`) sem desestabilizar o encoder `RoBERTaLexPT-base` e incorporar a predição de propriedades diretamente em `predict()`.

- **Variantes controladas:**
  1. **`E2a_diff_lr` (LR diferenciada encoder vs. cabeças):**
     - Param groups separados no `AdamW`:
       - `encoder`: `lr = 2e-5`, `weight_decay = 0.01`
       - `cabeças BIO (a, b) + cabeças de atributos (aux)`: `lr = 1e-4`, `weight_decay = 0.01`
     - Peso da loss auxiliar ajustável (`--aux-loss-weight`, testando `0.25` vs `0.50`).
  2. **`E2b_postproc_v3` (Pós-processamento determinístico de bordas na inferência):**
     - Ajuste em `decode_spans` / `predict` para:
       - Remover pontuação terminal isolada (`[.,;:]+$`) quando não fizer parte de abreviatura jurídica (`art.`, `inc.`, `fls.`, `n.`).
       - Fundir spans contíguos da mesma família separados apenas por espaço em branco (`gap <= 1`).

**Métrica de sucesso:** Ganho na acurácia de `Tipo de Pronunciamento` e `Procedencia` (tanto sobre spans de referência quanto ponta a ponta sobre spans preditos com IoU ≥ 0,50).

---

### Experimento 3 (E3) — Validação Cruzada por Grupos de Processo (5-Fold Group CV)
**Objetivo:** Substituir a decisão baseada em apenas 6 documentos (14 spans) por uma avaliação agregada em **5 grupos de processos distintos** (`--fold 0, 1, 2, 3, 4`), mantendo o grupo `test` original 100% fechado.

- **Protocolo:**
  - Executar o baseline atual (`compare_v2_robertalexpt_base_s42`) contra a melhor configuração combinada de `E1 + E2` nos 5 folds.
  - Consolidar em um único relatório (`artefatos/run1_cv5/comparacao_cv5.json`):
    - Média ± desvio-padrão do Macro F1 IoU ≥ 0,50 e F1 Exato nos 5 folds;
    - Micro F1 global somando todos os trechos dos 5 grupos de validação;
    - Matriz de confusão agregada para os subtipos de `Pronunciamento`, `Encaminhamento`, `Prazo` e `Procedencia`.

---

### Experimento 4 (E4) — Relações (`estabelecido`, `continuado`) e Adaptador para o Artemis
**Objetivo:** Completar a estrutura exigida pelo consumidor final (Artemis e visualizador brat/PDF).

1. **`E4a_relations` (Extração de Relações entre Spans):**
   - **Baseline estrutural (Regra v3):** Vincular `Prazo` ao `Encaminhamento` ou `Pronunciamento` imediatamente contíguo/envolvente na mesma sentença ou cláusula.
   - **Cabeça bilinear / par de spans (`PairClassifier`):** Usando o mean-pooling dos estados ocultos do `RoBERTaLexPT-base` para cada par `(span_origem, span_destino)` dentro da mesma janela, classificando `{Nenhuma, estabelecido, continuado, referenciado}`.
2. **`E4b_artemis_contract` (Adaptador de Inferência padrão Artemis):**
   - No Artemis (`ultimatum_ner.infer`), o fluxo de consumo carrega o snapshot do modelo via `load_model` e invoca `predict_documento(model, texto)`, que segmenta o documento em janelas sobrepostas, agrega predições e devolve entidades estruturadas.
   - Empacotar `predict_documento(model, texto)` em módulo compatível para que o extrator de pronunciamentos devolva no mesmo payload:
     - `entities`: `label`, `span`, `char_start_index`, `char_end_index`, `properties` (subtipo e procedência preditos pelas cabeças `aux` sobre o span extraído);
     - `relations`: lista de vínculos `{label, source, target}` validados contra os índices de `entities`.

---

## 3. Ordem Recomendada de Execução

| Etapa | Experimento | Tempo Estimado (RTX 5060 Ti) | Entregável |
| :---: | :--- | :---: | :--- |
| **1** | **E1 (`E1a`, `E1b`, `E1c`)** no Fold 0 | ~10 min (3 runs) | `COMPARACAO_FRONTEIRAS_RUN1.md` |
| **2** | **E2 (`E2a_diff_lr`, `E2b_postproc_v3`)** sobre o vencedor do E1 | ~7 min (2 runs) | `COMPARACAO_ATRIBUTOS_RUN1.md` |
| **3** | **E3 (5-Fold Group CV)** comparando Run 0 vs. `E1+E2` | ~25 min (2 × 5 folds) | `VALIDACAO_CRUZADA_5FOLDS.md` |
| **4** | **E4 (`E4a` Relações + `E4b` Adaptador Artemis)** | ~10 min | `inferir.py` atualizado + `artemis_pronunciamento.py` |
