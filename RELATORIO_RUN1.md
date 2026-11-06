# Relatório Técnico — Run 1 (Fronteiras, LR Diferenciada, Validação Multi-Fold e Relações)

Este documento consolida os resultados verificados da **Run 1**, executada sobre o corpus enriquecido (`enrichment_v2`, 138 documentos, 328 versões) com `eduagarcia/RoBERTaLexPT-base` (revisão `9dea30875c93dc2d33b35c2338dfc93c86715b9e`), seed 42, batch 2, janela 384 / stride 96 e teste histórico mantido 100% fechado (`--skip-test`).

---

## 1. Experimento 1 (E1) — Supervisão de Fronteiras Discordantes (`Fold 0`)

Na **Run 0**, o modelo treinava sobre todas as versões individuais em modo `comprehensive` mantendo os rótulos BIO das bordas onde o anotador individual divergia da curadoria (31,7% dos 964 spans pareados). A **Run 1** comparou três políticas de tratamento de fronteira:

| Execução | Política de Fronteira (`--boundary-policy`) | Melhor Época | Macro F1 (IoU ≥ 0,50) | Micro F1 (IoU ≥ 0,50) | Micro F1 Exato | Micro F1 Exato (`postproc_v3`) | Spans Preditos (Gold = 14) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `compare_v2_robertalexpt_base_s42` | `none` (Baseline Run 0) | 5 | 0,676 | 0,647 | 0,176 | 0,176 | 20 |
| **`run1_e1a_border_mask_s42`** | **`border_mask` (Máscara na borda divergente)** | **5** | **0,772** | **0,733** | **0,200** | **0,200** | **16** |
| `run1_e1b_border_soft_s42` | `border_soft` (Peso `0,25` na borda divergente) | 5 | 0,716 | 0,688 | **0,250** | **0,258** | 18 |
| `run1_e1c_curated_dedup_s42` | `curated_dedup` (Sem versões alternativas nos curados) | 5 | 0,636 | 0,541 | 0,054 | 0,054 | 23 |

### Desempenho por Família no `Fold 0` (`border_mask` vs. `Run 0`)

| Família | Suporte Gold | IoU ≥ 0,50 F1 (`Run 0`) | IoU ≥ 0,50 F1 (`E1a border_mask`) | F1 Exato (`Run 0`) | F1 Exato (`E1a border_mask`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pronunciamento** | 5 | 0,545 | 0,545 | 0,000 | 0,000 |
| **Encaminhamento** | 6 | 0,625 | **0,769** (+14,4 pp) | 0,125 | **0,308** (+18,3 pp) |
| **Prazo** | 3 | 0,857 | **1,000** (+14,3 pp) | **0,571** | 0,333 |

### Conclusão do E1
- **Eliminar totalmente as versões alternativas (`curated_dedup`, Macro F1 `0,636`) piora o modelo**, provando que as variações de janelas/exemplos adicionais são úteis.
- **Mascarar apenas os subtokens em que a versão individual contradiz o estado BIO da curadoria (`border_mask`) elevou o Macro F1 IoU de `0,676` para `0,772` (+9,6 pontos percentuais)**, o Micro F1 IoU de `0,647` para `0,733` (+8,6 pp), reduziu falsos positivos de 20 para 16 predições e subiu o acerto de `Tipo de Pronunciamento` de `0/5` para `1/5`.
- A variante suave (`border_soft`, peso `0,25`) também superou amplamente a Run 0 (`0,716` Macro IoU) e atingiu o maior F1 exato no Fold 0 (`0,258` com `postproc_v3`).

---

## 2. Experimento 2 (E2) — Learning Rate Diferenciada e Peso das Cabeças Auxiliares (`Fold 0`)

Aplicando `border_mask` no encoder (`2e-5`) e elevando a learning rate das cabeças lineares (`a`, `b`, `aux`) para `1e-4`:

| Execução | LR Encoder | LR Cabeças | Peso Loss Auxiliar | Melhor Época | Macro F1 (IoU ≥ 0,50) | Micro F1 (IoU ≥ 0,50) | Micro F1 Exato (`postproc_v3`) | Spans Preditos |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`run1_e1a_border_mask_s42`** | `2e-5` | `2e-5` | `0,25` | 5 | **0,772** | **0,733** | **0,200** | **16** |
| `run1_e2a_difflr_1e4_aux025_s42` | `2e-5` | `1e-4` | `0,25` | 5 | 0,724 | 0,710 | 0,133 | 17 |
| `run1_e2b_difflr_1e4_aux050_s42` | `2e-5` | `1e-4` | `0,50` | 5 | 0,673 | 0,645 | 0,194 | 17 |

- `E2a` (`0,724`) superou a Run 0 (`0,676`), mas manteve-se abaixo da taxa uniforme `2e-5` com `border_mask` (`0,772`). Aumentar o peso das cabeças auxiliares para `0,50` (`E2b`) competiu com o gradiente de localização BIO.

---

## 3. Experimento 3 (E3) — Validação Cruzada em 4 Grupos de Processo Balanceados (`24 Documentos`, `77 Spans`)

Para superar a limitação estatística do `Fold 0` (6 documentos, 14 spans), a configuração vencedora (`--boundary-policy border_mask`, LR `2e-5`) foi treinada e avaliada nos **4 grupos de validação que contêm todas as 3 famílias**:

| Fold | Docs Curados | Spans Gold | Melhor Época | Macro F1 (IoU ≥ 0,50) | Micro F1 (IoU ≥ 0,50) | Micro F1 Exato (Bruto) | Micro F1 Exato (`postproc_v3`) | Pronunciamento (Attr) | Encaminhamento (Attr) | Prazo (Attr) | Procedência (Attr) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 6 | 14 | 5 | **0,772** | **0,733** | 0,200 | 0,200 | 1/5 (20%) | 6/6 (100%) | 3/3 (100%) | 2/5 (40%) |
| **Fold 2** | 5 | 9 | 8 | **0,756** | **0,750** | 0,083 | 0,083 | 2/4 (50%) | 1/3 (33%) | 2/2 (100%) | — |
| **Fold 3** | 5 | 16 | 12 | **0,644** | **0,634** | 0,293 | **0,342** (+4,9 pp) | 2/6 (33%) | 4/6 (67%) | 3/3 (100%) | 0/1 (0%) |
| **Fold 5** | 8 | 38 | 12 | **0,712** | **0,708** | 0,354 | **0,458** (+10,4 pp) | 2/5 (40%) | 14/19 (74%) | 9/9 (100%) | 3/4 (75%) |
| **Média / Total** | **24** | **77** | — | **0,721 ± 0,049** | **0,706 ± 0,044** | **0,233** | **0,271** | **7/20 (35%)** | **25/34 (73,5%)** | **17/17 (100%)** | **5/10 (50%)** |

### Achados da Validação Multi-Fold
1. **Estabilidade acima de 0,70:** A média de Macro F1 (IoU ≥ 0,50) nos 4 processos foi de **0,721** (com 3 dos 4 folds acima de `0,71`).
2. **Efeito real do `postproc_v3` em grupos maiores:** Nos folds com mais spans (`Fold 3` e `Fold 5`), a limpeza de pontuação terminal residual (`postprocess_spans_v3`) elevou o **F1 Exato** de `0,293 -> 0,342` (+4,9 pp) no Fold 3 e de `0,354 -> 0,458` (+10,4 pp) no Fold 5, sem alterar o F1 IoU.
3. **Discriminação real entre subtipos (`Fold 5`):** No `Fold 5`, onde coexistem `Intimacao` (10) e `Notificacao` (9), além de `Explicito` (8) e `Implicito` (1), o modelo acertou **14/19 (73,7%)** em `Encaminhamento`, **9/9 (100%)** em `Prazo` e **3/4 (75%)** em `Procedencia`.

---

## 4. Experimento 4 (E4) — Extração de Relações e Contrato `predict_documento` (Artemis)

Foi implementada em `poc.py` / `train.py` / `inferir.py` a resolução estrutural de relações segundo o Guideline v3 (`predict_relations_v3`) e o contrato `predict_documento(model, text)` compatível com o pipeline do Artemis (incluindo predição ponta a ponta de `properties` e `relations` para cada entidade extraída):

| Relação (`source -> target`) | Suporte Curado (Treino + Dev) | Predições | Acertos (TP) | Precisão | Revocação | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`estabelecido` (`Prazo -> Encaminhamento/Pronunciamento`)** | 98 | 102 | 89 | 0,873 | **0,908** | **0,890** |
| **`vinculado` (`Encaminhamento -> Pronunciamento`)** | 48 | 54 | 32 | 0,593 | 0,667 | **0,627** |
| **`continuado` (`Segmento 1 -> Segmento 2 interrompido por Prazo`)** | 40 | 56 | 28 | 0,500 | 0,700 | **0,583** |

---

## 5. Rastreabilidade e Hashes dos Checkpoints (`Run 1`)

Todos os manifestos, logs e métricas detalhadas estão em `artefatos/enrichment_v2/comparacao_run1.json` e `resumo_run1_enxuta.json`:
- **Novo Melhor Checkpoint (`run1_e1a_border_mask_s42/best.pt`):**
  - SHA-256: `dba4a6d9b428d109c348090da5d38153a83269958eb2d93651c10f053ea9a3ea`
- **Segundo Melhor / Maior F1 Exato no Fold 0 (`run1_e1b_border_soft_s42/best.pt`):**
  - SHA-256: `84ea8479fb3293cd51b32c23b52f018721db32ed70edab2b791dc3346d6528fc`
