# Extra??o de Pronunciamentos Judiciais ? Run 0 (Experimentos Completos)

Esta branch (`experimentos-run0`) preserva todos os scripts de avalia??o, relat?rios t?cnicos, manifestos e curvas de treinamento (`training_log.json` e `best.json`) das **21 execu??es** realizadas na **Run 0**.

> Para a vers?o enxuta consolidando apenas o melhor modelo para integra??o no Artemis, consulte a branch [`main`](../../tree/main).

---

## Resumo das Baterias de Experimentos

1. **Ciclo Inicial e Supervis?o (`RELATORIO_TREINO.md`, `SUPERVISAO_CICLO_01.md`):**
   - Primeiras execu??es (`curated_s42`, `augmented_s42`, `curated_weighted_s42`, `augmented_s1337`, `comprehensive_s42`, `comprehensive_iou50_s42`) e avalia??o do peso de continua??o (`I-` = 2) em duas divis?es de valida??o (`supervisor_continuacao_s42`, `fold2_continuacao_1_s42`, `fold2_continuacao_2_s42`).
2. **Enriquecimento Guiado pela Curadoria (`SUPERVISAO_CICLO_02.md`, `COMPARACAO_ENRICHMENT_V2.md`):**
   - Proje??o integral de fam?lias, propriedades e rela??es da curadoria conclu?da sobre 485 vers?es individuais (`enrichment_v2`), elevando o Macro F1 (IoU ? 0,50) do RoBERTaLexPT-base de **0,571** (`compare_v2_raw_control_s42`) para **0,676** (`compare_v2_robertalexpt_base_s42`).
3. **Compara??o de Arquiteturas (`COMPARACAO_MODELOS_V2.md`, `HIPERPARAMETROS_LITERATURA.md`):**
   - Compara??o controlada partindo dos pesos pr?-treinados sobre `enrichment_v2`:
     - `compare_v2_robertalexpt_base_s42`: Macro F1 IoU **0,676** (?poca 5)
     - `compare_v2_bertimbau_large_s42`: Macro F1 IoU **0,583** (?poca 4)
     - `compare_v2_bertimbau_base_s42`: Macro F1 IoU **0,572** (?poca 5)
4. **Continua??o Longa e Varredura de Hiperpar?metros (`CONTINUACAO_48_EPOCAS.md`, `VARREDURA_HIPERPARAMETROS_V2.md`):**
   - Continua??o por 48 ?pocas com LR `1e-5` (`continue_v2_robertalexpt_base_48e_lr1e5_s42`, Macro F1 IoU **0,596**) e varredura de 4 combina??es de learning rate (`5e-6`, `2e-5`) e peso `I-` (`1`, `2`, `3`) em `sweep_v2_*` (melhor em **0,600**).
   - Conclus?o diagn?stica: o gargalo atual reside no conflito de supervis?o nas fronteiras discordantes entre anotador individual e curadoria (apenas 68,3% dos 964 pareamentos seguros possuem bordas id?nticas), e n?o em falta de ?pocas.
