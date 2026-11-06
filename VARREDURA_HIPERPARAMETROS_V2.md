# Variações a partir do melhor RoBERTaLexPT-base

As quatro execuções começaram do **mesmo checkpoint** RoBERTaLexPT-base da época 5 da comparação de arquiteturas (`artefatos/compare_v2_robertalexpt_base_s42/best.pt`). Usaram o mesmo `enrichment_v2`, 138 documentos e 328 versões, mesma divisão por processos e seed 42, janela 384/stride 96, batch 2, scheduler linear com warmup de 10%, 12 épocas completas e seleção por macro F1 IoU ≥ 0,50 no desenvolvimento. Só variaram taxa e peso da perda nos tokens `I-` de continuação. O teste histórico não foi consultado. Como o checkpoint guarda apenas pesos do modelo, o otimizador e o scheduler foram reiniciados em cada execução.

| Execução | Learning rate | Peso `I-` | Melhor época adicional | Macro F1 IoU ≥ 0,50 | Micro F1 IoU ≥ 0,50 | Micro F1 exato |
|---|---:|---:|---:|---:|---:|---:|
| Checkpoint de partida | — | 2 no treino anterior | 5 da rodada inicial | **0,676** | **0,647** | **0,176** |
| Continuação A | 5e-6 | 2 | 7 | 0,599 | 0,524 | 0,143 |
| Continuação B | 2e-5 | 2 | 5 | 0,546 | 0,526 | 0,158 |
| Continuação C | 2e-5 | 1 | 3 | 0,563 | 0,564 | 0,154 |
| Continuação D | 2e-5 | 3 | 3 | 0,600 | 0,595 | 0,162 |

Nenhuma continuação superou o checkpoint de partida nesta validação de seis documentos e 14 trechos. Entre as continuações, peso 3 deu o maior macro F1 IoU, mas a diferença para a taxa 5e-6 é de aproximadamente 0,001 e não sustenta preferência geral. Três variantes acertaram 1/5 tipos de Pronunciamento sobre spans humanos, contra 0/5 no checkpoint inicial; Procedência permaneceu 2/5 em todas. Esses suportes são pequenos, e os resultados foram obtidos no mesmo grupo usado para selecionar épocas e parâmetros.

A continuação separada de 48 épocas com taxa 1e-5 também não melhorou o checkpoint de partida (`CONTINUACAO_48_EPOCAS.md`). O conjunto de resultados indica que aumentar épocas ou ajustar apenas learning rate/peso de continuação **não foi suficiente nesta divisão**. O próximo experimento deve mudar a supervisão das bordas discordantes entre anotador e curador, mantendo propriedades curadas, e ser avaliado em processos ainda não usados para treino ou seleção. O F1 exato também deve ser interpretado junto com o acordo humano de fronteiras: entre 964 pareamentos seguros, apenas 658 (68,3%) têm as mesmas bordas na versão individual e curada.

Resultados exatos, hashes e verificações de comparabilidade: `artefatos/enrichment_v2/comparacao_hiperparametros.json`, produzido por `comparar_hiperparam_v2.py`. A rodada antiga de enriquecimento parcial continua excluída deste relatório.
