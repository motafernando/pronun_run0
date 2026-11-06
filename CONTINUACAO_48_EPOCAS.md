# Continuação longa do RoBERTaLexPT-base

O checkpoint de `artefatos/compare_v2_robertalexpt_base_s42/best.pt`, escolhido na época 5 da rodada iniciada dos pesos pré-treinados, foi usado para inicializar **48 épocas adicionais** sobre o mesmo `enrichment_v2`. AdamW e scheduler foram reiniciados, pois o checkpoint anterior guardava apenas pesos do modelo. A taxa foi reduzida de 2e-5 para 1e-5, com warmup linear de 10%, batch 2 e sem parada antecipada; por isso o experimento testa a combinação de mais treino, taxa menor e reinicialização do otimizador, não apenas a quantidade de épocas. A divisão por processos e a seleção por macro F1 IoU ≥ 0,50 permaneceram as mesmas. O teste histórico não foi consultado.

| Checkpoint | Melhor época na respectiva execução | Macro F1 IoU ≥ 0,50 | Micro F1 IoU ≥ 0,50 | Micro F1 exato |
|---|---:|---:|---:|---:|
| Inicial, antes da continuação | 5 | 0,676 | 0,647 | 0,176 |
| Melhor da continuação de 48 épocas | 3 | 0,596 | 0,537 | 0,146 |

A continuação executou as 48 épocas em cerca de 16 minutos. A última época marcou macro F1 IoU de 0,524. A perda média ponderada de treino foi de 0,0225 na primeira época adicional para 0,0054 na última, enquanto a validação não ultrapassou o checkpoint inicial. A escala dessa loss inclui pesos de versões e cabeças; ela não é comparável à de outra arquitetura sem o mesmo processamento. O padrão é compatível com ajuste excessivo aos exemplos de treino, mas a validação de um grupo, com seis documentos e 14 spans, é pequena demais para quantificar generalização. **Manter o checkpoint inicial como candidato de extração**; o checkpoint da continuação fica salvo como resultado experimental, não como substituto automático.

Há variação humana legítima de fronteira: em 964 pareamentos seguros entre anotador e curadoria no snapshot, 658 (68,3%) têm início e fim exatamente iguais. Essas ocorrências não são documentos independentes. No checkpoint inicial, Pronunciamento tem F1 exato 0 contra IoU 0,545 e Encaminhamento 0,125 contra IoU 0,625 nessa validação. O próximo experimento mais informativo é reduzir o conflito de rótulos nas bordas: usar a fronteira curada como alvo principal e mascarar ou atenuar a loss nos tokens de borda em que as versões individuais discordam, mantendo propriedades curadas como supervisão. Uma comparação pareada com mesmo conjunto e mesmo número de passos deve preceder qualquer conclusão sobre esse ajuste.

Manifesto, histórico e checkpoint da continuação: `artefatos/continue_v2_robertalexpt_base_48e_lr1e5_s42/`. Nenhuma integração à interface foi feita.
