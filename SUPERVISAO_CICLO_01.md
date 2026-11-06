# Supervisão inicial do modelo de pronunciamentos

## Escopo

Revisão qualitativa do checkpoint curado seed 42, escolhido na validação. O caso examinado pertence ao conjunto de treino, não à validação nem ao teste. O objetivo é levantar hipóteses para melhoria e orientar revisão, não medir generalização. O documento original e suas predições são materiais internos.

## Caso cres-16

O texto contém uma decisão seguida de uma ordem com prazo no meio. Na referência curada, o Pronunciamento inclui o conteúdo inteiro de "Defiro" até "prova testemunhal". A predição interrompeu o pronunciamento antes de "prova testemunhal" e classificou essa parte final como Encaminhamento. Conforme o guideline v3, isso sugere uma quebra indevida de fronteira e troca de família; precisa ser confirmado na revisão do caso completo.

O modelo identificou "Intimem-se as partes para que" como Encaminhamento, o prazo de quinze dias como Prazo e o trecho que continua a ordem como outro Encaminhamento. A estrutura fragmentada é compatível em princípio com o esquema do v3, mas o modelo atual não identifica a relação continuado nem estabelecido. Ele incluiu vírgula ou ponto além do limite usado na referência em três trechos. Uma correção de pontuação não deve ser aplicada globalmente sem verificar os critérios de fronteira de cada tipo e seus casos excepcionais.

O modelo ainda classificou uma referência ao retorno dos autos conclusos como Encaminhamento. Ela não consta na curadoria dessa versão. Não é possível decidir pela mera diferença se é falso positivo ou omissão da referência: encaminhar para revisão humana.

## Ações de supervisão

- Priorizar revisão de fragmentação indevida entre Pronunciamento e Encaminhamento.
- Medir fronteiras exatas e sobreposição como critérios complementares. Não transformar automaticamente uma predição parcialmente coincidente em acerto jurídico.
- Investigar uma regra explícita de pontuação à luz do v3, usando apenas o treino e a validação. Manter a comparação com spans originais.
- Tratar diferenças plausíveis de omissão em curadoria como casos para revisão; não converter predição em gold automaticamente.
- Para um treinamento orientado pela revisão, registrar proposta, justificativa, fonte, versão e decisão. Correções que mudam o gold exigem conferência; regras mecânicas de conversão devem ser declaradas e testadas.

Esta revisão foi realizada na an?lise t?cnica no contexto da conversa, sem chamada externa de LLM, e não equivale a adjudicação jurídica humana. Nenhuma alteração foi feita na instância do INCEpTION ou no snapshot de referência.

## Ampliação da revisão e experimento orientado

Uma fila privada foi gerada com oito documentos curados de treino escolhidos em ordem fixa e o checkpoint da rodada com todas as versões individuais. Ela contém 75 ocorrências sem correspondência exata de um lado ou do outro; uma divergência de fronteira pode gerar duas entradas, portanto não são 75 erros independentes.

Casos examinados: em cres-07, um prazo de dez dias foi fragmentado em "prazo de" e "10 (dez) dias"; em cres-16, outro prazo foi separado em "no" e no restante; em cres-23, "Intime-se" saiu fragmentado. Essas observações são de treino, não do teste. Elas fundamentam testar um peso de 2 para tokens de continuação I, mantendo o mesmo dado, encoder, seed e divisão. A hipótese é que dar mais importância à continuidade reduza fragmentação; ela pode piorar separações legítimas ou aumentar falsos positivos, então será medida na validação e preservada como experimento, não aplicada como correção automática.

## Resultado da comparação supervisionada

O experimento comparou, na mesma seed e divisão de cada rodada, peso de continuação 1 contra peso 2. O checkpoint foi escolhido pelo macro F1 das três famílias com IoU de caracteres ≥ 0,50. Em ambas as rodadas a referência foi a curadoria, o teste não foi consultado e o learning rate diminuiu por platô.

| Validação | Tamanho | Macro F1 IoU controle | Macro F1 IoU peso 2 | F1 exato geral controle | F1 exato geral peso 2 |
|---|---|---:|---:|---:|---:|
| Processo A | 6 documentos, 14 spans | 0,497 | 0,609 | 0,174 | 0,333 |
| Processo B | 5 documentos, 9 spans | 0,800 | 0,800 | 0,087 | 0,174 |

Em A, controle e peso 2 escolheram a época 4; em B, ambos escolheram a época 1. A diferença positiva em A não se repete na métrica tolerante em B. A validação é pequena e foi usada para seleção; os números são exploratórios, não uma estimativa independente. O teste das rodadas iniciais já foi observado e permaneceu bloqueado nas rodadas supervisionadas.

Arquivos verificáveis: artefatos/comparacao_supervisao.json e diretórios comprehensive_iou50_s42, supervisor_continuacao_s42, fold2_continuacao_1_s42 e fold2_continuacao_2_s42. Cada manifesto registra dados, agrupamento, seed e parâmetros disponíveis. Os arquivos `best.pt` são checkpoints experimentais, não versões publicadas.
