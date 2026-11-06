# Modelos de pronunciamento com a curadoria como fonte de verdade

## Dados e protocolo

Os três modelos foram iniciados dos respectivos pesos pré-treinados e ajustados com a cópia `enrichment_v2`: 138 documentos, 328 versões de treino, inclusive oito versões parciais de documentos já presentes. Famílias, propriedades e relações das versões individuais com curadoria concluída são as da curadoria; só fronteiras pareadas podem variar. O tokenizador produziu 558 janelas de treino para RoBERTaLexPT-base e 734 para cada BERTimbau. Os mesmos 17 grupos e documentos foram usados: 15 grupos para treino, um para desenvolvimento, e o teste histórico ficou fechado.

Configuração comum: seed 42, até 16 épocas, parada após cinco épocas sem melhora, batch 2, janela 384, stride 96, AdamW, learning rate 2e-5, weight decay 0,01, warmup de 10% com decaimento linear, peso 2 para tokens de continuação e escolha do checkpoint por macro F1 IoU ≥ 0,50 das três famílias. O desenvolvimento contém apenas seis documentos e 14 spans de um grupo. A avaliação de atributos usa **spans humanos** e, portanto, não mede extração e classificação ponta a ponta. Nenhum teste histórico foi reaberto.

| Modelo | Épocas executadas | Melhor época | Macro F1 IoU ≥ 0,50 | Micro F1 IoU ≥ 0,50 | Micro F1 exato | Spans preditos |
|---|---:|---:|---:|---:|---:|---:|
| RoBERTaLexPT-base | 10 | 5 | 0,676 | 0,647 | 0,176 | 20 |
| BERTimbau Base | 10 | 5 | 0,572 | 0,415 | 0,151 | 39 |
| BERTimbau Large | 9 | 4 | 0,583 | 0,545 | 0,121 | 19 |

| Propriedade sobre span humano | Suporte | RoBERTaLexPT-base | BERTimbau Base | BERTimbau Large |
|---|---:|---:|---:|---:|
| Tipo de Pronunciamento | 5 | 0/5 | 1/5 | 0/5 |
| Subtipo de Encaminhamento | 6 | 6/6 | 5/6 | 6/6 |
| Tipo de Prazo | 3 | 3/3 | 3/3 | 3/3 |
| Procedência | 5 | 2/5 | 3/5 | 3/5 |

Todos os seis Encaminhamentos humanos desta validação têm o mesmo subtipo (`Intimacao`), e os três Prazos têm `Implicito`. Os acertos nessas linhas não demonstram distinção entre classes. O RoBERTaLexPT-base é o candidato local para a prova de conceito de **localização de trechos** nesta divisão; para tipo e procedência, nenhum checkpoint tem evidência suficiente de confiabilidade. O Large não superou o Base de domínio jurídico na métrica de trechos escolhida, apesar de ter mais parâmetros. Esta é uma comparação exploratória na mesma validação usada para escolher época, não uma estimativa independente nem uma conclusão geral sobre arquiteturas.

Os checkpoints e manifestos estão em `artefatos/compare_v2_*_s42/`; hashes e métricas exatas estão em `artefatos/enrichment_v2/comparacao_modelos.json`. Tempo aproximado entre criação do manifesto e último log: 3,4 min (RoBERTaLexPT-base), 4,3 min (BERTimbau Base) e 9,7 min (BERTimbau Large), nesta GPU RTX 5060 Ti. São tempos observados, não previsão universal. A continuação anterior `aux_curated_truth_v2_s42` partiu de um checkpoint já treinado e não entra na tabela comparativa.

## Literatura e próximos cuidados

O protocolo de NER descrito pelos autores do BERTimbau usa até 50 épocas, seleção em validação, janelas sobrepostas, warmup de 10%, batch 16 e taxas distintas para encoder e classificador. A configuração local acima mantém seleção, warmup e janelas, mas fixa uma taxa conservadora e batch menor para esta GPU e este corpus. Detalhes e fontes primárias: `HIPERPARAMETROS_LITERATURA.md`.

Uma próxima avaliação independente requer novos processos anotados ou treinamento reiniciado com reserva definida antes de ajustar qualquer modelo. Relações curadas estão presentes no corpus, mas não são previstas por estas cabeças. A interface e o resultado pré-processado continuam para a etapa posterior definida no escopo do projeto.

Uma continuação posterior de 48 épocas do melhor RoBERTaLexPT-base, com taxa menor, não superou o checkpoint inicial. Ela está documentada separadamente em `CONTINUACAO_48_EPOCAS.md` e não altera a tabela comparativa das três arquiteturas iniciadas dos pesos pré-treinados.

A comparação pareada original versus `enrichment_v2` está em `COMPARACAO_ENRICHMENT_V2.md`. Quatro variações adicionais de taxa e peso de continuação, todas inicializadas do melhor checkpoint, estão em `VARREDURA_HIPERPARAMETROS_V2.md`; nenhuma superou o ponto de partida nesta validação.
