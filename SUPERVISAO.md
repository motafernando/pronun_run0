# Supervisão do treino de pronunciamentos

O supervisor examina dados e erros para propor melhorias no aprendizado supervisionado. As fontes de referência são o guideline v3, as anotações humanas e sua proveniência. A análise do supervisor pode orientar verificações, revisão de casos, funções de perda, taxa de aprendizagem e seleção de exemplos. Ela deve ser registrada e distinguida da curadoria humana.

## Como será feita a supervisão

1. Examinar exemplos e divergências somente do treino para propor correções de pipeline ou selecionar casos para revisão. Dados do teste não podem virar exemplos de treino, exemplos de prompt ou base de ajuste.
2. Examinar métricas agregadas e erros da validação para escolher checkpoints e decidir ajustes. Reutilizar a mesma validação diversas vezes pode provocar adaptação excessiva a esse processo; por isso, uma futura avaliação precisa de processos ainda não usados para seleção.
3. Registrar, para cada caso, identificador, origem, trecho e rótulo de referência, predição, motivo do desacordo e sugestão. Os textos com dados pessoais e os resultados permanecem em artefatos ignorados pelo git.
4. Tratar a sugestão do supervisor como hipótese, não como anotação humana. Casos cuja interpretação dependa de conhecimento jurídico ou de escolha entre alternativas permitidas pelo v3 seguem para revisão. Não alterar silenciosamente o gold.
5. Só promover correções ao conjunto de treino com regra declarada e evidência, preservando os registros originais. Rótulos produzidos pelo supervisor não servem como referência independente para avaliá-lo.

## Controle de aprendizagem

Além da loss supervisionada, uma métrica de validação pode atuar como sinal de seleção: macro F1 com correspondência por sobreposição mínima de 50%, calculado separadamente para as três famílias. Precisão e revocação são ambas consideradas, portanto marcações em excesso são penalizadas. A métrica não é uma recompensa aprendida nem altera os rótulos.

Uma rodada pode reduzir automaticamente o learning rate quando a métrica de validação estagnar e parar após uma paciência configurada. Essa abordagem é aprendizado supervisionado com controle por validação, não aprendizado por reforço. Um LLM como juiz poderia avaliar relevância e fronteiras de candidatos para priorizar revisão, mas seu parecer teria de ser comparado com referência humana e mantido fora do teste; não há necessidade de enviá-lo para um serviço externo nesta etapa.

Os dados usados nas quatro rodadas iniciais e o teste foram examinados como resultados agregados. Novos ajustes com a mesma validação são exploratórios; `--skip-test` impede consultar repetidamente o teste. A avaliação independente futura precisa reservar novos processos antes de qualquer análise.
