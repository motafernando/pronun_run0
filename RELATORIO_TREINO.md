# Primeira prova de conceito de pronunciamentos

## Continuação supervisionada

Foi definido o aproveitamento de as versões individuais inclusive de documentos curados. Foi implementado o modo comprehensive: curadoria continua com peso principal e versões individuais completas distintas do mesmo texto entram como supervisão positiva adicional, dividindo entre si peso total 0,5 por documento. Treino passa a ter 138 documentos, 414 versões e 685 janelas no split inicial. Uma primeira rodada desse regime selecionou checkpoint pelo F1 exato e terminou sem consultar novamente o teste.

Também foi preparado treinamento com até 20 épocas, paciência 5, seleção por macro F1 com IoU ≥ 0,50 na validação e redução automática da taxa de aprendizagem quando a métrica estaciona. O teste previamente observado é excluído dessas novas rodadas. Comparações sobre validação são exploratórias e precisam de avaliação futura em processos independentes.

As rodadas supervisionadas foram concluídas em duas divisões por processo. A comparação sem e com o peso de continuação deu macro F1 de validação 0,497 → 0,609 no primeiro processo e 0,800 → 0,800 no segundo. Os valores foram usados para selecionar checkpoints e cada grupo contém só um processo; não são resultados de teste independente. A supervisão, os erros examinados e o experimento estão em SUPERVISAO_CICLO_01.md. Não houve consulta ao teste nas rodadas novas.

Foram revisados exemplos do conjunto de treino contra o guideline v3 e registrou achados em SUPERVISAO_CICLO_01.md. Com base na fragmentação observada, foi proposta uma rodada controlada com peso 2 para tokens de continuação (I), mantendo os demais parâmetros fixos. Isso é supervisão por reponderação da loss, não aprendizado por reforço nem parecer jurídico validado. A referência humana original permaneceu intacta; propostas ambíguas aguardam revisão.

## Objetivo e estado

Em 6 de novembro de 2026 foram treinadas quatro execuções locais de um extrator independente do Artemis. Ele reconhece trechos das famílias Pronunciamento, Encaminhamento e Prazo com duas cabeças de classificação de tokens sobre RoBERTaLexPT-base. Os arquivos são locais e estão protegidos pelo gitignore. Ainda não produz subtipos, procedência, relações, vínculo com destinatário ou cálculo de vencimento. A integração com a interface foi adiada conforme orientação de a coordena??o do projeto.

## Dados e separação

Snapshot auditado do INCEpTION de 6 de novembro: 152 documentos candidatos e 150 textos únicos. Incluem os documentos usados para treinamento dos anotadores humanos, conforme orientação do responsável. Agrupamento conservador por texto idêntico e por qualquer CNJ citado no documento, com fechamento transitivo: 17 grupos. Uma ligação por CNJ citado não prova que o documento pertence àquele processo; a regra é conservadora e pode juntar processos diferentes.

Foram reservados um grupo para validação e um para teste; cada grupo tem sete documentos candidatos, mas somente seis versões de curadoria concluída entram na avaliação. O treinamento usa os demais 15 grupos. As versões dos anotadores dos grupos reservados não entram no treino. Os conjuntos foram salvos em manifest.json antes do treino e a ausência de grupos compartilhados foi reconferida nas quatro execuções. A unidade efetiva de validação é um processo e a do teste é outro processo; os resultados têm alta incerteza e não representam desempenho geral.

Controle curado: 119 documentos, 119 versões, 220 janelas. Ampliado: 138 documentos, 159 versões, 287 janelas; inclui anotações individuais concluídas de documentos sem curadoria concluída. Nestas versões, apenas os tokens positivos anotados supervisionam o treino; posições não anotadas são mascaradas. Cada versão recebe peso 1/n para n versões do mesmo documento. Ambiguidades de família e sobreposições incompatíveis na mesma cabeça são mascaradas, não corrigidas automaticamente. Documentos curados sem spans de pronunciamento não foram acrescentados como negativos, pois sua completude para a camada não foi estabelecida.

## Treinos e resultados

Base legal RoBERTaLexPT, janela de 384 subtokens, sobreposição de 96, batch 2, learning rate 2e-5, warmup 10%, AdamW, weight decay 0,01, fp16, até dez épocas e parada após três épocas sem melhorar. O checkpoint de cada execução foi escolhido pela média do F1 exato das três famílias na validação; o teste foi executado ao final de cada execução. A versão com pesos usa frequência de tokens na amostra de treino e expoente 0,5; classes sem suporte não recebem peso. Não se deve usar os resultados de teste já observados para escolher hiperparâmetros e continuar chamando este teste de cego.

| Treino | Seed | Melhor época | Macro F1 exato validação | F1 exato teste | F1 IoU ≥ 0,50 teste |
|---|---:|---:|---:|---:|---:|
| Curadoria, loss padrão | 42 | 2 | 0,222 | 0,056 | 0,444 |
| Curadoria + anotações individuais, loss padrão | 42 | 4 | 0,111 | 0,326 | 0,465 |
| Curadoria, pesos suavizados | 42 | 7 | 0,095 | 0,238 | 0,476 |
| Curadoria + anotações individuais, loss padrão | 1337 | 1 | 0,063 | 0,136 | 0,409 |

O teste contém 13 spans humanos nos seis documentos. No mesmo conjunto, o baseline lexical de regras obteve F1 exato 0,091 e F1 IoU ≥ 0,50 de 0,455. O pareamento IoU é guloso um a um por documento e família. Os resultados não provam superioridade de uma configuração: a validação e o teste têm apenas um processo cada, e a diferença entre as duas seeds ampliadas é grande. O teste foi usado para relatar cada rodada; não deve orientar uma nova rodada com a mesma separação sob rótulo de teste independente.

No treino ampliado seed 42, o teste teve precisão 0,233, revocação 0,538 e 30 predições para 13 spans de referência. F1 exato: Pronunciamento 0,500 (dois spans humanos), Encaminhamento 0,231 (sete) e Prazo 0,462 (quatro). São contagens pequenas. Falsos positivos e fronteiras são limitações importantes. No controle curado, apenas um trecho coincidiu exatamente no teste, embora oito tenham IoU ≥ 0,50.

## Reprodutibilidade e verificações

Os checkpoints estão em artefatos/<execução>/best.pt, junto com manifesto, histórico, best.json, test.json, predições e comparison.json. Os checkpoints das três primeiras rodadas usaram refs/main que o cache local indicava como 9dea30875c93dc2d33b35c2338dfc93c86715b9e após a execução; essa revisão não foi registrada no instante exato do carregamento. A quarta rodada já fixou e registrou explicitamente essa revisão. Versões de bibliotecas e hashes das três primeiras saídas estão em artefatos/proveniencia_primeiras_runs.json.

Foram verificados os quatro arquivos de checkpoint, as divisões de grupos, os testes unitários para agrupamento transitivo, sobreposição/aninhamento, máscara de categorias desconhecidas, decodificação e pesos, além de uma inferência real com o checkpoint curado. Isso não valida juridicamente os resultados. Nenhum dado foi alterado no INCEpTION, nem modelo publicado externamente.

## Pendências

- Avaliação com mais processos independentes e análise das divergências e fronteiras, respeitando os casos alternativos permitidos pelo guideline v3.
- Tratar subtipos, procedência e relações em experimentos próprios; não inferir uma relação causal apenas por proximidade sem identificá-la como heurística.
- Confirmar se as regiões não anotadas das curadorias são completas antes de usá-las como negativas definitivas.
- Comparar BERTimbau, outras seeds e divisões somente se forem úteis à continuação da pesquisa; uma nova amostra intocada será necessária para uma estimativa independente.
- Interface e resultado pré-processado foram explicitamente adiados para etapa posterior.

## Enriquecimento posterior

A curadoria concluída passou a ser a fonte integral de entidades, propriedades e relações nas cópias das versões individuais; apenas fronteiras individualmente pareadas podem variar. O processo e sua conferência estão em `SUPERVISAO_CICLO_02.md`. Uma comparação anterior, baseada em alinhamento parcial, foi descartada e não compõe os resultados deste relatório.

Três modelos novos foram treinados sobre a cópia corrigida e comparados com a mesma divisão: RoBERTaLexPT-base, BERTimbau Base e BERTimbau Large. A tabela, os hiperparâmetros e a interpretação estão em `COMPARACAO_MODELOS_V2.md`. Nenhuma dessas rodadas reabriu o teste histórico.

O efeito do enriquecimento foi medido em controle pareado (`COMPARACAO_ENRICHMENT_V2.md`). Uma continuação de 48 épocas (`CONTINUACAO_48_EPOCAS.md`) e quatro variações de taxa/peso de continuação (`VARREDURA_HIPERPARAMETROS_V2.md`) partiram do melhor RoBERTa e não superaram seu checkpoint inicial nessa validação pequena. Os resultados antigos da projeção parcial permanecem descartados.
