# Enriquecimento pela curadoria — versão corrigida

## Regra de verdade

A curadoria concluída determina **quais spans existem, suas famílias, todas as propriedades e todas as relações**. Para cada versão individual do mesmo documento e texto, o script cria uma cópia derivada dessa estrutura. Quando um span da mesma família tem correspondência única e recíproca com IoU de caracteres ≥ 0,50, a cópia usa apenas o início e o fim marcados pelo anotador; o conteúdo da propriedade continua sendo o da curadoria. Nos demais casos, usa-se também a fronteira curada. Spans extras do anotador não são transportados. As relações são copiadas integralmente da curadoria, usando a ordem canônica dos spans. Nenhuma anotação original ou estado do INCEpTION foi alterado.

## Conferência da cópia

O snapshot contém 713 versões; 485 versões individuais possuem curadoria concluída e foram projetadas. Houve 964 pareamentos de fronteira e 190 conjuntos de propriedades individuais diferentes dos curados foram substituídos. Outros 631 spans curados usaram a própria fronteira por falta de pareamento seguro; 316 spans extras de versões individuais não entraram na cópia. Em todas as 485 versões projetadas, a sequência de famílias e propriedades e a lista de relações são **idênticas às da respectiva curadoria**. Cada texto de span continua igual ao recorte da fronteira usada.

O arquivo derivado e o relatório de pareamento estão em `artefatos/enrichment_v2/`, protegidos pelo gitignore. O código é `enriquecimento.py`. O treino consulta os documentos candidatos e seus grupos conforme o snapshot original; versões individuais de documentos sem curadoria concluída continuam sendo supervisão fraca e não foram artificialmente igualadas a outro documento.

## Modelo e próximos limites

A cabeça de atributos de Encaminhamento foi ampliada para os valores `Despacho` e `DecisaoDeInstanciaSuperior`, presentes na própria curadoria concluída. Essa é uma decisão de **reproduzir o gold existente**; não é uma correção da curadoria nem uma nova interpretação normativa do guideline v3. A estrutura derivada registra propriedades e relações curadas integralmente. O extrator atual ainda não tem cabeça de relações; para aprendê-las, será preciso um experimento específico.

O enriquecimento anterior e a rodada comparativa baseada nele foram descartados por não cumprir a regra de verdade acima. Seus números não devem constar nos próximos relatórios. O teste histórico já observado permanece inadequado para seleção de nova configuração; uma estimativa independente requer dados novos ou novo treino com reserva prévia de processos.

Três treinos novos, iniciados dos pesos pré-treinados de RoBERTaLexPT-base, BERTimbau Base e BERTimbau Large, foram concluídos sobre `enrichment_v2`. Resultados e limitações estão em `COMPARACAO_MODELOS_V2.md` e `artefatos/enrichment_v2/comparacao_modelos.json`.
