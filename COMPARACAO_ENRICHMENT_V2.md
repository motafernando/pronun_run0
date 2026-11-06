# Efeito do enrichment corrigido no RoBERTaLexPT-base

Foram treinados dois modelos RoBERTaLexPT-base dos mesmos pesos pré-treinados, com seed 42, divisão por processos e hiperparâmetros idênticos. A seleção de documentos e versões foi fixada pela fonte `enrichment_v2`: **138 documentos e 328 versões** em ambos os lados. A única diferença de entrada é o conteúdo das anotações: originais versus estrutura projetada da curadoria. No lado original, cinco janelas não fornecem rótulos positivos e não contribuem para a loss; no lado enriquecido, esses rótulos são recuperados da curadoria. O teste histórico não foi consultado.

| Fonte | Melhor época | Macro F1 IoU ≥ 0,50 | Micro F1 IoU ≥ 0,50 | Micro F1 exato | Spans preditos |
|---|---:|---:|---:|---:|---:|
| Original | 7 | 0,571 | 0,541 | 0,162 | 23 |
| Enriquecida | 5 | **0,676** | **0,647** | **0,176** | 20 |

Os atributos, medidos sobre os spans humanos de referência, ficaram iguais: tipo de Pronunciamento 0/5, Encaminhamento 6/6, Prazo 3/3 e Procedência 2/5. Todas as seis referências de Encaminhamento são `Intimacao` e as três de Prazo são `Implicito`; os acertos não comprovam distinção entre subtipos. A validação contém só seis documentos e 14 trechos de um único grupo e também foi usada para selecionar a época. O ganho nesta divisão é um sinal exploratório, não uma estimativa independente de generalização nem uma medida isolada do efeito de propriedades versus spans. Relações estão no corpus corrigido, mas não entram na loss do modelo atual.

Manifestos, métricas exatas e hashes são conferidos por `comparar_enrichment_v2.py` e salvos em `artefatos/enrichment_v2/comparacao_enrichment.json`. A comparação antiga baseada em projeção parcial permanece descartada e não integra estes resultados.
