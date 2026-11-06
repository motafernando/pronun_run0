# Escolhas de treino e comparação de modelos

Referências primárias consultadas em 6 de novembro de 2026:

- Repositório dos autores do BERTimbau: https://github.com/neuralmind-ai/portuguese-bert
- Experimento de NER documentado pelos autores: https://raw.githubusercontent.com/neuralmind-ai/portuguese-bert/master/qualifying_exam-portuguese_named_entity_recognition_using_bert_crf.pdf
- Cartões oficiais BERTimbau Base e Large: https://huggingface.co/neuralmind/bert-base-portuguese-cased e https://huggingface.co/neuralmind/bert-large-portuguese-cased
- Cartão RoBERTaLexPT-base: https://huggingface.co/eduagarcia/RoBERTaLexPT-base

O estudo de NER do BERTimbau usa janelas com stride 128, IOB2, até 50 épocas para BERT sem CRF, checkpoint escolhido em validação, batch 16, taxa 5e-5 no encoder e 1e-3 no classificador, aquecimento de 10% e decaimento linear. Esse é um protocolo de outro corpus e outra arquitetura de cabeça, não uma regra de hiperparâmetros para o ULTIMATUM. Os cartões oficiais identificam BERTimbau Base (110M parâmetros) e Large (335M). Não foi encontrada variante oficial RoBERTaLexPT-large; a comparação local usa RoBERTaLexPT-base, BERTimbau Base e BERTimbau Large.

Para comparar arquiteturas neste piloto, as rodadas novas partem dos respectivos pesos pré-treinados, usam os mesmos documentos e grupos, seed 42, batch físico 2 se couber, janela 384 com stride 96, AdamW, taxa 2e-5, weight decay 0,01, warmup linear de 10%, até 16 épocas e parada após cinco épocas sem melhora do macro F1 IoU ≥ 0,50 no desenvolvimento. O máximo de 16 épocas foi escolhido para permitir mais iterações no corpus pequeno; a parada evita usar uma época arbitrária. Um batch menor no Large, se exigido pela memória, será registrado. A validação de um grupo continua pequena e não é teste independente.

Não se aplica automaticamente o peso 0,01 para a classe O do estudo de BERTimbau: nas rodadas locais anteriores já houve muitas predições extras, e reduzir esse peso tende a elevar ainda mais falsos positivos. Taxas separadas para encoder e cabeça e outros valores de classe exigem comparação controlada posterior. A primeira rodada corrigida de RoBERTaLexPT é uma **continuação** do checkpoint antigo e não entra na comparação inicializada do zero.
