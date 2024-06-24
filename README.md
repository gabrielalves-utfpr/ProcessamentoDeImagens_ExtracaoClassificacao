# Processamento De Imagens Extração e Classificação

> Projeto da Disciplina de Processamento de Imagens UTFPR-CP:
> https://github.com/gabrielalves-utfpr/ProcessamentoDeImagens_ExtracaoClassificacao

## Integrantes

- Gabriel Alves dos Santos
- Tiago Garcez Ferrari

## Descritor Implementado

**Local Binary Pattern (LBP)**
é um descritor de textura que captura informações sobre a variação local de intensidades em uma imagem.
Ele opera em nível de pixel, comparando os valores dos pixels vizinhos com o valor do pixel central.

- O LBP captura informações locais sobre a textura de uma imagem.
- Para cada vizinho, o LBP atribui um bit (1 ou 0) dependendo se o valor do pixel é maior ou menor que o valor do pixel central.
- Isso gera um padrão binário local para cada região da imagem.
- Os padrões binários locais são então convertidos para uma representação decimal, criando um histograma de frequência.
- O histograma resultante é um vetor de características que descreve a distribuição desses padrões binários locais na imagem.

## Classificadores e Acurácia

- **MLP** -> 89.28%
- **RF** -> 96.42%
- **SVM** -> 87.50%

## Bibliotecas Utilizadas

- tkinter
- matplotlib
- sklearn
- numpy
- skimage
- progress.bar
- cv2
