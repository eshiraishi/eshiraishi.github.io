---
title: '🇧🇷 Explicando Transformers'
date: '2025-07-12'
description: 'Guia completo em português sobre Transformers: da teoria à prática com PyTorch. Aprenda como funcionam os modelos por trás do ChatGPT, Claude e Gemini através de explicações conceituais e implementações funcionais.'
image: '/transformers.png'
---

Em 2017, a Google Brain lançou o artigo "Attention is All you Need", que introduziu para o mundo o Transformer, uma arquitetura para redes neurais para transdução de sequências baseada em atenção que permitiu a criação de modelos que superaram todos os outros modelos anteriores em tradução entre idiomas.

Anos depois, essa arquitetura gerou diversas variações que alcançaram o estado da arte em várias aplicações, em especial na criação de modelos de linguagem, que popularizaram fortemente o uso de IA Generativa para muitas aplicações de processamento de linguagem natural. No momento de escrita, são variações dos Transformers (como os Generative Pretrained Transformers ou GPTs) que estão sendo usadas por trás de inteligências artificiais avançadas como ChatGPT, Claude e Gemini, mostrando o potencial dessa arquitetura e o seu legado na história da inteligência artificial.

Quando eu estava aprendendo como esses modelos funcionavam, encontrei diversos bons recursos explicando a arquitetura. No entanto, senti falta de uma explicação conceitual e prática de ponta a ponta, que permitisse compreender o funcionamento desses modelos a partir de uma única fonte. Quando precisei aprender sobre o tema, tive que recorrer a várias referências simultaneamente para obter uma visão abrangente, o que é especialmente desafiador para quem busca um aprendizado mais prático e deseja criar uma versão funcional do modelo. Além disso, embora existam excelentes materiais em inglês e outros idiomas, praticamente não há recursos em português que abordem Transformers no nível de profundidade que considero ideal. Acredito que isso seja um fator limitante para explicar o assunto no Brasil e em outros países lusófonos.

Por isso, meu objetivo é apresentar uma explicação de ponta a ponta sobre o funcionamento dos Transformers, abordando o contexto em que  surgiram, o funcionamento conceitual de seus principais componentes e a arquitetura desses modelos. Vou ilustrar todos esses conceitos com exemplos práticos e implementações funcionais em PyTorch, de forma que você possa criar um modelo funcional usando apenas a teoria e o código apresentados aqui.

## Ementa

O guia é dividido em 7 partes:

1. [Introdução](/posts/transformers-introduction)
2. [Tokens e embeddings](/posts/transformers-tokens-embeddings)
3. [Atenção](/posts/transformers-attention)
4. [Positional Encoding](/posts/transformers-positional-encoding)
5. [Modelos autoregressivos](/posts/transformers-autoregressive-models)
6. [Juntando tudo](/posts/transformers-architecture)
7. [Treinamento](/posts/transformers-training)

## Pré-requisitos

Infelizmente, se eu não assumir absolutamente nenhum pré-requisito, o conteúdo ficará extenso demais para ser feito de uma vez. Pretendo escrever outros guias explicando esses pré-requisitos no futuro, e vou listar alguns recursos que você pode usar para estudar esses assuntos antes. Porém, para entender tudo que vou explicar aqui, é importante que você compreenda esses tópicos dos conteúdos a seguir:

### Álgebra linear

Vamos trabalhar com conceitos como vetores, matrizes, tensores, produto escalar, produto vetorial e transposição de matrizes. Eu aprendi álgebra linear pelo livro "Álgebra Linear" de J.L. Boldrini na faculdade, mas hoje recomendo buscar conteúdos em português como as aulas da [UNIVESP](https://youtube.com/playlist?list=PLTtZUJqLYbCmGfl498ASK1x7OmomkEB4x&si=tERp4r-Mbc_3Cl0q) ou da [Khan Academy](https://youtube.com/playlist?list=PLxI8Can9yAHdDIbEMgrt1n-FdoQfLu2-t&si=vN8GeZTYzawp-5vo), que são ótimas para revisar ou aprender do zero.

### Python e PyTorch

Você vai precisar conhecer o básico de Python: laços, estruturas de dados, condicionais, funções e um pouco de orientação a objetos (como criar classes, objetos e usar módulos). Todo o código do guia será feito em PyTorch, então é importante entender os conceitos principais do framework, como manipulação de tensores, uso de dispositivos, `autograd` e o submódulo `torch.nn`.

Se você nunca usou PyTorch antes, recomendo começar pelo [tutorial oficial](https://docs.pytorch.org/tutorials/beginner/basics/intro.html), que explica bem os conceitos básicos do framework. Se já tem alguma experiência e quer revisar, o curso aberto de Deep Learning da University of Amsterdam (UvA) tem um [tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.html) que resume de forma prática como programar redes neurais em PyTorch. Para quem prefere materiais em português, os primeiros capítulos da [versão traduzida](https://pt.d2l.ai/d2l-pt-pytorch.pdf) do livro "Dive into Deep Learning" (D2L) também é uma ótima opção, mesmo cobrindo mais assuntos do que o necessário para ler este guia.

Mas, principalmente quando o assunto é manipulação de tensores, muita coisa fará mais sentido na prática do que tentando decorar todos os métodos de uma vez. Se você já entende o básico, pode ir aprendendo os métodos novos conforme eles forem aparecendo ao longo do guia.

## Deep Learning

Como transformers são redes neurais, vamos usar todos os conceitos envolvendo redes feedfoward ou percéptrons multicamada (MLPs), como o uso do gradiente descendente, funções de ativação e backpropagation.

Como sugestão de formas de estudar sobre o tema, existem ótimos conteúdos sobre o tema em inglês. O curso [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning) na coursera é completo e muito acessível, e o capítulo 10 no livro [Introduction to Statistical Learning](https://www.statlearning.com/), que é gratuito e possui exemplos em Python e caso precise, tem explicações sobre vários outros temas de aprendizado de máquina, explica o tema conceitualmente em um nível que trará uma compreensão ampla sobre problemas de modelagem.

Porém, assim como discutido anteriormente, conteúdos disponíveis em inglês tornam o conteúdo menos acessível (em especial quando parte do conteúdo requer uma assinatura). Em português, além do livro "Dive into Deep Learning", sugiro os capítulos 7 e 8 do livro [Aprendizado de Máquina - Uma Abordagem Estatística](https://rafaelizbicki.com/ame/), que cobrem redes neurais (e caso ainda não tenha familiaridade com aprendizado de máquina, sugiro os capítulos anteriores também).

Para uma leitura mais prática, outra sugestão é a Parte II do livro "Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow", embora esse livro cubra muito mais assuntos do que redes neurais e outros frameworks no lugar de PyTorch.

Se quiser se aprofundar mais, os tutoriais [3](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html), [4](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial4/Optimization_and_Initialization.html) e [5](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html) do curso da UvA trazem explicações bem interessantes sobre outros componentes e arquiteturas de redes neurais. Eles ajudam a ampliar a visão sobre como criar e otimizar modelos similares, mostrando melhor o motivo de algumas escolhas na criação dos Transformers.

## Conclusão

Como o guia traz bastante código, você pode acessar tudo de forma separada na biblioteca [language-models](http://github.com/eshiraishi/language-models/). Todo o código das funções e classes apresentadas aqui está organizado lá. O projeto é open source e pode ser instalado como uma biblioteca, embora ainda não esteja disponível no PyPI.

E com isso, concluímos a apresentação. No próximo post, vamos iniciar o guia trazendo uma introdução sobre os Transformers e falando do seu contexto histórico.
