---
title: 'Explicando Transformers, Pt. I: Introdução'
date: '2025-07-13'
description: ''
---

Nesse post, será feita uma introdução aos Transformers, explicando os problemas que estamos tentando resolver com essa arquitetura, introduzindo o contexto histórico que levou à sua criação e os desafios presentes na criação desse tipo de modelo.

## Objetivo

De forma geral, Transformers são utilizados para resolver problemas de transdução de sequências, ou seja, tarefas em que é necessário gerar uma nova sequência de elementos a partir de uma sequência recebida. Essa abordagem permite que diversas aplicações sejam modeladas como transdução de sequências, incluindo tradução automática, geração de texto, sumarização e até mesmo síntese de moléculas.

Formalmente, um modelo de transdução de sequências estabelece uma relação entre uma sequência ordenada $s = \langle s_1, s_2, \cdots, s_n \rangle$, composta por elementos de um conjunto enumerável $S$, e outra sequência $t = \langle t_1, t_2, \cdots, t_m \rangle$, composta por elementos de um conjunto enumerável $T$.

Para facilitar a compreensão do funcionamento dos Transformers, muitos exemplos ao longo deste guia serão apresentados no contexto de uma aplicação específica. Considerando a relevância dos Transformers para o avanço de modelos de linguagem, como os Large Language Models, os exemplos deste guia focarão na construção de um modelo de linguagem.

O objetivo de um modelo de linguagem é prever os próximos trechos de um texto com base nos anteriores, de forma semelhante ao que fazem os corretores automáticos em teclados de smartphones.

Adaptando a definição formal, um modelo de linguagem relaciona um texto $s = \langle s_1, s_2, \cdots, s_n \rangle$, composto por caracteres do vocabulário $S$, a outro texto $s' = \langle s'_1, s'_2, \cdots, s'_m \rangle$, também formado por caracteres do mesmo vocabulário $S$. Vale notar que o fato de $S$ e $T$ serem iguais é uma particularidade de tarefas como modelos de linguagem. Em tradução automática, por exemplo, as sequências podem pertencer a idiomas diferentes, e portanto, os vocabulários também podem ser distintos.

Uma observação: além de criar um modelo de linguagem, outro grande desafio no desenvolvimento de redes neurais é garantir que o modelo seja eficiente do ponto de vista computacional. Embora seja possível implementar um Transformer usando apenas variáveis, listas e laços de repetição, esse tipo de abordagem costuma ser muito lenta para treinar e utilizar na prática, devido ao grande número de operações realizadas de forma sequencial e ineficiente.

Para obter bons desempenhos, é fundamental considerar o paralelismo desde o início da implementação. A maneira mais eficiente de fazer isso é aproveitar a capacidade dos dispositivos modernos (como GPUs e TPUs) de executar operações matemáticas sobre vetores, matrizes e tensores em paralelo. Por isso, ao projetar os algoritmos e a representação dos dados já pensando em estruturas multidimensionais, é possível criar implementações muito mais rápidas e escaláveis, sem aumentar a complexidade do código.

### Um pouco de história

Durante muitos anos, criar modelos eficientes para transdução de sequências foi um desafio em aberto para a comunidade científica. O principal obstáculo era superar limitações presentes nas arquiteturas anteriores aos Transformers. Mesmo após o surgimento dos Transformers, ainda não existem soluções plenamente satisfatórias para todas as aplicações desse tipo.

Antes dos Transformers, as técnicas baseadas em redes neurais recorrentes (RNNs) apresentavam o melhor desempenho em tarefas como tradução automática, sendo a base da arquitetura utilizada pelo Google Tradutor em 2014, conforme descrito no artigo "Sequence to Sequence Learning with Neural Networks". Embora a explicação detalhada dessas arquiteturas esteja fora do escopo deste guia, vale destacar que o processo de treinamento das RNNs enfrentava alguns problemas que motivaram a busca por alternativas mais eficientes:

1. A natureza recursiva das RNNs pode causar o problema de *gradient vanishing*, em que os gradientes calculados durante o backpropagation se tornam tão pequenos que o modelo não consegue aprender de forma eficiente, dificultando a convergência para o mínimo global da função de perda.

2. Como algumas operações precisam ser realizadas de forma sequencial, a inferência nesses modelos pode ser muito lenta, tornando o treinamento e o uso prático inviáveis em muitos casos.

3. Sem o uso de atenção, as RNNs têm dificuldade para capturar o significado de um trecho considerando o contexto completo da frase, o que pode prejudicar a qualidade do texto gerado.

Esses problemas eram críticos para o uso prático dessas arquiteturas, motivando a busca por alternativas que convergissem mais rápido e apresentassem melhor desempenho. Em especial, a necessidade de modelos capazes de interpretar palavras conforme o contexto impulsionou o estudo dos mecanismos de atenção, desenvolvidos justamente para esse fim.

Como em grande parte da pesquisa em redes neurais, muitas decisões e convenções adotadas nos Transformers derivam dos melhores resultados experimentais. No entanto, no caso dos Transformers, várias escolhas também foram feitas para garantir eficiência computacional tanto no treinamento quanto na inferência, além de evitar os problemas de convergência das arquiteturas anteriores.

O grande diferencial dos Transformers é serem baseados exclusivamente em redes neurais feedforward e mecanismos de atenção. Isso permite resolver os problemas mencionados e alcançar desempenho superior em tarefas de transdução de sequências.

Essa abordagem justifica o nome do artigo: do ponto de vista arquitetural, não são necessárias redes neurais recorrentes para criar modelos eficientes — basta o uso de mecanismos de atenção, ou seja, "Attention is All you Need".

## Conclusão

Os Transformers marcaram um avanço importante ao superar as limitações das arquiteturas anteriores em tarefas de transdução de sequências, tanto em desempenho quanto em eficiência computacional. Obter esses resultados usando apenas mecanismos de atenção combinados com redes neurais simples mostrou o quanto essa abordagem pode ser poderosa. Isso abriu caminho para grandes progressos em áreas como tradução automática, geração de texto e inteligência artificial conversacional, entre outros.

Na próxima parte deste guia, vamos ver como os dados são representados nos Transformers, entendendo melhor como esses modelos recebem sequências e explicando conceitos como tokens e embeddings durante o caminho.
