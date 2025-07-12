---
title: 'Explicando Transformers, Pt. V: Modelos Autoregressivos'
date: '2025-07-17'
description: ''
---

Antes de entendermos a estrutura de um Transformer, vale a pena entender primeiro como funcionam os modelos autoregressivos. Eles são uma classe de modelos que recebem sequências de dados e geram novas sequências, um elemento por vez. Isso deixará mais simples o entendimento da arquitetura dos Transformers em breve.

## Modelos autoregressivos

Transformers são modelos que realizam transdução de sequências utilizando um processo autoregressivo. Essa característica define como os diferentes componentes do modelo são organizados e combinados ao longo da arquitetura.

Quando um modelo de transdução de sequências é autoregressivo, ele é treinado para modificar a sequência recebida por meio de um processo chamado shift, no qual:

* O primeiro elemento é removido.
* Todos os elementos são deslocados uma posição para trás.
* A última posição é preenchida com um novo elemento gerado pelo modelo.

O modelo gera um novo elemento, que é adicionado ao início da sequência gerada. Em seguida, a sequência pós-shift se torna a nova sequência recebida, e o processo se repete: a cada iteração, um novo elemento é gerado e acrescentado à sequência, até que algum critério de parada seja atingido.

Os critérios de parada podem ser:

1. Definir um número máximo de iterações.
2. Encerrar o algoritmo quando o último elemento gerado for igual a um valor especial, como o token `<eos>`.

O exemplo a seguir mostra como funciona a tradução autoregressiva da palavra "cachorro" em Português para "dog" em Inglês. Os tokens especiais são usados para separar o texto de entrada do texto gerado, formando a sequência `<bos>cachorro<eos><bos>dog<eos>`.

$$
\underbrace{
 \begin{array}{c|cccccccccc}
    & 0              & 1              & 2              & 3              & 4              & 5              & 6              & 7              & 8              & 9              \\
    \hline
  1 & \texttt{<bos>} & \texttt{  c  } & \texttt{  a  } & \texttt{  c  } & \texttt{  h  } & \texttt{  o  } & \texttt{  r  } & \texttt{  r  } & \texttt{  o  } & \texttt{<eos>} \\
  2 & \texttt{  c  } & \texttt{  a  } & \texttt{  c  } & \texttt{  h  } & \texttt{  o  } & \texttt{  r  } & \texttt{  r  } & \texttt{  o  } & \texttt{<eos>} & \texttt{<bos>} \\
  3 & \texttt{  a  } & \texttt{  c  } & \texttt{  h  } & \texttt{  o  } & \texttt{  r  } & \texttt{  r  } & \texttt{  o  } & \texttt{<eos>} & \texttt{<bos>} & \texttt{  d  } \\
  4 & \texttt{  c  } & \texttt{  h  } & \texttt{  o  } & \texttt{  r  } & \texttt{  r  } & \texttt{  o  } & \texttt{<eos>} & \texttt{<bos>} & \texttt{  d  } & \texttt{  o  } \\
  5 & \texttt{  h  } & \texttt{  o  } & \texttt{  r  } & \texttt{  r  } & \texttt{  o  } & \texttt{<eos>} & \texttt{<bos>} & \texttt{  d  } & \texttt{  o  } & \texttt{  g  } \\
 \end{array}
 }_{\text{sequência recebida}}
$$

$$
\downarrow
$$

$$
\underbrace{
    \begin{array}{c|cccccccccc}
        & 0            & 1            & 2            & 3            & 4              & 5              & 6              & 7              & 8              & 9              \\
        \hline
        1 & \texttt{ c } & \texttt{ a } & \texttt{ c } & \texttt{ h } & \texttt{ o }   & \texttt{ r }   & \texttt{ r }   & \texttt{ o }   & \texttt{<eos>} & \texttt{<bos>} \\
        2 & \texttt{ a } & \texttt{ c } & \texttt{ h } & \texttt{ o } & \texttt{ r }   & \texttt{ r }   & \texttt{ o }   & \texttt{<eos>} & \texttt{<bos>} & \texttt{ d }   \\
        3 & \texttt{ c } & \texttt{ h } & \texttt{ o } & \texttt{ r } & \texttt{ r }   & \texttt{ o }   & \texttt{<eos>} & \texttt{<bos>} & \texttt{ d }   & \texttt{ o }   \\
        4 & \texttt{ h } & \texttt{ o } & \texttt{ r } & \texttt{ r } & \texttt{ o }   & \texttt{<eos>} & \texttt{<bos>} & \texttt{ d }   & \texttt{ o }   & \texttt{ g }   \\
        5 & \texttt{ o } & \texttt{ r } & \texttt{ r } & \texttt{ o } & \texttt{<eos>} & \texttt{<bos>} & \texttt{ d }   & \texttt{ o }   & \texttt{ g }   & \texttt{<eos>}
    \end{array}
}_{\text{Sequência após o shift}}
$$

Observe que, no primeiro shift, o resultado é sempre o mesmo: o token `<bos>` que está no início do texto é movido para o final. Esse padrão será útil para o treinamento dos Transformers no futuro.

Além disso, observe que o tamanho da sequência gerada depende apenas da quantidade de shifts realizados, e não do tamanho da sequência recebida. Assim, modelos autoregressivos permitem gerar sequências de qualquer comprimento.

Todos os mecanismos de atenção explicados utilizam todos os elementos da sequência recebida para gerar um novo elemento. Assim, o elemento na posição $i$ da sequência gerada pode depender dos elementos nas posições $i+1$, $i+2$ e seguintes. Durante o treinamento, isso pode introduzir um viés indesejado no modelo.

Esse viés acontece porque a função de perda compara a sequência pós-shift com a sequência gerada. Ou seja, para todos os elementos, exceto o último, a perda é calculada verificando se o i-ésimo elemento da sequência gerada corresponde ao i+1-ésimo elemento da sequência recebida. Assim, se o modelo puder acessar o i+1-ésimo elemento durante a geração, ele sempre irá copiar esse valor.

Esse padrão causa um vazamento de dados. Se não for corrigido, o modelo pode ter dificuldade para gerar corretamente o último elemento da sequência. Por isso, é importante limitar o acesso do modelo aos elementos posteriores durante a geração, permitindo que ele aprenda os padrões de atenção de forma adequada durante o treinamento.

Essa limitação é feita nos Transformers com o uso de uma attention mask, que zera parte dos pesos para garantir que um novo elemento não seja baseado nos valores dos elementos seguintes.

A aplicação da attention mask é feita somando $-\infty$ aos elementos da matriz triangular superior dos pesos de atenção. Assim, esses elementos se tornam 0 após a aplicação da função Softmax.

$$
    \text{SDPA-Mask}(Q,K,V,M) = \text{Softmax}\left(\frac{Q^TK}{\sqrt{d_{in}}}+M\right)V
$$

$$
    \underbrace{
    \begin{array}{c|ccccc}
        & 1      & 2      & 3      & \dots  & t-1    & t      \\
    \hline
    1      & 6.32   & -1.12  & 1.32   & \dots  & -2.16  & 1.92   \\
    2      & -1.81  & 0.96   & 0.89   & \dots  & 0.03   & 1.76   \\
    3      & -1.81  & 0.06   & -2.27  & \dots  & 1.01   & -1.32  \\
    \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
    t-1    & -0.11  & 1.55   & -0.18  & \dots  & 0.95   & 0.95   \\
    t      & 1.45   & -1.42  & 1.62   & \dots  & 2.06  & -0.23
    \end{array}
    }_{\text{Pesos de atenção}}
$$

$$
    +
$$

$$
    \underbrace{
        \begin{array}{c|ccccc}
            & 1      & 2          & 3        & \dots  & t-1      & t        \\
        \hline
        1        & 0      & -\infty  & -\infty  & \dots  & -\infty  & -\infty  \\
        2        & 0      & 0        & -\infty  & \dots  & -\infty  & -\infty  \\
        3        & 0      & 0        & 0        & \dots  & -\infty  & -\infty  \\
        \vdots   & \vdots & \vdots   & \vdots   & \ddots & \vdots   & \vdots   \\
        t-1      & 0      & 0        & 0        & \dots  & 0        & -\infty  \\
        t        & 0      & 0        & 0        & \dots  & 0        & 0
        \end{array}
    }_{\text{Máscara de atenção}}
$$

$$
    \downarrow
$$

$$
    \underbrace{
        \begin{array}{c|ccccc}
            & 1      & 2      & 3      & \dots  & t-1    & t      \\
        \hline
        1      & 6.32   & -\infty & -\infty    & \dots  & -\infty  & -\infty  \\
        2      & -1.81  & 0.96    & -\infty    & \dots  & -\infty  & -\infty  \\
        3      & -1.81  & 0.06    & -2.27      & \dots  & -\infty  & -\infty  \\
        \vdots & \vdots & \vdots  & \vdots     & \ddots & \vdots   & \vdots   \\
        t-1    & -0.11  & 1.55    & -0.18      & \dots  & 0.95     & -\infty  \\
        t      & 1.45   & -1.42   & 1.62       & \dots  & 2.06     & -0.23
        \end{array}
    }_{\text{Pesos mascarados}}
$$

Apesar da motivação explicada agora para o seu uso, em alguns casos, esse vazamento não causará problemas. Nesses casos, para compatibilidade, a attention mask aplicada será apenas um tensor nulo.

$$
    \underbrace{
        \begin{array}{c|ccccc}
            & 1      & 2      & 3      & \dots  & t-1    & t      \\
        \hline
        1      & 6.32   & -1.12  & 1.32   & \dots  & -2.16  & 1.92   \\
        2      & -1.81  & 0.96   & 0.89   & \dots  & 0.03   & 1.76   \\
        3      & -1.81  & 0.06   & -2.27  & \dots  & 1.01   & -1.32  \\
        \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
        t-1    & -0.11  & 1.55   & -0.18  & \dots  & 0.95   & 0.95   \\
        t      & 1.45   & -1.42  & 1.62   & \dots  & 2.06  & -0.23
        \end{array}
    }_{\text{Pesos de atenção}}
$$

$$
    +
$$

$$
    \underbrace{
        \begin{array}{c|ccccc}
                   & 1      & 2      & 3      & \dots  & t-1    & t      \\
            \hline
            1      & 0      & 0      & 0      & \dots  & 0      & 0      \\
            2      & 0      & 0      & 0      & \dots  & 0      & 0      \\
            3      & 0      & 0      & 0      & \dots  & 0      & 0      \\
            \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
            t-1    & 0      & 0      & 0      & \dots  & 0      & 0      \\
            t      & 0      & 0      & 0      & \dots  & 0      & 0
        \end{array}
    }_{\text{Matriz nula}}
$$

$$
    \downarrow
$$

$$
    \underbrace{
        \begin{array}{c|ccccc}
            & 1      & 2      & 3      & \dots  & t-1    & t      \\
        \hline
        1      & 6.32   & -1.12  & 1.32   & \dots  & -2.16  & 1.92   \\
        2      & -1.81  & 0.96   & 0.89   & \dots  & 0.03   & 1.76   \\
        3      & -1.81  & 0.06   & -2.27  & \dots  & 1.01   & -1.32  \\
        \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
        t-1    & -0.11  & 1.55   & -0.18  & \dots  & 0.95   & 0.95   \\
        t      & 1.45   & -1.42  & 1.62   & \dots  & 2.06  & -0.23
        \end{array}
    }_{\text{Pesos de atenção}}
$$

Você pode obter a attention mask de um tensor arbitrário da seguinte forma:

```python
def attn_mask_like(size: tuple[int]) -> torch.Tensor:
    mask = torch.ones(size)
    mask = mask.triu(diagonal=1)
    mask = mask.bool()
    return mask
```

## Conclusão

Usando um processo autogressivo, é possível criar modelos capazes de gerar sequências inteiras de elementos de forma iterativa. Esse é o segredo para criar inteligências artificiais capazes de traduzir textos, responder perguntas, e outras aplicações.

Agora temos todas as peças para montar nosso próprio modelo desse tipo. No próximo post, vamos ver como foi feita a arquitetura do primeiro Transformer, como descrito no artigo original.
