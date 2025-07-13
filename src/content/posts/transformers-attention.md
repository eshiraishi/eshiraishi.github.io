---
title: 'Explicando Transformers, Pt. III: Atenção'
date: '2025-07-15'
description: ''
image: '/transformers.png'
---

Neste post, vou explicar o que é atenção e como funcionam os mecanismos apresentados no artigo "Attention is All You Need". No caminho, vamos ver como funcionam os mecanismos Scaled Dot-Product Attention e Multihead Attention, e ao final, também vamos implementar esses mecanismos do zero em PyTorch com foco em eficiência computacional.

## O que é atenção?

No contexto de redes neurais para transdução de sequências, atenção refere-se à capacidade do modelo de considerar o contexto de cada elemento da sequência ao gerar um novo valor para cada elemento de entrada. Ou seja, o modelo pode "prestar atenção" em diferentes partes da sequência para produzir saídas mais precisas e contextualizadas.

Nesse processo, os pesos de atenção formam uma sequência intermediária que indica o quanto cada elemento da entrada deve ser considerado na geração de cada novo elemento. Ao aplicar esses pesos sobre os elementos originais, é gerada uma nova sequência de elementos para representar o conteúdo da sequência recebida, dessa vez representando melhor o contexto de cada elemento.

O uso de mecanismos de atenção dessa forma permite que os modelos compreendam melhor o contexto em que cada trecho está inserido. Isso torna o treinamento mais rápido e eficiente por realizar menos operações, além de ajudar o modelo a lidar com trechos ambíguos ao considerar o contexto completo ao interpretar cada elemento da sequência.

Por exemplo, considere o texto:

> João pensou: O trem estava cheio, mas ele conseguiu uma cadeira livre.

Para simplificar, vamos ignorar os tokens especiais e assumir que cada palavra já foi convertida em um embedding (representado por $\text{Emb}(X)$). Assim, o texto se transforma na seguinte sequência:

$$
    \begin{array}{ccccccccc}
        \text{João } & \text{pensou: } & \text{O }  & \text{trem } & \text{estava } & \text{cheio, } & \text{mas } & \text{ele } & \cdots \\
        \quad         \\
        \downarrow   & \downarrow      & \downarrow & \downarrow   & \downarrow     & \downarrow     & \downarrow  & \downarrow   \\
        \quad         \\
        64           & 23              & 28         & 91           & 12             & 44             & 57          & 72           \\
        \quad         \\
        \downarrow   & \downarrow      & \downarrow & \downarrow   & \downarrow     & \downarrow     & \downarrow  & \downarrow   \\
        \quad         \\
        0.02         & 0.68            & 0.46       & 1.49         & 0.6            & 1.36           & 0.7         & 0.38         \\
        0.12         & 1.05            & 1.7        & 1.59         & 0.88           & 0.3            & 1.85        & 0.94         \\
        0.13         & 1.57            & 0.75       & 0.69         & 0.7            & 1.75           & 0.7         & 0.63         \\
        \cdots       & \cdots          & \cdots     & \cdots       & \cdots         & \cdots         & \cdots      & \cdots      &        \\
        0.8          & 0.34            & 0.84       & 0.34         & 0.67           & 0.53           & 0.49        & 0.5
    \end{array}
$$

Então, um mecanismo de atenção recebe essa sequência e gera outra de mesmo comprimento, onde o embedding que representa a palavra "ele" será composto de embeddings mais próximos do trecho que compõe a palavra "João" do que da palavra "trem":

$$
    \underbrace{
        \begin{array}{ccccccccc}
            \text{João pensou: o trem estava cheio, mas ele...}
        \end{array}
    }_{X}
$$

$$
    \downarrow
$$

$$
    \underbrace{
        \begin{bmatrix}
            0.02   & 0.68   & 0.46   & 1.49   & 0.6    & 1.36   & 0.7    & 0.38   & \cdots & 1.23   \\
            0.12   & 1.05   & 1.7    & 1.59   & 0.88   & 0.3    & 1.85   & 0.94   & \cdots & 0.11   \\
            0.13   & 1.57   & 0.75   & 0.69   & 0.7    & 1.75   & 0.7    & 0.63   & \cdots & 1.04   \\
            \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
            0.8    & 0.34   & 0.84   & 0.34   & 0.67   & 0.53   & 0.49   & 0.5    & \cdots & 0.09
        \end{bmatrix}
    }_{\text{Emb}(X)}
$$

$$
    \downarrow
$$

$$
\underbrace{
    \begin{bmatrix}
        0.02   & 0.68   & 0.46   & 1.49   & 0.6    & 1.36   & 0.7    & 0.05   & \cdots 1.23   \\
        0.12   & 1.05   & 1.7    & 1.59   & 0.88   & 0.3    & 1.85   & 0.08   & \cdots 0.11   \\
        0.13   & 1.57   & 0.75   & 0.69   & 0.7    & 1.75   & 0.7    & 0.16   & \cdots 1.04   \\
        \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots \vdots \\
        0.8    & 0.34   & 0.84   & 0.34   & 0.67   & 0.53   & 0.49   & 0.11   & \cdots 0.09
    \end{bmatrix}
}_{\text{Atn}(X)}
$$

$$
    \text{Atn} \circ \text{Emb } (\text{ele}) \approx \text{Atn} \circ \text{Emb } (\text{João})
$$

### Self-Attention

Todos os mecanismos utilizados na arquitetura dos Transformers funcionam combinando os próprios elementos da sequência recebida entre si para calcular a atenção de cada elemento. Ou seja, a atenção atribuída a cada elemento da sequência é baseada apenas nos elementos da sequência recebida, sem depender de informações externas.

No entanto, nem todos os mecanismos de atenção funcionam dessa forma. Alguns dependem de informações externas, como variáveis adicionais ou outros modelos, para calcular a atenção. Por isso, dizemos que os mecanismos explicados aqui utilizam Self-Attention: a atenção de cada elemento é determinada apenas com base nos próprios elementos da sequência, sem depender de fontes externas.

### Queries, Keys e Values

Para explicar conceitualmente o funcionamento desses mecanismos, é importante destacar que, embora os termos dessas operações sejam numericamente iguais no início, cada um deles recebe um nome abstrato diferente para facilitar a compreensão do papel que desempenham no mecanismo de atenção.

De forma simplificada, um mecanismo de atenção pode ser comparado a um dicionário em Python: chaves (keys ou $K$) são associadas a valores (values ou $V$), e é possível recuperar um valor a partir de uma consulta (query ou $Q$). No contexto do mecanismo de atenção, os papéis de query, key e value são desempenhados pelos próprios elementos da sequência recebida, conforme descrito a seguir:

* Query: O elemento da sequência para o qual queremos representar de outra forma. No exemplo anterior, seria a palavra "ele".
* Keys: Todos os elementos da sequência recebida, que funcionam como possíveis referências para determinar o contexto da query.
* Value: Um novo elemento que representa o significado da query, calculado a partir do elemento original e das keys.
  * Se a query for ambígua, o value será ajustado para refletir melhor seu significado no contexto. No exemplo anterior, para "ele", o value ficará mais próximo do elemento que representa "João".
  * Se não houver ambiguidade, o value pode ser igual ou muito próximo ao embedding original da query. No exemplo anterior, para "João", o value praticamente não muda.

O exemplo a seguir ilustra essa analogia:

```python
sequence = keys = values = [
    'João',
    'pensou',
    'O',
    'trem',
    'estava',
    'cheio',
    'mas',
    'ele',
    ...
]

mechanism = AttentionMechanism(keys, values)
query = 'ele'

assert mechanism[query] == 'João'
```

No entanto, diferentemente do exemplo acima, retornar apenas um elemento das keys para uma query geralmente não é suficiente para capturar corretamente o contexto, especialmente em situações ambíguas. Por exemplo:

> Vi João e Maria ontem. Eles estavam juntos.

Nesse caso, a palavra "Eles" é ambígua, pois pode se referir tanto a "João" quanto a "Maria". Assim, um mecanismo de atenção não atribui todo o peso a um único value, mas distribui a atenção em diferentes proporções entre os possíveis valores, indicando o quanto cada um deve ser considerado:

```python
sequence = keys = values = [
    'Vi',
    'João',
    'e',
    'Maria',
    'ontem.',
    'Eles',
    ...
]

mechanism = AttentionMechanism(keys, values)
query = 'ele'

assert mechanism[query] == {
    'Vi': 0.01,
    'João': 0.45,
    'e': 0.01,
    'Maria': 0.45,
    'ontem.': 0.01,
    'Eles': 0.01,
    ...
}
```

Essas frações são chamadas de pesos de atenção e formam uma sequência normalizada, ou seja, a soma de todos os elementos é igual a 1. Como os pesos são normalizados e, nos Transformers, as sequências de entrada são representadas por embeddings, é possível calcular uma média ponderada desses embeddings usando os pesos de atenção para gerar uma nova representação. Essa média será a saída do mecanismo de atenção.

A normalização garante que a quantidade total de atenção distribuída entre os elementos seja limitada, permitindo que o mecanismo funcione corretamente. Assim, se o peso de um elemento aumenta em relação aos outros, pelo menos um dos demais pesos precisa diminuir proporcionalmente para que a somatória continue sendo igual a 1.

$$
    \text{Atn}(\text{Ele}) = 0.9 \cdot \text{Emb} (\text{João}) + 0.01 \cdot \text{Emb} (\text{pensou: }) + \cdots
$$

$$
    \underbrace{
        \begin{array}{c|cccc}
            & 1 & 2 & \dots & d \\
            \hline
            1  & 0.7 & 0.8 & \dots & 0.3 \\
            2  & 0.3 & 0.2 & \dots & 0.2 \\
            \vdots & \vdots & \vdots & \ddots & \vdots \\
            t  & 0.2 & 0.4 & \dots & 0.1
        \end{array}
    }_{V}
$$

$$
    \times
$$

$$
\underbrace{
    \begin{array}{c|cccc}
        & 1 & 2 & \dots & t \\
        \hline
        1  & 0.5 & 0.5 & \dots & 0.0 \\
        2  & 0.2 & 0.3 & \dots & 0.4 \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        t  & 0.1 & 0.2 & \dots & 0.5
    \end{array}
}_{\text{Pesos de atenção}}
$$

$$
    \downarrow
$$

$$
    \underbrace{
        \begin{array}{c|cccc}
            & 1 & 2 & \dots & d \\
            \hline
            1  & 0.4 & 0.3 & \dots & 0.8 \\
            2  & 0.9 & 0.1 & \dots & 0.1 \\
            \vdots & \vdots & \vdots & \ddots & \vdots \\
            t  & 0.2 & 0.5 & \dots & 0.3
        \end{array}
    }_{\text{Atn}(X)}
$$

Como nos Transformers é necessário transformar cada elemento da sequência recebida, o mecanismo de atenção pode ser acelerado ao ser aplicado em batch, utilizando toda a sequência como queries. Assim, as queries correspondem à própria sequência recebida, permitindo processar todos os elementos simultaneamente.

$$
    \underbrace{
        \begin{array}{c|cccc}
            & 1 & 2 & \dots & d \\
            \hline
            1  & 0.7 & 0.8 & \dots & 0.3 \\
            2  & 0.3 & 0.2 & \dots & 0.2 \\
            \vdots & \vdots & \vdots & \ddots & \vdots \\
            t  & 0.2 & 0.4 & \dots & 0.1
        \end{array}
    }_{\text{Q}}
    \qquad
    \underbrace{
        \begin{array}{c|cccc}
            & 1 & 2 & \dots & d \\
            \hline
            1  & 0.7 & 0.8 & \dots & 0.3 \\
            2  & 0.3 & 0.2 & \dots & 0.2 \\
            \vdots & \vdots & \vdots & \ddots & \vdots \\
            t  & 0.2 & 0.4 & \dots & 0.1
        \end{array}
    }_{\text{K}}
$$

$$
    \downarrow
$$

$$
    \underbrace{
    \begin{array}{c|cccc}
        & 1 & 2 & \dots & t \\
        \hline
        1  & 0.5 & 0.5 & \dots & 0.0 \\
        2  & 0.2 & 0.3 & \dots & 0.4 \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        t  & 0.1 & 0.2 & \dots & 0.5
    \end{array}
    }_{\text{Pesos de atenção}}
$$
Por isso, embora queries, keys e values sejam inicialmente derivados da mesma sequência, é importante destacar que cada um desempenha um papel específico e distinto para o mecanismo de atenção.

### Mecanismos de atenção

#### Dot-Product Attention (DPA)

Nesse mecanismo, os pesos de atenção de cada elemento são calculados usando o produto interno (dot product) entre as queries e as keys correspondentes.

$$
  \text{Scores-DPA}(Q,K) = Q^TK
$$

Ao calcular a nova sequência em batch, o DPA pode ser definido da seguinte forma:

$$
  \text{DPA}(Q,K,V) = \text{Scores-DPA}(Q,K) \cdot V = Q^TKV
$$

No entanto, dessa forma, o produto interno entre dois vetores pode assumir valores de $-\infty$ a $\infty$. Isso significa que o mecanismo de atenção pode, potencialmente, atribuir atenção excessiva a determinados elementos, o que pode enviesar o modelo e prejudicar seu funcionamento. Para evitar esse problema e garantir que os pesos de atenção sejam normalizados, aplica-se a função softmax sobre esses valores.

$$
  \text{DPA}(Q,K,V) = \text{Softmax}(Q^TK)V
$$

#### Scaled Dot-Product Attention (SDPA)

Essa variação do DPA inclui um termo de normalização nos pesos para estabilizar os gradientes durante o backpropagation.

$$
  \text{SDPA}(Q,K,V) = \text{Softmax}\left(\frac{Q^TK}{\sqrt{d}}\right)V
$$

A normalização pelo fator $\sqrt{d}$ tem base empírica. Antes dos Transformers, experimentos já mostravam que normalizar a variância dos gradientes gerados por camadas ocultas ajuda a evitar problemas como gradient vanishing e neurônios mortos durante o treinamento. Por isso, essa prática foi incorporada à arquitetura dos Transformers.

```python
def apply_scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    keys = keys.transpose(2, 3)
    scores = queries @ keys / (split_embed_dim**0.5)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    weights = F.softmax(scores, dim=3)
    outputs = weights @ values

    return outputs
```

#### Projeções lineares

Uma das maneiras para melhorar a performance dos Transformers é aplicar projeções lineares separadas às queries, keys e values antes do seu uso nos mecanismos de atenção, multiplicando cada uma por uma matrizes de parâmetros treinável específica ($W^Q$, $W^K$ e $W^V$). Isso faz com que queries, keys e values passem a pertencer a espaços diferentes, permitindo que o modelo aprenda, durante o treinamento, como transformar os embeddings originais em representações mais adequadas ao contexto de cada token. Essas projeções são otimizadas para melhorar a capacidade do Transformer de capturar relações contextuais relevantes entre os elementos da sequência.

$$
  \text{SDPA-Transformer}(Q,K,V) =  SDPA(QW^Q, KW^K,VW^V)
$$

```python
queries_projection = nn.Linear(embed_dim, embed_dim, bias=False)
keys_projection = nn.Linear(embed_dim, embed_dim, bias=False)
values_projection = nn.Linear(embed_dim, embed_dim, bias=False)

queries = queries_projection(queries)
keys = keys_projection(keys)
values = values_projection(values)

embeddings = apply_scaled_dot_product_attention(queries, keys, values)
```

#### Multihead Attention (MHA)

Essa variação do SDPA aplica o mecanismo de atenção $h$ vezes em paralelo, cada uma com diferentes projeções lineares dos elementos da sequência. O número de cabeças $h$ é um hiperparâmetro do modelo.

Como os pesos no SDPA são normalizados, cada cabeça de atenção não pode "prestar atenção" igualmente em todos os elementos ao mesmo tempo. O objetivo do uso de MHA é permitir que cada cabeça (ou seja, cada SDPA sendo realizado em paralelo) foque em diferentes padrões ou relações no contexto, enriquecendo a representação aprendida pelo modelo durante o treinamento.

Usando uma analogia, o MHA seria como ler um texto $h$ vezes, focando em partes diferentes a cada leitura para compreender melhor o contexto de cada palavra. No mecanismo, porém, todas essas "leituras" acontecem simultaneamente.

Embora esse mecanismo otimize o modelo, se implementado literalmente, seria necessário calcular o SDPA $h$ vezes, tornando o MHA $h$ vezes mais lento, o que pode tornar o mecanismo inviável computacionalmente. Para evitar esse problema, utiliza-se a adaptação a seguir no algoritmo, que permite que a sua complexidade computacional não cresça com o número de cabeças, mantendo a escalabilidade do modelo:

1. Dividir as queries, keys e values em $h$ partes, transformando as dimensões do batch de $b \times t \times d$ para $b \times t \times h \times \frac{d}{h}$.
2. Transpor o tensor para que a dimensão das cabeças venha antes da dimensão das sequências, mudando de $b \times t \times h \times \frac{d}{h}$ para $b \times h \times t \times \frac{d}{h}$.
3. Aplicar o SDPA separadamente em cada cabeça, usando projeções diferentes para cada uma.
4. Concatenar os embeddings resultantes de todas as cabeças.
5. Transpor o tensor para restaurar a ordem original das dimensões, voltando para $b \times t \times d$.
6. Aplicar uma projeção linear $W^O$ ao tensor.

Após a etapa 3, o funcionamento do algoritmo pode ser representado pela seguinte equação:

$$
  \text{MHA}(Q,K,V) = \left(\Big \Vert^h_{i=1} \text{SDPA}(QW^Q_i, KW^K_i,VW^V_i) \right) W^O
$$

Assim, o SDPA é aplicado $h$ vezes, mas como cada cabeça opera em uma dimensão $\frac{1}{h}$ menor, cada operação é proporcionalmente mais rápida. Isso garante que a complexidade computacional do MHA permaneça equivalente à do SDPA, mesmo com múltiplas cabeças.

Na prática, o MHA ainda é um pouco mais lento que o SDPA devido à projeção linear $W^O$ ao final, mas a complexidade computacional não cresce com o número de cabeças ou o tamanho das sequências.

```python
n_heads = 8
split_embed_dim = embed_dim // n_heads
outputs_projection = nn.Linear(embed_dim, embed_dim, bias=False)


def split_embeddings(
    embeddings: torch.Tensor,
    batch_size: int,
    n_tokens: int,
) -> torch.Tensor:
    splitted = embeddings.view(batch_size, n_tokens, n_heads, split_embed_dim)
    splitted = splitted.transpose(1, 2)

    return splitted


def join_embeddings(
    embeddings: torch.Tensor,
    batch_size: int,
    n_tokens: int,
) -> torch.Tensor:
    joined = embeddings.transpose(1, 2)
    joined = joined.contiguous()
    joined = joined.view(batch_size, n_tokens, embed_dim)
    return joined


def apply_multihead_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    batch_size, n_tokens, _ = queries.size()

    queries = queries_projection(queries)
    keys = keys_projection(keys)
    values = values_projection(values)

    queries = split_embeddings(queries, batch_size, n_tokens)
    keys = split_embeddings(keys, batch_size, n_tokens)
    values = split_embeddings(values, batch_size, n_tokens)

    keys = keys.transpose(2, 3)
    scores = queries @ keys / (split_embed_dim**0.5)
    weights = F.softmax(scores, dim=3)

    outputs = weights @ values
    outputs = join_embeddings(outputs, batch_size, n_tokens)
    outputs = outputs_projection(outputs)

    return outputs

class MultiheadAttention(nn.Module):
    def __init__(self: Self, embed_dim: int, n_heads: int) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.split_embed_dim = self.embed_dim // self.n_heads

        self.queries_projection = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.keys_projection = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.values_projection = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.outputs_projection = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def split_embeddings(
        self: Self,
        embeddings: torch.Tensor,
        batch_size: int,
        n_tokens: int,
    ) -> torch.Tensor:
        splitted = embeddings.view(
            batch_size,
            n_tokens,
            self.n_heads,
            self.split_embed_dim,
        )

        splitted = splitted.transpose(1, 2)

        return splitted

    def join_embeddings(
        self: Self,
        embeddings: torch.Tensor,
        batch_size: int,
        n_tokens: int,
    ) -> torch.Tensor:
        joined = embeddings.transpose(1, 2)
        joined = joined.contiguous()
        joined = joined.view(batch_size, n_tokens, self.embed_dim)

        return joined

    def forward(
        self: Self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_tokens, _ = queries.size()

        queries = self.queries_projection(queries)
        keys = self.keys_projection(keys)
        values = self.values_projection(values)

        queries = self.split_embeddings(queries, batch_size, n_tokens)
        keys = self.split_embeddings(keys, batch_size, n_tokens)
        values = self.split_embeddings(values, batch_size, n_tokens)

        keys = keys.transpose(2, 3)

        scores = queries @ keys / (self.split_embed_dim**0.5)
        weights = F.softmax(scores, dim=3)

        outputs = weights @ values
        outputs = self.join_embeddings(outputs, batch_size, n_tokens)
        outputs = self.outputs_projection(outputs)

        return outputs
```

## Conclusão

Os mecanismos de atenção são uma maneira eficiente de ajustar as representações dos valores para que reflitam melhor seu significado no contexto. A eficiência do MHA nessa tarefa faz toda a diferença quando combinada com outras técnicas para criar Transformers capazes de realizar bem várias tarefas.

Apesar disso, a atenção sozinha pode ter algumas limitações. No próximo post, vamos ver como o Positional Encoding pode ajudar a representar a posição dos elementos em uma sequência e superar esses problemas.
