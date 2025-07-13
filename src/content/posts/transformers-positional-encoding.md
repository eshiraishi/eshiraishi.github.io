---
title: 'Explicando Transformers, Pt. IV: Positional Encoding'
date: '2025-07-16'
description: ''
image: '/transformers.png'
---

No post anterior, vimos como funcionam mecanismos de atenção e concluímos que existem algumas limitações no seu uso direto. Nesse post, explicarei como funciona o Positional Encoding, uma técnica complementar à atenção para representar a posição dos elementos na sequência recebida.

## Positional Encoding (PE)

Os mecanismos de atenção apresentados até aqui não levam em conta a posição dos elementos na sequência recebida ao gerar a sequência de saída. Isso significa que, se a ordem dos elementos for alterada, o resultado permanecerá o mesmo. Esse efeito é indesejado e pode introduzir vieses no modelo durante o treinamento.

Antes de aplicar qualquer mecanismo de atenção, é necessário representar na sequência recebida a posição de cada elemento de alguma forma. Na arquitetura, essa representação é feita usando Positional Encoding (PE), técnica que gera uma sequência de embeddings especiais: cada um representa a posição de um elemento na sequência, permitindo que o modelo entenda a ordem dos elementos a partir apenas dos elementos da sequência recebida.

A técnica de PE utilizada nos Transformers é baseada na seguinte função:

$$
    \text{PE}(i, j, p) = \begin{cases}
        \sin \dfrac{p}{\theta^{\frac{2i}{d}}} & \text{se }j \text{ é par}, \\
        \cos \dfrac{p}{\theta^{\frac{2i}{d}}} & \text{se }j \text{ é ímpar}.
    \end{cases}
$$

Onde:

* $p$ é a posição de um elemento na sequência recebida.
* $j$ é a posição de um item escalar em um elemento da sequência recebida.
* $i$ é um índice auxiliar tal que $0 \leq i < \frac{d}{2}$, e $i$ é incrementado a cada dois itens consecutivos de $j$.
* $\theta$ é um hiperparâmetro que representa a escala das posições.

Os embeddings gerados pelo PE são incorporados à sequência de embeddings original ao serem somados com o valor dos token embeddings. Nos Transformers, essa função foi escolhida porque equivale a aplicar uma matriz de rotação aos embeddings dos elementos da sequência, em que o ângulo de rotação de cada elemento é determinado por sua posição.

O uso de PE é uma forma de codifição relativa, ou seja, ao analisar os valores para toda a sequência simultaneamente, essa técnica prioriza ser mais efetiva em representar a posição de um elemento em relação aos seus vizinhos ao invés de ser mais efetiva em representar a ordem de todos os elementos de forma absoluta.

Para ilustrar o cálculo de $PE$ em batch, considere uma sequência de embeddings de comprimento $t = 5$, dimensão $d = 4$ e $\theta = 10000$. A matriz de embeddings posicionais que será somada à sequência recebida será calculada da seguinte forma:

$$
    i =
    \begin{bmatrix}
        0 & 0 & 1 & 1 \\
        0 & 0 & 1 & 1 \\
        0 & 0 & 1 & 1 \\
        0 & 0 & 1 & 1 \\
        0 & 0 & 1 & 1
    \end{bmatrix}
$$

$$
    j =
    \begin{bmatrix}
        0 & 1 & 2 & 3 \\
        0 & 1 & 2 & 3 \\
        0 & 1 & 2 & 3 \\
        0 & 1 & 2 & 3 \\
        0 & 1 & 2 & 3
    \end{bmatrix}
$$

$$
    p =
    \begin{bmatrix}
        0 & 0 & 0 & 0 \\
        1 & 1 & 1 & 1 \\
        2 & 2 & 2 & 2 \\
        3 & 3 & 3 & 3 \\
        4 & 4 & 4 & 4
    \end{bmatrix}
$$

$$
    PE(i,j,p) =
    \begin{bmatrix}
         0.0000 &  1.0000 & 0.0000 & 1.0000 \\
         0.8415 &  0.5403 & 0.0100 & 0.9999 \\
         0.9093 & -0.4161 & 0.0200 & 0.9998 \\
         0.1411 & -0.9899 & 0.0300 & 0.9996 \\
        -0.7568 & -0.6536 & 0.0400 & 0.9992 \\
    \end{bmatrix}
$$

No PyTorch, é possível calcular o PE em batch de forma vetorizada, sem a necessidade de laços de repetição, o que torna a operação muito mais eficiente:

```python
@dataclass
class PositionalEncoderConfig:
    embed_dim: int
    theta: int = 10000
```

&nbsp;

```python
class PositionalEncoder(nn.Module):
    def __init__(self: Self, config: PositionalEncoderConfig) -> None:
        super().__init__()
        self.config = config

    @torch.no_grad()
    def forward(self: Self, embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, n_tokens, _ = embeddings.size()

        indexes = torch.arange(self.config.embed_dim, dtype=torch.float)

        positions = torch.arange(n_tokens, dtype=torch.float)
        positions = positions.view(n_tokens, 1)

        i = torch.arange(self.config.embed_dim // 2)
        i = i.float()
        i = i.repeat_interleave(2)

        cos_indexes = indexes % 2
        cos_indexes = cos_indexes.bool()
        cos_indexes = cos_indexes.expand((n_tokens, self.config.embed_dim))

        sin_indexes = ~cos_indexes

        encodings = positions / (self.config.theta ** (2 * i / self.config.embed_dim))

        encodings[sin_indexes] = encodings[sin_indexes].sin()
        encodings[cos_indexes] = encodings[cos_indexes].cos()

        encodings = encodings.expand((batch_size, n_tokens, self.config.embed_dim))

        return embeddings + encodings
```

## Conclusão

O Positional Encoding é fundamental para que os mecanismos de atenção consigam entender a ordem dos elementos em uma sequência. Com isso, estamos quase prontos para montar um Transformer completo.

No próximo post, para facilitar a explicação da arquitetura, vamos entender como funcionam modelos autoregressivos, e ver como toda essa representação de dados pode ser usada para gerar outras formas de dados complexos, como textos e outros tipos de sequências.
