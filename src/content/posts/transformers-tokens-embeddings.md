---
title: "🇧🇷 Explicando Transformers (Pt. 2): Tokens e embeddings"
date: "2025-07-14"
draft: true
description: "Neste post, vou mostrar como os dados são representados em modelos como os Transformers, explicando como inteligências artificiais como o ChatGPT entendem nossos prompts. Vamos passar por conceitos importantes como tokens, embeddings e padding, criar um tokenizador simples em Python e como gerar embeddings de forma prática usando PyTorch."
image: "/transformers.png"
---

Neste post, vou mostrar como os dados são representados em modelos como os Transformers, explicando como inteligências artificiais como o ChatGPT entendem nossos prompts. Vamos passar por conceitos importantes como tokens, embeddings e padding, criar um tokenizador simples em Python e como gerar embeddings de forma prática usando PyTorch.

## Representação dos dados

Para trabalhar com textos usando redes neurais, precisamos convertê-los para uma forma numérica que os modelos consigam processar. O método mais comum é transformar o texto em uma sequência de tokens, que são então convertidos em embeddings.

### Tokens

Textos são compostos por um alfabeto finito e conhecido, o que permite enumerar todos os caracteres desse alfabeto e associar cada um a um número único. Essa representação numérica é chamada de token, e a função que faz essa associação é chamada de tokenizer (ou "tokenizador").

Além de mapear caracteres individuais, também é possível criar tokens para sequências de caracteres, como palavras ou n-gramas (sequências de $n$ caracteres), aumentando o número de tokens disponíveis, o que pode melhorar a performance do modelo. Essa abordagem é comum em modelos de linguagem para melhorar a performance, utilizando algoritmos como Byte Pair Encoding, WordPiece e SentencePiece, que podem gerar tokenizers com centenas de milhares de tokens.

No entanto, como o modelo de linguagem pressupõe que o texto já foi "tokenizado", a escolha do método de tokenização não altera a arquitetura do modelo. Por isso, para simplificar, neste guia será utilizado um tokenizer básico, composto apenas pelos caracteres imprimíveis da tabela ASCII, disponíveis no objeto `printable` do módulo `string` em Python.

| Letra | Token |
| ----- | ----- |
| `a`   | `1`   |
| `b`   | `2`   |
| `c`   | `3`   |
| ...   | ...   |
| `z`   | `26`  |
| `␣`   | `27`  |
| `.`   | `28`  |
| `,`   | `29`  |

Uma sequência de tokens gerada a partir de um texto pode ser representada por um vetor, onde cada elemento corresponde ao valor numérico de um token na ordem em que aparece no texto. Quando reunimos vários desses vetores (um para cada texto de um batch) e os organizamos como linhas de uma matriz, conseguimos paralelizar as operações envolvidas na predição através de operações vetorizadas, o que maximiza a performance do modelo.

$$
\begin{array}{cccc}
    \texttt{g} & \texttt{a} & \texttt{t} & \texttt{o}
\end{array}
$$

$$
\downarrow
$$

$$
\begin{bmatrix}
    7  &  1  & 20 & 15
\end{bmatrix}
$$

Além disso, o tokenizer insere dois tokens especiais em cada sequência: `<bos>` (ou beginning of sentence), que marca o início do texto, e `<eos>` (ou end of sentence), que marca o final do texto.

Nestes exemplos, `<bos>` e `<eos>` são sempre representados pelos números `1` e `2`, respectivamente. Por isso, os demais tokens do vocabulário começam a ser numerados a partir de `3`.

| Letra   | Token |
| ------- | ----- |
| `<bos>` | `1`   |
| `<eos>` | `2`   |
| `a`     | `3`   |
| `b`     | `4`   |
| `c`     | `5`   |
| ...     | ...   |
| `z`     | `28`  |
| `␣`     | `29`  |
| `.`     | `30`  |
| `,`     | `31`  |

$$
\begin{array}{cccc}
    \texttt{g} & \texttt{a} & \texttt{t} & \texttt{o}
\end{array}
$$

$$
    \downarrow
$$

$$
    \begin{array}{cccccc}
        \texttt{<bos>} & \texttt{g} & \texttt{a} & \texttt{t} & \texttt{o}  & \texttt{<eos>}
    \end{array}
$$

$$
    \downarrow
$$

$$

\begin{bmatrix}
    1 & 9  &  3  & 22 & 17 & 2
\end{bmatrix}
$$

Os tokens `<bos>` e `<eos>` também podem ser utilizados para separar a sequência recebida da sequência gerada, o que será importante durante o treinamento dos Transformers. Por exemplo, se o modelo recebe o texto "cachorro" e deve gerar o texto "dog", ambas as sequências podem ser representadas juntas da seguinte forma:

$$
\begin{array}{ccccccccccccccc}
\texttt{<bos>} & \texttt{c} & \texttt{a} & \texttt{c} & \texttt{h} & \texttt{o} & \texttt{r} & \texttt{r} & \texttt{o} & \texttt{<eos>} & \texttt{<bos>} & \texttt{d} & \texttt{o} & \texttt{g} & \texttt{<eos>}
\end{array}
$$

```python
from string import printable

string_to_int = {
    '<bos>': 1,
    '<eos>': 2,
}

string_to_int.update({
    char: index + 2
    for index, char in enumerate(printable, 1)
})

special_tokens = {'<bos>', '<eos>'}

def split(text: str) -> Generator[str, None, None]:
    index = 0

    while index < len(text):
        token = text[index]
        for special_token in special_tokens:
            if text.startswith(special_token, index):
                token = special_token
                break

        yield token

        index += len(token)

def tokenize(self: Self, text: str) -> torch.Tensor:
    tokens = [string_to_int[token] for token in split(text)]
    tokens = torch.tensor(tokens)
    return tokens
```

### Padding

Como mencionado anteriormente, para desenvolver modelos rápidos de forma simples, é importante maximizar o número de operações realizadas em batch utilizando operações tensoriais.

Após um texto ser transformado em uma sequência de tokens, ele pode ser representado como um vetor. Ao aplicar essa transformação em todos os textos de um batch e organizá-los juntos, obtemos uma matriz onde cada linha representa um texto do batch. Essa matriz permite acelerar as operações tensoriais nas etapas seguintes. No entanto, para que essa matriz seja válida, todos os vetores (textos) precisam ter o mesmo comprimento. Como os textos originais geralmente têm tamanhos diferentes, é necessário equalizar seus tamanhos para criar uma matriz válida.

Para garantir que todos os vetores de um batch tenham o mesmo comprimento, é usado um processo de padding, onde tokens especiais são adicionados ao início ou final de cada sequência até que todas tenham $n$ elementos. O token especial adicionado é chamado de padding token, representado por `<pad>`, e nos exemplos será sempre representado pelo número `0`. Considere também que cada batch será composto sempre de $b$ sequências.

| Letra   | Token |
| ------- | ----- |
| `<pad>` | `0`   |
| `<bos>` | `1`   |
| `<eos>` | `2`   |
| `a`     | `4`   |
| `b`     | `5`   |
| `c`     | `6`   |
| ...     | ...   |
| `z`     | `28`  |
| `␣`     | `29`  |
| `.`     | `30`  |
| `,`     | `31`  |

Por fim, existem duas formas principais de definir o comprimento $t$ de cada sequência após o padding:

1. Definir $t$ como o comprimento da maior sequência presente no batch atual.
2. Definir $t$ como um valor fixo e arbitrário, previamente estabelecido.

Por exemplo, ao transformar um batch de textos em tokens e aplicar padding seguindo a opção 1, o procedimento seria:

$$
\begin{array}{c}
    \text{gato} \\
    \text{elefante} \\
    \text{peixe} \\
    \text{pássaro} \\
    \text{cão}
\end{array}
$$

$$
\downarrow
$$

$$
\begin{array}{cccccccccc}
    \texttt{<bos>} & \texttt{ g } & \texttt{ a } & \texttt{ t } & \texttt{ o } & \texttt{<eos>} & \texttt{<pad>} & \texttt{<pad>} & \texttt{<pad>} & \texttt{<pad>} \\
    \texttt{<bos>} & \texttt{e} & \texttt{l} & \texttt{e} & \texttt{f} & \texttt{a} & \texttt{n} & \texttt{t} & \texttt{e}  & \texttt{<eos>} \\
    \texttt{<bos>} & \texttt{p} & \texttt{e} & \texttt{i} & \texttt{x} & \texttt{e}  & \texttt{<eos>} & \texttt{<pad>} & \texttt{<pad>} & \texttt{<pad>} \\
    \texttt{<bos>} & \texttt{p} & \texttt{á} & \texttt{s} & \texttt{s} & \texttt{a} & \texttt{r} & \texttt{o}  & \texttt{<eos>} & \texttt{<pad>} \\
    \texttt{<bos>} & \texttt{c} & \texttt{ã} & \texttt{o} & \texttt{<eos>} & \texttt{<pad>} & \texttt{<pad>} & \texttt{<pad>} & \texttt{<pad>} & \texttt{<pad>}
\end{array}
$$

$$
\downarrow
$$

$$
\begin{bmatrix}
   1 &  7  &  1  & 20 & 15 &  2 &  0 &  0 &  0 &  0 \\
   1 &  5  & 12  &  5 &  6 &  1 & 14 & 20 &  5  &  2\\
   1 & 16  &  5  &  9 & 24 &  5  &  2 &  0 &  0 &  0 \\
   1 & 16  & 27  & 19 & 19 &  1 & 18 & 15  &  2 &  0 \\
   1 &  3  & 28  & 15  &  2 &  0 &  0 &  0 &  0 &  0
\end{bmatrix}
$$

De modo geral, o ganho de velocidade proporcionado pelo uso de padding costuma compensar o aumento no consumo de memória, já que o tempo de processamento geralmente é um gargalo maior do que a memória utilizada durante o treinamento e a inferência de redes neurais.

A escolha entre as duas alternativas pode impactar a performance dependendo do hardware utilizado. Em CPUs e GPUs, a opção 1 tende a ser mais vantajosa por economizar memória, pois a opção 2 não traz ganhos de velocidade significativos nesses casos. Já em TPUs, operações com batches de tamanho fixo podem ser mais rápidas do que com batches de tamanhos variados, tornando a opção 2 potencialmente mais eficiente.

```python
char_ids = {
    '<pad>': 0,
    '<bos>': 1,
    '<eos>': 2,
}

char_ids.update({
    char: index + 3
    for index, char in enumerate(printable)
})

special_tokens = {'<pad>', '<bos>', '<eos>'}

def pad(
    self: Self,
    tokens: torch.Tensor,
    amount: int,
    fill_value: int,
    side: Literal["left", "right"],
) -> torch.Tensor:
    if amount == 0:
        return tokens

    padding = torch.full(
        size=(amount,),
        fill_value=fill_value,
    )
    padded_tensor = (tokens, padding) if side == "right" else (padding, tokens)
    padded_tensor = torch.cat(padded_tensor)

    return padded_tensor

def batch_encode(
    texts: list[str],
    side: Literal["left", "right"] = "right",
    strategy: Literal["max", "fixed"] = "max",
    amount: int | None = None,
    truncate: bool = False,
) -> torch.Tensor:
    token_lists = [encode(text) for text in texts]
    lengths = [len(tokens) for tokens in token_lists]

    max_length = max(lengths)
    max_length = max_length if amount is None else max(max_length, amount)

    token_lists = [
        pad(
            tokens=tokens,
            amount=max_length - length,
            fill_value=0,
            side=side,
        )
        for tokens, length in zip(token_lists, lengths)
    ]
    token_lists = torch.stack(token_lists)

    if strategy == "fixed" and truncate:
        token_lists = token_lists[:, :amount]

    return token_lists
```

Uma curiosidade: o módulo `torch.nested` permite representar matrizes como listas de vetores com comprimentos diferentes, evitando a necessidade de padding. No entanto, esse módulo ainda é experimental e, na prática, seu uso pode tornar as operações mais lentas do que as alternativas tradicionais.

### Truncação

Mesmo ao optar pela alternativa 1, é comum definir um tamanho máximo para as sequências. Se o comprimento de uma sequência ultrapassar esse limite, ela será truncada, ou seja, os tokens excedentes no final serão descartados.

Em ambos os casos, o valor de $t$ costuma ser determinado com base na memória disponível ou de forma empírica, escolhendo um comprimento que maximize o número de elementos que um modelo consiga levar em consideração durante a geração.

### Implementação completa do Tokenizer

```python
class Tokenizer:
    def __init__(
        self: Self,
        vocab: set[str],
        config: TokenizerConfig,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.config = config

        self.string_to_int = {
            self.config.pad_token: 0,
            self.config.bos_token: 1,
            self.config.eos_token: 2,
        }
        self.string_to_int.update(
            {char: (index + 3) for index, char in enumerate(self.vocab)}
        )

        self.pad_token_int = self.string_to_int[self.config.pad_token]
        self.bos_token_int = self.string_to_int[self.config.bos_token]
        self.eos_token_int = self.string_to_int[self.config.eos_token]

        self.vocab_size = len(self.string_to_int)

        self.int_to_string = {
            0: self.config.pad_token,
            1: self.config.bos_token,
            2: self.config.eos_token,
        }
        self.int_to_string.update(
            {(index + 3): char for index, char in enumerate(self.vocab)}
        )

    def split(self: Self, chars: str) -> Generator[str, None, None]:
        index = 0
        while index < len(chars):
            token = chars[index]
            for special_token in self.config.special_tokens:
                if chars.startswith(special_token, index):
                    token = special_token
                    break

            yield token
            index += len(token)

    def encode(self: Self, text: str) -> torch.Tensor:
        tokens = [self.string_to_int[token_string] for token_string in self.split(text)]
        tokens = torch.tensor(tokens)
        return tokens

    def pad(
        self: Self,
        tokens: torch.Tensor,
        amount: int,
        fill_value: int,
        side: Literal["left", "right"],
    ) -> torch.Tensor:
        if amount == 0:
            return tokens

        padding = torch.full(
            size=(amount,),
            fill_value=fill_value,
        )
        padded_tensor = (tokens, padding) if side == "right" else (padding, tokens)
        padded_tensor = torch.cat(padded_tensor)

        return padded_tensor

    def batch_encode(
        self: Self,
        texts: list[str],
        side: Literal["left", "right"] = "right",
        strategy: Literal["max", "fixed"] = "max",
        amount: int | None = None,
        truncate: bool = False,
    ) -> torch.Tensor:
        token_lists = [self.encode(text) for text in texts]
        lengths = [len(tokens) for tokens in token_lists]

        max_length = max(lengths)
        max_length = max_length if amount is None else max(max_length, amount)

        token_lists = [
            self.pad(
                tokens=tokens,
                amount=max_length - length,
                fill_value=self.pad_token_int,
                side=side,
            )
            for tokens, length in zip(token_lists, lengths)
        ]
        token_lists = torch.stack(token_lists)

        if strategy == "fixed" and truncate:
            token_lists = token_lists[:, :amount]

        return token_lists

    def decode(self: Self, tokens: torch.Tensor) -> str:
        outputs = [self.int_to_string[token] for token in tokens.tolist()]
        outputs = "".join(outputs)
        return outputs

    def batch_decode(self: Self, tokens: torch.Tensor) -> list[str]:
        outputs = [self.decode(item) for item in tokens]
        return outputs

    def batch_shift(
        self: Self,
        input_tokens: torch.Tensor,
        output_tokens: torch.Tensor,
    ) -> str:
        outputs = torch.cat((input_tokens[:, 1:], output_tokens), dim=1)
        return outputs

    def add_special_tokens(self: Self, text: str) -> str:
        return self.config.bos_token + text + self.config.eos_token
```

### Embeddings

A representação numérica dos tokens, ou seja, simplesmente converter caracteres em números inteiros, é uma forma prática de representar sequências para serem usadas como dados de entrada. No entanto, usar esses valores inteiros diretamente como representação dos elementos da sequência pode introduzir vieses indesejados durante o treinamento.

Esses vieses ocorrem porque, ao utilizar os inteiros literalmente, o modelo pode interpretar que um token com valor $i$ tem alguma relação de ordem ou proximidade com o token $i+1$, quando na verdade essa relação não existe no contexto do problema. Isso pode levar o modelo a aprender padrões artificiais que não refletem a estrutura real dos dados, dificultando o aprendizado. Ao transformar esses inteiros em vetores de dimensão $d$ em um espaço bem escolhido, chamados de embeddings, essas relações de ordem deixam de existir, permitindo que o modelo aprenda relações reais entre os tokens de uma sequência.

Embeddings são representações numéricas que mapeiam elementos de um espaço original (por exemplo, inteiros que identificam tokens) para um novo espaço vetorial de dimensão $d$. Em aprendizado de máquina, esse novo espaço é escolhido para facilitar o aprendizado de padrões pelos modelos. O valor de $d$ é um hiperparâmetro, e normalmente é diferente da dimensão do espaço original.

Independentemente da técnica utilizada para gerar os embeddings, quando cada token é convertido em um vetor de $d$ dimensões, um batch de sequências originalmente representado por uma matriz de dimensão $b \times t$ (comprimento do batch por comprimento da sequência) será transformado em um tensor de dimensão $b \times t \times d$ (comprimento do batch, pelo comprimento da sequência pela dimensão dos embeddings).

$$
    \begin{bmatrix}
        1   & 14    & \dots  & 10 & 2 \\
        1   & 25    & \dots  & 0 & 2 \\
        \vdots & \vdots  & \ddots & \vdots & \vdots \\
        1   & 5    & \dots & 0 & 2
    \end{bmatrix}
$$

$$
    \downarrow
$$

$$
    \begin{bmatrix}
        \begin{bmatrix}
            0.23   & 1.45    & \dots  & 2.67 \\
            3.14   & 4.56    & \dots  & 5.78 \\
            \vdots & \vdots  & \ddots & \vdots \\
            1.76   & 0.65    & \dots  & 0.13 \\
            6.89 & 7.01 & \dots & 8.23
        \end{bmatrix} \\
        \quad \\
        \begin{bmatrix}
            0.23   & 1.45    & \dots  & 2.67 \\
            9.34   & 0.12    & \dots  & 1.34 \\
            \vdots & \vdots  & \ddots & \vdots \\
            1.49   & 5.59    & \dots  & 0.33 \\
            6.89   & 7.01    & \dots  & 8.23
        \end{bmatrix} \\
        \vdots \\
        \begin{bmatrix}
            0.23   & 1.45    & \dots  & 2.67 \\
            5.79 & 6.80 & \dots & 7.91 \\
            \vdots & \vdots  & \ddots & \vdots \\
            1.49   & 5.59    & \dots  & 0.33 \\
            6.89 & 7.01 & \dots & 8.23
        \end{bmatrix}
    \end{bmatrix}
$$

Uma das formas mais simples e eficientes de transformar tokens em embeddings é aplicar One-Hot Encoding em cada elemento, o que os converte em vetores de alta dimensionalidade com muitos valores nulos (chamados de vetores esparsos).

Se $|T|$ representa o tamanho do vocabulário, a representação do texto "baba" usando One-Hot Encoding será:

$$
    \begin{array}{cccccccccc}
        \texttt{<bos>} & \texttt{b} & \texttt{a} & \texttt{b} & \texttt{a} & \texttt{<eos>}
    \end{array}
$$

$$
    \downarrow
$$

$$
    \begin{bmatrix}
        1 & 4 & 3 & 4 & 3 & 2
    \end{bmatrix}
$$

$$
    \downarrow
$$

$$
\begin{array}{c|cccccc}
    & 0 & 1 & 2 & 3 & 4 & \dots & |T| \\
    \hline
    1 & 0 & 1 & 0 & 0 & 0 & \dots & 0 \\
    4 & 0 & 0 & 0 & 0 & 1 & \dots & 0 \\
    3 & 0 & 0 & 0 & 1 & 0 & \dots & 0 \\
    4 & 0 & 0 & 0 & 0 & 1 & \dots & 0 \\
    3 & 0 & 0 & 0 & 1 & 0 & \dots & 0 \\
    2 & 0 & 0 & 1 & 0 & 0 & \dots & 0
\end{array}
$$

One-Hot Encoding é utilizado porque permite acessar rapidamente os embeddings correspondentes a cada token usando seus índices. Ao multiplicar uma linha da sequência codificada (de dimensão $1 \times |T|$) por uma matriz de embeddings (de dimensão $|T| \times d$) o resultado é simplesmente a linha da matriz de embeddings correspondente ao token daquela posição. Essa matriz é chamada de matriz de consulta, pois essa operação funciona como consultar uma tabela: cada linha da sequência seleciona diretamente o vetor de embedding associado a ela, de forma semelhante ao acesso a elementos de uma lista ou array usando índices.

Assim, ao multiplicar toda a sequência codificada (com dimensões $t \times |T|$) pela matriz de consulta (com dimensões $|T| \times d$), cada linha da sequência seleciona diretamente o embedding correspondente ao token daquela posição. O resultado é uma matriz onde cada linha é o embedding do respectivo token da sequência recebida, preservando a ordem dos tokens.

No exemplo, considere que cada token tem sempre o mesmo embedding após essa transformação. Nesse cenário, basta concatenar todos os $|T|$ embeddings possíveis em ordem para formar a matriz de consulta. Dessa forma, a multiplicação representa o processo de converter a sequência de tokens em seus embeddings.

$$
    \underbrace{
        \begin{array}{c|cccccc}
            & 0 & 1 & 2 & 3 & 4 & \dots & |T| \\
            \hline
            1 & 0 & 1 & 0 & 0 & 0 & \dots & 0 \\
            4 & 0 & 0 & 0 & 0 & 1 & \dots & 0 \\
            3 & 0 & 0 & 0 & 1 & 0 & \dots & 0 \\
            4 & 0 & 0 & 0 & 0 & 1 & \dots & 0 \\
            3 & 0 & 0 & 0 & 1 & 0 & \dots & 0 \\
            2 & 0 & 0 & 1 & 0 & 0 & \dots & 0
        \end{array}
    }_{\text{One-Hot Encoding}}
$$

$$
    \times
$$

$$
    \underbrace{
        \begin{array}{c|cccc}
            & 1 & 2 & \dots & d \\
            \hline
            0      & 0.1    & 0.2    & \dots  & 0.6 \\
            1      & 0.7    & 0.8    & \dots  & 0.3 \\
            2      & 0.4    & 0.5    & \dots  & 0.9 \\
            3      & 0.9    & 0.1    & \dots  & 0.5 \\
            4      & 0.3    & 0.9    & \dots  & 0.1 \\
            \vdots & \vdots & \vdots & \ddots & \vdots \\
            |T|    & 0.2    & 0.4    & \dots  & 0.1
        \end{array}
    }_{\text{Matriz de consulta}}
$$

$$
    \downarrow
$$

$$
    \underbrace{
        \begin{array}{c|cccc}
            & 1 & 2 & \dots & d \\
            \hline
            1  & 0.7 & 0.8 & \dots & 0.3 \\
            4  & 0.3 & 0.9 & \dots & 0.1 \\
            3  & 0.9 & 0.1 & \dots & 0.5 \\
            4  & 0.3 & 0.9 & \dots & 0.1 \\
            3  & 0.9 & 0.1 & \dots & 0.5 \\
            2  & 0.4 & 0.5 & \dots & 0.9 \\
        \end{array}
    }_{\text{Embeddings}}
$$

Esse método de acesso aos embeddings permite que seus valores sejam ajustados automaticamente durante o treinamento do modelo para melhorar a performance. Isso é feito tratando a matriz de consulta como os pesos de uma camada linear em uma rede neural feedforward, otimizando seus valores via backpropagation a partir da função de perda do modelo. Dessa forma, o próprio treinamento encontra a melhor representação vetorial para cada token, sendo necessário apenas definir previamente o valor de $d$.

Além do impacto na performance do modelo, todas as operações envolvidas são aceleráveis via hardware (por dependerem apenas de operações de álgebra linear), e como os embeddings são densos, a dimensão $d$ (que geralmente é um valor na casa das centenas ou milhares) costuma ser muito menor que o tamanho do vocabulário $|T|$ (que pode chegar a centenas de milhares em Large Language Models, por exemplo), o que reduz significativamente o uso de memória em comparação com representações esparsas e de alta dimensionalidade como one-hot encoding. Isso torna o modelo mais eficiente e evita problemas de performance no modelo associados à alta dimensionalidade durante o treinamento (conhecidos popularmente como a "maldição da dimensionalidade").

No PyTorch, todo esse mecanismo é implementado pela classe `torch.nn.Embedding`.

```python
embed_dim = 512

embedder = nn.Embedding(
    num_embeddings=len(printable) + 1,
    embedding_dim=embed_dim,
)

embedding = embedder(42)
```

## Conclusão

Neste post, vimos como transformar textos em sequências numéricas usando tokens e embeddings, de um jeito prático e eficiente. Esses conceitos podem parecer um pouco abstratos agora, mas logo vão fazer todo sentido quando começarmos a explorar como os Transformers realmente funcionam.

No próximo post, vamos entender o que é atenção e como funcionam os mecanismos de atenção usados originalmente na arquitetura Transformer, Scaled Dot-Product Attention e Multihead Attention, como apresentados no artigo "Attention is All You Need".
