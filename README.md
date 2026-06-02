# Siri-LLM: Building a Large Language Model (LLM) From Scratch

Siri-LLM is an educational project focused on understanding and implementing the fundamental building blocks of modern Large Language Models (LLMs) from first principles.

The objective of this project is not merely to use existing AI frameworks, but to deeply understand how models such as GPT, Gemini, Llama, Claude, and BERT process language internally.

The project starts with raw text data and gradually builds toward a complete Transformer-based language model by implementing each component step-by-step in Python.

---

# Why This Project?

Modern AI systems appear magical on the surface, but internally they are built from a sequence of mathematical and engineering components.

Most developers use pre-trained models without understanding:

- How text becomes numbers
- How words become vectors
- How Transformers understand context
- How Attention works
- How GPT predicts the next token

This project aims to bridge that gap.

Instead of treating LLMs as black boxes, Siri-LLM explores every major component individually and then combines them into a complete architecture.

---

# End Goal

Build a GPT-style Language Model completely from scratch.

Final Pipeline:

```text
Raw Text
    │
    ▼
Corpus Collection
    │
    ▼
Text Cleaning
    │
    ▼
Tokenizer
    │
    ▼
Token IDs
    │
    ▼
Token Embeddings
    │
    ▼
Positional Embeddings
    │
    ▼
Self-Attention
    │
    ▼
Multi-Head Attention
    │
    ▼
Transformer Block
    │
    ▼
Stacked Transformer Layers
    │
    ▼
Language Model Head
    │
    ▼
Next Token Prediction
    │
    ▼
Mini GPT Model
```

---

# Understanding LLMs Step By Step

A Large Language Model is essentially a machine that learns patterns in text.

Example:

Input:

```text
5G technology is transforming
```

Desired Prediction:

```text
wireless communication
```

To make this prediction possible, an LLM must go through multiple stages.

---

# Stage 1: Data Collection

An LLM first requires text data.

For this project, domain-specific content is collected from:

- Telecom Research
- Wireless Communication
- Artificial Intelligence
- Machine Learning
- Signal Processing

Sources include:

- iSignal Research
- iSignal Academy
- iSignal Learning Platform
- iSignal Technical Blogs

Output:

```text
raw_corpus.txt
```

---

# Stage 2: Text Cleaning

Raw internet text contains:

- HTML
- URLs
- Symbols
- Duplicate spaces
- Noise

Example:

Before:

```text
Visit https://example.com for more info!!!
```

After:

```text
Visit for more info
```

Output:

```text
isignal_cleaned_corpus.txt
```

---

# Stage 3: Building a Tokenizer

Neural Networks do not understand text.

They only understand numbers.

Tokenizer converts:

```text
5G AI telecom research
```

into:

```text
['5G', 'AI', 'telecom', 'research']
```

and then:

```text
[111, 156, 674, 514]
```

### Implemented

- Byte Pair Encoding (BPE)
- Vocabulary Learning
- Token Encoding
- Token Decoding
- Vocabulary Storage

Special Tokens:

```text
[PAD]
[UNK]
[CLS]
[SEP]
[MASK]
```

Output:

```text
isignal_tokenizer.json
```

---

# Stage 4: Token Embeddings

Token IDs still have no meaning.

Example:

```text
AI = 156
Telecom = 674
```

The numbers themselves do not represent semantics.

Embeddings convert token IDs into dense vectors.

Example:

```text
AI

↓

[-0.27, 0.79, 0.34, ...]
```

Each token receives a mathematical representation.

Purpose:

- Capture meaning
- Learn relationships
- Enable neural processing

Output:

```text
Embedding Matrix
Shape:

[Vocabulary Size × Embedding Dimension]
```

---

# Stage 5: Positional Encoding

Transformers process all tokens simultaneously.

Without position information:

```text
AI loves telecom
```

and

```text
telecom loves AI
```

look identical.

Positional Encoding injects order information.

Implemented:

### Sinusoidal Positional Encoding

```text
Token Embedding
        +
Position Embedding
        =
Final Input Embedding
```

Output:

```text
Final Input Embeddings
```

---

# Stage 6: Self-Attention

Self-Attention is the breakthrough that made Transformers possible.

Instead of processing words sequentially, every token can interact with every other token.

Example:

```text
The signal failed because it was noisy.
```

The model learns:

```text
it → signal
```

even when the words are far apart.

---

## Query, Key, Value

Every token creates:

### Query (Q)

What am I looking for?

### Key (K)

What information do I provide?

### Value (V)

What information do I contain?

---

## Attention Computation

```text
Q × Kᵀ
```

produces similarity scores.

These scores are normalized using:

```text
Softmax
```

The result becomes:

```text
Attention Weights
```

These weights determine how much information should be gathered from other tokens.

Output:

```text
Context Vectors
```

---

# Stage 7: Multi-Head Attention

One attention mechanism is not enough.

Different relationships exist simultaneously.

Examples:

Head 1:

```text
Grammar
```

Head 2:

```text
Meaning
```

Head 3:

```text
Long-range context
```

Head 4:

```text
Position relationships
```

Multiple heads operate in parallel.

Their outputs are combined into a richer representation.

Implemented:

- Multiple Heads
- Head Splitting
- Head Concatenation
- Output Projection
- Causal Masking

---

# Stage 8: Transformer Block

This is the core building block of GPT.

Architecture:

```text
Input
 │
 ▼
Multi-Head Attention
 │
 ▼
Add & Normalize
 │
 ▼
Feed Forward Network
 │
 ▼
Add & Normalize
 │
 ▼
Output
```

Components:

### Multi-Head Attention

Context Learning

### Residual Connections

Stable gradient flow

### Layer Normalization

Training stability

### Feed Forward Network

Non-linear learning

---

# Stage 9: Stacking Transformer Blocks

One Transformer block is useful.

Many Transformer blocks create an LLM.

Example:

```text
Transformer Block 1
        │
        ▼
Transformer Block 2
        │
        ▼
Transformer Block 3
        │
        ▼
...
        ▼
Transformer Block N
```

Each layer learns increasingly abstract representations.

---

# Stage 10: Language Modeling Head

Final Transformer outputs are projected into vocabulary space.

Example:

Input:

```text
5G AI is changing
```

Output Probabilities:

```text
wireless      0.72
telecom       0.18
research      0.06
networks      0.04
```

Highest probability becomes the predicted token.

---

# Stage 11: Text Generation

Prediction Loop:

```text
Input Text
     │
     ▼
Predict Next Token
     │
     ▼
Append Prediction
     │
     ▼
Predict Again
```

Example:

```text
5G AI is changing
```

↓

```text
5G AI is changing wireless
```

↓

```text
5G AI is changing wireless communication
```

↓

```text
5G AI is changing wireless communication systems
```

This is how GPT-style models generate text.

---

# Educational Notebook

Before implementing Attention, this project includes:

```text
Self_Attention_Deep_Dive.ipynb
```

The notebook explains:

- Why Transformers were invented
- Problems with RNNs
- Problems with LSTMs
- Query, Key, Value
- Attention Scores
- Softmax
- Context Vectors
- Multi-Head Attention
- GPT Intuition

This notebook serves as the conceptual foundation for the implementation.

---

# Project Structure

```text
Siri-LLM/
│
├── README.md
├── LICENSE
│
├── Self_Attention_Deep_Dive.ipynb
│
├── tokenizer/
│
├── embeddings/
│
├── positional_encoding/
│
├── attention/
│
├── transformer/
│
└── datasets/
```

---

# Technologies Used

- Python
- NumPy
- PyTorch
- Hugging Face Tokenizers
- BeautifulSoup4
- Requests
- Jupyter Notebook

---

# Current Progress

## Completed

✅ Corpus Collection

✅ Data Cleaning

✅ BPE Tokenizer

✅ Token Embeddings

✅ Positional Encoding

✅ Self-Attention

✅ Multi-Head Attention

---

## In Progress

🚧 Transformer Block

---

## Upcoming

- Layer Normalization
- Feed Forward Networks
- Residual Connections
- Transformer Decoder
- GPT Architecture
- Text Generation

---

# Learning Outcomes

After completing this project you will understand:

- How text becomes tokens
- How embeddings represent meaning
- How positional encoding works
- How attention discovers relationships
- How multi-head attention improves learning
- How Transformer blocks are built
- How GPT predicts text
- How modern LLMs function internally

---

# Author

**Krishna**

Building a Transformer-based Large Language Model from scratch to understand the foundations of modern AI systems.

---

# License

MIT License
