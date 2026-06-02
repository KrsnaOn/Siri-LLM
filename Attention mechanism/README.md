# Siri-LLM

Siri-LLM is a custom Large Language Model (LLM) learning project developed from scratch using domain-specific telecom, wireless communication, and AI datasets collected from iSignal platforms.

The project focuses on understanding how modern Transformer-based architectures work internally by implementing every major component step-by-step in Python.

Rather than using pre-built deep learning frameworks for the core logic, the project builds the Transformer pipeline from first principles, including tokenization, embeddings, positional encoding, self-attention, and multi-head attention.

---

## Project Goals

- Build a custom BPE tokenizer
- Train on telecom and AI-specific corpora
- Generate token embeddings
- Implement sinusoidal positional encoding
- Understand Self-Attention mechanisms
- Implement Multi-Head Attention
- Build Transformer blocks from scratch
- Develop a mini GPT-style architecture
- Learn the foundations of modern LLMs

---

## Features Implemented

### 1. Data Collection and Corpus Creation

- Scrapes text from iSignal platforms
- Builds a domain-specific telecom and AI corpus
- Stores raw corpus for tokenizer training

### 2. Text Cleaning and Preprocessing

- Removes unwanted symbols
- Removes URLs and noisy content
- Normalizes whitespace
- Creates a clean training corpus

### 3. Custom BPE Tokenizer

- Byte Pair Encoding (BPE) implementation
- Vocabulary generation
- Token ID generation
- Token encoding and decoding
- Special token support

Special Tokens:

- `[PAD]`
- `[UNK]`
- `[CLS]`
- `[SEP]`
- `[MASK]`

### 4. Token Embeddings

- Converts token IDs into dense vectors
- Creates embedding lookup tables
- Generates semantic token representations

### 5. Positional Embeddings

- Implements Sinusoidal Positional Encoding
- Preserves sequence order information
- Combines token and positional embeddings

### 6. Self-Attention

- Query (Q) matrix generation
- Key (K) matrix generation
- Value (V) matrix generation
- Attention score computation
- Softmax normalization
- Context vector generation

### 7. Multi-Head Attention

- Parallel attention heads
- Head splitting and concatenation
- Scaled Dot Product Attention
- Output projection layer
- Causal masking

---

## Transformer Learning Pipeline

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
BPE Tokenizer
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
Self Attention
    │
    ▼
Multi-Head Attention
    │
    ▼
Transformer Block
    │
    ▼
Mini GPT-Style Model
```

---

## Project Structure

```text
Siri-LLM/
│
├── README.md
├── LICENSE
│
├── tokenizer/
│   ├── tokenizer.py
│   ├── tokenizing.ipynb
│   ├── raw_corpus.txt
│   ├── isignal_cleaned_corpus.txt
│   ├── isignal_tokenizer.json
│   └── tokenizer_test_output.txt
│
├── embeddings/
│   ├── embedding_vectors.py
│   └── notes.md
│
├── positional_encoding/
│   ├── positional_encoding.py
│   └── notes.md
│
├── attention/
│   ├── attention.py
│   ├── isignal_attention_output.txt
│   └── README.md
│
├── transformer/
│   ├── transformer_block.py
│   └── README.md
│
└── datasets/
```

---

## Learning Resources

To better understand the concepts before implementing them in code, this project also includes a beginner-friendly Jupyter Notebook:

### Self Attention Deep Dive

File:

```text
Self_Attention_Deep_Dive.ipynb
```


## Technologies Used

### Programming Language

- Python

### Libraries

- NumPy
- PyTorch
- Hugging Face Tokenizers
- BeautifulSoup4
- Requests
- Jupyter Notebook

---

## Installation

Clone the repository:

```bash
git clone https://github.com/isignalNR/Siri-LLM.git
cd Siri-LLM
```

Install dependencies:

```bash
pip install numpy
pip install torch
pip install tokenizers
pip install requests
pip install beautifulsoup4
```

Or:

```bash
pip install numpy torch tokenizers requests beautifulsoup4
```

---

## Running the Project

Run the complete pipeline:

```bash
python isignalattentionpipeline.py
```

The pipeline automatically performs:

1. Website Scraping
2. Text Cleaning
3. BPE Tokenizer Training
4. Tokenization
5. Token Embedding Generation
6. Positional Encoding
7. Self-Attention
8. Multi-Head Attention
9. Output Generation

---

## Sample Pipeline Output

Example Input:

```text
5G AI telecom research and machine learning
```

Processing Steps:

```text
Input Text
    ↓
Tokenizer
    ↓
Token IDs
    ↓
Embedding Vectors
    ↓
Positional Encoding
    ↓
Self Attention
    ↓
Multi-Head Attention
    ↓
Context-Aware Representations
```

---

## Attention Mechanism

Self-Attention allows each token to understand its relationship with every other token.

### Query (Q)

What information am I searching for?

### Key (K)

What information do I provide?

### Value (V)

What information do I contain?

### Attention Formula

Attention(Q,K,V)

= Softmax((QKᵀ) / √dₖ)V

---

## Multi-Head Attention

Multi-Head Attention enables the model to learn multiple relationships simultaneously.

Different heads may learn:

- Semantic relationships
- Context relationships
- Positional relationships
- Long-range dependencies

This is the core mechanism used in GPT, BERT, Gemini, and Llama architectures.

---

## Current Progress

### Completed

- Domain Corpus Creation
- Data Cleaning
- Custom BPE Tokenizer
- Token Embeddings
- Positional Embeddings
- Self-Attention
- Multi-Head Attention

### In Progress

- Transformer Block

### Upcoming

- Layer Normalization
- Feed Forward Network
- Residual Connections
- Transformer Decoder
- GPT-Style Language Model
- Text Generation

---

## Learning Outcomes

This project demonstrates:

- NLP preprocessing
- Tokenizer construction
- Embedding generation
- Positional encoding
- Attention mechanisms
- Multi-head attention
- Transformer architecture fundamentals
- GPT-style model foundations

---

## Future Roadmap

### Transformer Block

- LayerNorm
- Residual Connections
- Feed Forward Network

### Decoder Architecture

- Masked Self-Attention
- Stacked Transformer Blocks

### Language Model

- Next-token prediction
- Text generation

### Fine-Tuning

- Domain adaptation
- Telecom-specific conversational AI

---

## Author

**Krishna**

Building a Transformer-based Mini LLM from first principles using Python.

---

## License

This project is released under the MIT License.