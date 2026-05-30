# Siri-LLM

Siri-LLM is a custom Large Language Model (LLM) learning project developed from scratch using domain-specific telecom, wireless communication, and AI datasets collected from iSignal platforms.

The project focuses on understanding the internal building blocks of modern Transformer-based models by implementing each component step-by-step in Python.

---

## Project Goals

* Build a custom BPE tokenizer
* Generate token embeddings
* Implement positional embeddings
* Understand self-attention mechanisms
* Implement multi-head attention
* Build Transformer blocks from scratch
* Create a mini educational LLM architecture

---

## Features Implemented

### Custom BPE Tokenizer

* Web scraping based corpus collection
* Text cleaning and preprocessing
* Vocabulary generation
* Byte Pair Encoding (BPE) tokenizer training
* Token encoding and decoding

### Token Embeddings

* Vocabulary-to-vector mapping
* Embedding lookup table creation
* Token ID to dense vector conversion
* Embedding matrix generation

### Positional Embeddings

* Sinusoidal positional encoding
* Position-aware token representations
* Combination of token embeddings and positional vectors

The tokenizer successfully learns domain-specific tokens such as:

* 5G
* AI
* telecom
* RAN
* Wireless
* Machine Learning
* Deep Learning

from the custom telecom and AI corpus.

---

## Technologies Used

* Python
* NumPy
* Hugging Face Tokenizers
* BeautifulSoup4
* Requests
* Jupyter Notebook

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
│   └── embedding_vectors.py
│
├── positional_encoding/
│   └── positional_encoding.py
│
├── self_attention/
│   └── self_attention.py
│
├── multi_head_attention/
│   └── multihead_attention.py
│
├── transformer/
│   └── transformer_block.py
│
└── datasets/
```

---

## Current Progress

### Completed

* Corpus Collection
* Text Cleaning
* BPE Tokenizer Training
* Tokenization Testing
* Token Embedding Generation
* Sinusoidal Positional Encoding

Example Output: The sentence

"5G AI telecom research and machine learning"

is tokenized into learned subword tokens and converted into token embeddings and positional embeddings.

### In Progress

* Self Attention

### Upcoming

* Multi-Head Attention
* Transformer Block
* Mini Transformer Model

---

## Installation

```bash
pip install tokenizers
pip install requests
pip install beautifulsoup4
pip install numpy
```

---

## Run the Project

```bash
isignal_tokenizer_with_embeddings (1).py
```

The script performs:

1. Corpus Collection
2. Text Cleaning
3. Tokenizer Training
4. Tokenization Testing
5. Token Embedding Generation
6. Positional Embedding Generation

The implementation combines tokenizer training, token embeddings, and positional embeddings into a single educational pipeline.

---

## Learning Outcomes

After completing this project, you will understand:

* How tokenizers work internally
* How embedding vectors represent meaning
* Why positional encoding is required
* How attention mechanisms work
* How Transformers process language
* Foundations of GPT-style architectures

---

## Author

Krishna

Building a Transformer-based Mini LLM from first principles using Python.
