# ============================================================
#  ISIGNAL — FULL NLP PIPELINE
#  Scraping → Tokenization → Embeddings → Self-Attention
#                                       → Multi-Head Attention
# ============================================================
#
#  What this script does, top to bottom:
#
#  PART 1 — Scrape text from iSignal websites
#  PART 2 — Clean the scraped text
#  PART 3 — Train a BPE tokenizer on the corpus
#  PART 4 — Token Embeddings  (what each token MEANS)
#  PART 5 — Positional Embeddings (WHERE each token sits)
#  PART 6 — Self-Attention  (tokens look at each other)
#  PART 7 — Multi-Head Attention (parallel attention heads)
#  PART 8 — Final Summary
#
#  Install:  pip install requests beautifulsoup4 tokenizers numpy torch
# ============================================================

import re
import math
import requests
import numpy as np
import torch
import torch.nn as nn
from bs4 import BeautifulSoup
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Fix seeds so random numbers are reproducible every run
torch.manual_seed(42)
np.random.seed(42)


# ╔══════════════════════════════════════════════════════════════╗
# ║              PART 1 — SCRAPE WEBSITES                       ║
# ╚══════════════════════════════════════════════════════════════╝
#
# We send an HTTP GET request to each URL, parse the HTML with
# BeautifulSoup, strip out non-text tags (scripts, styles, etc.),
# and collect all visible text into a list called 'corpus'.
#
# Browser-like User-Agent headers help bypass basic bot-blocking.
# Some sites may still return 403 Forbidden — that is handled
# gracefully with a fallback domain-specific corpus below.
# ============================================================

print("\n" + "=" * 60)
print("  PART 1 — SCRAPING WEBSITES")
print("=" * 60)

urls = [
    "https://isignalresearch.com/",
    "https://academy.isignalresearch.com",
    "https://learn.isignalresearch.com",
    "https://isignaltechblog.blogspot.com/"
]

# Pretend to be a real browser so websites don't block us
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

corpus = []   # Will hold all scraped text (one string per page)

for url in urls:
    try:
        print(f"\n  Scraping → {url}")
        response = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
        print(f"  Status   : {response.status_code}")

        if response.status_code == 200:
            # Parse the HTML and remove non-content tags
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "header", "footer", "svg"]):
                tag.extract()

            text = soup.get_text(separator=" ", strip=True)
            print(f"  Characters extracted : {len(text)}")

            if len(text) > 100:
                corpus.append(text)
                print("  ✓ Added to corpus")
            else:
                print("  ✗ Too short — skipped")
        else:
            print(f"  ✗ HTTP {response.status_code} — could not access website")

    except Exception as e:
        print(f"  ERROR: {e}")

# ── Fallback corpus ──────────────────────────────────────────────────────────
# If scraping failed (403 in restricted environments, no internet, etc.),
# we inject representative iSignal-domain sentences so the rest of the
# pipeline (tokenizer training, attention demo) still runs correctly.
if len(corpus) == 0:
    print("\n  [!] No pages scraped (403 or network restricted).")
    print("      Using built-in iSignal domain corpus as fallback.\n")
    corpus = [
        """
        iSignal Research is a leading telecom and 5G research organization.
        We provide AI-powered signal processing research, machine learning
        solutions for wireless networks, and deep learning models for
        next-generation 5G and 6G telecommunications.
        Our academy offers courses on antenna design, RF engineering,
        beamforming, OFDM, MIMO systems, and spectrum analysis.
        Students learn signal modulation, channel estimation, and network
        optimization using Python and MATLAB simulations.
        Artificial intelligence and machine learning are transforming the
        telecom industry with intelligent base stations and edge computing.
        Deep learning models help predict network congestion, optimize
        handover decisions, and improve quality of service in LTE and 5G networks.
        Research topics include massive MIMO, millimeter wave propagation,
        cognitive radio, software defined networking, and network slicing.
        The iSignal blog covers practical tutorials on wireless communication
        fundamentals, signal detection algorithms, and Python programming
        for telecommunications engineers and researchers.
        Natural language processing, computer vision, and reinforcement
        learning are applied to automate network management tasks.
        """,
        """
        5G New Radio technology enables ultra-reliable low latency
        communication for autonomous vehicles, industrial IoT, and
        augmented reality applications. Signal intelligence research
        focuses on adaptive modulation, coding schemes, and power control
        algorithms for heterogeneous networks. The academy curriculum
        includes digital signal processing, Fourier transforms, wavelet
        analysis, and filter design for communication systems. Machine
        learning methods such as neural networks, support vector machines,
        and random forests are used for signal classification and
        anomaly detection in telecommunication networks.
        """,
    ]

print(f"\n  Total documents in corpus: {len(corpus)}")

# Save raw corpus to disk
with open("raw_corpus.txt", "w", encoding="utf-8") as f:
    for page in corpus:
        f.write(page + "\n\n")
print("  raw_corpus.txt saved")


# ╔══════════════════════════════════════════════════════════════╗
# ║              PART 2 — CLEAN THE TEXT                        ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Raw web text contains noise: URLs, special Unicode characters,
# bracket annotations, and runs of whitespace. We strip all of
# these out so the tokenizer trains on clean, readable text.
# ============================================================

print("\n" + "=" * 60)
print("  PART 2 — CLEANING TEXT")
print("=" * 60)

def clean_text(text):
    """
    Remove noise from raw scraped text.
      - URLs (http://...) removed entirely
      - Content inside [square brackets] removed
      - Special symbols removed; keep letters, digits, basic punctuation
      - Multiple spaces / newlines collapsed to a single space
    """
    text = re.sub(r'http\S+', '', text)                    # Remove URLs
    text = re.sub(r'\[[^\]]+\]', '', text)                  # Remove [bracket content]
    text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', ' ', text)       # Keep only clean chars
    text = re.sub(r'\s+', ' ', text)                        # Collapse whitespace
    return text.strip()

cleaned_corpus = []
for page in corpus:
    cleaned = clean_text(page)
    if len(cleaned.split()) > 10:       # Drop near-empty fragments
        cleaned_corpus.append(cleaned)

print(f"\n  Cleaned documents : {len(cleaned_corpus)}")
if cleaned_corpus:
    print("\n  Sample (first 300 chars):")
    print("  " + cleaned_corpus[0][:300])

# Save cleaned corpus — this is what the tokenizer trains on
with open("isignal_cleaned_corpus.txt", "w", encoding="utf-8") as f:
    for doc in cleaned_corpus:
        f.write(doc + "\n")
print("\n  isignal_cleaned_corpus.txt saved")


# ╔══════════════════════════════════════════════════════════════╗
# ║         PART 3 — TRAIN A BPE TOKENIZER                     ║
# ╚══════════════════════════════════════════════════════════════╝
#
# What is a Tokenizer?
# --------------------
# Neural networks only understand numbers, not words. A tokenizer
# converts text → sequences of integer IDs.
#
# BPE (Byte-Pair Encoding) Algorithm:
#   1. Start with individual characters as the initial vocabulary.
#   2. Count all adjacent character pairs in the corpus.
#   3. Merge the most frequent pair into a new token.
#   4. Repeat until vocab_size tokens have been created.
#
# Result: common words become single tokens; rare words are split
# into subword pieces.
#   "telecom"  → ["telecom"]          (frequent → kept whole)
#   "beamform" → ["beam", "##form"]   (rare → split)
#
# Special tokens:
#   [PAD]  → padding short sequences to equal length in a batch
#   [UNK]  → any word not seen during training
#   [CLS]  → classification marker (placed at sentence start in BERT)
#   [SEP]  → separator between two sentences
#   [MASK] → token the model must predict (used in masked LM training)
# ============================================================

print("\n" + "=" * 60)
print("  PART 3 — TRAINING BPE TOKENIZER")
print("=" * 60)

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()      # First split on whitespace

trainer = BpeTrainer(
    vocab_size=5000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

print("\n  Training on isignal_cleaned_corpus.txt ...")
tokenizer.train(files=["isignal_cleaned_corpus.txt"], trainer=trainer)
tokenizer.save("isignal_tokenizer.json")
VOCAB_SIZE = tokenizer.get_vocab_size()
print(f"  ✓ Tokenizer trained  |  Vocabulary size: {VOCAB_SIZE} tokens")

# ── Encode a representative iSignal domain sentence ──────────────────────────
sample_text = "5G AI telecom research and machine learning signal processing"
encoded     = tokenizer.encode(sample_text)
tokens      = encoded.tokens
token_ids   = encoded.ids

print(f"\n  Demo sentence : \"{sample_text}\"")
print(f"\n  Tokens  : {tokens}")
print(f"  IDs     : {token_ids}")
print(f"  Count   : {len(tokens)} tokens")

# Guard: if sentence produced no tokens (extreme edge case), inject defaults
if len(tokens) == 0:
    print("  [!] Tokenizer produced 0 tokens for the sample sentence.")
    print("      Injecting placeholder tokens for the demo.\n")
    tokens    = ["5G", "AI", "telecom", "research", "machine", "learning", "signal"]
    token_ids = list(range(len(tokens)))   # fake IDs 0,1,2,...


# ╔══════════════════════════════════════════════════════════════╗
# ║         PART 4 — TOKEN EMBEDDINGS                           ║
# ╚══════════════════════════════════════════════════════════════╝
#
# What is a Token Embedding?
# --------------------------
# An embedding converts a discrete integer ID into a dense vector
# of real numbers. This vector represents the token's *meaning*.
#
# Embedding Table (lookup table):
#   Shape = [VOCAB_SIZE  ×  EMBED_DIM]
#            ↑                 ↑
#          5000 rows         16 columns
#       (one per token)   (16 numbers per token)
#
# To get token 312's embedding: take row 312 of the table.
# In PyTorch: table[token_id]  — an O(1) lookup.
#
# In real models (BERT, GPT) this table is LEARNED via gradient
# descent so similar-meaning tokens end up with similar vectors.
# Here we randomly initialize to demonstrate the concept.
# ============================================================

print("\n" + "=" * 60)
print("  PART 4 — TOKEN EMBEDDINGS")
print("=" * 60)

EMBED_DIM = 16    # Dimension of each embedding vector
                  # Real models: BERT=768, GPT-2=1024, GPT-3=12288

# Build random embedding table — shape: [VOCAB_SIZE × EMBED_DIM]
token_embed_table = np.random.randn(VOCAB_SIZE, EMBED_DIM)

print(f"\n  Embedding table : shape {token_embed_table.shape}")
print(f"  → {VOCAB_SIZE} rows   (one per vocabulary token)")
print(f"  → {EMBED_DIM} columns (floating-point dimensions per token)")

# Look up every token in our sample sentence
# Result shape: [num_tokens × EMBED_DIM]
token_embeddings = token_embed_table[token_ids]

print(f"\n  Sample sentence token embeddings  ({len(token_ids)} tokens × {EMBED_DIM} dims):\n")
print(f"  {'Pos':<5} {'Token':<18} {'ID':<7} {'Embedding — first 6 of {EMBED_DIM} dims'}")
print("  " + "-" * 65)
for i, (tok, tid, emb) in enumerate(zip(tokens, token_ids, token_embeddings)):
    print(f"  {i:<5} {tok:<18} {tid:<7} {np.round(emb[:6], 4)}")

print(f"\n  ↑ Each row is the '{EMBED_DIM}-number meaning vector' for that token.")
print("  In a trained model, 'AI' and 'machine' would have similar vectors.")


# ╔══════════════════════════════════════════════════════════════╗
# ║         PART 5 — POSITIONAL EMBEDDINGS (Sinusoidal)         ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Problem:
# Self-Attention (next section) processes all tokens simultaneously
# in parallel — it has NO built-in sense of order.
# "5G AI research" and "research AI 5G" would look the same!
#
# Solution: ADD a positional encoding vector to each token embedding.
# Position 0 always gets the same vector, position 1 always gets
# the same vector, etc.  The model can then learn to use these
# unique "coordinate stamps" to understand word order.
#
# Sinusoidal Formula (Vaswani et al. "Attention is All You Need"):
#
#   PE[pos, 2i]   = sin( pos / 10000^(2i / d_model) )
#   PE[pos, 2i+1] = cos( pos / 10000^(2i / d_model) )
#
# Why sin/cos?
#   • Values always in [-1, 1] — no scaling issues
#   • Each dimension oscillates at a DIFFERENT frequency
#   • Low i → high frequency (nearby positions look different)
#   • High i → low frequency (only very distant positions differ)
#   • The model can learn any relative-position offset
#     from a linear combination of the sin/cos values
#
# Alternative: LEARNED positional embeddings (used in GPT-2, BERT)
#   — a trainable table updated during training just like token embeddings.
# ============================================================

print("\n" + "=" * 60)
print("  PART 5 — POSITIONAL EMBEDDINGS (Sinusoidal)")
print("=" * 60)

def sinusoidal_positional_encoding(seq_len, d_model):
    """
    Build a fixed sinusoidal positional encoding matrix.

    Args:
        seq_len  : number of tokens (rows to produce)
        d_model  : embedding dimension (must equal EMBED_DIM)

    Returns:
        PE : numpy array of shape [seq_len × d_model]
             All values guaranteed in [-1, 1].
    """
    PE = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(d_model // 2):
            # Denominator grows exponentially → lower frequency for higher i
            denom = 10000 ** (2 * i / d_model)

            PE[pos, 2 * i]     = math.sin(pos / denom)   # even index → sin
            PE[pos, 2 * i + 1] = math.cos(pos / denom)   # odd  index → cos

    return PE

num_tokens     = len(token_ids)
pos_embeddings = sinusoidal_positional_encoding(num_tokens, EMBED_DIM)

print(f"\n  Positional embeddings : shape {pos_embeddings.shape}")
print(f"  → One {EMBED_DIM}-dim vector per position — FIXED (not learned)\n")

print(f"  {'Pos':<5} {'Token':<18} {'Positional Embedding (all {EMBED_DIM} dims)'}")
print("  " + "-" * 70)
for i, (tok, pemb) in enumerate(zip(tokens, pos_embeddings)):
    print(f"  {i:<5} {tok:<18} {np.round(pemb, 3)}")

print("\n  ↑ Notice sin/cos alternation and that all values are in [-1, 1].")

# ── Combine Token + Positional → Final Input Embedding ───────────────────────
# Element-wise addition: both matrices are [num_tokens × EMBED_DIM]
# Result: each vector carries BOTH meaning (what) and position (where).
# This combined tensor is the INPUT that goes into the attention layers.
final_embeddings = token_embeddings + pos_embeddings

print(f"\n  Token Embeddings    : {token_embeddings.shape}")
print(f"  + Positional Embed  : {pos_embeddings.shape}")
print(f"  = Final Input       : {final_embeddings.shape}  ← fed into Self-Attention")

print(f"\n  Final Input Embeddings (first 6 dims shown):\n")
print(f"  {'Pos':<5} {'Token':<18} {'Final = Token Emb + Pos Emb (first 6 dims)'}")
print("  " + "-" * 65)
for i, (tok, emb) in enumerate(zip(tokens, final_embeddings)):
    print(f"  {i:<5} {tok:<18} {np.round(emb[:6], 4)}")


# ╔══════════════════════════════════════════════════════════════╗
# ║         PART 6 — SELF-ATTENTION                             ║
# ╚══════════════════════════════════════════════════════════════╝
#
# What does Self-Attention do?
# ----------------------------
# It lets every token look at every other token and compute a
# weighted combination of their value vectors.
#
# Intuition: when processing the word "it" in the sentence
# "The signal failed because it was too noisy", self-attention
# helps the model link "it" back to "signal" — even though they
# are several words apart.
#
# Three Roles  (implemented as three linear projections):
# ─────────────────────────────────────────────────────────────
#  QUERY  W_Q :  "What am I searching for?"
#  KEY    W_K :  "What do I offer / advertise as a key?"
#  VALUE  W_V :  "What information do I actually carry?"
#
# Library analogy:
#   Q = your search query  ("machine learning papers")
#   K = book index keywords / titles
#   V = actual book contents
#
# Forward Pass — 5 Steps:
# ─────────────────────────────────────────────────────────────
#  1.  Q = X · W_Q           project each token to "query space"
#      K = X · W_K           project each token to "key space"
#      V = X · W_V           project each token to "value space"
#
#  2.  scores = Q · Kᵀ       dot product → similarity matrix (T × T)
#               ↑ entry [i,j] = how much token i should attend to token j
#
#  3.  scores /= √d_k        scale to prevent exploding gradients
#               ↑ d_k = key dimension; without this, large dims → near-one-hot softmax
#
#  4.  weights = softmax(scores, dim=-1)
#               ↑ each row becomes a probability distribution (sum to 1)
#
#  5.  output = weights · V  weighted sum of value vectors
#               ↑ each output vector is a blend of all tokens' values,
#                 weighted by how relevant each one is
# ============================================================

print("\n" + "=" * 60)
print("  PART 6 — SELF-ATTENTION  (Q, K, V Mechanism)")
print("=" * 60)

# Convert final_embeddings numpy array → PyTorch float tensor
# Shape: [num_tokens × EMBED_DIM]
X = torch.tensor(final_embeddings, dtype=torch.float32)

D_IN  = EMBED_DIM   # input dim  = 16
D_OUT = 8           # output dim = 8  (can differ from D_IN)
                    # In GPT-2: D_IN = D_OUT = 768 / num_heads

print(f"\n  Input X shape     : {tuple(X.shape)}")
print(f"  D_IN (input dim)  : {D_IN}")
print(f"  D_OUT (output dim): {D_OUT}")

# ── Three Learnable Projection Matrices ──────────────────────────────────────
# nn.Linear(in, out, bias=False) = a matrix multiply: y = x · Wᵀ
# In a real model these weights are updated by backpropagation.
W_Q = nn.Linear(D_IN, D_OUT, bias=False)
W_K = nn.Linear(D_IN, D_OUT, bias=False)
W_V = nn.Linear(D_IN, D_OUT, bias=False)

# ── STEP 1: Compute Q, K, V ──────────────────────────────────────────────────
Q = W_Q(X)   # Shape: [T × D_OUT]   each token's "search query"
K = W_K(X)   # Shape: [T × D_OUT]   each token's "index keyword"
V = W_V(X)   # Shape: [T × D_OUT]   each token's "actual content"

print(f"\n  STEP 1 — Project X into Q, K, V")
print(f"  {'Matrix':<8} {'Shape':<22} Role")
print("  " + "-" * 55)
print(f"  {'Q':<8} {str(tuple(Q.shape)):<22} What each token is searching for")
print(f"  {'K':<8} {str(tuple(K.shape)):<22} What each token offers as a key")
print(f"  {'V':<8} {str(tuple(V.shape)):<22} What each token actually holds")

# ── STEP 2: Attention Scores = Q · Kᵀ ───────────────────────────────────────
# Matrix multiply (T×D_OUT) × (D_OUT×T) → (T×T)
# attn_scores[i][j] = dot product of query_i with key_j
# High value = token i finds token j highly relevant
attn_scores = Q @ K.transpose(-2, -1)   # Shape: [T × T]

print(f"\n  STEP 2 — Attention Scores (Q · Kᵀ)  shape: {tuple(attn_scores.shape)}")
print(f"  Entry [i,j] = dot product of query_i with key_j\n")

score_np = attn_scores.detach().numpy()
col_hdr  = "  " + " " * 20 + "".join(f"{t[:7]:>9}" for t in tokens)
print(col_hdr)
for i, tok in enumerate(tokens):
    row = "  " + f"{tok[:16]:<18}  " + "".join(f"{score_np[i,j]:>9.3f}" for j in range(len(tokens)))
    print(row)

# ── STEP 3: Scale by √d_k ────────────────────────────────────────────────────
# Without scaling, as d_k grows the dot products grow in magnitude
# → softmax produces near-one-hot distributions → gradients vanish.
# Dividing by √d_k keeps variance ≈ 1 regardless of dimension size.
d_k         = K.shape[-1]                   # = D_OUT = 8
attn_scaled = attn_scores / math.sqrt(d_k) # Normalize scores

print(f"\n  STEP 3 — Scale by √d_k = √{d_k} ≈ {math.sqrt(d_k):.3f}")
print(f"  (Keeps gradient flow healthy as embedding dimension grows)")

# ── STEP 4: Softmax → Attention Weights ──────────────────────────────────────
# Softmax(x_i) = exp(x_i) / Σ exp(x_j)
# Each row becomes a probability distribution summing to 1.0.
# Row i = "when producing output for token i, what % of attention
#          to pay to each other token?"
attn_weights = torch.softmax(attn_scaled, dim=-1)   # Shape: [T × T]

print(f"\n  STEP 4 — Softmax → Attention Weights  shape: {tuple(attn_weights.shape)}")
print(f"  Each row sums to 1.0  (like a probability distribution)\n")

w_np = attn_weights.detach().numpy()
print(col_hdr)
for i, tok in enumerate(tokens):
    row = "  " + f"{tok[:16]:<18}  " + "".join(f"{w_np[i,j]:>9.3f}" for j in range(len(tokens)))
    print(row)

row_sums = attn_weights.sum(dim=-1).detach().numpy()
print(f"\n  Row sums : {np.round(row_sums, 4)}  ← should all be 1.0")

# ── STEP 5: Context Vectors = Attention Weights × V ──────────────────────────
# (T×T) × (T×D_OUT) → (T×D_OUT)
# Each output vector is a weighted average of ALL value vectors.
# Tokens that received high attention weights contribute more.
context_vecs = attn_weights @ V   # Shape: [T × D_OUT]

print(f"\n  STEP 5 — Context Vectors  (Attn Weights × V)  shape: {tuple(context_vecs.shape)}")
print(f"\n  {'Pos':<5} {'Token':<18} {'Context Vector (all {D_OUT} dims)'}")
print("  " + "-" * 60)
cv_np = context_vecs.detach().numpy()
for i, tok in enumerate(tokens):
    print(f"  {i:<5} {tok:<18} {np.round(cv_np[i], 4)}")

print("\n  ↑ Each context vector is a rich blend of ALL tokens' information,")
print("  proportional to how much attention each token received.")

# ── Most attended token pair ──────────────────────────────────────────────────
# Find the strongest off-diagonal attention weight (exclude self-attention)
if len(tokens) > 1:
    w_tmp = w_np.copy()
    np.fill_diagonal(w_tmp, 0)   # zero out self-attention diagonal
    max_i, max_j = np.unravel_index(w_tmp.argmax(), w_tmp.shape)
    print(f"\n  Strongest cross-token attention:")
    print(f"  '{tokens[max_i]}' → '{tokens[max_j]}'  weight = {w_np[max_i, max_j]:.4f}")
    print(f"  ('{tokens[max_i]}' pays the most attention to '{tokens[max_j]}')")


# ╔══════════════════════════════════════════════════════════════╗
# ║         PART 7 — MULTI-HEAD ATTENTION                       ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Why Multiple Heads?
# -------------------
# A single self-attention head learns ONE kind of relationship
# between tokens at a time.  But text has many different kinds
# of structure simultaneously:
#
#   Head 1 might focus on → grammatical subject–verb links
#   Head 2 might focus on → semantic similarity / synonyms
#   Head 3 might focus on → pronoun co-reference ("it" → "signal")
#   Head 4 might focus on → neighboring / positional proximity
#
# By running H heads IN PARALLEL, the model captures all of these
# at once and combines the evidence before producing its output.
#
# Architecture:
#   Input X
#     ├─ Head 1: W_Q1, W_K1, W_V1 → context_1
#     ├─ Head 2: W_Q2, W_K2, W_V2 → context_2
#     │   ...
#     └─ Head H: W_QH, W_KH, W_VH → context_H
#               ↓
#          Concatenate [context_1, ..., context_H]
#               ↓
#          W_out (linear projection)
#               ↓
#          Final Output
#
# Efficient Implementation — the Weight-Splitting Trick:
# ───────────────────────────────────────────────────────
# Instead of H separate attention modules, we:
#   1. Project X into a LARGE Q, K, V  (dim = d_out = H × head_dim)
#   2. RESHAPE / SPLIT that large dim into H chunks using .view()
#   3. Run ALL H heads simultaneously using 4-D tensor operations
# This is mathematically identical to H separate heads but uses
# ONE matrix multiply instead of H → much faster on GPU.
#
# Causal (look-ahead) Mask:
# ─────────────────────────
# For language modeling (like GPT), when predicting token at
# position i, the model must NOT be allowed to see tokens at
# positions i+1, i+2, ...  (that would be cheating).
# We enforce this by setting future attention scores to -∞
# before softmax.  exp(-∞) = 0, so those weights become zero.
#
# Dropout on Attention Weights:
# ─────────────────────────────
# Randomly zeros out some attention weights during training.
# Prevents the model from over-relying on any single connection.
# At inference time, dropout is turned off (model.eval()).
#
# Tensor Shape Journey through MultiHeadAttention:
# ─────────────────────────────────────────────────
#   Input X              : (B, T, d_in)
#   After W_Q/K/V        : (B, T, d_out)        d_out = H × head_dim
#   After .view()        : (B, T, H, head_dim)
#   After .transpose(1,2): (B, H, T, head_dim)  ← H heads in parallel
#   Q·Kᵀ scores         : (B, H, T, T)
#   After mask + softmax : (B, H, T, T)
#   × V                  : (B, H, T, head_dim)
#   After .transpose(1,2): (B, T, H, head_dim)
#   After .view()        : (B, T, d_out)         heads merged (concatenated)
#   After out_proj       : (B, T, d_out)          final linear mix
# ============================================================

print("\n" + "=" * 60)
print("  PART 7 — MULTI-HEAD ATTENTION")
print("=" * 60)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention Module.

    Uses the weight-splitting trick for efficiency:
    one large W_Q/K/V matrix is split into H heads via reshape,
    rather than creating H separate attention modules.

    Args:
        d_in           : input embedding dimension
        d_out          : total output dimension  (= num_heads × head_dim)
        context_length : maximum sequence length (for the causal mask)
        num_heads      : number of parallel attention heads (H)
        dropout        : attention dropout rate (0.0 = no dropout)
    """

    def __init__(self, d_in, d_out, context_length, num_heads, dropout=0.0):
        super().__init__()

        # Each head must get an equal share of the output dimension
        assert d_out % num_heads == 0, \
            f"d_out ({d_out}) must be divisible by num_heads ({num_heads})"

        self.d_out     = d_out
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads   # private dimension per head

        # ONE large projection per role instead of H separate ones.
        # bias=False: standard in most transformer implementations.
        self.W_Q = nn.Linear(d_in, d_out, bias=False)   # all heads' Q weights combined
        self.W_K = nn.Linear(d_in, d_out, bias=False)   # all heads' K weights combined
        self.W_V = nn.Linear(d_in, d_out, bias=False)   # all heads' V weights combined

        # Projects the concatenated head outputs back to d_out.
        # This allows heads to "communicate" before passing to next layer.
        self.out_proj = nn.Linear(d_out, d_out, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Causal mask: upper triangle = 1 means "future position — block this"
        # register_buffer: not a learnable param; travels with model to GPU/CPU.
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """
        Args:
            x : (B, T, d_in)  — B=batch size, T=seq length, d_in=input dim
        Returns:
            out          : (B, T, d_out)   enriched token representations
            attn_weights : (B, H, T, T)    attention distribution per head
        """
        B, T, d_in = x.shape

        # ── A: Project to Q, K, V  → (B, T, d_out) ──────────────────────────
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # ── B: Split into H heads ─────────────────────────────────────────────
        # (B, T, d_out)
        #   .view(B, T, H, head_dim) → split last dim into H chunks
        #   .transpose(1, 2)         → (B, H, T, head_dim)  heads axis second
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape now: (B, H, T, head_dim) — all H heads live side by side

        # ── C: Scaled dot-product attention (all H heads at once) ─────────────
        # (B,H,T,hd) × (B,H,hd,T) → (B,H,T,T)
        attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)

        # ── D: Apply causal mask — block future positions ─────────────────────
        # mask[:T,:T] selects the T×T top-left submatrix for sequences < context_length
        # masked_fill: wherever mask==1 (future), set score to -infinity
        attn_scores = attn_scores.masked_fill(
            self.mask[:T, :T].bool(), float('-inf')
        )

        # ── E: Softmax + dropout ──────────────────────────────────────────────
        # exp(-inf) = 0 → masked future positions get zero weight automatically
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # ── F: Weighted sum of values ─────────────────────────────────────────
        # (B,H,T,T) × (B,H,T,hd) → (B,H,T,hd)
        out = attn_weights @ V

        # ── G: Merge H heads back into single tensor ──────────────────────────
        # .transpose(1,2): (B,H,T,hd) → (B,T,H,hd)
        # .contiguous():   ensure memory is laid out sequentially (needed by .view)
        # .view(B,T,d_out): merge H×hd → d_out  (equivalent to concatenating heads)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_out)

        # ── H: Final projection — mix information across heads ────────────────
        out = self.out_proj(out)

        return out, attn_weights


# ── Instantiate Multi-Head Attention ─────────────────────────────────────────
NUM_HEADS      = 2
D_MHA_OUT      = 8       # total output dim  =  num_heads × head_dim  =  2 × 4
CONTEXT_LENGTH = 128

mha = MultiHeadAttention(
    d_in           = EMBED_DIM,       # 16
    d_out          = D_MHA_OUT,       # 8   (=2 heads × 4 head_dim)
    context_length = CONTEXT_LENGTH,
    num_heads      = NUM_HEADS,
    dropout        = 0.0              # 0 = no dropout for this demo
)

# Add batch dimension: (T, d_in) → (1, T, d_in)
# Real training: batch_size > 1 (many sentences processed together)
X_batch = X.unsqueeze(0)   # Shape: (1, num_tokens, EMBED_DIM)

mha_out, mha_weights = mha(X_batch)

print(f"\n  Configuration:")
print(f"  {'Num heads':<22}: {NUM_HEADS}")
print(f"  {'Head dim':<22}: {mha.head_dim}   ({D_MHA_OUT} / {NUM_HEADS})")
print(f"  {'Causal mask':<22}: YES  (upper triangle blocked)")
print(f"\n  Shapes:")
print(f"  {'Input':<22}: {tuple(X_batch.shape)}")
print(f"  {'Output':<22}: {tuple(mha_out.shape)}")
print(f"  {'Attn weights':<22}: {tuple(mha_weights.shape)}")
print(f"                          ↑ (batch=1, heads={NUM_HEADS}, tokens={num_tokens}, tokens={num_tokens})")

# ── Per-head attention weight tables ─────────────────────────────────────────
for h in range(NUM_HEADS):
    print(f"\n  ─── Head {h+1} Attention Weights ───")
    print(f"  (causal mask applied: each token can only see PAST + SELF)\n")

    head_w = mha_weights[0, h].detach().numpy()   # (T, T)

    col_hdr = "  " + " " * 20 + "".join(f"{t[:7]:>9}" for t in tokens)
    print(col_hdr)

    for i, tok in enumerate(tokens):
        row = "  " + f"{tok[:16]:<18}  "
        for j in range(len(tokens)):
            if j > i:
                row += f"{'───':>9}"       # future token — masked
            else:
                row += f"{head_w[i,j]:>9.3f}"
        print(row)

    # Report what each token attends to most
    print(f"\n  Head {h+1} strongest attention per token:")
    for i, tok in enumerate(tokens):
        visible = head_w[i, :i+1]       # only past + self visible
        top_j   = int(np.argmax(visible))
        print(f"    '{tok}'  →  most attends to  '{tokens[top_j]}'  ({visible[top_j]:.3f})")

# ── Final context vectors from MHA ───────────────────────────────────────────
print(f"\n  ─── Multi-Head Attention Output Context Vectors ───")
print(f"  shape {tuple(mha_out[0].shape)}  ({num_tokens} tokens × {D_MHA_OUT} output dims)\n")
print(f"  {'Pos':<5} {'Token':<18} {'Output Vector (all {D_MHA_OUT} dims)'}")
print("  " + "-" * 60)
mha_out_np = mha_out[0].detach().numpy()
for i, tok in enumerate(tokens):
    print(f"  {i:<5} {tok:<18} {np.round(mha_out_np[i], 4)}")

print(f"\n  ↑ Each token now carries a {D_MHA_OUT}-dim vector that blends")
print(f"  information from ALL visible past tokens, as judged by")
print(f"  {NUM_HEADS} independent attention heads working in parallel.")


# ╔══════════════════════════════════════════════════════════════╗
# ║         PART 8 — FINAL SUMMARY                              ║
# ╚══════════════════════════════════════════════════════════════╝

print("\n" + "=" * 60)
print("  PART 8 — PIPELINE SUMMARY")
print("=" * 60)

print(f"""
  ╭─────────────────────────────────────────────────────╮
  │  iSignal NLP Pipeline — Data Flow                   │
  ╰─────────────────────────────────────────────────────╯

  [1] Raw Text  (scraped from iSignal URLs)
        ↓  clean_text(): remove URLs, symbols, extra spaces
  [2] Cleaned Corpus  → isignal_cleaned_corpus.txt

        ↓  BpeTrainer(vocab_size=5000)
  [3] BPE Tokenizer  → isignal_tokenizer.json
        Vocab size: {VOCAB_SIZE} tokens

        ↓  tokenizer.encode(sentence)
  [4] Token IDs  →  {token_ids}
         Tokens  →  {tokens}

        ↓  token_embed_table[token_ids]
  [5] Token Embeddings  →  shape {token_embeddings.shape}
        Each token = a {EMBED_DIM}-dim vector of real numbers
        (WHAT each token means)

        ↓  + sinusoidal_positional_encoding()
  [6] Final Input Embeddings  →  shape {final_embeddings.shape}
        (WHAT + WHERE combined in one vector)

        ↓  W_Q, W_K, W_V  →  Q·Kᵀ/√d_k  →  softmax  →  ×V
  [7] Self-Attention  →  context shape {tuple(context_vecs.shape)}
        Each token has attended to ALL other tokens once.

        ↓  {NUM_HEADS} parallel heads, causal mask, out_proj
  [8] Multi-Head Attention  →  output shape {tuple(mha_out_np.shape)}
        {NUM_HEADS} perspectives combined into one enriched representation.

  ╭─────────────────────────────────────────────────────╮
  │  What each stage contributes:                        │
  │  Token Embedding   → WHAT  (meaning)                 │
  │  Positional Embed  → WHERE (order)                   │
  │  Self-Attention    → HOW   (relationships)           │
  │  Multi-Head Attn   → WHY   (multiple relationship    │
  │                             types at once)           │
  ╰─────────────────────────────────────────────────────╯

  What to build next (Transformer Decoder / GPT):
    • Layer Normalization  (stabilizes training)
    • Feed-Forward Network (2-layer MLP per token)
    • Residual (skip) connections
    • Stack N Transformer blocks
    • Linear + Softmax language model head
    ──────────────────────────────────────────
    → You have a GPT!
""")

# ── Save all key tensors to a log file ───────────────────────────────────────
with open("isignal_attention_output.txt", "w", encoding="utf-8") as f:
    f.write(f"Sample sentence      : {sample_text}\n")
    f.write(f"Tokens               : {tokens}\n")
    f.write(f"Token IDs            : {token_ids}\n\n")
    f.write(f"Token Embeddings     shape={token_embeddings.shape}:\n{np.round(token_embeddings,4)}\n\n")
    f.write(f"Positional Embeddings shape={pos_embeddings.shape}:\n{np.round(pos_embeddings,4)}\n\n")
    f.write(f"Final Input Embeddings shape={final_embeddings.shape}:\n{np.round(final_embeddings,4)}\n\n")
    f.write(f"Self-Attention Weights shape={tuple(attn_weights.shape)}:\n{np.round(w_np,4)}\n\n")
    f.write(f"Self-Attn Context Vectors shape={tuple(context_vecs.shape)}:\n{np.round(cv_np,4)}\n\n")
    for h in range(NUM_HEADS):
        hw = mha_weights[0,h].detach().numpy()
        f.write(f"MHA Head {h+1} Weights shape={tuple(hw.shape)}:\n{np.round(hw,4)}\n\n")
    f.write(f"MHA Output Vectors shape={mha_out_np.shape}:\n{np.round(mha_out_np,4)}\n")

print("  Files saved:")
print("  1. raw_corpus.txt                — Raw scraped / fallback text")
print("  2. isignal_cleaned_corpus.txt    — Cleaned training corpus")
print("  3. isignal_tokenizer.json        — Trained BPE tokenizer")
print("  4. isignal_attention_output.txt  — All tensors & attention weights")
print("\n  ✓ Complete pipeline finished successfully!\n")
