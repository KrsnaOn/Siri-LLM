# ============================================================
# ISIGNAL CUSTOM TOKENIZER TRAINING PIPELINE
# WITH TOKEN EMBEDDINGS + POSITIONAL EMBEDDINGS
# ============================================================

# ============================================================
# INSTALL REQUIRED LIBRARIES FIRST
# pip install requests beautifulsoup4 tokenizers numpy
# ============================================================

import requests
from bs4 import BeautifulSoup
import re
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# ============================================================
# URL DATASET
# ============================================================

urls = [
    "https://isignalresearch.com/",
    "https://academy.isignalresearch.com",
    "https://learn.isignalresearch.com",
    "https://isignaltechblog.blogspot.com/"
]

# ============================================================
# HEADERS — Browser-like headers to bypass basic bot blocking
# ============================================================

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ============================================================
# EMPTY LIST TO STORE SCRAPED TEXT
# ============================================================

corpus = []

# ============================================================
# WEBSITE SCRAPING LOOP
# ============================================================

for url in urls:

    try:

        print("\n================================================")
        print(f"Scraping Website: {url}")

        response = requests.get(
            url,
            headers=headers,
            timeout=20,
            allow_redirects=True
        )

        print("Status Code:", response.status_code)

        if response.status_code == 200:

            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup(["script", "style", "noscript", "header", "footer", "svg"]):
                tag.extract()

            text = soup.get_text(separator=" ", strip=True)

            print("Extracted Characters:", len(text))

            if len(text) > 100:
                corpus.append(text)
                print("SUCCESSFULLY ADDED TO CORPUS")
            else:
                print("Very small text skipped")

        else:
            print("FAILED TO ACCESS WEBSITE")

    except Exception as e:
        print("ERROR OCCURRED:")
        print(e)

# ============================================================
# CHECK SCRAPED DATA
# ============================================================

print("\n================================================")
print("TOTAL PAGES SCRAPED:", len(corpus))

# ============================================================
# SAVE RAW CORPUS
# ============================================================

with open("raw_corpus.txt", "w", encoding="utf-8") as f:
    for line in corpus:
        f.write(line + "\n\n")

print("\nRAW CORPUS SAVED")

# ============================================================
# TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):
    text = re.sub(r'http\S+', '', text)           # Remove URLs
    text = re.sub(r'\[[^\]]+\]', '', text)         # Remove bracket content
    text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', ' ', text)  # Remove special symbols
    text = re.sub(r'\s+', ' ', text)               # Remove extra spaces
    return text.strip()

# ============================================================
# CLEAN ALL SCRAPED TEXT
# ============================================================

cleaned_corpus = []

for line in corpus:
    cleaned = clean_text(line)
    if len(cleaned.split()) > 10:
        cleaned_corpus.append(cleaned)

print("\n================================================")
print("CLEANED DOCUMENTS:", len(cleaned_corpus))

if len(cleaned_corpus) > 0:
    print("\nSAMPLE CLEANED TEXT:\n")
    print(cleaned_corpus[0][:500])

with open("isignal_cleaned_corpus.txt", "w", encoding="utf-8") as f:
    for line in cleaned_corpus:
        f.write(line + "\n")

print("\nCLEANED CORPUS SAVED")

# ============================================================
# CREATE & TRAIN TOKENIZER
# ============================================================

print("\n================================================")
print("CREATING TOKENIZER")

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=5000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

print("\nTRAINING TOKENIZER...")

tokenizer.train(files=["isignal_cleaned_corpus.txt"], trainer=trainer)
tokenizer.save("isignal_tokenizer.json")

print("\nTOKENIZER TRAINED AND SAVED")

# ============================================================
# TEST TOKENIZER — Encode a sample sentence
# ============================================================

print("\n================================================")
print("TESTING TOKENIZER")

sample_text = "5G AI telecom research and machine learning"

encoded = tokenizer.encode(sample_text)

print("\nORIGINAL TEXT:")
print(sample_text)

print("\nTOKENS (subword pieces the BPE model split words into):")
print(encoded.tokens)

print("\nTOKEN IDs (integer index of each token in the vocabulary):")
print(encoded.ids)


# ============================================================
# ============================================================
# SECTION: TOKEN EMBEDDINGS
# ============================================================
# ============================================================
#
# What is a Token Embedding?
# --------------------------
# After tokenization, each token (like "5G", "AI", "telecom")
# gets converted into a dense numeric vector.
#
# This vector has a fixed size called EMBEDDING_DIM.
# In real transformers (BERT, GPT), these vectors are LEARNED
# during training so similar words get similar vectors.
#
# Here, we use random initialization (np.random.randn) to
# simulate this — the embedding table is a 2D matrix of shape:
#
#     [VOCAB_SIZE  x  EMBEDDING_DIM]
#
# To get a token's embedding, we simply index this matrix
# using the token's integer ID.
# ============================================================

print("\n================================================")
print("TOKEN EMBEDDINGS")
print("================================================")

# --- Settings ---

VOCAB_SIZE    = tokenizer.get_vocab_size()  # Total unique tokens in vocabulary
EMBEDDING_DIM = 16                          # Size of each embedding vector
                                            # (BERT uses 768, GPT-2 uses 1024)
                                            # We use 16 for readable terminal output

# --- Build the Embedding Table ---
# Shape: [VOCAB_SIZE x EMBEDDING_DIM]
# Each row = one token's embedding vector (randomly initialized here)

np.random.seed(42)  # Seed for reproducibility

token_embedding_table = np.random.randn(VOCAB_SIZE, EMBEDDING_DIM)

print(f"\nEmbedding Table Shape  : {token_embedding_table.shape}")
print(f"  → Rows               : {VOCAB_SIZE}  (one row per vocabulary token)")
print(f"  → Columns            : {EMBEDDING_DIM}  (dimensions per embedding vector)")

# --- Look Up Each Token's Embedding ---
# token_ids = list of integer IDs from tokenizer
# We index the embedding table with these IDs

token_ids = encoded.ids  # e.g. [312, 87, 1045, ...]

# token_embedding_table[token_ids] gives shape: [NUM_TOKENS x EMBEDDING_DIM]
token_embeddings = token_embedding_table[token_ids]

print(f"\nTokens in Sample Sentence : {len(token_ids)}")
print(f"Token Embeddings Shape    : {token_embeddings.shape}")
print(f"  → Rows                  : {len(token_ids)}  (one row per token in sentence)")
print(f"  → Columns               : {EMBEDDING_DIM}  (embedding dimensions)")

# --- Print Each Token's Embedding Vector ---

print("\n--- Token Embeddings (each token → its vector) ---\n")

for i, (token, tid, emb) in enumerate(zip(encoded.tokens, token_ids, token_embeddings)):

    print(f"  Position {i:>2} | Token: '{token:<12}' | ID: {tid:>5} | Embedding (first 8 dims): {np.round(emb[:8], 4)}")
    #                                                                              ↑
    #   Showing only first 8 of 16 dims to keep output readable.
    #   In practice all 16 (or 768) dims are used.


# ============================================================
# ============================================================
# SECTION: POSITIONAL EMBEDDINGS
# ============================================================
# ============================================================
#
# Why Positional Embeddings?
# --------------------------
# Transformers process ALL tokens simultaneously (no loops).
# This means the model has NO built-in sense of word order.
#
# Positional Embeddings inject order information by adding
# a unique position vector to each token's embedding.
#
# Two common approaches:
#
# 1. LEARNED Positional Embeddings (used in BERT, GPT-2)
#    → A trainable table of shape [MAX_SEQ_LEN x EMBEDDING_DIM]
#    → The model learns the best position vectors during training
#
# 2. SINUSOIDAL Positional Embeddings (original "Attention is All You Need")
#    → Fixed mathematical formula using sin/cos waves
#    → No training needed — positions are computed analytically
#    → Formula:
#         PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
#         PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
#
# We implement BOTH below so you can compare them.
# ============================================================

print("\n\n================================================")
print("POSITIONAL EMBEDDINGS")
print("================================================")

MAX_SEQ_LEN = 128  # Maximum number of tokens a sequence can have
                   # (BERT uses 512, GPT-2 uses 1024)
                   # We use 128 as a reasonable demo value

# ============================================================
# METHOD 1 — LEARNED Positional Embeddings
# ============================================================
# Just like the token embedding table, this is a random matrix
# that would be updated via backpropagation during training.
# Shape: [MAX_SEQ_LEN x EMBEDDING_DIM]

print("\n--- METHOD 1: Learned Positional Embedding Table ---")

learned_pos_table = np.random.randn(MAX_SEQ_LEN, EMBEDDING_DIM)

print(f"\nLearned Position Table Shape : {learned_pos_table.shape}")
print(f"  → Rows    : {MAX_SEQ_LEN}  (one row per possible position 0 to {MAX_SEQ_LEN-1})")
print(f"  → Columns : {EMBEDDING_DIM}  (same dimension as token embeddings)")

# Look up positions [0, 1, 2, ..., num_tokens-1] for our sentence
num_tokens = len(token_ids)
positions  = np.arange(num_tokens)            # [0, 1, 2, ...]

learned_pos_embeddings = learned_pos_table[positions]  # Shape: [num_tokens x EMBEDDING_DIM]

print(f"\nPositions for our sentence    : {positions.tolist()}")
print(f"Learned Pos Embeddings Shape  : {learned_pos_embeddings.shape}")

print("\n  (First 8 dims of each position's learned vector)")
for pos, pemb in enumerate(learned_pos_embeddings):
    print(f"  Position {pos} → {np.round(pemb[:8], 4)}")


# ============================================================
# METHOD 2 — SINUSOIDAL Positional Embeddings
# ============================================================

print("\n\n--- METHOD 2: Sinusoidal Positional Embeddings (Fixed Formula) ---")

def sinusoidal_positional_encoding(seq_len, d_model):
    """
    Compute sinusoidal positional encodings.

    Parameters:
        seq_len : number of positions (tokens)
        d_model : embedding dimension size

    Returns:
        PE matrix of shape [seq_len x d_model]

    Formula:
        PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
    """

    PE = np.zeros((seq_len, d_model))  # Initialize with zeros

    # pos = token position index (0, 1, 2, ...)
    # i   = dimension index (0, 1, 2, ..., d_model/2 - 1)

    for pos in range(seq_len):
        for i in range(d_model // 2):

            # Denominator grows exponentially with dimension index i
            # This creates different frequencies for each dimension
            denominator = 10000 ** (2 * i / d_model)

            PE[pos, 2 * i]     = np.sin(pos / denominator)  # Even dimensions → sine
            PE[pos, 2 * i + 1] = np.cos(pos / denominator)  # Odd  dimensions → cosine

    return PE

# Compute sinusoidal encodings for the exact number of tokens in our sentence
sin_pos_embeddings = sinusoidal_positional_encoding(num_tokens, EMBEDDING_DIM)

print(f"\nSinusoidal PE Shape : {sin_pos_embeddings.shape}")
print(f"  → No training needed — values computed from fixed math formula")

print("\n  (All 16 dims of each position's sinusoidal vector)")
for pos, pemb in enumerate(sin_pos_embeddings):
    print(f"  Position {pos} → {np.round(pemb, 4)}")

# Notice: values are always in range [-1, 1] due to sin/cos
print("\n  Note: All values are in [-1, 1] range (property of sin/cos)")


# ============================================================
# ============================================================
# SECTION: FINAL INPUT EMBEDDINGS
# ============================================================
# ============================================================
#
# In a real transformer model, the final embedding fed into
# the attention layers is simply:
#
#     Final Input = Token Embedding + Positional Embedding
#
# Both matrices have the same shape [num_tokens x EMBEDDING_DIM]
# so they can be added element-wise.
#
# This combined vector carries BOTH:
#   → What the token means (token embedding)
#   → Where the token sits in the sequence (positional embedding)
# ============================================================

print("\n\n================================================")
print("FINAL INPUT EMBEDDINGS  =  Token Emb + Positional Emb")
print("================================================")

# Using sinusoidal positional embeddings for the final combination
# (You can swap sin_pos_embeddings with learned_pos_embeddings)

final_embeddings = token_embeddings + sin_pos_embeddings
#                  ↑                   ↑
#   Shape: [N x 16]    +   Shape: [N x 16]   =   Shape: [N x 16]
#   (token meaning)         (position info)         (combined)

print(f"\nToken Embeddings Shape      : {token_embeddings.shape}")
print(f"Positional Embeddings Shape : {sin_pos_embeddings.shape}")
print(f"Final Input Embeddings Shape: {final_embeddings.shape}")
print(f"\n  → These final vectors are what gets fed into the Transformer attention layers")

print("\n--- Final Embeddings Per Token (first 8 dims shown) ---\n")

for i, (token, emb) in enumerate(zip(encoded.tokens, final_embeddings)):
    print(f"  [{i}] '{token:<12}' → {np.round(emb[:8], 4)}")

# ============================================================
# SUMMARY TABLE — All Embedding Stages Side By Side
# ============================================================

print("\n\n================================================")
print("SUMMARY — EMBEDDING PIPELINE FOR EACH TOKEN")
print("================================================")
print(f"\n{'Pos':<5} {'Token':<14} {'Token ID':<10} {'Token Emb [0]':<18} {'Sin Pos Emb [0]':<20} {'Final Emb [0]'}")
print("-" * 80)

for i, (token, tid) in enumerate(zip(encoded.tokens, token_ids)):
    t_emb  = round(token_embeddings[i][0], 5)
    p_emb  = round(sin_pos_embeddings[i][0], 5)
    f_emb  = round(final_embeddings[i][0], 5)
    print(f"  {i:<4} {token:<14} {tid:<10} {str(t_emb):<18} {str(p_emb):<20} {f_emb}")

print("\n  [0] = Only the first dimension shown for compactness")
print("  Final = Token Emb + Sinusoidal Positional Emb")


# ============================================================
# SAVE ALL OUTPUTS TO FILE
# ============================================================

with open("tokenizer_test_output.txt", "w", encoding="utf-8") as f:

    f.write("Original Text:\n")
    f.write(sample_text + "\n\n")

    f.write("Tokens:\n")
    f.write(str(encoded.tokens) + "\n\n")

    f.write("Token IDs:\n")
    f.write(str(encoded.ids) + "\n\n")

    f.write("Token Embeddings (shape: num_tokens x embedding_dim):\n")
    f.write(str(token_embeddings) + "\n\n")

    f.write("Sinusoidal Positional Embeddings:\n")
    f.write(str(sin_pos_embeddings) + "\n\n")

    f.write("Final Input Embeddings (Token + Positional):\n")
    f.write(str(final_embeddings) + "\n\n")

print("\nTEST OUTPUT SAVED TO tokenizer_test_output.txt")

# ============================================================
# FINISHED
# ============================================================

print("\n================================================")
print("PROCESS COMPLETED SUCCESSFULLY")
print("Generated Files:")
print("""
1. raw_corpus.txt                   — Raw scraped text
2. isignal_cleaned_corpus.txt       — Cleaned text for training
3. isignal_tokenizer.json           — Trained BPE tokenizer
4. tokenizer_test_output.txt        — Token IDs + all embeddings
""")
