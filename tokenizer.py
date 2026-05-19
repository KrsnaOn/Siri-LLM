# ============================================================
# ISIGNAL CUSTOM TOKENIZER TRAINING PIPELINE
# FULL WORKING VERSION
# ============================================================

# ============================================================
# INSTALL REQUIRED LIBRARIES FIRST
# ============================================================

# pip install requests
# pip install beautifulsoup4
# pip install tokenizers

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import requests
from bs4 import BeautifulSoup
import re
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# ============================================================
# URL DATASET
# ============================================================

# Use only public websites
# Social media websites often block scraping

urls = [

    "https://isignalresearch.com/",
    "https://academy.isignalresearch.com",
    "https://learn.isignalresearch.com",
    "https://isignaltechblog.blogspot.com/"
]

# ============================================================
# HEADERS
# ============================================================

# Browser-like headers help bypass basic bot blocking

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

        # ====================================================
        # SEND REQUEST
        # ====================================================

        response = requests.get(
            url,
            headers=headers,
            timeout=20,
            allow_redirects=True
        )

        print("Status Code:", response.status_code)

        # ====================================================
        # CHECK SUCCESS
        # ====================================================

        if response.status_code == 200:

            # ====================================================
            # PARSE HTML
            # ====================================================

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            # ====================================================
            # REMOVE UNWANTED TAGS
            # ====================================================

            for tag in soup([
                "script",
                "style",
                "noscript",
                "header",
                "footer",
                "svg"
            ]):
                tag.extract()

            # ====================================================
            # EXTRACT VISIBLE TEXT
            # ====================================================

            text = soup.get_text(
                separator=" ",
                strip=True
            )

            print("Extracted Characters:", len(text))

            # ====================================================
            # STORE LARGE TEXT ONLY
            # ====================================================

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

with open(
    "raw_corpus.txt",
    "w",
    encoding="utf-8"
) as f:

    for line in corpus:

        f.write(line + "\n\n")

print("\nRAW CORPUS SAVED")

# ============================================================
# TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove brackets content
    text = re.sub(r'\[[^\]]+\]', '', text)

    # Remove special symbols
    text = re.sub(
        r'[^a-zA-Z0-9\s.,!?-]',
        ' ',
        text
    )

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# ============================================================
# CLEAN ALL SCRAPED TEXT
# ============================================================

cleaned_corpus = []

for line in corpus:

    cleaned = clean_text(line)

    # Keep meaningful sentences only
    if len(cleaned.split()) > 10:

        cleaned_corpus.append(cleaned)

# ============================================================
# CHECK CLEANED DATA
# ============================================================

print("\n================================================")
print("CLEANED DOCUMENTS:", len(cleaned_corpus))

# Print sample cleaned text
if len(cleaned_corpus) > 0:

    print("\nSAMPLE CLEANED TEXT:\n")
    print(cleaned_corpus[0][:500])

# ============================================================
# SAVE CLEANED CORPUS
# ============================================================

with open(
    "isignal_cleaned_corpus.txt",
    "w",
    encoding="utf-8"
) as f:

    for line in cleaned_corpus:

        f.write(line + "\n")

print("\nCLEANED CORPUS SAVED")

# ============================================================
# CREATE TOKENIZER
# ============================================================

print("\n================================================")
print("CREATING TOKENIZER")

# Create BPE tokenizer
tokenizer = Tokenizer(BPE())

# Split words using spaces
tokenizer.pre_tokenizer = Whitespace()

# ============================================================
# TRAINER SETTINGS
# ============================================================

trainer = BpeTrainer(

    vocab_size=5000,

    special_tokens=[

        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]"
    ]
)

# ============================================================
# TRAIN TOKENIZER
# ============================================================

print("\nTRAINING TOKENIZER...")

tokenizer.train(

    files=["isignal_cleaned_corpus.txt"],
    trainer=trainer
)

# ============================================================
# SAVE TOKENIZER
# ============================================================

tokenizer.save("isignal_tokenizer.json")

print("\nTOKENIZER TRAINED SUCCESSFULLY")
print("TOKENIZER SAVED")

# ============================================================
# TEST TOKENIZER
# ============================================================

print("\n================================================")
print("TESTING TOKENIZER")

sample_text = (
    "5G AI telecom research and machine learning"
)

# Encode text
encoded = tokenizer.encode(sample_text)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\nORIGINAL TEXT:")
print(sample_text)

print("\nTOKENS:")
print(encoded.tokens)

print("\nTOKEN IDS:")
print(encoded.ids)

# ============================================================
# OPTIONAL: SAVE TOKENS TEST
# ============================================================

with open(
    "tokenizer_test_output.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("Original Text:\n")
    f.write(sample_text + "\n\n")

    f.write("Tokens:\n")
    f.write(str(encoded.tokens) + "\n\n")

    f.write("Token IDs:\n")
    f.write(str(encoded.ids))

print("\nTEST OUTPUT SAVED")

# ============================================================
# FINISHED
# ============================================================

print("\n================================================")
print("PROCESS COMPLETED SUCCESSFULLY")
print("Generated Files:")

print("""
1. raw_corpus.txt
2. isignal_cleaned_corpus.txt
3. isignal_tokenizer.json
4. tokenizer_test_output.txt
""")