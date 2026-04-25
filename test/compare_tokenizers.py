import os
from transformers import AutoTokenizer

def highlight_boundaries(tokenizer, text):
    """Encodes and decodes token by token to show boundaries"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    # decode each token separately
    decoded = [tokenizer.decode([t]) for t in tokens]
    return " | ".join(decoded)

def main():
    print("Loading Tokenizers...")
    bpe_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    wordpiece_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    passages = [
        "In prokaryotic cells, most cellular activities take place directly in the cytoplasm.",
        "Structurally, the cell membrane is extremely thin, about 7 to 10 nanometres (nm) thick.",
        "Without a rigid cell wall, animal cells can change shape easily. This cellular flexibility supports movement.",
        "The plant cell wall is primarily made of cellulose, a type of carbohydrate formed by many glucose units.",
        "Some microorganisms, like fungi and bacteria also have a cell wall to provide protection."
    ]

    print("="*60)
    print("Tokenizer Comparison Report")
    print("="*60 + "\n")

    for i, p in enumerate(passages):
        print(f"--- Passage {i+1} ---")
        print(f"RAW TEXT: {p}")
        
        bpe_tokens = bpe_tokenizer.encode(p, add_special_tokens=False)
        wp_tokens = wordpiece_tokenizer.encode(p, add_special_tokens=False)
        
        print(f"\n[GPT-2 BPE] Count: {len(bpe_tokens)}")
        print(f"Boundaries: {highlight_boundaries(bpe_tokenizer, p)}")
        
        print(f"\n[BERT WordPiece] Count: {len(wp_tokens)}")
        print(f"Boundaries: {highlight_boundaries(wordpiece_tokenizer, p)}")
        print("\n" + "-"*40 + "\n")

if __name__ == "__main__":
    main()
