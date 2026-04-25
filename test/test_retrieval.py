import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingest import PDFParser, ContentClassifier, Tokenizer
from src.store import Chunker, ChunkStore, BM25Index

def main():
    print("Initializing components based on Strategy Report (WordPiece, 150 Tokens)...")
    pdf_path = r"C:\Users\abeku\OneDrive\Desktop\Studybuddy\StudyBuddy-AI\data\raw\iesc102.pdf"
    
    parser = PDFParser(pdf_path)
    pages = parser.extract()
    print(f"Extracted {len(pages)} pages from Corpus.")

    classifier = ContentClassifier()
    tokenizer = Tokenizer('bert-base-uncased')
    
    chunker = Chunker(tokenizer, classifier, max_tokens=150, overlap=30)
    chunks = chunker.chunk(pages)
    
    store = ChunkStore()
    store.add_chunks(chunks)

    output_path = r"C:\Users\abeku\OneDrive\Desktop\Studybuddy\StudyBuddy-AI\test\output.jsonl"
    store.save_jsonl(output_path)
    print(f"\nSaved standardized chunks to {output_path}")

    # Stage 2: Retrieval Validation
    index = BM25Index(tokenizer)
    index.build_index(store)

    questions = [
        "What is the cell membrane?",
        "What is the function of the nucleus?",
        "Explain the process of osmosis.",
        "What is the difference between prokaryotic and eukaryotic cells?",
        "What is cytoplasm?",
        "Why is the plasma membrane called a selectively permeable membrane?",
        "What are the characteristics of a plant cell wall?",
        "Where do cellular activities basically take place?",
        "How do substances like carbon dioxide move in and out of a cell?",
        "What is the definition of diffusion?",
        "Do ancient single-celled biological entities lack defined genetic cores?",
        "Why do plant borders stand firm but animal edges wobble?",
        "If I dunk a living unit into heavily salted liquid, why does it shrivel?",
        "How does a quantum computer process qubits?",
        "Explain the long-term economic effects of the 2008 financial crisis.",
        "What are the rules of playing professional basketball?"
    ]

    print("\n" + "="*50)
    print("STAGE 2 RETRIEVAL VALIDATION (ALL 16 QS)")
    print("="*50)

    for i, q in enumerate(questions):
        print(f"\nQUERY {i+1}: {q}")
        print("+"*30)
        results = index.search(q, top_k=2)
        for j, res in enumerate(results):
            score = res['score']
            text = res['chunk']['text'].replace("\n", " ")[:200]
            print(f"  -> Rank {j+1} (BM25 Score: {score:.2f})")
            print(f"     Text Preview: {text}...")
        print("-" * 50)

if __name__ == "__main__":
    main()
