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
        "What is the definition of a eukaryotic cell?",
        "How is the process of osmosis defined?",
        "What is the structure of the cell membrane?"
    ]

    print("\n" + "="*50)
    print("STAGE 2 RETRIEVAL VALIDATION")
    print("="*50)

    for i, q in enumerate(questions):
        print(f"\nQUERY {i+1}: {q}")
        print("+"*30)
        results = index.search(q, top_k=2)
        for j, res in enumerate(results):
            score = res['score']
            text = res['chunk']['text'].replace("\n", " ")[:150]
            print(f"  -> Rank {j+1} (BM25 Score: {score:.2f})")
            print(f"     Text Preview: {text}...")
        print("-" * 50)

if __name__ == "__main__":
    main()
