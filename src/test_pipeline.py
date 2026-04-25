import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingest import PDFParser, ContentClassifier, Tokenizer
from src.store import Chunker, ChunkStore, BM25Index

def main():
    print("1. Testing Ingestion Layer...")
    pdf_path = r"C:\Users\abeku\OneDrive\Desktop\Studybuddy\StudyBuddy-AI\data\raw\iesc102.pdf"
    
    parser = PDFParser(pdf_path)
    pages = parser.extract()
    print(f"Extracted {len(pages)} pages.")

    print("2. Initializing Classifiers and Tokenizers...")
    classifier = ContentClassifier()
    tokenizer = Tokenizer('bert-base-uncased')

    print("3. Chunking Data (Stage 2)...")
    chunker = Chunker(tokenizer, classifier, max_tokens=100, overlap=20)
    # Using only 5 pages for fast test
    chunks = chunker.chunk(pages[:5])
    print(f"Generated {len(chunks)} chunks.")

    print("4. Storing Chunks in ChunkStore...")
    store = ChunkStore()
    store.add_chunks(chunks)
    out_path = r"C:\Users\abeku\OneDrive\Desktop\Studybuddy\StudyBuddy-AI\data\processed\chunks_v2.jsonl"
    store.save_jsonl(out_path)
    print(f"Saved to {out_path}")

    print("5. Building BM25 Index...")
    index = BM25Index(tokenizer)
    index.build_index(store)
    
    print("6. Searching BM25 Index for 'cell membrane'...")
    results = index.search("cell membrane", top_k=2)
    for i, res in enumerate(results):
        print(f"Result {i+1} (Score: {res['score']:.2f}):")
        print(res['chunk']['text'][:150] + "...\n")

if __name__ == "__main__":
    main()
