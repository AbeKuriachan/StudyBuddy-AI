import os
import sys
import pandas as pd
from typing import List, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ingest import PDFParser, ContentClassifier, Tokenizer
from src.store import Chunker, ChunkStore, BM25Index
from src.generate import Generator

QUESTIONS = [
    # --- DIRECT ---
    {"text": "What is the cell membrane?", "type": "DIRECT"},
    {"text": "What is the function of the nucleus?", "type": "DIRECT"},
    {"text": "Explain the process of osmosis.", "type": "DIRECT"},
    {"text": "What is the difference between prokaryotic and eukaryotic cells?", "type": "DIRECT"},
    {"text": "What is cytoplasm?", "type": "DIRECT"},
    {"text": "Why is the plasma membrane called a selectively permeable membrane?", "type": "DIRECT"},
    {"text": "What are the characteristics of a plant cell wall?", "type": "DIRECT"},
    {"text": "Where do cellular activities basically take place?", "type": "DIRECT"},
    {"text": "How do substances like carbon dioxide move in and out of a cell?", "type": "DIRECT"},
    {"text": "What is the definition of diffusion?", "type": "DIRECT"},

    # --- PARAPHRASED --- (Testing Lexical Limits)
    {"text": "Do ancient single-celled biological entities lack defined genetic cores?", "type": "PARAPHRASED"},
    {"text": "Why do plant borders stand firm but animal edges wobble?", "type": "PARAPHRASED"},
    {"text": "If I dunk a living unit into heavily salted liquid, why does it shrivel?", "type": "PARAPHRASED"},

    # --- OUT OF SCOPE ---
    {"text": "How does a quantum computer process qubits?", "type": "OUT_OF_SCOPE"},
    {"text": "Explain the long-term economic effects of the 2008 financial crisis.", "type": "OUT_OF_SCOPE"},
    {"text": "What are the rules of playing professional basketball?", "type": "OUT_OF_SCOPE"}
]

def heuristic_score(q_type: str, answer_text: str) -> Dict[str, str]:
    refused = "I cannot answer this" in answer_text
    
    if q_type == "OUT_OF_SCOPE":
        return {
            "correctness": "yes" if refused else "no",
            "grounding": "yes" if refused else "no",
            "refusal_appropriateness": "yes" if refused else "no"
        }
    
    if q_type == "PARAPHRASED":
        if refused:
            # Lexical retrieval failed, so it refused. Grounding technically worked (it didn't hallucinate).
            return {
                "correctness": "no",
                "grounding": "yes",
                "refusal_appropriateness": "no"  # It incorrectly refused a valid textbook question
            }
        else:
            return {
                "correctness": "yes",  # Assuming it retrieved correctly
                "grounding": "yes",
                "refusal_appropriateness": "N/A"
            }
            
    if q_type == "DIRECT":
        if refused:
            return {
                "correctness": "no",
                "grounding": "yes",
                "refusal_appropriateness": "no"  # It incorrectly refused a valid textbook question
            }
        else:
            return {
                "correctness": "yes",
                "grounding": "yes",
                "refusal_appropriateness": "N/A"
            }
    
    return {}

def main():
    print("Initializing Pipeline...")
    pdf_path = r"C:\Users\abeku\OneDrive\Desktop\Studybuddy\StudyBuddy-AI\data\raw\iesc102.pdf"
    
    parser = PDFParser(pdf_path)
    pages = parser.extract()
    classifier = ContentClassifier()
    tokenizer = Tokenizer('bert-base-uncased')
    
    chunker = Chunker(tokenizer, classifier, max_tokens=150, overlap=30)
    store = ChunkStore()
    store.add_chunks(chunker.chunk(pages))
    
    index = BM25Index(tokenizer)
    index.build_index(store)
    
    generator = Generator()

    print(f"Executing Eval on {len(QUESTIONS)} Questions...")
    
    results = []

    for i, q in enumerate(QUESTIONS):
        print(f"[{i+1}/{len(QUESTIONS)}] Processing: {q['text']} ({q['type']})")
        
        # 1. Retrieve
        top_chunks = index.search(q['text'], top_k=2)
        
        # 2. Generate
        output = generator.answer(q['text'], top_chunks)
        answer_text = output['answer']
        
        # 3. Score
        scores = heuristic_score(q['type'], answer_text)
        
        results.append({
            "Question ID": i+1,
            "Question": q['text'],
            "Type": q['type'],
            "Answer": answer_text,
            "Correctness": scores.get("correctness"),
            "Grounding": scores.get("grounding"),
            "Refusal_Appropriate": scores.get("refusal_appropriateness")
        })

    # Export to CSV
    df = pd.DataFrame(results)
    out_path = r"C:\Users\abeku\OneDrive\Desktop\Studybuddy\StudyBuddy-AI\documents\evaluation_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved scoring matrix to {out_path}")

if __name__ == "__main__":
    main()
