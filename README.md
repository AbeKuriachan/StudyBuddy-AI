# StudyBuddy-AI: Modular RAG Pipeline

StudyBuddy-AI is an educational Retrieval-Augmented Generation (RAG) pipeline optimized for analyzing structured textbook PDFs (such as NCERT Biology/Physics). It seamlessly bridges raw document extraction to rigidly grounded LLM output generation.

## Architecture (`src/`)
Our pipeline is decoupled into discrete stages mapping directly to the RAG lifecycle:

- **`src/ingest.py` (Stage 1: Corpus Extraction)**
  - **`PDFParser`**: Extracts text maintaining underlying page geography (avoiding tabular destruction) via PyMuPDF.
  - **`ContentClassifier`**: Uses regex heuristics to classify text blocks dynamically as `Concept`, `Example`, or `End-of-chapter Q`.
  - **`Tokenizer`**: Wraps HuggingFace's `transformers` (`bert-base-uncased`) to calculate subword metrics mathematically seamlessly.
- **`src/store.py` (Stage 2: Chunk & Store)**
  - **`Chunker`**: Limits contextual drift by splitting paragraphs cleanly at 150-token constraints with 30-token overlap padding based on the Tokenizer logic.
  - **`ChunkStore`**: Tracks chunk payloads and metadata before dumping to standard `.jsonl` schemas.
  - **`BM25Index`**: Uses lexical retrieval via `rank_bm25` to statistically score overlaps and fetch relevant context instantly. 
- **`src/generate.py` (Stage 3: Inference)**
  - **`Generator`**: A strict inference wrapper communicating natively with the Groq API (`llama-3.1-8b-instant`). It operates at `temperature=0` with a heavily guarded prompt ensuring the model rigidly refuses out-of-scope outputs.

## 📄 Evaluation Artifacts (`documents/`)
A evaluation tests was executed to validate chunk boundaries, exact retrieval hits, and LLM generative grounding (Stage 4). You can review these comprehensive reports in the `documents/` directory:

| Document Formats | Description |
|------------------|-------------|
| **`strategy_report.md`** | Theoretical justification breaking down our logic behind adopting `bert-base-uncased` WordPiece slicing instead of BPE, and contextualizing the 150-size limits. |
| **`tokenizer_output.txt`** | Raw log dumps comparing how `gpt2` (BPE) vs `bert-base-uncased` (WordPiece) tokenizers diverge structurally across complex biological terms. |
| **`retrieval_output.txt`** | A standalone preview text log highlighting the exact isolated chunk overlaps the `BM25Index` returned against a matrix of 16 queries. |
| **`evaluation_results.csv`** | A matrix grading the LLM across 16 varied questions (Direct, Paraphrased, Out-of-Scope) on robust axes: Correctness, Grounding, and Refusal Appropriateness. |
| **`eval_summary.md`** | An evaluation summary pinpointing 3 successful queries that passed verification, and diagnosing the probable lexical retrieval faults of 2 heavily-paraphrased queries. |

##  Running the Tests Locally
The active testing scripts used to populate the `documents/` directory remain inside the `test/` folder.
* **Compare Tokenizers**: `python test/compare_tokenizers.py`
* **Evaluate Lexical Chunk Hits**: `python test/test_retrieval.py`
* **Automate the Model Evaluations**: `python test/eval_llm_generation.py` (Ensure you have a `.env` configured mapped to `GROQ_API_KEY=xxx` in your roots first!).