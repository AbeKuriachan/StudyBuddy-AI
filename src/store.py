import uuid
import re
import json
from rank_bm25 import BM25Okapi

class Chunker:
    def __init__(self, tokenizer, classifier, max_tokens=500, overlap=80):
        """
        Takes instantiated Tokenizer and ContentClassifier from ingest.
        """
        self.tokenizer = tokenizer
        self.classifier = classifier
        self.max_tokens = max_tokens
        self.overlap = overlap

    def chunk(self, parsed_pages):
        chunks = []
        current_chapter = None
        current_section = None
        buffer = ""
        buffer_pages = set()

        for page in parsed_pages:
            text = page["text"]
            page_num = page["page"]

            chapter = self._detect_chapter(text)
            section = self._detect_section(text)

            if chapter: current_chapter = chapter
            if section: current_section = section

            buffer += " " + text
            buffer_pages.add(page_num)

            while self.tokenizer.count_tokens(buffer) > self.max_tokens:
                chunk_text, remainder_text = self.tokenizer.split_to_token_limit(buffer, self.max_tokens)
                
                # Classify chunk text
                content_type = self.classifier.classify(chunk_text)

                chunks.append(
                    self._create_chunk(
                        chunk_text,
                        current_chapter,
                        current_section,
                        content_type,
                        list(buffer_pages),
                        self.max_tokens
                    )
                )

                buffer = self.tokenizer.get_overlap_text(chunk_text, remainder_text, self.overlap)
                buffer_pages = {page_num}

        if buffer.strip():
            token_count = self.tokenizer.count_tokens(buffer)
            content_type = self.classifier.classify(buffer)
            chunks.append(
                self._create_chunk(
                    buffer,
                    current_chapter,
                    current_section,
                    content_type,
                    list(buffer_pages),
                    token_count
                )
            )

        return chunks

    def _detect_chapter(self, text):
        lines = text.split("\n")
        for i, line in enumerate(lines[:10]):
            line_clean = line.strip().lower()
            match = re.match(r'^chapter\s+(\d+)$', line_clean)
            if match:
                return {"number": int(match.group(1)), "title": None}

            if line_clean == "chapter" and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.isdigit():
                    return {"number": int(next_line), "title": None}
        return None

    def _detect_section(self, text):
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            match = re.match(r'^(\d+\.\d+)\s+([A-Za-z][A-Za-z\s\-\—]+)', line)
            if match:
                return {"number": match.group(1), "title": match.group(2).strip()}
        return None

    def _create_chunk(self, text, chapter, section, content_type, pages, token_count):
        return {
            "id": str(uuid.uuid4()),
            "text": text.strip(),
            "metadata": {
                "chapter": chapter,
                "section": section,
                "content_type": content_type,
                "token_count": token_count,
                "page_range": [min(pages), max(pages)] if pages else []
            }
        }


class ChunkStore:
    def __init__(self):
        self.chunks = []

    def set_chunks(self, chunks):
        self.chunks = chunks
        
    def add_chunks(self, chunks):
        self.chunks.extend(chunks)

    def save_jsonl(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                
    def load_jsonl(self, path):
        self.chunks = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.chunks.append(json.loads(line))
        return self.chunks


class BM25Index:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer  # Shared WordPiece tokenizer
        self.bm25 = None
        self.store = None

    def build_index(self, chunk_store):
        """
        Initializes the rank_bm25 corpus from a ChunkStore.
        """
        self.store = chunk_store
        

        corpus = []
        for chunk in self.store.chunks:
            # Map robust token IDs directly to strings so BM25 counts them natively 
            token_ids = self.tokenizer.tokenizer.encode(chunk['text'], add_special_tokens=False)
            token_strs = [str(tid) for tid in token_ids]
            corpus.append(token_strs)
            
        self.bm25 = BM25Okapi(corpus)

    def search(self, query, top_k=5):
        if not self.bm25:
            raise ValueError("BM25 index not built. Call build_index first.")
            
        query_token_ids = self.tokenizer.tokenizer.encode(query, add_special_tokens=False)
        query_strs = [str(tid) for tid in query_token_ids]
        
        scores = self.bm25.get_scores(query_strs)
        
        # Sort and retrieve top K chunks
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "score": scores[idx],
                "chunk": self.store.chunks[idx]
            })
            
        return results
