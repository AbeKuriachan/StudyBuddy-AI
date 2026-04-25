import re
import fitz
from transformers import AutoTokenizer

class PDFParser:
    def __init__(self, pdf_path):
        self.doc = fitz.open(pdf_path)
        self.pdf_path = pdf_path

    def extract(self):
        parsed_pages = []

        for page_num, page in enumerate(self.doc, start=1):
            blocks = page.get_text("blocks")
            # b[4] contains the actual text of the block
            page_text = "\n".join([b[4] for b in blocks if isinstance(b[4], str)])

            parsed_pages.append({
                "page": page_num,
                "text": self._clean_text(page_text)
            })

        return parsed_pages

    def _clean_text(self, text):
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{2,}', '\n', text)
        return text.strip()


class ContentClassifier:
    """Classifies chunk text conceptually."""
    def classify(self, text: str) -> str:
        text_lower = text.lower()
        if re.search(r'\b(example|for instance|suppose|consider)\b', text_lower):
            return "Example"
        if re.search(r'\b(question|exercise|q\d+|\?)\b', text_lower) and len(text.split()) < 100:
            return "End-of-chapter Q"
        return "Concept"


class Tokenizer:
    def __init__(self, model_name='bert-base-uncased'):
        # Increase model_max_length to suppress warnings when we pass entire pages into .encode() 
        # since we are only using it for token slicing, not inference.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=100000)
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def split_to_token_limit(self, text: str, max_tokens: int) -> tuple:
        """
        Splits text into a chunk of max_tokens and the remainder.
        Returns: (chunk_text, remaining_text)
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return text, ""
            
        chunk_tokens = list(tokens)[:max_tokens]
        chunk_text = self.tokenizer.decode(chunk_tokens)
        remainder_tokens = list(tokens)[max_tokens:]
        remainder_text = self.tokenizer.decode(remainder_tokens)
        
        return chunk_text, remainder_text

    def get_overlap_text(self, chunk_text: str, remainder_text: str, overlap: int) -> str:
        """
        Appends the last `overlap` words/tokens of the chunk back onto the remainder
        to maintain context between splits.
        """
        tokens = self.tokenizer.encode(chunk_text, add_special_tokens=False)
        overlap_tokens = tokens[-overlap:] if len(tokens) > overlap else tokens
        overlap_text = self.tokenizer.decode(overlap_tokens)
        
        # Merge overlap back to remainder
        return overlap_text + " " + remainder_text
