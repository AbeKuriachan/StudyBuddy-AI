# Pipeline Strategy Justification

Based on Stage 1 requirements, this report outlines and justifies the selected chunking and tokenization strategies for processing the NCERT PDFs logic.

### Tokenizer Selection: BERT WordPiece over GPT-2 BPE
After testing both GPT-2 (BPE) and BERT (WordPiece) tokenizers on biological passages, we chose **WordPiece (`bert-base-uncased`)**. 

1. **Morphological Safety**: In biology contexts, BPE tends to memorize full words (e.g., `carbohydrate` parses as 1 token), whereas WordPiece aggressively maps to morphological roots (e.g., `car | ##bo | ##hy | ##dra | ##te`). While WordPiece yields mathematically higher token counts, splitting scientific terminology into affixes prevents out-of-vocabulary (OOV) drops and provides robust overlap matching for the downstream BM25 retrieval index.
2. **Casing Standardization**: Utilizing an `uncased` tokenizer reduces semantic fragmentation caused by sentence caps vs inline caps, which is highly prevalent when parsing PDFs that contain headers, titles, and uppercase diagram callouts.

### Chunking Strategy
Our chosen strategy employs a target size of **150 tokens** with an **overlap of 30 tokens**, anchored by semantic constraints.
1. **Size (150 Tokens)**: Empirically, 150 token chunks encapsulate exactly one or two dense conceptual textbook paragraphs without heavily diluting the internal keyword density. If chunks are too wide (e.g., 500+ tokens), BM25 term frequencies will struggle to pinpoint specific answers.
2. **Overlap (30 Tokens)**: By tracking 30 overlapping tokens backward when making a hard boundary, we prevent structural nouns/verbs spanning sentences from being orphaned.
3. **Handling**: We parse the PDF aggressively into purely spatial "blocks" ahead of time to safeguard layout integrity (saving tables natively). We then classify metadata conceptually (`Concept`/`Example`/`End-of-chapter Q`) before writing it into the `ChunkStore` to power targeted filtering logic downstream if required.
