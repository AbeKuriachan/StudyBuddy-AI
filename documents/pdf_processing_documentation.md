# PDF Processing Pipeline Documentation

This document outlines the architecture, history of issues, and current state of the PDF parsing and chunking pipeline used in the `StudyBuddy-AI` project.

## 1. Overall Working of the Classes

The pipeline consists of two main classes designed to extract text from raw PDFs and break them into manageable chunks suitable for Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG).

### `PDFParser`
The purpose of the `PDFParser` class is to read a PDF file and extract its textual content page-by-page.
- **Initialization**: It uses the PyMuPDF library (`fitz`) to open the document via the file path.
- **Extraction (`extract` method)**: It iterates through every page in the PDF. Currently, it uses `page.get_text("blocks")` which extracts text based on geometric layout boxes. This helps preserve row structures in tables and separates distinct paragraphs better than simple top-down line-reading. 
- **Cleaning (`_clean_text` method)**: It strips out excessive contiguous spaces and reduces chained line-breaks to keep the text density tight while preserving fundamental paragraph boundaries.

### `Chunker`
The purpose of the `Chunker` class is to take the extracted pages and split them into overlapping token-sized windows (chunks), while trying to append relevant structural metadata (like Chapter or Section).
- **Initialization**: Accepts a `max_tokens` limit (e.g., 500) and an `overlap` threshold (e.g., 80) to maintain context between split chunks.
- **Chunking Logic (`chunk` method)**: 
  - Accumulates text from the `parsed_pages` into a continuous `buffer`. 
  - Iteratively checks if the buffer exceeds `max_tokens`.
  - Uses a `while` loop to split the buffer into a chunk of maximum size.
  - Retains a specific amount of the end of the newly formed chunk (`overlap`) alongside any remaining un-chunked text in the buffer, carrying this overlap into the next iteration to ensure continuity.
- **Metadata Detection**: Scans the text dynamically to identify `Chapter` labels and numeric `Sections` via regex to tag chunks with their surrounding hierarchical context.


---

## 2. Previous Issues & How We Addressed Them

During the initial review of the pipeline, three severe issues were detected affecting the usability and integrity of the parsed text.

### Issue A: Total Data Loss on Large Pages
**The Problem**: The `Chunker` used an `if` statement to process chunks. When a page was added to the buffer, if the buffer exceeded the maximum token size, it simply created one chunk and then deleted the rest of the text, only keeping the final 80 "overlapping" words of the page. Hundreds of words in the middle of long pages were completely erased.
**The Solution**: Rewrote the logic to use a `while` loop. The script now continually chops the buffer into `max_token` blocks until the buffer is safely below the limit, ensuring 100% of the text is retained and chunked.

### Issue B: formatting Destruction (Loss of Newlines)
**The Problem**: The chunking splits (`_trim_to_token_limit`, `_apply_overlap` and `_token_len`) used `string.split()` without arguments. This stripped every single newline character (`\n`) from the output, turning lists, tables, and paragraphs into a single unreadable wall of text.
**The Solution**: Updated the token-length and splitting logic to strictly use `.split(" ")` (space-only). This preserves newline characters natively within the chunk strings.

### Issue C: Shredded Tabular Data
**The Problem**: The `PDFParser` initially used `page.get_text("text")`. Because this method reads purely from left to right, top to bottom without robust structural awareness, it dismantled tables. A 3-column table would be read completely down column 1, then completely down column 2, destroying the relationship between rows.
**The Solution**: Refactored the parser to use `page.get_text("blocks")`. This extracts text inside physical bounding boxes first, maintaining horizontal row integrity for tables and lists much better.

---

## 3. Current Known Issues (Future Improvements)

While data loss and total formatting failures are resolved, analyzing the generated `chunks.jsonl` against complex textbook PDFs reveals two lingering extraction artifacts.

### Issue A: Font Encoding & Garbage Characters
**The Problem**: Complex PDF layouts (like textbooks) often map characters like smart quotes (`‘`), dashes (`—`), or scientific symbols (`µm`) to custom font embeddings without standard Unicode translation. `PyMuPDF` extracts these as garbage characters like `?~mosaic?T` or `?%?"?%`.
**Potential Fixes**: 
1. Implement a regex pre-processing step inside `_clean_text` to manually sanitize known rogue sequences.
2. If scale is required, migrate to an OCR-backed solution (`pdf2image` + `pytesseract`) or a more robust document AI parser like `Unstructured.io`.

### Issue B: Reading Order of Sidebars, Headers, and Footers
**The Problem**: Although `blocks` fixes tabular rows, PyMuPDF still orders these blocks based on their physical Y-coordinates (top to bottom). This means side-bar callout boxes (e.g., "What if...") and rigid page footers/headers (e.g., "Grade 8... Page 13") get abruptly injected right into the middle of paragraph sentences that happen to sit adjacent to them vertically.
**Potential Fixes**:
1. **Geometric Filtering**: Update `PDFParser.extract()` to check the block coordinates (`b[0]` to `b[3]`). Automatically skip/delete blocks that exist in the top 5% or bottom 5% of the page to remove headers/footers.
2. **Layout AI Models**: For advanced sidebar reading-order logic, utilize Layout parsing models like `LayoutLM` or Amazon Textract.
