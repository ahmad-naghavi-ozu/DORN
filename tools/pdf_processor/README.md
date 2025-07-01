# PDF Processor Tools

This directory contains tools for processing PDF files.

## pdf_to_text.py

Converts PDF files to text format using multiple methods for best results.

### Usage

```bash
python3 pdf_to_text.py <path_to_pdf_file>
```

### Features

- **Primary method**: Uses `pdftotext` (poppler-utils) which preserves layout and formatting
- **Fallback method**: Uses PyPDF2 for text extraction when pdftotext fails
- **Automatic output naming**: Creates `<pdf_name>_extracted.txt` in the same directory
- **Error handling**: Provides clear error messages and fallback options

### Dependencies

- `pdftotext` (from poppler-utils package) - preferred method
- `PyPDF2` Python package - fallback method

At least one of these should be available on the system.

### Example

```bash
# Convert a PDF to text
python3 tools/pdf_processor/pdf_to_text.py documents/research_paper.pdf

# This will create: research_paper_extracted.txt
```

### Output Format

The extracted text file will contain:
- Page separators (`--- Page X ---`)
- Preserved text layout (when using pdftotext)
- UTF-8 encoding for proper character support
