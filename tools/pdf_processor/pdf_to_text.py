#!/usr/bin/env python3
"""
PDF to Text Converter
Converts PDF files to text using multiple methods for best results.
"""

import sys
import os
import subprocess
from pathlib import Path

def convert_with_pdftotext(pdf_path, output_path):
    """Convert PDF to text using pdftotext (poppler-utils)"""
    try:
        subprocess.run(['pdftotext', '-layout', pdf_path, output_path], 
                      check=True, capture_output=True, text=True)
        return True, "Successfully converted using pdftotext"
    except subprocess.CalledProcessError as e:
        return False, f"pdftotext failed: {e.stderr}"
    except FileNotFoundError:
        return False, "pdftotext not found"

def convert_with_pypdf2(pdf_path, output_path):
    """Convert PDF to text using PyPDF2"""
    try:
        import PyPDF2
        
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_content = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += f"\n--- Page {page_num + 1} ---\n"
                text_content += page.extract_text()
                text_content += "\n"
        
        with open(output_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text_content)
        
        return True, "Successfully converted using PyPDF2"
    except ImportError:
        return False, "PyPDF2 not available"
    except Exception as e:
        return False, f"PyPDF2 failed: {str(e)}"

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 pdf_to_text.py <pdf_file_path>")
        print("Output will be saved as <pdf_name>_extracted.txt")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found")
        sys.exit(1)
    
    # Generate output filename
    pdf_name = Path(pdf_path).stem
    output_path = f"{pdf_name}_extracted.txt"
    
    print(f"Converting PDF: {pdf_path}")
    print(f"Output file: {output_path}")
    print("-" * 50)
    
    # Try pdftotext first (usually gives better formatting)
    success, message = convert_with_pdftotext(pdf_path, output_path)
    if success:
        print(f"✓ {message}")
        print(f"Text extracted and saved to: {output_path}")
        return
    else:
        print(f"✗ {message}")
        print("Trying PyPDF2 as fallback...")
    
    # Fallback to PyPDF2
    success, message = convert_with_pypdf2(pdf_path, output_path)
    if success:
        print(f"✓ {message}")
        print(f"Text extracted and saved to: {output_path}")
    else:
        print(f"✗ {message}")
        print("All conversion methods failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
