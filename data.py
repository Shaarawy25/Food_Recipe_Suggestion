import os
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from typing import List, Dict
import hashlib
from difflib import SequenceMatcher
from cleantext import clean

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file."""
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
    return text

def clean_and_normalize_text(text: str) -> str:
    """Clean text using cleantext library."""
    if not text:
        return ""
    
    cleaned = clean(text,
                   fix_unicode=True,
                   to_ascii=False,
                   lower=False,
                   no_line_breaks=False,
                   no_urls=True,
                   no_emails=True,
                   no_phone_numbers=True,
                   no_numbers=False,
                   no_digits=False,
                   no_currency_symbols=False,
                   no_punct=False,
                   normalize_whitespace=True,
                   lang="en")
    
    return cleaned.strip()

def extract_recipe_metadata(text: str, source_file: str) -> Dict:
    """Extract basic metadata from recipe text."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    recipe_title = ""
    for line in lines[5:15]: 
        line_lower = line.lower()
       
        if any(skip in line_lower for skip in ['table of contents', 'copyright', 'author', 
                                                'cookbook', 'recipe book', 'page', 'chapter']):
            continue
        if 15 <= len(line) <= 60 :
            recipe_title = line
            break
    
    return {
        "source": source_file,
        "recipe_title": recipe_title,
        "cuisine_type": "healthy" if "healthy" in source_file.lower() else ""
    }

def chunk_text(text: str) -> List[str]:
    """Split text into chunks for vector store."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return text_splitter.split_text(text)

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two text chunks."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()



def process_pdfs_in_directory(directory_path: str) -> Dict:
    """Process all PDF files and create chunks."""
    processed_data = {}
    all_chunks = []
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    
    print("Found", len(pdf_files), "PDF files to process...")
    
    for pdf_file in pdf_files:
        print("Processing:", pdf_file)
        pdf_path = os.path.join(directory_path, pdf_file)
        
        extracted_text = extract_text_from_pdf(pdf_path)
        
        if extracted_text.strip():
            cleaned_text = clean_and_normalize_text(extracted_text)
            metadata = extract_recipe_metadata(cleaned_text, pdf_file)
            chunks = chunk_text(cleaned_text)
            
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    "text": chunk,
                    "metadata": {
                        **metadata,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                }
                all_chunks.append(chunk_data)
            
            processed_data[pdf_file] = {
                "num_chunks": len(chunks),
                "total_length": len(extracted_text)
            }
            
            print("  - Created", len(chunks), "chunks")
        else:
            print("  - No text extracted")
    
    
    final_output = {
        "processed_pdfs": processed_data,
        "all_chunks": all_chunks,
        "statistics": {
            "total_pdfs": len(processed_data),
            "total_chunks": len(all_chunks)
        }
    }
    
    output_path = os.path.join(directory_path, "processed_recipes.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print("Processed data saved to:", output_path)
    print("Final chunks:", len(all_chunks))
    
    return final_output

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    process_pdfs_in_directory(current_dir)
