import fitz  # PyMuPDF
import os




def extract_relevant_sections(text):
    """
    Extracts title, authors, and abstract from the first page text using improved regex.
    Returns a string containing these sections.
    """
    import re
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Title: first line, Authors: second line (if not all caps)
    title = lines[0] if lines else ""
    authors = ""
    if len(lines) > 1 and not lines[1].isupper():
        authors = lines[1]
    # Abstract: match 'Abstract' and capture until next section or double newline
    abstract = ""
    abstract_match = re.search(r'(?i)abstract[:\s]*([\s\S]*?)(?:\n\n|\n[A-Z][a-z]+)', text)
    if abstract_match:
        abstract = abstract_match.group(1).strip()
    # Compose relevant info
    result = f"Title: {title}\nAuthors: {authors}\nAbstract: {abstract}"
    return result





def extract_text_from_pdf(pdf_path):
    """
    Extracts and returns the full text from all pages of a PDF file using PyMuPDF.
    """
    import fitz
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def batch_extract_from_folder(folder_path):
    """
    Extracts relevant info from all PDF files in a folder.
    Returns a dict: {filename: relevant_text}
    """
    results = {}
    for fname in os.listdir(folder_path):
        if fname.lower().endswith('.pdf'):
            fpath = os.path.join(folder_path, fname)
            results[fname] = extract_text_from_pdf(fpath)
    return results

# Example usage:
# texts = batch_extract_from_folder('path/to/pdf_folder')
# print(texts)
