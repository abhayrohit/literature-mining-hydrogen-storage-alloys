# main.py
"""
Pipeline to extract text from PDFs, query DeepSeek R1, and save structured results to CSV.
"""
import os
import csv
from extract import batch_extract_from_folder
from llm_prompt import query_deepseek_ollama

PDF_FOLDER = "./pdfs"  # Change to your PDF folder path
CSV_OUTPUT = "output.csv"


# Set your Ollama model name (as shown in `ollama list`)
OLLAMA_MODEL_NAME = "deepseek-r1"  

FIELDS = [
    "Title",
    "Authors",
    "Year",
    "Journal",
    "Type of Study",
    "Alloy Composition",
    "Synthesis/Processing Method",
    "Hydrogen Storage Capacity (wt%)",
    "Operating Conditions",
    "Key Findings",
    "Reference Link or DOI"
]


def process_papers():
    texts = batch_extract_from_folder(PDF_FOLDER)
    results = []
    for fname, text in texts.items():
        print(f"Processing {fname}...")
        response = query_deepseek_ollama(text, model_name=OLLAMA_MODEL_NAME)
        # Simple fallback: try to split by lines and match fields
        row = ["N/A"] * len(FIELDS)
        for line in response.splitlines():
            for i, field in enumerate(FIELDS):
                if line.startswith(field):
                    row[i] = line.split(":", 1)[-1].strip()
        results.append(row)
    # Write to CSV
    with open(CSV_OUTPUT, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(results)
    print(f"Saved results to {CSV_OUTPUT}")

if __name__ == "__main__":
    process_papers()
