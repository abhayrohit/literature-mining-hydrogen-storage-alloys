# literature-mining-hydrogen-storage-alloys

End-to-end pipeline to extract structured hydrogen storage alloy data from PDF research papers using a local Ollama LLM (default: gpt-oss:120b-cloud). Includes:

- PDF text extraction (PyMuPDF)
- LLM-based structured extraction of alloy compositions & properties
- Rule-based fallback extraction
- FastAPI backend + minimal HTML frontend for uploading a PDF and downloading the extracted CSV table

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Recommended) Create Virtual Environment (Windows PowerShell)
```powershell
py -3.12 -m venv .venv
./.venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.12 not available, substitute `py -3.13`.

### 3. Pull/Prepare an Ollama Model
```powershell
ollama run gpt-oss:120b-cloud
```
This first run pulls the model. For lower resource usage, pick a smaller variant (e.g., `gpt-oss:7b`).

### 4. Run the API
```powershell
$env:MODEL_NAME="gpt-oss:120b-cloud"
uvicorn main_fastapi:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the Frontend
Navigate to: http://localhost:8000

Upload a PDF; after processing you will see a table and a link to download the CSV.
Health check: http://localhost:8000/health

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| OLLAMA_URL | Base URL of Ollama server | http://localhost:11434 |
| MODEL_NAME | Ollama model name | gpt-oss:120b-cloud |

Example (PowerShell) select smaller model:
```powershell
$env:MODEL_NAME="gpt-oss:7b"
```

## Output Table Schema (Alloy Table Mode)

```
Alloy Name | Storage Capacity | Temperature Range | Pressure Range | Synthesis Method
```
Empty cells are left blank, no N/A tokens.

## Directory Structure Created at Runtime

- `uploads/` – raw uploaded PDFs
- `data/raw_text/` – extracted plain text
- `outputs/` – generated alloy CSV tables

## Fallback & Reliability

If the LLM output is malformed, the backend sanitizes lines, enforces the header, and deduplicates alloys.

## Future Enhancements

- Add batch processing endpoint
- Integrate database persistence (SQLite/Postgres)
- Add authentication & job queue for large PDFs

## Citation / Academic Use

Please verify extracted data manually before publishing results; LLM outputs may contain omissions or subtle parsing errors.

