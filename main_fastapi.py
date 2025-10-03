"""Minimal API backend for hydrogen storage alloy extraction.

Provides only JSON endpoints:
    GET /health   -> status
    GET /config   -> model/config info
    POST /extract -> accepts multipart/form-data PDF file, returns JSON + CSV text

The frontend (pure HTML/CSS/JS) should be served separately (e.g., from /frontend via any static server) and call POST /extract.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import fitz  # PyMuPDF
import os
import uuid
import csv
import io
import requests
import traceback
import re
from typing import List, Dict, Tuple, Any

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-oss:120b-cloud")

app = FastAPI(title="Hydrogen Storage Alloy Extractor API", version="0.2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RAW_TEXT_DIR = os.path.join(BASE_DIR, "data", "raw_text")
RESULT_DIR = os.path.join(BASE_DIR, "outputs")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RAW_TEXT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

ALLOY_TABLE_HEADER = ["Alloy Name", "Storage Capacity", "Temperature Range", "Pressure Range", "Synthesis Method"]

# ----- NEW MULTI-CHUNK EXTRACTION LOGIC -----
# Rationale: Single-pass prompt was truncating papers (12000 char cap) causing missed alloys.
# We now split the paper text into overlapping chunks, extract JSON-structured alloy data per chunk,
# and aggregate + deduplicate across the full paper.

MAX_CHARS_PER_CHUNK = int(os.environ.get("MAX_CHARS_PER_CHUNK", 8000))  # conservative for many local models
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 600))
MAX_CHUNKS = int(os.environ.get("MAX_CHUNKS", 15))  # safety limit
LLM_TIMEOUT_SECONDS = int(os.environ.get("LLM_TIMEOUT", 600))
RETRY_COUNT = int(os.environ.get("LLM_RETRIES", 2))
KEEP_EMPTY_ALLOYS = bool(int(os.environ.get("KEEP_EMPTY_ALLOYS", "0")))  # set to 1 to keep placeholder-only alloys

def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split long text into overlapping character chunks to fit model context."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    end = max_chars
    while start < len(text) and len(chunks) < MAX_CHUNKS:
        chunk = text[start:end]
        chunks.append(chunk)
        # Move start forward leaving an overlap to maintain context continuity
        start = end - overlap
        if start < 0:
            start = 0
        end = start + max_chars
    return chunks

def build_chunk_prompt(chunk_text: str, chunk_index: int, total_chunks: int) -> str:
    """Prompt instructing model to return ONLY JSON with list of alloy objects for a chunk."""
    return f"""
You are an expert materials science extraction assistant focused on hydrogen storage alloys.
You will be given PART {chunk_index+1} of {total_chunks} of a research paper.

Extract ONLY factual hydrogen storage alloy data explicitly present in THIS PART (do NOT guess content from other parts).
Return STRICT JSON with this schema (no markdown, no comments):
{{
  "alloys": [
    {{
      "name": "string",                  # alloy composition or identifier; keep Unicode subscripts/superscripts
      "storage_capacity": "string|null",  # value + units exactly as written (e.g., "2.3 wt% H2"), or null
      "temperature_range": "string|null", # exact temperature or range with units, or null
      "pressure_range": "string|null",    # exact pressure or range with units, or null
      "synthesis_method": "string|null"   # method/process as stated (e.g., mechanical alloying, arc melting), or null
    }}
  ]
}}

Rules:
1. If no alloys appear in this part, output {{"alloys": []}}.
2. Do NOT invent or infer values not explicitly written.
3. Avoid duplicate alloy entries inside the same chunk.
4. Keep numbers & units exactly.
5. One alloy entry per composition (merge repeated mentions).

TEXT PART START\n---\n{chunk_text[:max(0, MAX_CHARS_PER_CHUNK)]}\n---\nTEXT PART END

Return ONLY the JSON object now:"""

def query_ollama_raw(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.05, "top_p": 0.9, "top_k": 40}
    }
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=LLM_TIMEOUT_SECONDS)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Ollama error {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    return data.get("response", "").strip()

def extract_json_alloys(response_text: str) -> List[Dict[str, Any]]:
    """Attempt to parse JSON alloys list from model response which should be pure JSON.
    More defensive parsing in case of minor deviations."""
    if not response_text:
        return []
    # Try direct JSON first
    import json
    try:
        obj = json.loads(response_text)
        if isinstance(obj, dict) and isinstance(obj.get("alloys"), list):
            return sanitize_alloy_entries(obj["alloys"])
    except json.JSONDecodeError:
        pass
    # Fallback: regex curly braces capture
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if match:
        snippet = match.group(0)
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict) and isinstance(obj.get("alloys"), list):
                return sanitize_alloy_entries(obj["alloys"])
        except Exception:
            return []
    return []

def sanitize_alloy_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    cleaned = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        if not name:
            continue
        cleaned.append({
            "name": name,
            "storage_capacity": _clean_nullable(e.get("storage_capacity")),
            "temperature_range": _clean_nullable(e.get("temperature_range")),
            "pressure_range": _clean_nullable(e.get("pressure_range")),
            "synthesis_method": _clean_nullable(e.get("synthesis_method")),
        })
    return cleaned

def _clean_nullable(val: Any) -> str:
    if val is None:
        return ""
    val = str(val).strip()
    # Strip stray quotes/backticks
    return val.strip('`"')

def aggregate_alloys(list_of_lists: List[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """Merge alloys across chunks, deduplicating by normalized name.
    If the same alloy appears with differing non-empty attributes, merge them (semicolon join unique)."""
    merged: Dict[str, Dict[str, str]] = {}
    for chunk_entries in list_of_lists:
        for alloy in chunk_entries:
            norm = normalize_alloy_name(alloy["name"])
            if norm not in merged:
                merged[norm] = {
                    "Alloy Name": alloy["name"],
                    "Storage Capacity": alloy["storage_capacity"],
                    "Temperature Range": alloy["temperature_range"],
                    "Pressure Range": alloy["pressure_range"],
                    "Synthesis Method": alloy["synthesis_method"],
                }
            else:
                existing = merged[norm]
                for key_map in [
                    ("Storage Capacity", "storage_capacity"),
                    ("Temperature Range", "temperature_range"),
                    ("Pressure Range", "pressure_range"),
                    ("Synthesis Method", "synthesis_method"),
                ]:
                    out_key, src_key = key_map
                    new_val = alloy[src_key]
                    if new_val and new_val not in existing[out_key].split("; "):
                        if existing[out_key]:
                            existing[out_key] += "; " + new_val
                        else:
                            existing[out_key] = new_val
    # Return sorted list by alloy name for determinism
    return sorted(merged.values(), key=lambda r: r["Alloy Name"].lower())

def filter_informative_alloys(alloys: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Keep only alloys that have at least one non-empty property field (excluding name)."""
    informative = []
    for a in alloys:
        if any(a.get(col, "").strip() for col in ALLOY_TABLE_HEADER[1:]):
            informative.append(a)
    return informative

def normalize_alloy_name(name: str) -> str:
    # Lowercase, remove spaces & common separators for dedup (keep subscripts/superscripts characters)
    return re.sub(r'[\s\-]', '', name.lower())

def alloys_to_csv_rows(alloys: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
    if not alloys:
        header_line = " | ".join(ALLOY_TABLE_HEADER)
        return header_line, []
    header_line = " | ".join(ALLOY_TABLE_HEADER)
    lines = [header_line]
    for alloy in alloys:
        # Use '-' for unspecified/blank values at output stage (keep internal representation unchanged)
        row = [alloy.get(col, "") or '-' for col in ALLOY_TABLE_HEADER]
        lines.append(" | ".join(row))
    # Also prepare rows with '-' for API response consistency
    output_rows = []
    for alloy in alloys:
        output_rows.append({col: (alloy.get(col, "") or '-') for col in ALLOY_TABLE_HEADER})
    return "\n".join(lines), output_rows

# ----- END MULTI-CHUNK EXTRACTION LOGIC -----

# Serve static frontend (optional). If 'frontend' folder exists, mount at root.
if os.path.isdir(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

    @app.get("/", response_class=HTMLResponse)
    def root_index():
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                content = f.read()
            inject = "<script>window.API_BASE='';</script>"  # same origin
            # Rewrite relative asset paths to mounted /frontend path
            content = content.replace('href="styles.css"', 'href="/frontend/styles.css"')
            content = content.replace("src=\"script.js\"", "src=\"/frontend/script.js\"")
            # Basic favicon injection if absent
            if 'rel="icon"' not in content:
                content = content.replace('</head>', '<link rel="icon" href="/favicon.ico" /></head>')
            return inject + content
        return "<h1>Frontend not found</h1>"

    @app.get('/favicon.ico')
    def favicon():
        # Provide a 1x1 transparent gif in base64 (no file dependency)
        from fastapi import Response
        gif_b64 = 'R0lGODlhAQABAPAAAAAAAAAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=='
        import base64
        data = base64.b64decode(gif_b64)
        return Response(content=data, media_type='image/gif')

@app.get("/health")
def health():
    """Simple health/status endpoint."""
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "ollama_url": OLLAMA_URL,
        "upload_dir_exists": os.path.isdir(UPLOAD_DIR),
        "outputs_dir_exists": os.path.isdir(RESULT_DIR)
    }

@app.get("/config")
def config():
    return {"model": MODEL_NAME, "ollama_url": OLLAMA_URL}

def extract_text_from_pdf(path: str) -> str:
    text = []
    with fitz.open(path) as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)

def build_alloy_table_prompt(paper_text: str) -> str:
    """Legacy single-pass prompt (kept for fallback)."""
    instructions = f"You are an assistant. Return ONLY the CSV table. Text truncated.\n{paper_text[:8000]}"
    return instructions

def query_ollama(prompt: str) -> str:
    """Backward-compatible wrapper using new raw query function."""
    try:
        return query_ollama_raw(prompt)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Connection to Ollama failed: {e}")

def clean_csv_output(raw: str) -> str:
    # Remove accidental markdown fencing
    raw = raw.strip()
    if raw.lower().startswith("```"):
        raw = raw.strip('`')
    # Keep only lines that have separators or header
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    # Ensure header exists and is first
    header_idx = None
    for i,l in enumerate(lines):
        if l.lower().startswith("alloy name"):
            header_idx = i
            break
    if header_idx is None:
        return " | ".join(ALLOY_TABLE_HEADER)
    lines = lines[header_idx:]
    # Filter out lines that are not part of table (no separator)
    table_lines = [l for l in lines if '|' in l]
    if not table_lines:
        return " | ".join(ALLOY_TABLE_HEADER)
    # Normalize spacing around pipes
    normalized = []
    for l in table_lines:
        cells = [c.strip() for c in l.split('|')]
        row = " | ".join(cells)
        normalized.append(row)
    # Deduplicate header duplicates
    final = []
    seen_alloys = set()
    for row in normalized:
        if row.lower().startswith("alloy name"):
            if not final:
                final.append(row)
            continue
        parts = [p.strip() for p in row.split('|')]
        if parts:
            alloy = parts[0]
            if alloy and alloy.lower() not in seen_alloys:
                seen_alloys.add(alloy.lower())
                final.append(row)
    if not final:
        final.append(" | ".join(ALLOY_TABLE_HEADER))
    return "\n".join(final)

def csv_to_rows(csv_text: str) -> List[Dict[str,str]]:
    reader = csv.reader(io.StringIO(csv_text), delimiter='|')
    rows = list(reader)
    if not rows:
        return []
    header = [h.strip() for h in rows[0]]
    data_rows = []
    for r in rows[1:]:
        if len(r) != len(header):
            # Try to skip malformed rows
            continue
        data_rows.append({header[i].strip(): r[i].strip() for i in range(len(header))})
    return data_rows

## Note: Root path intentionally not implemented to keep backend minimal.

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    """Enhanced extraction endpoint with multi-chunk alloy aggregation.
    Returns: JSON containing rows + csv_text + diagnostic info about chunk processing.
    """
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    file_id = str(uuid.uuid4())
    pdf_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    with open(pdf_path, 'wb') as f:
        f.write(await file.read())

    # Extract text & persist raw text
    try:
        text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        tb = traceback.format_exc(limit=2)
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {e}; trace: {tb}")
    raw_text_path = os.path.join(RAW_TEXT_DIR, f"{file_id}.txt")
    with open(raw_text_path, 'w', encoding='utf-8') as f:
        f.write(text)

    # Multi-chunk processing
    chunks = chunk_text(text)
    all_chunk_results: List[List[Dict[str, str]]] = []
    chunk_errors = 0
    for idx, chunk in enumerate(chunks):
        prompt = build_chunk_prompt(chunk, idx, len(chunks))
        alloys = []
        for attempt in range(RETRY_COUNT + 1):
            try:
                resp = query_ollama_raw(prompt)
                alloys = extract_json_alloys(resp)
                # If response empty but we expected maybe some content, retry once
                if alloys or attempt == RETRY_COUNT:
                    break
            except Exception:
                if attempt == RETRY_COUNT:
                    chunk_errors += 1
        all_chunk_results.append(alloys)

    aggregated = aggregate_alloys(all_chunk_results)
    pre_filter_count = len(aggregated)
    if not KEEP_EMPTY_ALLOYS:
        aggregated = filter_informative_alloys(aggregated)
    csv_text, final_rows = alloys_to_csv_rows(aggregated)

    # Fallback: legacy single-pass if no alloys detected
    fallback_used = False
    if not final_rows:
        fallback_used = True
        try:
            legacy_prompt = build_alloy_table_prompt(text)
            legacy_output = query_ollama(legacy_prompt)
            csv_text = clean_csv_output(legacy_output)
            parsed_rows = csv_to_rows(csv_text)
            # Normalize blanks to '-'
            for r in parsed_rows:
                for k,v in r.items():
                    if not v.strip():
                        r[k] = '-'
            # Rebuild csv_text to ensure consistency with '-'
            if parsed_rows:
                header = " | ".join(ALLOY_TABLE_HEADER)
                lines = [header]
                for r in parsed_rows:
                    lines.append(" | ".join([r.get(col,'') or '-' for col in ALLOY_TABLE_HEADER]))
                csv_text = "\n".join(lines)
            # Apply filtering on fallback result too if configured
            if not KEEP_EMPTY_ALLOYS:
                parsed_rows = [r for r in parsed_rows if any(r.get(col,'').strip() for col in ALLOY_TABLE_HEADER[1:])]
            final_rows = parsed_rows
        except Exception:
            # Keep empty result with header
            csv_text = " | ".join(ALLOY_TABLE_HEADER)
            final_rows = []

    # Save CSV
    csv_filename = f"alloy_table_{file_id}.csv"
    csv_path = os.path.join(RESULT_DIR, csv_filename)
    # Write a proper comma-delimited CSV for spreadsheet tools (keep pipe version in API response)
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:  # utf-8-sig adds BOM for Excel Unicode
        writer = csv.writer(f)
        writer.writerow(ALLOY_TABLE_HEADER)
        for r in final_rows:
            writer.writerow([r.get(col, '') for col in ALLOY_TABLE_HEADER])

    return JSONResponse({
        "file_id": file_id,
        "rows": final_rows,
        "csv_text": csv_text,
        "download_url": f"/download/{csv_filename}",
        "model": MODEL_NAME,
        "diagnostics": {
            "chunks": len(chunks),
            "chunk_errors": chunk_errors,
            "fallback_used": fallback_used,
            "total_alloys": len(final_rows),
            "pre_filter_alloys": pre_filter_count,
            "kept_empty_alloys": KEEP_EMPTY_ALLOYS
        }
    })

@app.get("/download/{filename}")
def download_csv(filename: str):
    path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type='text/csv', filename=filename)

if __name__ == "__main__":
    uvicorn.run("main_fastapi:app", host="0.0.0.0", port=8000, reload=True)
