"""
Advanced LLM prompting for structured data extraction from hydrogen storage alloy literature.
Supports multiple LLM backends and implements Chain-of-Thought reasoning.
"""

import json
import requests
import logging
from typing import Dict, Any, Optional, List
import re


logger = logging.getLogger(__name__)

# Minimal extract_from_pdf for FastAPI integration
import fitz  # PyMuPDF
import pandas as pd
def extract_from_pdf(pdf_path):
    """
    Full wrapper for your LLM pipeline: extracts text from PDF, runs LLM extraction, returns DataFrame with all structured fields.
    """
    # Extract text from PDF
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()

    # Use the filename (without extension) as paper_id
    import os
    paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
    raw_text_path = f"data/raw_text/{paper_id}.txt"

    # Save raw text for traceability (optional, can skip if not needed)
    os.makedirs(os.path.dirname(raw_text_path), exist_ok=True)
    with open(raw_text_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Run LLM extraction
    extractor = LLMExtractor()
    result = extractor.extract_from_text(text, paper_id, raw_text_path)
    if not result:
        # Fallback if LLM fails
        result = extractor.create_fallback_extraction(paper_id, raw_text_path)

    # Add additional metadata fields for output compatibility
    result["title"] = None
    result["authors"] = None
    result["year"] = None
    result["doi"] = None
    result["abstract"] = None
    result["pdf_path"] = pdf_path

    # Reorder columns to match your desired output
    columns = [
        "title", "authors", "year", "doi", "abstract", "pdf_path", "alloy_name", "storage_capacity_wt_percent", "storage_capacity_note", "synthesis_method", "operating_conditions", "advantages", "limitations", "extracted_from", "confidence_score", "raw_text_path"
    ]
    df = pd.DataFrame([{col: result.get(col) for col in columns}])
    return df

# Define the expected output schema
OUTPUT_SCHEMA = {
    "alloy_name": "string|null",
    "storage_capacity_wt_percent": "float|null", 
    "storage_capacity_note": "string|null",
    "synthesis_method": "string|null",
    "operating_conditions": "string|null",
    "advantages": "string|null",
    "limitations": "string|null",
    "extracted_from": "string",
    "confidence_score": "float",
    "raw_text_path": "string"
}

# Few-shot examples with Chain-of-Thought reasoning
FEW_SHOT_EXAMPLES = [
    {
        "input": """Title: Hydrogen absorption and desorption properties of Li-Mg-Al-Ti-M based lightweight high entropy alloys
Authors: H. Hashimoto, S. Isobe, T. Minami, K. Asano, K. Sakaki, N. Hashimoto, H. Oka
Abstract: The hydrogen absorption and desorption properties of the Li-Mg-Al-Ti-V and Li-Mg-Al-Ti-Nb lightweight high entropy alloys (HEAs) synthesized by mechanical alloying were investigated to develop hydrogen storage alloys with high gravimetric capacity. An equiatomic Li₂₀Mg₂₀Al₂₀Ti₂₀V₂₀ consists of two phases; a Ti-V based bodycentered cubic (BCC) phase and a Li-Mg based BCC phase. The alloy absorbs hydrogen up to H/M = 0.74 (2.3 wt% H₂) at 250 ℃, although the hydrogen absorption and desorption induced phase separation of Mg from the Li-Mg based BCC phase.""",
        
        "reasoning": "Let me analyze this step by step: 1) The alloy composition is clearly Li-Mg-Al-Ti-V high entropy alloy with specific composition Li₂₀Mg₂₀Al₂₀Ti₂₀V₂₀. 2) Synthesis method is mechanical alloying. 3) Storage capacity is explicitly stated as 2.3 wt% H₂ at 250°C. 4) Operating conditions are 250°C (temperature given). 5) The advantage is high gravimetric capacity for lightweight applications. 6) Limitation is phase separation during cycling.",
        
        "output": {
            "alloy_name": "Li₂₀Mg₂₀Al₂₀Ti₂₀V₂₀ (Li-Mg-Al-Ti-V HEA)",
            "storage_capacity_wt_percent": 2.3,
            "storage_capacity_note": "H/M = 0.74 equivalent to 2.3 wt% H₂",
            "synthesis_method": "Mechanical alloying",
            "operating_conditions": "250°C", 
            "advantages": "High gravimetric capacity, lightweight high entropy alloy structure",
            "limitations": "Phase separation of Mg during hydrogen absorption/desorption cycling",
            "extracted_from": "abstract",
            "confidence_score": 0.95,
            "raw_text_path": "data/raw_text/1.txt"
        }
    },
    {
        "input": """Title: Unlocking improved hydrogen storage: Thermodynamic tuning and ionic conductivity boost in Fe-doped Mg2NiH4
Authors: I. Belkoufa, A. Assila, S. Sebbahi, A. Alaoui-Belghiti, S. Laasri, M. Tlemçani, E.K. Hlil, A.E. Hajjaji
Abstract: Mg₂Ni is considered a promising candidate for hydrogen storage materials due to its reasonable hydrogenation and dehydrogenation kinetics and cost-effectiveness. However, the high thermodynamic stability of Mg₂NiH₄ poses a significant challenge in terms of the operating temperature required for hydrogen release. This study investigates the crystal and electronic structure, and thermodynamic stability of Iron-doped Mg₂NiH₄ and their alloys using first-principles calculations based on density functional theory. The results demonstrate that by replacing one in sixteen Mg atoms and one in eight Ni atoms with Fe, the enthalpy of hydrogen desorption can be reduced from 65.173 to 57.58 and 50.72 kJ/mol H₂, respectively.""",
        
        "reasoning": "Analyzing this systematically: 1) Base alloy is Mg₂Ni with Fe doping. 2) This is a theoretical/computational study using DFT calculations. 3) No direct storage capacity mentioned, but focuses on thermodynamic improvements. 4) Operating temperature reduction is the main advantage through reduced desorption enthalpy. 5) Fe doping is the modification method. 6) The study shows ionic conductivity improvements up to 5×10⁻¹ S/cm.",
        
        "output": {
            "alloy_name": "Fe-doped Mg₂NiH₄",
            "storage_capacity_wt_percent": None,
            "storage_capacity_note": "Storage capacity not explicitly reported; focus on thermodynamic stability",
            "synthesis_method": "Theoretical study - DFT calculations, Fe substitution in Mg₂Ni structure",
            "operating_conditions": "400 K for ionic conductivity measurements",
            "advantages": "Reduced desorption enthalpy (from 65.173 to 50.72 kJ/mol H₂), improved ionic conductivity (up to 5×10⁻¹ S/cm), cost-effective base material",
            "limitations": "High thermodynamic stability of base Mg₂NiH₄ requiring elevated temperatures",
            "extracted_from": "abstract",
            "confidence_score": 0.90,
            "raw_text_path": "data/raw_text/2.txt"
        }
    }
]

class LLMExtractor:
    def __init__(self, model_name: str = "gpt-oss:120b-cloud", api_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.api_url = api_url
        
    def build_extraction_prompt(self, text: str, paper_id: str) -> str:
        """
        Build a comprehensive prompt for structured data extraction.
        """
        prompt = f"""You are an expert materials science researcher specializing in hydrogen storage alloys. Your task is to extract structured information from scientific literature.

IMPORTANT INSTRUCTIONS:
1. Analyze the text using Chain-of-Thought reasoning
2. Extract ONLY factual information explicitly stated in the text
3. Return ONLY valid JSON - no explanations, no markdown, no additional text
4. Use null for missing information
5. Be conservative with confidence scores

REQUIRED OUTPUT SCHEMA:
{json.dumps(OUTPUT_SCHEMA, indent=2)}

FEW-SHOT EXAMPLES:

Example 1:
Input: {FEW_SHOT_EXAMPLES[0]['input']}
Reasoning: {FEW_SHOT_EXAMPLES[0]['reasoning']}
Output: {json.dumps(FEW_SHOT_EXAMPLES[0]['output'], indent=2)}

Example 2:
Input: {FEW_SHOT_EXAMPLES[1]['input']}
Reasoning: {FEW_SHOT_EXAMPLES[1]['reasoning']}
Output: {json.dumps(FEW_SHOT_EXAMPLES[1]['output'], indent=2)}

NOW EXTRACT FROM THIS TEXT:

Input: {text[:3000]}{'...' if len(text) > 3000 else ''}

Think step by step about:
1. What is the main alloy composition?
2. What synthesis/processing methods are described?  
3. What storage capacity values are reported?
4. What operating conditions are mentioned?
5. What are the key advantages and limitations?

Output ONLY JSON:"""

        return prompt
    
    def query_ollama(self, prompt: str) -> Optional[str]:
        """
        Query local Ollama model.
        """
        try:
            url = f"{self.api_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for deterministic output
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            logger.error(f"Error querying Ollama: {str(e)}")
            return None
    
    def extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Extract and validate JSON from LLM response.
        """
        if not response:
            return None
        
        # Try to find JSON in the response
        json_patterns = [
            r'\{.*\}',  # Look for JSON object
            r'```json\s*(\{.*\})\s*```',  # JSON in code blocks
            r'```\s*(\{.*\})\s*```',  # JSON in generic code blocks
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    # Clean the match
                    json_str = match.strip()
                    if not json_str.startswith('{'):
                        continue
                    
                    # Parse JSON
                    data = json.loads(json_str)
                    
                    # Validate required fields
                    if self.validate_extracted_data(data):
                        return data
                        
                except json.JSONDecodeError:
                    continue
        
        # If no valid JSON found, try parsing the entire response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Could not extract valid JSON from response")
            return None
    
    def validate_extracted_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate that extracted data matches expected schema.
        """
        required_fields = ["extracted_from", "confidence_score"]
        
        for field in required_fields:
            if field not in data:
                return False
        
        # Validate confidence score
        if not isinstance(data.get("confidence_score"), (int, float)) or not 0 <= data["confidence_score"] <= 1:
            return False
        
        # Validate storage capacity if present
        if data.get("storage_capacity_wt_percent") is not None:
            if not isinstance(data["storage_capacity_wt_percent"], (int, float)):
                return False
        
        return True
    
    def extract_from_text(self, text: str, paper_id: str, raw_text_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract structured data from text using LLM.
        """
        prompt = self.build_extraction_prompt(text, paper_id)
        response = self.query_ollama(prompt)
        
        if not response:
            logger.error(f"No response from LLM for paper {paper_id}")
            return None
        
        extracted_data = self.extract_json_from_response(response)
        
        if not extracted_data:
            logger.error(f"Could not extract valid JSON for paper {paper_id}")
            return None
        
        # Add metadata
        extracted_data["raw_text_path"] = raw_text_path
        
        logger.info(f"Successfully extracted data for paper {paper_id}")
        return extracted_data
    
    def create_fallback_extraction(self, paper_id: str, raw_text_path: str) -> Dict[str, Any]:
        """
        Create a fallback extraction when LLM fails.
        """
        return {
            "alloy_name": None,
            "storage_capacity_wt_percent": None,
            "storage_capacity_note": None,
            "synthesis_method": None,
            "operating_conditions": None,
            "advantages": None,
            "limitations": None,
            "extracted_from": "fallback",
            "confidence_score": 0.0,
            "raw_text_path": raw_text_path
        }


if __name__ == "__main__":
    # Test the extractor
    extractor = LLMExtractor()
    test_text = """Title: Test paper on Mg-Ni alloys
    Abstract: Mg-Ni alloys show 3.5 wt% hydrogen storage capacity at 300°C."""
    
    result = extractor.extract_from_text(test_text, "test", "test.txt")
    if result:
        print(json.dumps(result, indent=2))