# llm_prompt.py
"""
Prompt template and LLM query function for hydrogen storage alloy literature mining.
Assumes local DeepSeek R1 model accessible via HuggingFace Transformers or similar.
"""


import requests
from typing import List, Dict

FEW_SHOT_EXAMPLES = [
    {
        "input": "Title: Hydrogen Storage in Mg-based Alloys. Authors: A. Smith, B. Lee. Journal: Journal of Materials Science, 2021. Abstract: Mg alloys synthesized by ball milling show 6.5 wt% hydrogen capacity at 300°C, 5 MPa. DOI: 10.1007/s10853-021-05823-4",
        "output": {
            "Title": "Hydrogen Storage in Mg-based Alloys",
            "Authors": "A. Smith, B. Lee",
            "Year": 2021,
            "Journal": "Journal of Materials Science",
            "Type of Study": "Experimental",
            "Alloy Composition": "Mg-based alloys",
            "Synthesis/Processing Method": "Ball milling",
            "Hydrogen Storage Capacity (wt%)": "6.5",
            "Operating Conditions": "300°C, 5 MPa",
            "Key Findings": "Mg alloys show improved hydrogen absorption after ball milling.",
            "Reference Link or DOI": "10.1007/s10853-021-05823-4"
        }
    },
    {
        "input": "Title: High-Pressure Hydrogen Storage in TiFe Alloys. Authors: J. Doe, M. Chan. Journal: International Journal of Hydrogen Energy, 2020. Abstract: TiFe alloys synthesized via arc melting show hydrogen storage capacity of 1.8 wt% at 25°C and 30 bar. DOI: 10.1016/j.ijhydene.2020.01.123",
        "output": {
            "Title": "High-Pressure Hydrogen Storage in TiFe Alloys",
            "Authors": "J. Doe, M. Chan",
            "Year": 2020,
            "Journal": "International Journal of Hydrogen Energy",
            "Type of Study": "Experimental",
            "Alloy Composition": "TiFe alloys",
            "Synthesis/Processing Method": "Arc melting",
            "Hydrogen Storage Capacity (wt%)": "1.8",
            "Operating Conditions": "25°C, 30 bar",
            "Key Findings": "Stable cycling performance.",
            "Reference Link or DOI": "10.1016/j.ijhydene.2020.01.123"
        }
    },
    {
            "input": '''Highlights
    Hydrogen storage performance of V-Ti-based solid solution alloys is related to the elementary composition, phase structure, and homogeneity.

    Micro-strain accumulation is responsible for capacity degradation.

    Low-cost and high-performance V-Ti-based solid solution alloys with high reversible hydrogen storage capacity, good cyclic durability, and excellent activation performance should be developed.

    Abstract
    This review details the advancement in the development of V–Ti-based hydrogen storage materials for using in metal hydride (MH) tanks to supply hydrogen to fuel cells at relatively ambient temperatures and pressures. V–Ti-based solid solution alloys are excellent hydrogen storage materials among many metal hydrides due to their high reversible hydrogen storage capacity which is over 2 wt% at ambient temperature. The preparation methods, structure characteristics, improvement methods of hydrogen storage performance, and attenuation mechanism are systematically summarized and discussed. The relationships between hydrogen storage properties and alloy compositions as well as phase structures are discussed emphatically. For large-scale applications on MH tanks, it is necessary to develop low-cost and high-performance V–Ti-based solid solution alloys with high reversible hydrogen storage capacity, good cyclic durability, and excellent activation performance.
    ''',
            "output": {
                "Title": "Advancement in the development of V–Ti-based hydrogen storage materials",
                "Authors": "N/A",
                "Year": "N/A",
                "Journal": "N/A",
                "Type of Study": "Review",
                "Alloy Composition": "V–Ti-based solid solution alloys",
                "Synthesis/Processing Method": "Arc melting, vacuum magnetic levitation induction melting, powder sintering, aluminothermy process, ball milling, suction casting, floating zone melting, rapid solidification, laser engineered net shaping, heat treatment",
                "Hydrogen Storage Capacity (wt%)": "Over 2 wt% at ambient temperature (theoretical up to 4 wt%, effective about 2.6 wt%)",
                "Operating Conditions": "Ambient temperature, relatively low pressure",
                "Key Findings": "V–Ti-based alloys have high reversible hydrogen storage capacity, good cyclic durability, and excellent activation performance. Micro-strain accumulation degrades capacity. Preparation methods and alloy composition strongly affect performance.",
                "Reference Link or DOI": "N/A"
            }
        }
    ]

PROMPT_TEMPLATE = """
You are an expert assistant extracting structured data from scientific literature about hydrogen storage alloys.\n\nThe input may contain sections like Highlights, Abstract, Introduction, etc. Extract all relevant information for each field, even if the information is spread across multiple sections.\n\nExtract the following fields from the provided text:\n- Title\n- Authors\n- Year\n- Journal\n- Type of Study (Experimental / Theoretical / Review / Simulation)\n- Alloy Composition\n- Synthesis/Processing Method\n- Hydrogen Storage Capacity (wt%)\n- Operating Conditions (Temperature, Pressure, etc.)\n- Key Findings\n- Reference Link or DOI\n\nIf a field is missing, output "N/A".\n\nFew-shot examples:\n"""
for ex in FEW_SHOT_EXAMPLES:
    PROMPT_TEMPLATE += f"\nInput: {ex['input']}\nOutput: {ex['output']}\n"
PROMPT_TEMPLATE += "\nNow, extract the fields from the following text:\nInput: {{input_text}}\nOutput:"


def build_prompt(input_text: str) -> str:
    """Builds the full prompt for the LLM."""
    return PROMPT_TEMPLATE.replace("{{input_text}}", input_text)



def query_deepseek_ollama(input_text: str, model_name: str = "deepseek") -> str:
    """
    Sends the prompt to the local DeepSeek R1 model via Ollama API and returns the output string.
    """
    prompt = build_prompt(input_text)
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()
    return result.get("response", "")

# Example usage:
# from transformers import AutoModelForCausalLM, AutoTokenizer
# model = AutoModelForCausalLM.from_pretrained('path/to/deepseek-r1')
# tokenizer = AutoTokenizer.from_pretrained('path/to/deepseek-r1')
# result = query_deepseek(model, tokenizer, "Your abstract text here")
# print(result)
