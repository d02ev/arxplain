import json
import os
import sys
from ai_integration import init, call_llm


SYSTEM_PROMPT = """
You are an expert research paper analyst.

Your task is to extract:
- the main problem statement
- the motivation (why this matters)
- the claimed contributions
- key claims made by the authors

Rules:
- Only use content from the provided text.
- Do NOT invent claims.
- Every extracted item MUST have a trace with page number and a direct snippet from the text.
- Contributions must be short and atomic.
- Key claims must be atomic and specific.
- Output must be valid JSON only. No markdown.
"""

def find_section_pages(outline_sections, target_names):
    """
    Returns list of (start_page, end_page) for matching section names.
    """
    ranges = []
    for sec in outline_sections:
        name = sec.get("name", "").lower().strip()
        for target in target_names:
            if target in name:
                ranges.append((sec["start_page"], sec["end_page"]))
    return ranges

def collect_text_from_page_ranges(pages, page_ranges):
    """
    Collect text from specific page ranges.
    """
    collected = []
    for start, end in page_ranges:
        for p in pages:
            if start <= p["page_number"] <= end:
                collected.append(f"[PAGE {p['page_number']}]\n{p['text']}")
    return "\n\n".join(collected)

def build_user_prompt(data: dict) -> str:
    outline = data.get("outline", {})
    sections = outline.get("sections", [])
    pages = data.get("pages", [])
    abstract = outline.get("abstract", "")

    intro_ranges = find_section_pages(sections, ["introduction"])
    concl_ranges = find_section_pages(sections, ["conclusion", "discussion", "limitations"])

    intro_text = collect_text_from_page_ranges(pages, intro_ranges)
    concl_text = collect_text_from_page_ranges(pages, concl_ranges)

    # fallback: if no intro found, use first 2 pages
    if not intro_text.strip():
        intro_text = "\n\n".join([f"[PAGE {p['page_number']}]\n{p['text']}" for p in pages[:2]])

    return f"""
ABSTRACT:
{abstract}

INTRODUCTION_TEXT:
{intro_text[:8000]}

CONCLUSION_DISCUSSION_LIMITATIONS_TEXT:
{concl_text[:6000]}

Return ONLY JSON in this format:
{{
  "problem_statement": {{
    "text": string,
    "trace": {{
      "page": number,
      "snippet": string
    }}
  }},
  "motivation": {{
    "text": string,
    "trace": {{
      "page": number,
      "snippet": string
    }}
  }},
  "contributions": [
    {{
      "contribution_id": "CON1",
      "text": string,
      "trace": {{
        "page": number,
        "snippet": string
      }}
    }}
  ],
  "key_claims": [
    {{
      "claim_id": "C1",
      "type": "performance|novelty|efficiency|theory|other",
      "text": string,
      "evidence_hint": string,
      "trace": {{
        "page": number,
        "snippet": string
      }},
      "confidence": "high|medium|low"
    }}
  ]
}}

Constraints:
- Every item MUST include trace.page and trace.snippet.
- trace.snippet MUST be a direct quote fragment from the provided text.
- If evidence_hint is unknown, use "".
- Do not hallucinate.
"""

def extract_claims(input_json: str, output_json: str, model: str):
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    client = init()

    user_prompt = build_user_prompt(data)

    extracted_claims = call_llm(
        client=client,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model
    )

    data["claims"] = extracted_claims

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)