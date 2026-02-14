import json
import os
import sys
from ai_integration import init, call_llm

SYSTEM_PROMPT = """
You are an expert academic research paper parser.

Your task is to refine and normalize a research paper outline.

You must ONLY output valid JSON. No markdown. No commentary.

Rules:
- Normalize section names to Title Case.
- Remove non-section noise (copyright, arXiv, preprint info).
- Do NOT invent new sections.
- Use ONLY the extracted section candidates provided.
- Keep correct ordering.
- Ensure start_page <= end_page.
- Page ranges must be non-overlapping.
- trace.snippet should resemble one of the candidate snippets.
"""

def build_user_prompt(outline_raw_data: dict) -> str:
  pages = outline_raw_data.get("pages", [])
  outline = outline_raw_data.get("outline", {})

  first_page_text = pages[0]["text"] if pages else ""
  abstract = outline.get("abstract", "")
  candidates = outline.get("section_candidates", [])

  page_count = outline_raw_data.get("source", {}).get("page_count", len(pages))

  return f"""
FIRST_PAGE_TEXT:
{first_page_text[:4000]}

ABSTRACT_EXTRACTED:
{abstract[:2000]}

SECTION_CANDIDATES_EXTRACTED:
{json.dumps(candidates, indent=2)}

PAGE_COUNT: {page_count}

Return ONLY JSON in this format:
{{
  "title": string | null,
  "authors": [string],
  "keywords": [string],
  "sections": [
    {{
      "name": string,
      "start_page": number,
      "end_page": number,
      "trace": {{
        "page": number,
        "snippet": string
      }}
    }}
  ]
}}

IMPORTANT:
- Sections MUST be derived from candidates.
- Do not add fake sections.
- Use correct page ranges.
"""

def apply_outline_refinement(outline_raw_data: dict, refined_outline: dict) -> dict:
  old_outline = outline_raw_data.get("outline", {})
  abstract = old_outline.get("abstract")

  outline_raw_data["outline"] = {
      "title": refined_outline.get("title"),
      "authors": refined_outline.get("authors", []),
      "abstract": abstract,
      "keywords": refined_outline.get("keywords", []),
      "sections": refined_outline.get("sections", []),
      "section_candidates": old_outline.get("section_candidates", [])
  }

  return outline_raw_data

def refine_outline(output_s2_json: str, output_path: str, model: str):
  if not os.path.exists(output_s2_json):
    raise FileNotFoundError(f"Input JSON not found: {output_s2_json}")

  with open(output_s2_json, "r", encoding="utf-8") as f:
    stage2_data = json.load(f)

    client = init()

    user_prompt = build_user_prompt(stage2_data)

    refined_outline = call_llm(
        client=client,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model
    )

    updated_data = apply_outline_refinement(stage2_data, refined_outline)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)