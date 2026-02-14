import json
import os
import sys
from ai_integration import init, call_llm


SYSTEM_PROMPT = """
You are an expert research paper technical extractor.

Your job is to extract the method and experimental results from a research paper.

Rules:
- Only use information explicitly present in the provided text.
- Do NOT hallucinate numbers.
- If a number is not explicitly present, leave it null.
- Every extracted item must include trace.page and trace.snippet.
- trace.snippet must be a direct quote fragment from the input text.
- Output must be valid JSON only.
"""

def find_section_ranges(sections, keywords):
    ranges = []
    for sec in sections:
        name = sec.get("name", "").lower()
        if any(k in name for k in keywords):
            ranges.append((sec["start_page"], sec["end_page"]))
    return ranges


def collect_text_from_ranges(pages, ranges, max_chars=12000):
    collected = []
    for start, end in ranges:
        for p in pages:
            if start <= p["page_number"] <= end:
                collected.append(f"[PAGE {p['page_number']}]\n{p['text']}")
    combined = "\n\n".join(collected)
    return combined[:max_chars]


def build_user_prompt(data: dict) -> str:
    outline = data.get("outline", {})
    sections = outline.get("sections", [])
    pages = data.get("pages", [])

    method_ranges = find_section_ranges(sections, ["method", "methodology", "approach", "model", "architecture", "self-attention", "training", "attention"])
    exp_ranges = find_section_ranges(sections, ["experiment", "results", "evaluation", "benchmark"])
    limit_ranges = find_section_ranges(sections, ["limitation", "discussion", "conclusion"])

    method_text = collect_text_from_ranges(pages, method_ranges, max_chars=12000)
    exp_text = collect_text_from_ranges(pages, exp_ranges, max_chars=12000)
    limit_text = collect_text_from_ranges(pages, limit_ranges, max_chars=6000)

    # fallback if not found
    if not method_text.strip():
        method_text = "\n\n".join([f"[PAGE {p['page_number']}]\n{p['text']}" for p in pages[:4]])[:12000]

    if not exp_text.strip():
        exp_text = "\n\n".join([f"[PAGE {p['page_number']}]\n{p['text']}" for p in pages[-4:]])[:12000]

    return f"""
METHOD_SECTION_TEXT:
{method_text}

EXPERIMENTS_RESULTS_TEXT:
{exp_text}

LIMITATIONS_CONCLUSION_TEXT:
{limit_text}

Return ONLY JSON in this format:
{{
  "method": {{
    "high_level_summary": {{
      "text": string,
      "trace": {{ "page": number, "snippet": string }}
    }},
    "core_idea": {{
      "text": string,
      "trace": {{ "page": number, "snippet": string }}
    }},
    "step_by_step": [
      {{
        "step": number,
        "text": string,
        "trace": {{ "page": number, "snippet": string }}
      }}
    ],
    "architecture_components": [
      {{
        "name": string,
        "purpose": string,
        "trace": {{ "page": number, "snippet": string }}
      }}
    ],
    "equations": [
      {{
        "equation": string,
        "meaning": string,
        "trace": {{ "page": number, "snippet": string }}
      }}
    ]
  }},
  "experiments": {{
    "datasets": [
      {{
        "name": string,
        "trace": {{ "page": number, "snippet": string }}
      }}
    ],
    "metrics": [
      {{
        "name": string,
        "trace": {{ "page": number, "snippet": string }}
      }}
    ],
    "baselines": [
      {{
        "name": string,
        "trace": {{ "page": number, "snippet": string }}
      }}
    ],
    "results": [
      {{
        "dataset": string,
        "metric": string,
        "baseline": string,
        "baseline_value": number | null,
        "proposed_value": number | null,
        "delta": number | null,
        "trace": {{ "page": number, "snippet": string }}
      }}
    ],
    "limitations": [
      {{
        "text": string,
        "trace": {{ "page": number, "snippet": string }}
      }}
    ]
  }}
}}

Constraints:
- Do not fabricate baselines, datasets, or numbers.
- If you cannot find exact numeric values, use null.
- Ensure every list item has trace.page and trace.snippet.
"""

def method_result_extraction(input_json: str, output_json: str, model: str):
  if not os.path.exists(input_json):
    raise FileNotFoundError(f"Input JSON not found: {input_json}")

  with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

  client = init()
  user_prompt = build_user_prompt(data)

  extracted = call_llm(
    client=client,
    system_prompt=SYSTEM_PROMPT,
    user_prompt=user_prompt,
    model=model
  )

  data["method"] = extracted.get("method", {})
  data["experiments"] = extracted.get("experiments", {})

  with open(output_json, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)