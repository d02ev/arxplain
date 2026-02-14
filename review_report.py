import json
import os
import sys
from ai_integration import init, call_llm


SYSTEM_PROMPT = """
You are a strict research paper reviewer.

You are given:
- extracted ground-truth structured data from a paper
- a generated markdown explanation report

Your job is to critique the report.

Rules:
- Do NOT rewrite the report.
- Only evaluate it.
- Identify missing content, unclear explanations, and hallucinations.
- Hallucination means: any claim not supported by the extracted structured data.
- Be strict and skeptical.
- Output must be valid JSON only.
"""


def build_user_prompt(data: dict) -> str:
    outline = data.get("outline", {})
    claims = data.get("claims", {})
    method = data.get("method", {})
    experiments = data.get("experiments", {})
    report_md = data.get("explanation_report", {}).get("content", "")

    return f"""
GROUND_TRUTH_EXTRACTED_DATA:

OUTLINE_JSON:
{json.dumps(outline, indent=2)}

CLAIMS_JSON:
{json.dumps(claims, indent=2)}

METHOD_JSON:
{json.dumps(method, indent=2)}

EXPERIMENTS_JSON:
{json.dumps(experiments, indent=2)}

GENERATED_MARKDOWN_REPORT:
{report_md}

Now critique the report.

Return ONLY JSON in this format:
{{
  "overall_score": number,
  "section_scores": {{
    "problem_explanation": number,
    "core_idea_explanation": number,
    "method_explanation": number,
    "results_explanation": number,
    "limitations": number,
    "clarity": number,
    "structure": number,
    "hallucination_risk": number
  }},
  "missing_sections": [string],
  "hallucinated_statements": [
    {{
      "statement": string,
      "reason": string,
      "severity": "high|medium|low"
    }}
  ],
  "weak_explanations": [
    {{
      "section": string,
      "problem": string,
      "fix_suggestion": string
    }}
  ],
  "rewrite_instructions": [string]
}}

Constraints:
- overall_score must be 0 to 100.
- Each section score must be 0 to 10.
- hallucination_risk: 10 means extremely risky, 0 means fully grounded.
- Be strict.
"""


def review_report(input_json_path: str, output_json_path: str, model: str):
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    client = init()
    user_prompt = build_user_prompt(data)

    review_json = call_llm(
        client=client,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model
    )

    # Append review into JSON
    data["review"] = review_json

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
