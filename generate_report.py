import json
import os
import sys
from ai_integration import init, call_llm


SYSTEM_PROMPT = """
You are a senior research mentor and technical educator.

Your job is to explain a research paper clearly and deeply.

Rules:
- You MUST explain, not summarize.
- Write like teaching an engineer who knows programming but not the paper topic.
- Use simple language.
- Avoid academic fluff.
- Use analogies where helpful.
- Do NOT invent numbers, datasets, or claims.
- Use only the extracted structured data provided.
- Output must be valid JSON only.
"""


def build_user_prompt(data: dict) -> str:
    outline = data.get("outline", {})
    claims = data.get("claims", {})
    method = data.get("method", {})
    experiments = data.get("experiments", {})

    title = outline.get("title") or data.get("source", {}).get("file_name", "Unknown Paper")

    return f"""
You are given extracted structured data from a research paper.

PAPER_TITLE:
{title}

ABSTRACT:
{outline.get("abstract", "")}

CLAIMS_JSON:
{json.dumps(claims, indent=2)}

METHOD_JSON:
{json.dumps(method, indent=2)}

EXPERIMENTS_JSON:
{json.dumps(experiments, indent=2)}

Generate an explanation report in Markdown.

Return ONLY JSON in this format:
{{
  "markdown_report": "string"
}}

Markdown must follow this exact structure:

# {title}

## TL;DR (max 5 lines)
...

## 1. What problem does this paper solve?
...

## 2. Why is this problem hard?
...

## 3. What is the main contribution?
...

## 4. Core idea (intuitive explanation)
...

## 5. How the method works (step-by-step)
1. ...
2. ...

## 6. Architecture / Components
- **Component**: purpose

## 7. Experiments and Results (What matters)
...

## 8. What do the results actually prove?
...

## 9. Limitations / Assumptions
...

## 10. Practical takeaways (for engineers)
...

## Glossary (simple definitions)
- Term: meaning

## Skeptical reviewer notes
...

Constraints:
- If results are missing, say "Not clearly extracted".
- If limitations are missing, say "Not explicitly stated".
- Keep it clear and concise.
"""


def generate_report(input_json_path: str, output_json_path: str, report_md_path: str, model: str):
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    client = init()
    user_prompt = build_user_prompt(data)

    response_json = call_llm(
        client=client,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model
    )

    markdown_report = response_json.get("markdown_report", "").strip()

    if not markdown_report:
        raise ValueError("Stage #05 failed: markdown_report is empty.")

    # Save markdown report
    os.makedirs(os.path.dirname(report_md_path), exist_ok=True)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    # Append report into JSON
    data["explanation_report"] = {
        "format": "markdown",
        "path": report_md_path
    }

    # Save final JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)