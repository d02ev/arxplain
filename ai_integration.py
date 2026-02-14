import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def init() -> OpenAI:
    api_key = os.getenv("GITHUB_AI_TOKEN")
    base_url = os.getenv("GITHUB_AI_ENDPOINT")
    model = os.getenv("GITHUB_AI_MODEL")

    return OpenAI(api_key=api_key, base_url=base_url)

def sanitize_json_response(raw_text: str) -> str:
    raw_text = raw_text.strip()

    # Remove triple backtick fenced blocks
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```json\s*", "", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"^```\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

    return raw_text.strip()

def call_llm(client: OpenAI, system_prompt: str, user_prompt: str, model: str) -> dict:
    response = client.chat.completions.create(
        messages=[
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_prompt }
        ],
        temperature=0,
        model=model
    )
    content = response.choices[0].message.content.strip()
    content = sanitize_json_response(content)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}\nRaw content: {content}")