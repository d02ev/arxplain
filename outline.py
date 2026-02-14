import json
import os
import re
import sys
from typing import List, Dict, Any


# Common research paper section names (regex-friendly)
SECTION_PATTERNS = [
    r"^abstract$",
    r"^introduction$",
    r"^related work$",
    r"^background$",
    r"^method$",
    r"^methodology$",
    r"^approach$",
    r"^model$",
    r"^architecture$",
    r"^experiments$",
    r"^experimental setup$",
    r"^results$",
    r"^discussion$",
    r"^conclusion$",
    r"^limitations$",
    r"^future work$",
    r"^references$",
    r"^acknowledgements?$",
    r"^appendix$"
]

SECTION_REGEX = re.compile(
    r"^(\d+(\.\d+)*)?\s*(" + "|".join(SECTION_PATTERNS) + r")\s*$",
    re.IGNORECASE
)

ABSTRACT_HEADER_REGEX = re.compile(r"^abstract\s*$", re.IGNORECASE)


def normalize_heading(text: str) -> str:
    """Normalize heading text into consistent title-case form."""
    text = re.sub(r"^\d+(\.\d+)*\s*", "", text.strip())
    text = re.sub(r"\s+", " ", text)
    return text.title()


def extract_heading_candidates(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = []

    for page in pages:
        page_num = page["page_number"]
        lines = [ln.strip() for ln in page["text"].split("\n") if ln.strip()]

        for ln in lines:
            if SECTION_REGEX.match(ln):
                candidates.append({
                    "page": page_num,
                    "raw_heading": ln.strip(),
                    "normalized_heading": normalize_heading(ln),
                    "snippet": ln.strip()
                })

    # Deduplicate based on normalized heading + page
    seen = set()
    unique = []
    for c in candidates:
        key = (c["page"], c["normalized_heading"])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def guess_title(pages: List[Dict[str, Any]]) -> str | None:
    """
    Naive title guess:
    Take the first page, grab the first 15 non-empty lines,
    ignore lines that look like Abstract/Authors, return longest line.
    """
    if not pages:
        return None

    first_page_text = pages[0]["text"]
    lines = [ln.strip() for ln in first_page_text.split("\n") if ln.strip()]

    # Only consider first ~15 lines
    lines = lines[:15]

    blacklist = {"abstract", "arxiv", "proceedings", "conference", "journal"}
    filtered = []

    for ln in lines:
        low = ln.lower()
        if any(word in low for word in blacklist):
            continue
        if len(ln) < 8:
            continue
        if re.match(r"^\d+(\.\d+)*\s+", ln):  # looks like numbered section heading
            continue
        filtered.append(ln)

    if not filtered:
        return None

    # Pick the longest line as title candidate
    return max(filtered, key=len)


def extract_abstract(pages: List[Dict[str, Any]]) -> str | None:
    """
    Rule-based abstract extraction:
    - find "Abstract" heading
    - capture subsequent lines until next section heading
    """
    full_lines = []

    # Flatten all pages into (page_number, line) tuples
    for page in pages:
        page_num = page["page_number"]
        lines = [ln.strip() for ln in page["text"].split("\n") if ln.strip()]
        for ln in lines:
            full_lines.append((page_num, ln))

    abstract_start_index = None

    for i, (_, line) in enumerate(full_lines):
        if ABSTRACT_HEADER_REGEX.match(line):
            abstract_start_index = i + 1
            break

    if abstract_start_index is None:
        return None

    abstract_lines = []

    for i in range(abstract_start_index, len(full_lines)):
        _, line = full_lines[i]

        # stop if next section starts
        if SECTION_REGEX.match(line) and not ABSTRACT_HEADER_REGEX.match(line):
            break

        abstract_lines.append(line)

    abstract_text = " ".join(abstract_lines).strip()
    abstract_text = re.sub(r"\s+", " ", abstract_text)

    return abstract_text if abstract_text else None


def build_sections_from_candidates(candidates: List[Dict[str, Any]], page_count: int):
    """
    Convert heading candidates into a section list with page ranges.
    Example:
    Introduction starts at page 1, ends at page 2 (until next section starts).
    """
    if not candidates:
        return []

    # Sort by page ascending
    candidates = sorted(candidates, key=lambda x: x["page"])

    sections = []
    for idx, c in enumerate(candidates):
        start_page = c["page"]

        if idx < len(candidates) - 1:
            next_page = candidates[idx + 1]["page"]
            end_page = max(start_page, next_page - 1)
        else:
            end_page = page_count

        sections.append({
            "section_id": f"S{idx + 1}",
            "name": c["normalized_heading"],
            "start_page": start_page,
            "end_page": end_page,
            "trace": {
                "page": c["page"],
                "snippet": c["snippet"]
            }
        })

    return sections


def stage2_generate_outline(input_json_path: str, output_json_path: str):
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages = data.get("pages", [])
    page_count = data.get("source", {}).get("page_count", len(pages))

    candidates = extract_heading_candidates(pages)
    title = guess_title(pages)
    abstract = extract_abstract(pages)

    sections = build_sections_from_candidates(candidates, page_count)

    outline = {
        "title": title,
        "authors": [],
        "abstract": abstract,
        "keywords": [],
        "section_candidates": candidates,  # useful debug info for later
        "sections": sections
    }

    data["outline"] = outline

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)