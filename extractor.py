import fitz
import json
import os
import re
import sys
from datetime import datetime


CAPTION_REGEX = re.compile(r"^(Figure|Fig\.|Table)\s+\d+[:\.]?\s+.*", re.IGNORECASE)


def extract_captions_from_text(page_text: str, page_number: int):
    captions = []
    lines = [line.strip() for line in page_text.split("\n") if line.strip()]

    caption_id_counter = 1

    for line in lines:
        if CAPTION_REGEX.match(line):
            cap_type = "figure" if line.lower().startswith(("figure", "fig")) else "table"

            captions.append({
                "caption_id": f"CAP_P{page_number}_{caption_id_counter}",
                "page_number": page_number,
                "type": cap_type,
                "text": line
            })
            caption_id_counter += 1

    return captions


def extract_pdf(pdf_path: str, output_dir: str = "output"):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)
    assets_dir = os.path.join(output_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    doc = fitz.open(pdf_path)

    pages_data = []
    all_text_parts = []
    figures_data = []
    captions_data = []

    figure_counter = 1

    for page_index in range(doc.page_count):
        page_number = page_index + 1
        page = doc.load_page(page_index)

        text = page.get_text("text") or ""
        text = text.strip()

        char_count = len(text)
        word_count = len(text.split()) if text else 0

        pages_data.append({
            "page_id": f"P{page_number}",
            "page_number": page_number,
            "text": text,
            "char_count": char_count,
            "word_count": word_count
        })

        all_text_parts.append(text)

        # Extract captions from this page
        captions_data.extend(extract_captions_from_text(text, page_number))

        # Extract embedded images
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)

            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_filename = f"fig_page{page_number}_{figure_counter}.{image_ext}"
            image_path = os.path.join(assets_dir, image_filename)

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            figures_data.append({
                "figure_id": f"FIG{figure_counter}",
                "page_number": page_number,
                "image_path": f"assets/{image_filename}",
                "width": base_image.get("width"),
                "height": base_image.get("height")
            })

            figure_counter += 1

    # full_text = "\n\n".join([t for t in all_text_parts if t])

    # Extraction notes
    extraction_notes = {
        "has_images": len(figures_data) > 0,
        "has_captions": len(captions_data) > 0,
        "warnings": []
    }

    # if not full_text.strip():
    #     extraction_notes["warnings"].append("No extractable text found in PDF.")

    result = {
        "schema_version": "paper-extract-v1",
        "source": {
            "file_name": os.path.basename(pdf_path),
            "file_path": pdf_path,
            "file_type": "pdf",
            "page_count": doc.page_count,
            "extracted_at": datetime.utcnow().isoformat() + "Z"
        },
        "metadata": {
            "title_guess": None,
            "author_guess": None,
            "creation_date": None
        },
        "pages": pages_data,
        "figures": figures_data,
        "tables_raw": [],
        "captions": captions_data,
        "extraction_notes": extraction_notes
    }

    doc.close()
    return result
