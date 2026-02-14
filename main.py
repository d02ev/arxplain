import sys
import os
import json
from claim_extraction import extract_claims
from extractor import extract_pdf
from generate_report import generate_report
from review_report import review_report
from method_result_extraction import method_result_extraction
from outline import stage2_generate_outline
from outline_refinement import refine_outline

def main():
    if len(sys.argv) < 2:
        print("Usage: python extractor.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    output_dir = "output"
    output_s1_json_path = os.path.join(output_dir, "output_s1.json")
    output_s2_json_path = os.path.join(output_dir, "output_s2.json")
    output_s3_json_path = os.path.join(output_dir, "output_s3.json")
    output_s4_json_path = os.path.join(output_dir, "output_s4.json")
    output_s5_json_path = os.path.join(output_dir, "output_s5.json")
    output_s6_json_path = os.path.join(output_dir, "output_s6.json")
    output_report_md_path = os.path.join(output_dir, "explanation_report.md")


    print(f"Stage#01: PDF extraction started.")
    extracted_data = extract_pdf(pdf_path, output_dir=output_dir)

    os.makedirs(output_dir, exist_ok=True)

    with open(output_s1_json_path, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    print(f"Stage#01: PDF extraction completed.")

    print(f"Stage#02: Outline generation started.")
    stage2_generate_outline(output_s1_json_path, output_s2_json_path)
    print(f"Stage#02: Outline generation completed.")

    print(f"Stage#2.3: Outline refinement started.")
    refine_outline(output_s2_json_path, output_s2_json_path, "openai/gpt-4.1-mini")
    print(f"Stage#2.3: Outline refinement completed.")

    print(f"Stage#03: Claim extraction started.")
    extract_claims(output_s2_json_path, output_s3_json_path, "openai/gpt-4.1-mini")
    print(f"Stage#03: Claim extraction completed.")

    print(f"Stage#04: Method and result extraction started.")
    method_result_extraction(output_s3_json_path, output_s4_json_path, "openai/gpt-4.1-mini")
    print(f"Stage#04: Method and result extraction completed.")

    print(f"Stage#05: Explanation report generation started.")
    generate_report(output_s4_json_path, output_s5_json_path, output_report_md_path, "openai/gpt-4.1-mini")
    print(f"Stage#05: Explanation report generation completed.")

    print(f"Stage#06: Explanation report review started.")
    review_report(output_s5_json_path, output_s6_json_path, "openai/gpt-4.1-mini")
    print(f"Stage#06: Explanation report review completed.")


if __name__ == "__main__":
    main()