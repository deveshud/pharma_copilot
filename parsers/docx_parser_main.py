from dataclasses import asdict
from pathlib import Path
import json

from docx_parser import DocxParser


def parse_docx_folder(folder_path: str, parser: DocxParser):
    folder = Path(folder_path)
    parsed_results = {}

    for file_path in sorted(folder.glob("*.docx")):
        blocks = parser.parse(str(file_path))
        parsed_results[file_path.name] = blocks

    return parsed_results


def combine_all_blocks(parsed_results):
    all_blocks = []

    for file_name, blocks in parsed_results.items():
        all_blocks.extend(blocks)

    return all_blocks


def save_results_to_json(parsed_results, output_file="parsed_output.json"):
    json_ready = {}

    for file_name, blocks in parsed_results.items():
        json_ready[file_name] = [asdict(block) for block in blocks]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = DocxParser(heading_threshold=3)

    folder_path = "data\\sample_docs\\"

    folder = Path(folder_path)

    print("Looking for DOCX files in:", folder.resolve())
    print("DOCX files found:")
    for file_path in sorted(folder.glob("*.docx")):
        print("-", file_path.name)

    parsed_results = parse_docx_folder(folder_path, parser)

    print("FILES PARSED:")
    for file_name, blocks in parsed_results.items():
        print(f"{file_name}: {len(blocks)} blocks")

    print("\nSAMPLE OUTPUT:")
    for file_name, blocks in parsed_results.items():
        print(f"\n--- {file_name} ---")
        for block in blocks[:3]:
            print(asdict(block))
        break

    all_blocks = combine_all_blocks(parsed_results)
    print(f"\nTOTAL BLOCKS ACROSS ALL FILES: {len(all_blocks)}")

    save_results_to_json(parsed_results, output_file="parsed_outputs//docx_parsed_output.json")
    print("\nParsed output also saved to docx_parsed_output.json")