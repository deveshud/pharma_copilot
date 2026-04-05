from __future__ import annotations

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parsers.ppt_parser import PptParser, build_slide_level_text, parse_ppt_folder, save_results_to_json


if __name__ == "__main__":
    folder = PROJECT_ROOT / "data" / "sample_docs"
    print(f"Looking for PPTX files in: {folder.resolve()}")

    if not folder.exists():
        print(f"Folder not found: {folder.resolve()}")
    else:
        parsed_results = parse_ppt_folder(
            str(folder),
            PptParser(include_notes=True, include_images=True),
        )
        for file_name, blocks in parsed_results.items():
            print(f"{file_name}: {len(blocks)} blocks")

        save_results_to_json(parsed_results, "parsed_outputs\\normalized_ppt_blocks.json")
        slide_level_text = build_slide_level_text(parsed_results)
        with open(PROJECT_ROOT / "parsed_outputs" / "normalized_ppt_slides.json", "w", encoding="utf-8") as f:
            json.dump(slide_level_text, f, indent=2, ensure_ascii=False)

        print("Saved: normalized_ppt_blocks.json")
        print("Saved: normalized_ppt_slides.json")
