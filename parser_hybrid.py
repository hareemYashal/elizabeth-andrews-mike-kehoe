"""Hybrid parser combining PyMuPDF (structured JSON) + LlamaParse (rich markdown).

This approach:
1. Extracts structured tables using PyMuPDF (reliable, deterministic)
2. Extracts full context using LlamaParse (rich formatting)
3. Combines both for Gemini to intelligently merge and fill gaps
4. Exports to Excel with complete, accurate data

Benefits:
- PyMuPDF provides the "skeleton" - reliable structure
- LlamaParse provides the "context" - rich descriptions and relationships
- Gemini intelligently merges both, using structure as anchor
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
import fitz  # PyMuPDF
from llama_parse import LlamaParse

# Import PyMuPDF extraction from parser_incremental
from parser_incremental import extract_text_from_pdf, parse_structured_text_tables, parse_position_based_tables, classify_tables_by_content

load_dotenv()

# LlamaParse configuration
LLAMA_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
if not LLAMA_API_KEY:
    raise ValueError("LLAMA_CLOUD_API_KEY not found. Add it to your .env file before running.")

LLAMA_PARSER = LlamaParse(
    result_type="markdown",
    premium_mode=True,
    page_separator="\n---PAGE_BREAK---\n"
)

TARGET_PDFS = [
    "ALPH072025.pdf",
    "AUGU062025.pdf",
    "COLU052025.pdf",
    "COLU062025.pdf",
    "MACO052025.pdf"
]

INPUT_DIR = Path("input_pdfs")
OUTPUT_DIR = Path("hybrid_output")
PYMUPDF_DIR = OUTPUT_DIR / "pymupdf_structured"
LLAMA_DIR = OUTPUT_DIR / "llama_markdown"
COMBINED_DIR = OUTPUT_DIR / "combined"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"


def ensure_output_dirs() -> None:
    """Create output directories."""
    for directory in (PYMUPDF_DIR, LLAMA_DIR, COMBINED_DIR, ANALYSIS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def resolve_target_pdfs() -> List[Path]:
    """Find all target PDFs."""
    pdf_paths: List[Path] = []
    for name in TARGET_PDFS:
        path = INPUT_DIR / name
        if not path.exists():
            print(f"⚠️  Skipping missing PDF: {name}")
            continue
        pdf_paths.append(path)
    if not pdf_paths:
        raise FileNotFoundError("No target PDFs were found.")
    return pdf_paths


def extract_with_pymupdf(pdf_path: Path) -> Dict:
    """Extract structured tables using PyMuPDF."""
    print(f"  📊 PyMuPDF: Extracting structured tables...")
    
    # Use functions from parser_incremental
    text_content, positioned_blocks = extract_text_from_pdf(pdf_path)
    
    # Parse tables
    tables = parse_structured_text_tables(text_content)
    position_tables = parse_position_based_tables(positioned_blocks)
    tables.extend(position_tables)
    
    # Classify
    structured_tables = classify_tables_by_content(tables, text_content)
    
    # Calculate stats
    total_tables = sum(len(tables) for tables in structured_tables.values())
    total_rows = sum(len(row) for tables in structured_tables.values() for table in tables for row in table.get("rows", []))
    
    print(f"     ✅ Found {total_tables} tables ({total_rows} rows)")
    print(f"        FEES: {len(structured_tables['FEES'])}, INTERCHANGE: {len(structured_tables['INTERCHANGE'])}, SUMMARY_CARD: {len(structured_tables['SUMMARY_CARD'])}")
    
    return {
        "structured_tables": structured_tables,
        "raw_text": text_content,
        "stats": {
            "total_tables": total_tables,
            "total_rows": total_rows,
            "by_section": {k: len(v) for k, v in structured_tables.items()}
        }
    }


def extract_with_llamaparse(pdf_path: Path) -> Dict:
    """Extract rich markdown using LlamaParse."""
    print(f"  📝 LlamaParse: Extracting rich context...")
    
    with pdf_path.open("rb") as file_handle:
        documents = LLAMA_PARSER.load_data(file_handle, extra_info={"file_name": pdf_path.name})
    
    # Concatenate markdown
    markdown_parts = []
    for doc in documents:
        text = getattr(doc, "text", "") or ""
        markdown_parts.append(text.rstrip())
    
    full_markdown = "\n\n".join(markdown_parts).strip() + "\n"
    
    print(f"     ✅ Extracted {len(documents)} document chunks ({len(full_markdown)} chars)")
    
    return {
        "markdown": full_markdown,
        "chunks": len(documents),
        "length": len(full_markdown)
    }


def combine_results(pymupdf_data: Dict, llama_data: Dict, pdf_name: str) -> Dict:
    """Combine PyMuPDF structured data with LlamaParse markdown."""
    print(f"  🔗 Combining results...")
    
    combined = {
        "pdf_name": pdf_name,
        "extraction_method": "hybrid",
        
        # Structured data from PyMuPDF (the "anchor")
        "structured_data": {
            "fees": pymupdf_data["structured_tables"]["FEES"],
            "interchange": pymupdf_data["structured_tables"]["INTERCHANGE"],
            "summary_by_card": pymupdf_data["structured_tables"]["SUMMARY_CARD"],
            "other": pymupdf_data["structured_tables"]["OTHER"]
        },
        
        # Rich context from LlamaParse (for filling gaps)
        "rich_context": {
            "markdown": llama_data["markdown"],
            "length": llama_data["length"]
        },
        
        # Metadata
        "metadata": {
            "pymupdf_stats": pymupdf_data["stats"],
            "llama_chunks": llama_data["chunks"]
        },
        
        # Instructions for LLM
        "llm_instructions": """
EXTRACTION STRATEGY:
1. Use 'structured_data' as your PRIMARY source - this is reliable and deterministic
2. Use 'rich_context.markdown' to:
   - Fill in missing descriptions
   - Understand relationships between rows
   - Extract additional context (dates, merchant info, etc.)
   - Clarify ambiguous values
3. Cross-reference both sources to ensure accuracy
4. If there's a conflict, trust the structured_data for numbers, but use markdown for descriptions

PRIORITY ORDER:
- Numbers/amounts: structured_data first
- Descriptions/text: markdown first (better formatting)
- Counts: structured_data first
- Context: markdown only
"""
    }
    
    print(f"     ✅ Combined data ready for LLM")
    return combined


def analyze_coverage(combined_data: Dict) -> Dict:
    """Analyze how well each parser covered the document."""
    structured = combined_data["structured_data"]
    
    # Count structured elements
    fees_rows = sum(len(table.get("rows", [])) for table in structured["fees"])
    interchange_rows = sum(len(table.get("rows", [])) for table in structured["interchange"])
    card_rows = sum(len(table.get("rows", [])) for table in structured["summary_by_card"])
    
    # Analyze markdown coverage
    markdown = combined_data["rich_context"]["markdown"]
    has_fees_section = "FEES" in markdown or "Fee" in markdown
    has_interchange_section = "INTERCHANGE" in markdown or "Interchange" in markdown
    has_card_section = "Card Type" in markdown or "CARD TYPE" in markdown
    
    analysis = {
        "structured_coverage": {
            "fees_rows": fees_rows,
            "interchange_rows": interchange_rows,
            "card_summary_rows": card_rows,
            "total_structured_rows": fees_rows + interchange_rows + card_rows
        },
        "markdown_coverage": {
            "has_fees_section": has_fees_section,
            "has_interchange_section": has_interchange_section,
            "has_card_section": has_card_section,
            "total_length": len(markdown)
        },
        "recommendations": []
    }
    
    # Generate recommendations
    if fees_rows == 0:
        analysis["recommendations"].append("⚠️  No structured FEES rows - rely heavily on markdown")
    if card_rows == 0:
        analysis["recommendations"].append("⚠️  No structured CARD SUMMARY - rely heavily on markdown")
    if not has_fees_section:
        analysis["recommendations"].append("⚠️  Markdown missing FEES section - use structured data only")
    
    if not analysis["recommendations"]:
        analysis["recommendations"].append("✅ Both parsers have good coverage - hybrid approach optimal")
    
    return analysis


def process_pdf_hybrid(pdf_path: Path) -> None:
    """Process a single PDF using hybrid approach."""
    print(f"\n{'='*70}")
    print(f"Processing {pdf_path.name}")
    print(f"{'='*70}")
    
    # Stage 1: PyMuPDF extraction
    pymupdf_data = extract_with_pymupdf(pdf_path)
    
    # Stage 2: LlamaParse extraction
    llama_data = extract_with_llamaparse(pdf_path)
    
    # Stage 3: Combine results
    combined_data = combine_results(pymupdf_data, llama_data, pdf_path.name)
    
    # Stage 4: Analyze coverage
    analysis = analyze_coverage(combined_data)
    
    # Save outputs
    stem = pdf_path.stem
    
    pymupdf_path = PYMUPDF_DIR / f"{stem}.structured.json"
    llama_path = LLAMA_DIR / f"{stem}.markdown.md"
    combined_path = COMBINED_DIR / f"{stem}.combined.json"
    analysis_path = ANALYSIS_DIR / f"{stem}.analysis.json"
    
    pymupdf_path.write_text(json.dumps(pymupdf_data["structured_tables"], indent=2), encoding="utf-8")
    llama_path.write_text(llama_data["markdown"], encoding="utf-8")
    combined_path.write_text(json.dumps(combined_data, indent=2), encoding="utf-8")
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    
    print(f"\n  💾 Files saved:")
    print(f"     PyMuPDF:  {pymupdf_path}")
    print(f"     LlamaParse: {llama_path}")
    print(f"     Combined: {combined_path}")
    print(f"     Analysis: {analysis_path}")
    
    print(f"\n  📈 Analysis:")
    for rec in analysis["recommendations"]:
        print(f"     {rec}")


def main() -> None:
    """Main entry point."""
    ensure_output_dirs()
    pdf_paths = resolve_target_pdfs()
    
    print(f"🚀 Hybrid Parser: PyMuPDF + LlamaParse")
    print(f"Processing {len(pdf_paths)} PDF(s)...\n")
    
    for pdf_path in pdf_paths:
        process_pdf_hybrid(pdf_path)
    
    print(f"\n{'='*70}")
    print("🎉 Hybrid extraction complete!")
    print(f"{'='*70}")
    print("\n📂 Output structure:")
    print(f"   {OUTPUT_DIR}/")
    print(f"   ├── pymupdf_structured/  - Reliable JSON tables")
    print(f"   ├── llama_markdown/      - Rich markdown context")
    print(f"   ├── combined/            - Merged data for Gemini")
    print(f"   └── analysis/            - Coverage analysis")
    print("\n💡 Next step: Feed 'combined/*.json' files to Gemini for intelligent extraction!")


if __name__ == "__main__":
    main()

