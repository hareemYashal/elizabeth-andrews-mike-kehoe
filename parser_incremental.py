"""Incremental parsing workflow for credit card statements using PyMuPDF.

This script uses PyMuPDF (fitz) for 100% local PDF parsing - no API calls needed!
It breaks the pipeline into explicit stages so we can inspect intermediate outputs:

1. Extract text and tables directly from PDF using PyMuPDF's built-in table detection
2. Classify tables by section (FEES, INTERCHANGE, SUMMARY_CARD) based on content
3. Analyse extraction quality and provide suggestions

Benefits of PyMuPDF:
- Runs completely locally (no API keys or rate limits)
- Fast and reliable
- Built-in table detection
- Already in your requirements.txt

Run this file directly to generate artefacts under `debug_json/`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

from dotenv import load_dotenv
import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Configuration & environment
# ---------------------------------------------------------------------------

load_dotenv()

# PyMuPDF doesn't require API keys - works locally

TARGET_PDFS = [
    "ALPH072025.pdf",
    "AUGU062025.pdf",
    "COLU052025.pdf",
    "COLU062025.pdf",
    "MACO052025.pdf",
]

INPUT_DIR = Path("input_pdfs")
OUTPUT_DIR = Path("debug_json")
RAW_JSON_DIR = OUTPUT_DIR / "raw"
STRUCTURED_DIR = OUTPUT_DIR / "structured"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"


# ---------------------------------------------------------------------------
# Stage 1: Parse PDFs -> JSON artefacts
# ---------------------------------------------------------------------------


def ensure_output_dirs() -> None:
    for directory in (RAW_JSON_DIR, STRUCTURED_DIR, ANALYSIS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def resolve_target_pdfs() -> List[Path]:
    pdf_paths: List[Path] = []
    for name in TARGET_PDFS:
        path = INPUT_DIR / name
        if not path.exists():
            print(f"⚠️  Skipping missing PDF: {name}")
            continue
        pdf_paths.append(path)

    if not pdf_paths:
        raise FileNotFoundError(
            "No target PDFs were found. Make sure the files are in input_pdfs/."
        )
    return pdf_paths


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, List[Dict]]:
    """Extract both plain text and structured blocks with positions from PDF."""
    print(f"➡️  Parsing {pdf_path.name} with PyMuPDF...")
    
    doc = fitz.open(pdf_path)
    text_parts = []
    all_blocks = []
    page_count = len(doc)
    
    for page_num in range(page_count):
        page = doc[page_num]
        
        # Get plain text
        text = page.get_text("text")
        text_parts.append(f"# PAGE {page_num + 1}\n\n{text}")
        
        # Get text blocks with positions (x, y coordinates)
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    
                    if line_text.strip():
                        all_blocks.append({
                            "page": page_num + 1,
                            "text": line_text.strip(),
                            "bbox": line.get("bbox", (0, 0, 0, 0)),  # (x0, y0, x1, y1)
                            "y": line.get("bbox", (0, 0, 0, 0))[1]  # y-coordinate for sorting
                        })
    
    doc.close()
    full_text = "\n\n---PAGE_BREAK---\n\n".join(text_parts)
    
    print(f"   ✅ Extracted text from {page_count} pages ({len(all_blocks)} positioned blocks)")
    return full_text, all_blocks


def parse_structured_text_tables(text: str) -> List[Dict]:
    """Parse tables from structured text by detecting patterns."""
    import re
    
    all_tables = []
    pages = text.split("---PAGE_BREAK---")
    
    for page_num, page_text in enumerate(pages, 1):
        lines = [line.strip() for line in page_text.split("\n") if line.strip()]
        
        # Look for table-like patterns (rows with consistent structure)
        current_table = None
        table_lines = []
        
        for idx, line in enumerate(lines):
            # Skip headers/titles (all caps, short lines)
            if len(line) < 5 or line.isupper() and len(line) < 50:
                if table_lines and current_table:
                    # End of table
                    parsed = parse_table_from_lines(table_lines, current_table, page_num)
                    if parsed and len(parsed.get("rows", [])) > 0:
                        all_tables.append(parsed)
                    table_lines = []
                    current_table = None
                continue
            
            # Detect table starts by looking for common headers
            if any(keyword in line for keyword in [
                "Card Type", "Average Ticket", "Items", "Gross Sales",
                "Product/Description", "Sales Total", "Number of Transactions",
                "Description", "Type", "Amount", "Fee", "Charge",
                "Date", "Submitted"
            ]):
                if table_lines and current_table:
                    # Save previous table
                    parsed = parse_table_from_lines(table_lines, current_table, page_num)
                    if parsed and len(parsed.get("rows", [])) > 0:
                        all_tables.append(parsed)
                
                current_table = line
                table_lines = []
            elif current_table and line:
                # Check if line looks like data (contains numbers, currencies)
                if re.search(r'[\d$,\.\-]+', line):
                    table_lines.append(line)
        
        # Don't forget the last table on the page
        if table_lines and current_table:
            parsed = parse_table_from_lines(table_lines, current_table, page_num)
            if parsed and len(parsed.get("rows", [])) > 0:
                all_tables.append(parsed)
    
    print(f"   ✅ Parsed {len(all_tables)} tables from text structure")
    return all_tables


def parse_position_based_tables(positioned_blocks: List[Dict]) -> List[Dict]:
    """Parse tables using positional information (for multi-column layouts)."""
    tables = []
    
    # Find "Card Type" in all blocks (don't group by page yet)
    for idx, block in enumerate(positioned_blocks):
        if "Card Type" == block["text"].strip() and idx + 1 < len(positioned_blocks):
            # Check if next few lines look like table headers
            next_lines = positioned_blocks[idx+1:idx+20]  # Look ahead 20 lines
            
            # Detect if this is the "Summary by Card Type" table
            if any("Average" in b["text"] for b in next_lines[:5]):
                print(f"   🔍 Found Card Type table on page {block['page']} at block {idx}")
                table_data = extract_card_type_table(positioned_blocks, idx)
                if table_data and table_data.get("rows"):
                    print(f"   ✅ Extracted {len(table_data['rows'])} card type rows")
                    tables.append(table_data)
                else:
                    print(f"   ⚠️  Card Type table extraction returned no rows")
    
    return tables


def extract_card_type_table(blocks: List[Dict], start_idx: int) -> Dict:
    """Extract the 'Summary by Card Type' table using position-based parsing."""
    import re
    
    # The table has this structure (from the PDF):
    # Card Type | Average Ticket | Items | Amount (Gross Sales) | Items | Amount (Refunds) | Items | Amount (Total)
    # Each value is on its own line in the raw text
    
    page_num = blocks[start_idx]["page"]
    
    # Collect lines after "Card Type" header
    data_lines = []
    for i in range(start_idx + 1, min(start_idx + 60, len(blocks))):
        text = blocks[i]["text"].strip()
        # Stop at section breaks
        if any(stop_word in text for stop_word in ["SUMMARY BY DAY", "FEES", "INTERCHANGE", "Page 1 of", "Page 2 of"]):
            break
        if text:
            data_lines.append(text)
    
    # Card names to look for
    card_names = ["Mastercard", "VISA", "Discover", "American Express", "Debit/Atm", "Total"]
    rows = []
    
    i = 0
    while i < len(data_lines):
        line = data_lines[i]
        
        # Skip column header lines
        if line in ["Average", "Ticket", "Items", "Amount", "Total Gross Sales You Submitted", 
                    "Refunds", "Total Amount You Submitted"]:
            i += 1
            continue
        
        # Check if this line is a card name
        matched_card = None
        for card in card_names:
            if card.lower() in line.lower():
                matched_card = line
                break
        
        if matched_card:
            # Extract next 7 values (avg ticket, items, amount, items, amount, items, amount)
            values = []
            j = i + 1
            while j < len(data_lines) and len(values) < 7:
                next_line = data_lines[j].strip()
                
                # Stop if we hit another card name
                if any(card.lower() in next_line.lower() for card in card_names):
                    break
                
                # Skip header lines
                if next_line not in ["Average", "Ticket", "Items", "Amount"]:
                    values.append(next_line)
                
                j += 1
            
            if len(values) >= 7:
                rows.append({
                    "Card Type": matched_card,
                    "Average Ticket": values[0],
                    "Gross Sales Items": values[1],
                    "Gross Sales Amount": values[2],
                    "Refunds Items": values[3],
                    "Refunds Amount": values[4],
                    "Total Items": values[5],
                    "Total Amount": values[6]
                })
            
            i = j
        else:
            i += 1
    
    if rows:
        return {
            "page": page_num,
            "headers": ["Card Type", "Average Ticket", "Gross Sales Items", "Gross Sales Amount", 
                       "Refunds Items", "Refunds Amount", "Total Items", "Total Amount"],
            "rows": rows,
            "header_hint": "Summary by Card Type"
        }
    
    return {}


def parse_table_from_lines(data_lines: List[str], header_hint: str, page_num: int) -> Dict:
    """Convert a list of data lines into a structured table."""
    import re
    
    # Detect if this is a multi-column table by looking for multiple values per line
    rows = []
    
    for line in data_lines:
        # Split by multiple spaces or tabs
        cells = re.split(r'\s{2,}|\t', line)
        cells = [cell.strip() for cell in cells if cell.strip()]
        
        if cells:
            rows.append(cells)
    
    if not rows:
        return {}
    
    # Try to infer headers from the most common column count
    from collections import Counter
    col_counts = Counter(len(row) for row in rows)
    most_common_cols = col_counts.most_common(1)[0][0] if col_counts else 0
    
    # Generate generic headers
    headers = [f"Column_{i+1}" for i in range(most_common_cols)]
    
    # Convert rows to dictionaries
    row_dicts = []
    for row in rows:
        if len(row) == len(headers):
            row_dict = dict(zip(headers, row))
            row_dicts.append(row_dict)
        elif len(row) > 0:
            # Handle variable length rows
            row_dict = {}
            for idx, cell in enumerate(row):
                if idx < len(headers):
                    row_dict[headers[idx]] = cell
            row_dicts.append(row_dict)
    
    return {
        "page": page_num,
        "headers": headers,
        "rows": row_dicts,
        "header_hint": header_hint
    }




def classify_tables_by_content(tables: List[Dict], text_content: str) -> Dict[str, List[Dict]]:
    """Classify tables by analyzing their content and header hints."""
    structured = {
        "FEES": [],
        "INTERCHANGE": [],
        "SUMMARY_CARD": [],
        "OTHER": []
    }
    
    for table in tables:
        # Get header hint and sample data
        header_hint = table.get("header_hint", "").upper()
        
        # Check first few rows for additional context
        sample_data = ""
        for row in table.get("rows", [])[:3]:
            sample_data += " ".join(str(v) for v in row.values()).upper()
        
        combined_text = header_hint + " " + sample_data
        
        # Classify based on keywords with priority order
        section = "OTHER"
        
        # Check for Summary by Card Type (highest priority for avg ticket + items)
        if any(kw in combined_text for kw in ["CARD TYPE", "AVERAGE TICKET", "GROSS SALES"]):
            if "MASTERCARD" in combined_text or "VISA" in combined_text or "DISCOVER" in combined_text:
                section = "SUMMARY_CARD"
        
        # Check for Interchange
        elif any(kw in combined_text for kw in ["INTERCHANGE", "PROGRAM FEE", "SALES TOTAL", "PRODUCT/DESCRIPTION"]):
            if "INTERCHANGE" in combined_text or "SALES TOTAL" in header_hint:
                section = "INTERCHANGE"
        
        # Check for Fees (broad category)
        elif any(kw in combined_text for kw in ["FEES", "FEE", "SERVICE CHARGE", "DISCOUNT", "TRANSACTION FEE", "ASSESSMENT"]):
            if "INTERCHANGE" not in combined_text:
                section = "FEES"
        
        # Special handling for tables with "Description" + "Amount"
        elif "DESCRIPTION" in header_hint and "AMOUNT" in header_hint:
            section = "FEES"
        
        # Add classification metadata
        table["classified_as"] = section
        table["header_hint_used"] = header_hint[:100]
        
        structured[section].append(table)
    
    return structured


# ---------------------------------------------------------------------------
# Stage 2: JSON analysis & suggestions
# ---------------------------------------------------------------------------


def analyse_table_structure(structured_tables: Dict[str, List[Dict]]) -> Dict[str, object]:
    """Analyse extracted table quality and provide suggestions."""
    
    total_tables = sum(len(tables) for tables in structured_tables.values())
    section_counts = {section: len(tables) for section, tables in structured_tables.items()}
    
    # Count total rows across all tables
    total_rows = 0
    for tables in structured_tables.values():
        for table in tables:
            total_rows += len(table.get("rows", []))
    
    suggestions: List[str] = []
    
    # Check for empty sections
    if not structured_tables.get("FEES"):
        suggestions.append("⚠️  No FEES tables detected - check classification rules")
    if not structured_tables.get("INTERCHANGE"):
        suggestions.append("⚠️  No INTERCHANGE tables detected - check classification rules")
    if not structured_tables.get("SUMMARY_CARD"):
        suggestions.append("⚠️  No SUMMARY BY CARD TYPE tables detected - check classification rules")
    
    # Check table completeness
    for section, tables in structured_tables.items():
        for idx, table in enumerate(tables):
            if isinstance(table, dict):
                if not table.get("headers"):
                    suggestions.append(
                        f"Table {idx+1} in {section} has no headers"
                    )
                if not table.get("rows"):
                    suggestions.append(
                        f"Table {idx+1} in {section} has no data rows"
                    )
                
                # Check for mismatched column counts
                headers = table.get("headers", [])
                rows = table.get("rows", [])
                for row_idx, row in enumerate(rows):
                    if len(row) != len(headers):
                        suggestions.append(
                            f"Table {idx+1} in {section}, row {row_idx+1}: "
                            f"{len(row)} cells vs {len(headers)} headers"
                        )
                        break  # Only report first mismatch per table
    
    if not suggestions:
        suggestions.append("✅ All tables extracted successfully")
    
    return {
        "total_tables": total_tables,
        "total_rows": total_rows,
        "section_counts": section_counts,
        "suggestions": suggestions,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def process_pdf(pdf_path: Path) -> None:
    # Extract text and positioned blocks
    text_content, positioned_blocks = extract_text_from_pdf(pdf_path)
    
    # Parse tables from text structure
    tables = parse_structured_text_tables(text_content)
    
    # Parse position-based tables (like Card Type summary)
    position_tables = parse_position_based_tables(positioned_blocks)
    tables.extend(position_tables)
    
    # Classify tables by content
    structured_tables = classify_tables_by_content(tables, text_content)
    
    # Analyze results
    analysis = analyse_table_structure(structured_tables)

    raw_path = RAW_JSON_DIR / f"{pdf_path.stem}.raw.txt"
    structured_path = STRUCTURED_DIR / f"{pdf_path.stem}.structured.json"
    analysis_path = ANALYSIS_DIR / f"{pdf_path.stem}.analysis.json"

    raw_path.write_text(text_content, encoding="utf-8")
    structured_path.write_text(json.dumps(structured_tables, indent=2), encoding="utf-8")
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    print(f"   📄 Raw text saved to         {raw_path}")
    print(f"   🗂️  Structured tables saved to {structured_path}")
    print(f"   🧭 Analysis saved to         {analysis_path}")
    print(f"   📊 Found {analysis['total_tables']} tables ({analysis['total_rows']} rows): {analysis['section_counts']}")
    print("   Suggestions:")
    for suggestion in analysis["suggestions"]:
        print(f"     - {suggestion}")


def main() -> None:
    ensure_output_dirs()
    pdf_paths = resolve_target_pdfs()

    print(f"Processing {len(pdf_paths)} PDF(s) incrementally…")
    for pdf_path in pdf_paths:
        print("\n" + "=" * 70)
        process_pdf(pdf_path)

    print("\n🎉 Stage 1 (PyMuPDF extraction) & Stage 2 (analysis) complete.")
    print("📂 Outputs saved to debug_json/:")
    print("   - raw/*.txt - Raw text extracted from PDF")
    print("   - structured/*.json - Tables organized by section with all data")
    print("   - analysis/*.json - Quality metrics and suggestions")
    print("\n💡 PyMuPDF runs 100% locally - no API calls, no rate limits!")


if __name__ == "__main__":
    main()

