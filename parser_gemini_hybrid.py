"""Gemini-powered processor for hybrid parser output.

Takes combined JSON (PyMuPDF structure + LlamaParse markdown) and uses Gemini
to intelligently extract and merge data into Excel format.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

INPUT_DIR = Path("hybrid_output/combined")
OUTPUT_DIR = Path("hybrid_output/excel_output")


def ensure_output_dir():
    """Create output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_fees_with_gemini(combined_data: Dict) -> List[Dict]:
    """Use Gemini to extract FEES data from hybrid sources."""
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a financial data extraction expert. You will receive:
1. STRUCTURED DATA: Reliable JSON tables from PyMuPDF (use for numbers/structure)
2. MARKDOWN CONTEXT: Rich text from LlamaParse (use for descriptions/context)

Your task: Extract ALL FEE entries with complete information.

RULES:
- Use structured_data as the PRIMARY source for amounts and counts
- Use markdown to fill in missing descriptions and understand context
- Return JSON array with these fields for each fee:
  * "description": Fee name/description (prefer markdown for clarity)
  * "fee_type": Type of fee (Authorization, Settlement, Monthly, etc.)
  * "count": Number of occurrences (from structured data)
  * "rate_percent": Percentage rate if applicable
  * "rate_dollar": Dollar rate if applicable  
  * "amount": Total fee amount (from structured data)
  * "source": Which data source you used (structured/markdown/both)

IMPORTANT: Extract ALL fees you can find, not just those in structured_data.fees.
Look in the markdown for additional fee details that might be missing from structured tables.

Return ONLY valid JSON array, no markdown formatting."""),
        ("user", """STRUCTURED DATA:
{structured_fees}

MARKDOWN CONTEXT (search for fees here too):
{markdown_snippet}

Extract all fees as JSON array:""")
    ])
    
    # Prepare data
    structured_fees = json.dumps(combined_data["structured_data"]["fees"], indent=2)
    markdown = combined_data["rich_context"]["markdown"]
    
    # Take relevant markdown snippet (fees section)
    markdown_snippet = ""
    if "FEES" in markdown:
        start = markdown.find("FEES")
        markdown_snippet = markdown[max(0, start-500):min(len(markdown), start+3000)]
    else:
        markdown_snippet = markdown[:2000]  # First 2000 chars
    
    chain = prompt_template | llm
    response = chain.invoke({
        "structured_fees": structured_fees,
        "markdown_snippet": markdown_snippet
    })
    
    # Parse response
    response_text = response.content
    # Remove markdown code blocks if present
    response_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
    
    try:
        fees_data = json.loads(response_text)
        print(f"     ✅ Extracted {len(fees_data)} fee entries")
        return fees_data
    except json.JSONDecodeError as e:
        print(f"     ❌ Failed to parse Gemini response: {e}")
        print(f"     Response: {response_text[:200]}...")
        return []


def extract_interchange_with_gemini(combined_data: Dict) -> List[Dict]:
    """Use Gemini to extract INTERCHANGE data."""
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """Extract ALL INTERCHANGE/PROGRAM FEE entries.

Return JSON array with:
- "product_description": Product/service description
- "sales_total": Total sales amount  
- "number_of_transactions": Transaction count
- "interchange_amount": Interchange fee amount
- "source": Data source used

Use structured data for numbers, markdown for descriptions."""),
        ("user", """STRUCTURED DATA:
{structured_interchange}

MARKDOWN CONTEXT:
{markdown_snippet}

Extract all interchange entries as JSON array:""")
    ])
    
    structured_interchange = json.dumps(combined_data["structured_data"]["interchange"], indent=2)
    markdown = combined_data["rich_context"]["markdown"]
    
    # Get interchange section
    markdown_snippet = ""
    if "INTERCHANGE" in markdown:
        start = markdown.find("INTERCHANGE")
        markdown_snippet = markdown[max(0, start-300):min(len(markdown), start+2500)]
    else:
        markdown_snippet = markdown[:2000]
    
    chain = prompt_template | llm
    response = chain.invoke({
        "structured_interchange": structured_interchange,
        "markdown_snippet": markdown_snippet
    })
    
    response_text = re.sub(r'```json\s*|\s*```', '', response.content).strip()
    
    try:
        interchange_data = json.loads(response_text)
        print(f"     ✅ Extracted {len(interchange_data)} interchange entries")
        return interchange_data
    except json.JSONDecodeError as e:
        print(f"     ❌ Failed to parse response: {e}")
        return []


def extract_card_summary_with_gemini(combined_data: Dict) -> List[Dict]:
    """Use Gemini to extract SUMMARY BY CARD TYPE data."""
    
    # For card summary, structured data is already very good
    # Just clean it up and ensure all fields are present
    
    card_data = combined_data["structured_data"]["summary_by_card"]
    
    if not card_data:
        print(f"     ⚠️  No structured card summary found")
        return []
    
    # Extract rows from first table (should only be one card summary table)
    rows = card_data[0].get("rows", [])
    
    cleaned_rows = []
    for row in rows:
        cleaned = {
            "card_type": row.get("Card Type", ""),
            "average_ticket": row.get("Average Ticket", ""),
            "gross_sales_items": row.get("Gross Sales Items", ""),
            "gross_sales_amount": row.get("Gross Sales Amount", ""),
            "refunds_items": row.get("Refunds Items", ""),
            "refunds_amount": row.get("Refunds Amount", ""),
            "total_items": row.get("Total Items", ""),
            "total_amount": row.get("Total Amount", "")
        }
        cleaned_rows.append(cleaned)
    
    print(f"     ✅ Extracted {len(cleaned_rows)} card types")
    return cleaned_rows


def parse_filename_metadata(filename: str) -> Dict:
    """Extract metadata from filename."""
    # Example: ALPH072025.pdf -> Location: ALPH, Month: 07, Year: 2025
    stem = Path(filename).stem
    
    # Try to extract location code (first 4 chars) and date (last 6 chars)
    if len(stem) >= 10:
        location = stem[:4]
        date_part = stem[-6:]  # MMYYYY
        try:
            month = date_part[:2]
            year = "20" + date_part[2:4] if len(date_part) >= 4 else ""
            return {
                "location_code": location,
                "month": month,
                "year": year,
                "statement_period": f"{month}/{year}"
            }
        except:
            pass
    
    return {
        "location_code": "",
        "month": "",
        "year": "",
        "statement_period": ""
    }


def process_combined_file(combined_path: Path) -> None:
    """Process a single combined JSON file and generate Excel."""
    print(f"\n{'='*70}")
    print(f"Processing {combined_path.name}")
    print(f"{'='*70}")
    
    # Load combined data
    with combined_path.open("r", encoding="utf-8") as f:
        combined_data = json.load(f)
    
    pdf_name = combined_data["pdf_name"]
    metadata = parse_filename_metadata(pdf_name)
    
    print(f"  📋 Metadata: {metadata['location_code']} - {metadata['statement_period']}")
    
    # Extract data using Gemini
    print(f"  🤖 Gemini: Extracting FEES...")
    fees = extract_fees_with_gemini(combined_data)
    
    print(f"  🤖 Gemini: Extracting INTERCHANGE...")
    interchange = extract_interchange_with_gemini(combined_data)
    
    print(f"  🤖 Gemini: Extracting CARD SUMMARY...")
    card_summary = extract_card_summary_with_gemini(combined_data)
    
    # Create DataFrames
    df_fees = pd.DataFrame(fees) if fees else pd.DataFrame()
    df_interchange = pd.DataFrame(interchange) if interchange else pd.DataFrame()
    df_card_summary = pd.DataFrame(card_summary) if card_summary else pd.DataFrame()
    
    # Add metadata to all sheets
    for df in [df_fees, df_interchange, df_card_summary]:
        if not df.empty:
            df.insert(0, "Location", metadata["location_code"])
            df.insert(1, "Month", metadata["month"])
            df.insert(2, "Year", metadata["year"])
    
    # Save to Excel
    output_path = OUTPUT_DIR / f"{Path(pdf_name).stem}_extracted.xlsx"
    
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        if not df_fees.empty:
            df_fees.to_excel(writer, sheet_name="FEES", index=False)
        if not df_interchange.empty:
            df_interchange.to_excel(writer, sheet_name="INTERCHANGE", index=False)
        if not df_card_summary.empty:
            df_card_summary.to_excel(writer, sheet_name="CARD_SUMMARY", index=False)
    
    print(f"\n  💾 Excel saved: {output_path}")
    print(f"     Sheets: FEES ({len(df_fees)} rows), INTERCHANGE ({len(df_interchange)} rows), CARD_SUMMARY ({len(df_card_summary)} rows)")


def main():
    """Main entry point."""
    ensure_output_dir()
    
    combined_files = list(INPUT_DIR.glob("*.json"))
    
    if not combined_files:
        print("❌ No combined JSON files found in hybrid_output/combined/")
        print("   Run parser_hybrid.py first!")
        return
    
    print(f"🤖 Gemini Hybrid Processor")
    print(f"Processing {len(combined_files)} file(s)...\n")
    
    for combined_path in combined_files:
        try:
            process_combined_file(combined_path)
        except Exception as e:
            print(f"\n❌ Error processing {combined_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("🎉 Excel generation complete!")
    print(f"{'='*70}")
    print(f"\n📂 Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

