from llama_parse import LlamaParse
import os
import re
from tabulate import tabulate
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
llama_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Validate that API keys are present
if not llama_api_key:
    raise ValueError("LLAMA_CLOUD_API_KEY not found in environment variables. Please check your .env file.")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")

# Set environment variables for the libraries
os.environ["LLAMA_CLOUD_API_KEY"] = llama_api_key
os.environ["GOOGLE_API_KEY"] = google_api_key

# Initialize LLM once for reuse across all functions
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# LLM usage toggles
USE_LLM_PRIMARY = True   # Use LLM as the primary extractor from markdown sections
USE_LLM_POST = False     # Disable post-processor; rely on LLM for clean output

# Initialize LlamaParse
parser = LlamaParse(result_type="markdown")

# Load the bank statement PDFs from input_pdfs
import glob

# Specific PDFs to process
target_pdfs = ["ALPH072025.pdf", "AUGU062025.pdf", "COLU052025.pdf", "COLU062025.pdf", "MACO052025.pdf"]

# Find all PDF files in input_pdfs directory
pdf_files = glob.glob(os.path.join("input_pdfs", "*.pdf"))

if not pdf_files:
    print("No PDF files found in input_pdfs directory.")
    print("Please add a PDF file to the input_pdfs directory and run the script again.")
    exit(1)

# Filter to only process the target PDFs
target_files = []
for target in target_pdfs:
    target_path = os.path.join("input_pdfs", target)
    if os.path.exists(target_path):
        target_files.append(target_path)
    else:
        print(f"⚠️ Warning: {target} not found in input_pdfs directory")

if not target_files:
    print("No target PDF files found.")
    exit(1)

print(f"Processing {len(target_files)} PDF files: {[os.path.basename(f) for f in target_files]}")

def parse_filename_for_metadata(pdf_path):
    filename = os.path.basename(pdf_path)
    base = os.path.splitext(filename)[0]
    # Try MMYYYY at end
    m = re.search(r"(\d{2})(\d{4})$", base)
    date_val = ""
    if m:
        mm, yyyy = m.groups()
        # Use the last day of the month instead of first day
        if mm in ['01', '03', '05', '07', '08', '10', '12']:
            last_day = '31'
        elif mm in ['04', '06', '09', '11']:
            last_day = '30'
        elif mm == '02':
            # Simple leap year check
            year = int(yyyy)
            if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                last_day = '29'
            else:
                last_day = '28'
        else:
            last_day = '30'  # fallback
        date_val = f"{mm}/{last_day}/{yyyy}"
    return {
        "Account Name": base.upper(),
        "Account ID": "",
        "Date": date_val,
        "Account Holder": "",
        "Filename": filename,
        "Account Short ID": "",
        "Gateway": "",
        "Account Type": "",
        "Processor": "",
        "User Permissions": ""
    }


def process_single_pdf(documents, input_pdf_name, file_path):
    """Process a single PDF and save results to Excel."""

    metadata = parse_filename_for_metadata(file_path)

    print("🔄 Extracting account information from document header...")
    account_info = extract_account_info_from_header(documents[0].text)
    metadata.update(account_info)

    statement_text = documents[0].text
    if "Statement Period:" in statement_text:
        import re

        period_match = re.search(
            r"Statement Period:\s*(\d{2}/\d{2}/\d{2})\s*-\s*(\d{2}/\d{2}/\d{2})",
            statement_text,
        )
        if period_match:
            _, end_date = period_match.groups()
            month, day, year = end_date.split("/")
            year = f"20{year}" if int(year) < 50 else f"19{year}"
            metadata["Date"] = f"{month}/{day}/{year}"
            print(f"   ✅ Updated Date from statement period: {metadata['Date']}")

    print(f"   ✅ Account Name: {metadata.get('Account Name', 'Unknown')}")
    print(f"   ✅ Account Holder: {metadata.get('Account Holder', 'Unknown')}")
    print(f"   ✅ Date: {metadata.get('Date', 'Unknown')}")

    sections_markdown, statement_data = build_statement_from_documents(documents)

    print("\n🔍 DEBUGGING: Looking for Summary by Card Type in raw markdown...")
    found_summary = False
    for doc in documents:
        text = doc.text
        if "SUMMARY BY CARD TYPE" not in text:
            continue

        found_summary = True
        start = text.find("# SUMMARY BY CARD TYPE")
        if start == -1:
            continue

        end = text.find("\n# ", start + 1)
        if end == -1:
            end = len(text)

        summary_section = text[start:end]
        print("📋 Found Summary by Card Type section:")
        print(summary_section[:1000])
        if len(summary_section) > 1000:
            print("...")
        break

    if not found_summary:
        print("❌ No 'SUMMARY BY CARD TYPE' found in any document")

    print("\n" + "=" * 50 + "\n")

    fees_rows = []
    interchange_rows = []
    summary_card_rows = []

    if sections_markdown.get("FEES"):
        fees_rows = process_section_with_llm("FEES", sections_markdown["FEES"], metadata)
        fees_rows = populate_missing_metadata(fees_rows, metadata)
        validate_extracted_data(fees_rows, "FEES")
        print("   🔍 FEES rows after processing:")
        for i, row in enumerate(fees_rows[:5]):
            print(
                f"     {i+1}. '{row.get('Statement Description', '')[:50]}...' | Fee Type: '{row.get('Fee Type', '')}'"
            )

    if sections_markdown.get("INTERCHANGE"):
        interchange_rows = process_section_with_llm("INTERCHANGE", sections_markdown["INTERCHANGE"], metadata)
        interchange_rows = populate_missing_metadata(interchange_rows, metadata)
        validate_extracted_data(interchange_rows, "INTERCHANGE")

    if sections_markdown.get("SUMMARY_CARD"):
        summary_card_rows = process_section_with_llm("SUMMARY_CARD", sections_markdown["SUMMARY_CARD"], metadata)
        summary_card_rows = populate_missing_metadata(summary_card_rows, metadata)
        validate_extracted_data(summary_card_rows, "SUMMARY_CARD")

    print(
        "\n📊 SECTION PROCESSING COMPLETE:\n"
        f"   📥 FEES rows: {len(fees_rows)} | INTERCHANGE rows: {len(interchange_rows)} | "
        f"SUMMARY_CARD rows: {len(summary_card_rows)}"
    )

    all_rows = fees_rows + summary_card_rows
    print("\n🔄 Post-processing descriptions, fee types, and fee signs...")
    all_rows = post_process_rows_with_llm(all_rows, metadata)
    print(f"   ✅ Final row count (FEES + SUMMARY_CARD): {len(all_rows)}")

    output_excel = f"output_excels/{input_pdf_name.replace('.pdf', '_tables.xlsx')}"
    save_extracted_data_to_excel(all_rows, output_excel)


def process_all_pdfs():
    for file_path in target_files:
        input_pdf_name = os.path.basename(file_path)

        print(f"\n{'='*60}")
        print(f"Processing PDF: {input_pdf_name}")
        print(f"{'='*60}")

        with open(file_path, "rb") as f:
            documents = parser.load_data(f, extra_info={"file_name": file_path})

        process_single_pdf(documents, input_pdf_name, file_path)

# Function to parse markdown tables
def parse_markdown_table(text):
    lines = text.split("\n")
    headers = []
    rows = []
    for line in lines:
        if line.strip().startswith("|") and not line.strip().startswith("|-"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not headers:
                headers = cells
            else:
                rows.append(cells)
    return headers, rows

# Initialize data structures for the statement
def _extract_section_block(full_text, header):
    try:
        start_idx = full_text.find(header)
        if start_idx == -1:
            return ""
        rest = full_text[start_idx:]
        # Cut at next header starting with "# " (level-1) or "## " (level-2)
        import re as _re
        m = _re.search(r"\n#\s|\n##\s", rest[1:])
        if m:
            end = 1 + m.start()
            return rest[:end]
        return rest
    except Exception:
        return ""

def _collect_sections_markdown(full_text):
    """Collect FEES/INTERCHANGE/SUMMARY sections from a markdown document."""

    import re as _re

    sections = {"FEES": [], "INTERCHANGE": [], "SUMMARY_CARD": []}

    headers = [
        (match.group(0), match.start())
        for match in _re.finditer(r"^#{1,3}\s.*$", full_text, _re.MULTILINE)
    ]
    headers.append(("__END__", len(full_text)))
    
    print(f"   🔍 Found headers: {[h[0] for h in headers[:10]]}")

    for i in range(len(headers) - 1):
        header_line, start = headers[i]
        _, end = headers[i + 1]
        block = full_text[start:end]
        header_upper = header_line.upper()

        if any(keyword in header_upper for keyword in ["INTERCHANGE", "PROGRAM FEE"]):
            sections["INTERCHANGE"].append(block)
            print(f"   ✅ Matched INTERCHANGE: {header_line}")
        elif (
            any(keyword in header_upper for keyword in ["SUMMARY BY CARD TYPE", "SUMMARY BY CARD", "CARD TYPE"])
            or ("SUMMARY" in header_upper and "CARD" in header_upper)
        ):
            sections["SUMMARY_CARD"].append(block)
            print(f"   ✅ Matched SUMMARY_CARD: {header_line}")
        elif any(keyword in header_upper for keyword in ["FEE", "FEES", "SERVICE CHARGE", "DISCOUNT"]):
            if "SUMMARY" not in header_upper and "THIRD PARTY" not in header_upper and "CHARGEBACK" not in header_upper:
                sections["FEES"].append(block)
                print(f"   ✅ Matched FEES: {header_line}")

    return {
        "FEES": "\n\n".join(sections["FEES"]).strip(),
        "INTERCHANGE": "\n\n".join(sections["INTERCHANGE"]).strip(),
        "SUMMARY_CARD": "\n\n".join(sections["SUMMARY_CARD"]).strip(),
    }


def build_statement_from_documents(documents):
    sections_markdown = {"FEES": "", "INTERCHANGE": "", "SUMMARY_CARD": ""}
    statement_data = {
        "summary": {},
        "summary_by_card_type": [],
        "third_party_transactions": [],
        "fees": [],
        "interchange_charges": [],
    }

    for doc in documents:
        text = doc.text

        block_sections = _collect_sections_markdown(text)
        for name, content in block_sections.items():
            cleaned_content = content.strip()
            if not cleaned_content:
                continue
            if sections_markdown[name]:
                sections_markdown[name] = (
                    sections_markdown[name] + "\n\n" + cleaned_content
                ).strip()
            else:
                sections_markdown[name] = cleaned_content
            print(f"   🔍 Found {name} section: {len(cleaned_content)} chars")

        if "# SUMMARY" in text:
            headers, rows = parse_markdown_table(text)
            for row in rows:
                if len(row) >= 3:
                    statement_data["summary"][row[0]] = {
                        "description": row[1],
                        "amount": row[2],
                    }

        if "# SUMMARY BY CARD TYPE" in text:
            headers, rows = parse_markdown_table(text)
            for row in rows:
                if len(row) >= 8:
                    statement_data["summary_by_card_type"].append(
                        {
                            "card_type": row[0],
                            "average_ticket": row[1],
                            "items": row[2],
                            "gross_sales": row[3],
                            "refunds_items": row[4],
                            "refunds_amount": row[5],
                            "total_items": row[6],
                            "total_amount": row[7],
                        }
                    )

        if "# THIRD PARTY TRANSACTIONS" in text:
            headers, rows = parse_markdown_table(text)
            for row in rows:
                if len(row) >= 3:
                    statement_data["third_party_transactions"].append(
                        {
                            "date": row[0],
                            "description": row[1],
                            "amount": row[2],
                        }
                    )

        if "# FEES" in text:
            headers, rows = parse_markdown_table(text)
            for row in rows:
                if len(row) >= 3:
                    statement_data["fees"].append(
                        {
                            "description": row[0],
                            "type": row[1],
                            "amount": row[2],
                        }
                    )

        if "# INTERCHANGE CHARGES/PROGRAM FEES" in text or "INTERCHANGE" in text:
            headers, rows = parse_markdown_table(text)
            for row in rows:
                if len(row) >= 8:
                    statement_data["interchange_charges"].append(
                        {
                            "product_description": row[0],
                            "sales_total": row[1],
                            "percent_of_sales": row[2],
                            "number_of_transactions": row[3],
                            "percent_of_total_transactions": row[4],
                            "interchange_rate": row[5],
                            "cost_per_transaction": row[6],
                            "sub_total": row[7],
                            "total_charges": row[8] if len(row) > 8 else "",
                        }
                    )

    return sections_markdown, statement_data

# Function to format and display the statement
def display_statement(data):
    print("# Bank Statement")
    print(f"Merchant: {data['header']['merchant_name']} (Store #{data['header']['store_number']})")
    print(f"Merchant Number: {data['header']['merchant_number']}")
    print(f"Statement Period: {data['header']['statement_period']}")
    print(f"Address: {data['header']['address']}")
    print(f"Customer Service: Phone: {data['header']['customer_service_phone']}, Website: {data['header']['customer_service_website']}")
    print("\n")

    # Display Summary
    print("## Summary")
    summary_table = [[k, v["description"], v["amount"]] for k, v in data["summary"].items()]
    print(tabulate(summary_table, headers=["Page", "Description", "Amount"], tablefmt="grid"))

    # Display Summary by Card Type
    print("\n## Summary by Card Type")
    card_type_table = [[row["card_type"], row["average_ticket"], row["items"], row["gross_sales"],
                        row["refunds_items"], row["refunds_amount"], row["total_items"], row["total_amount"]]
                       for row in data["summary_by_card_type"]]
    print(tabulate(card_type_table, headers=["Card Type", "Average Ticket", "Items", "Gross Sales",
                                            "Refunds Items", "Refunds Amount", "Total Items", "Total Amount"],
                   tablefmt="grid"))

    # Display Third Party Transactions
    print("\n## Third Party Transactions")
    third_party_table = [[row["date"], row["description"], row["amount"]]
                         for row in data["third_party_transactions"]]
    print(tabulate(third_party_table, headers=["Date", "Description", "Amount"], tablefmt="grid"))

    # Display Fees
    print("\n## Fees")
    fees_table = [[row["description"], row["type"], row["amount"]] for row in data["fees"]]
    print(tabulate(fees_table, headers=["Description", "Type", "Amount"], tablefmt="grid"))

    # Display Interchange Charges
    print("\n## Interchange Charges/Program Fees")
    interchange_table = [[row["product_description"], row["sales_total"], row["percent_of_sales"],
                         row["number_of_transactions"], row["percent_of_total_transactions"],
                         row["interchange_rate"], row["cost_per_transaction"], row["sub_total"],
                         row["total_charges"]]
                        for row in data["interchange_charges"]]
    print(tabulate(interchange_table, headers=["Product/Description", "Sales Total", "% of Sales",
                                              "Number of Transactions", "% of Total Transactions",
                                              "Interchange Rate", "Cost per Transaction", "Sub Total",
                                              "Total Charges"], tablefmt="grid"))

    # Save to markdown file
    with open("cleaned_statement.md", "w", encoding="utf-8") as f:
        f.write(f"# Bank Statement\n\n")
        f.write(f"**Merchant**: {data['header']['merchant_name']} (Store #{data['header']['store_number']})\n")
        f.write(f"**Merchant Number**: {data['header']['merchant_number']}\n")
        f.write(f"**Statement Period**: {data['header']['statement_period']}\n")
        f.write(f"**Address**: {data['header']['address']}\n")
        f.write(f"**Customer Service**: Phone: {data['header']['customer_service_phone']}, Website: {data['header']['customer_service_website']}\n\n")

        f.write("## Summary\n")
        f.write(tabulate(summary_table, headers=["Page", "Description", "Amount"], tablefmt="markdown") + "\n\n")

        f.write("## Summary by Card Type\n")
        f.write(tabulate(card_type_table, headers=["Card Type", "Average Ticket", "Items", "Gross Sales",
                                                  "Refunds Items", "Refunds Amount", "Total Items", "Total Amount"],
                         tablefmt="markdown") + "\n\n")

        f.write("## Third Party Transactions\n")
        f.write(tabulate(third_party_table, headers=["Date", "Description", "Amount"], tablefmt="markdown") + "\n\n")

        f.write("## Fees\n")
        f.write(tabulate(fees_table, headers=["Description", "Type", "Amount"], tablefmt="markdown") + "\n\n")

        f.write("## Interchange Charges/Program Fees\n")
        f.write(tabulate(interchange_table, headers=["Product/Description", "Sales Total", "% of Sales",
                                                    "Number of Transactions", "% of Total Transactions",
                                                    "Interchange Rate", "Cost per Transaction", "Sub Total",
                                                    "Total Charges"], tablefmt="markdown") + "\n")

def infer_brand(description):
    if not description:
        return "DEBIT"
    desc_upper = description.upper()
    if desc_upper.startswith("MC ") or desc_upper.startswith("MASTERCARD") or "MASTERCARD" in desc_upper:
        return "MASTERCARD"
    if desc_upper.startswith("VS ") or desc_upper.startswith("VISA") or "VISA" in desc_upper:
        return "VISA"
    if desc_upper.startswith("DS ") or desc_upper.startswith("DISCOVER") or "DISCOVER" in desc_upper:
        return "DISCOVER"
    if desc_upper.startswith("AX ") or desc_upper.startswith("AMERICAN EXPRESS") or "AMERICAN EXPRESS" in desc_upper or "AMEX" in desc_upper:
        return "American Express"
    return "DEBIT"

def clean_numeric_value(value):
    if value is None:
        return ""
    return str(value).strip()

def fix_fee_signs(amount_str):
    if not amount_str:
        return amount_str
    try:
        amount = float(str(amount_str).replace(",", "").replace("$", ""))
        flipped_amount = -amount
        return f"{flipped_amount:,.2f}"
    except Exception:
        return amount_str

def clean_description_with_amounts(desc):
    """Remove amount and rate portions from descriptions like 'VI CEDP COMM ENH DATA PGM FEE $64,212.52 AT .000500'"""
    if not desc:
        return desc
    
    # Pattern to match: FEE $amount AT rate
    import re
    # Remove patterns like "FEE $64,212.52 AT .000500" or similar
    cleaned = re.sub(r'\s+FEE\s+\$[\d,]+\.?\d*\s+AT\s+[\d.]+', '', desc)
    cleaned = re.sub(r'\s+\$[\d,]+\.?\d*\s+AT\s+[\d.]+', '', cleaned)
    cleaned = re.sub(r'\s+AT\s+[\d.]+', '', cleaned)
    
    return cleaned.strip()

def extract_account_info_from_header(full_text):
    """Extract Account Name and Account Holder from document header using LLM"""
    try:
        # Use the global LLM instance
        
        system_prompt = """
        Extract Account Name and Account Holder from bank statement header.
        
        Look for patterns like:
        - "Merchant: JOHNSTONE SUPPLY-GA_ALPH (Store #5)"
        - "Merchant: COMPANY NAME-LOCATION (Store #X)"
        - "Merchant: MERCHANT NAME (Store #X)"
        
        Rules:
        - Account Name: Extract the location part (after the last dash) and store number
          Example: "Merchant: JOHNSTONE SUPPLY-GA_ALPH (Store #5)" → "GA_ALPH STORE# 5"
        - Account Holder: Extract the company name part (before the last dash) and add " CORPORATE"
          Example: "Merchant: JOHNSTONE SUPPLY-GA_ALPH (Store #5)" → "JOHNSTONE SUPPLY CORPORATE"
        
        Return JSON: {{"Account Name": "...", "Account Holder": "..."}}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Extract from this document header:\n\n{header_text}")
        ])
        
        # Get first 2000 characters (likely contains header)
        header_text = full_text[:2000]
        
        chain = prompt | llm
        result = chain.invoke({"header_text": header_text})
        
        # Parse JSON response
        import json
        try:
            account_info = json.loads(result.content)
            return account_info
        except json.JSONDecodeError as e:
            print(f"   ⚠️ JSON Parse Error: {e}")
            # Try to extract manually if JSON parsing fails
            content = result.content.strip()
            if "Account Name" in content and "Account Holder" in content:
                # Try to extract values using regex
                import re
                name_match = re.search(r'"Account Name":\s*"([^"]*)"', content)
                holder_match = re.search(r'"Account Holder":\s*"([^"]*)"', content)
                if name_match and holder_match:
                    return {
                        "Account Name": name_match.group(1),
                        "Account Holder": holder_match.group(1)
                    }
            raise e
        
    except Exception as e:
        print(f"   ⚠️ Failed to extract account info: {e}")
        return {
            'Account Name': 'Unknown',
            'Account Holder': 'Corporate'
        }

def populate_missing_metadata(rows, metadata):
    """Populate missing metadata fields in all rows"""
    for row in rows:
        # Fill in missing metadata fields
        for key, value in metadata.items():
            if key not in row or row[key] is None or row[key] == "":
                row[key] = value
        
        # Ensure all required fields have default values
        if row.get('Count #') is None:
            row['Count #'] = 1
        if row.get('Amount $') is None:
            row['Amount $'] = 0
        if row.get('Rate %') is None:
            row['Rate %'] = 0
        if row.get('Rate $') is None:
            row['Rate $'] = 0
        if row.get('Fees $') is None:
            row['Fees $'] = 0
    
    return rows

def validate_extracted_data(rows, section_name):
    """Validate that all required fields are populated"""
    if not rows:
        return
    
    required_fields = ['Statement Description', 'Brand', 'Fee Type', 'Count #', 'Amount $', 'Rate %', 'Rate $', 'Fees $']
    missing_counts = {field: 0 for field in required_fields}
    
    for row in rows:
        for field in required_fields:
            if field not in row or row[field] is None or row[field] == "":
                missing_counts[field] += 1
    
    print(f"   🔍 {section_name} validation:")
    for field, count in missing_counts.items():
        if count > 0:
            print(f"     ⚠️ {field}: {count} missing values")
        else:
            print(f"     ✅ {field}: All values present")

    total_missing = sum(missing_counts.values())
    if total_missing > 0:
        print(f"     ⚠️ Total missing values: {total_missing}")
    else:
        print(f"     ✅ All required fields populated")

def to_rows(statement_data, metadata):
    rows = []
    # Fees
    for fee in statement_data.get("fees", []):
        desc = fee.get("description", "")
        amt = fee.get("amount", "")
        brand = infer_brand(desc)
        rows.append({
            'Account Name': metadata['Account Name'],
            'Account ID': metadata['Account ID'],
            'Statement Description': desc,
            'Brand': brand,
            'Count #': "",
            'Amount $': "",
            'Rate %': "",
            'Rate $': "",
            'Fees $': fix_fee_signs(amt),
            'Fee Type': 'Fees',
            'Date': metadata['Date'],
            'Account Holder': metadata['Account Holder'],
            'Account Short ID': '',
            'Gateway': '',
            'Account Type': '',
            'Filename': metadata['Filename'],
            'Processor': '',
            'User Permissions': ''
        })
    # Interchange charges
    for ic in statement_data.get("interchange_charges", []):
        desc = ic.get("product_description", "")
        brand = infer_brand(desc)
        rows.append({
            'Account Name': metadata['Account Name'],
            'Account ID': metadata['Account ID'],
            'Statement Description': desc,
            'Brand': brand,
            'Count #': clean_numeric_value(ic.get("number_of_transactions", "")),
            'Amount $': clean_numeric_value(ic.get("sales_total", "")),
            'Rate %': clean_numeric_value(ic.get("interchange_rate", "")),
            'Rate $': clean_numeric_value(ic.get("cost_per_transaction", "")),
            'Fees $': clean_numeric_value(ic.get("sub_total", "")),
            'Fee Type': 'Interchange Charges',
            'Date': metadata['Date'],
            'Account Holder': metadata['Account Holder'],
            'Account Short ID': '',
            'Gateway': '',
            'Account Type': '',
            'Filename': metadata['Filename'],
            'Processor': '',
            'User Permissions': ''
        })
    # Summary by card type → Gross Sales and Refunds
    for s in statement_data.get("summary_by_card_type", []):
        brand = infer_brand(s.get("card_type", ""))
        # Gross Sales
        rows.append({
            'Account Name': metadata['Account Name'],
            'Account ID': metadata['Account ID'],
            'Statement Description': 'Gross Sales',
            'Brand': brand,
            'Count #': clean_numeric_value(s.get("items", "")),
            'Amount $': clean_numeric_value(s.get("gross_sales", "")),
            'Rate %': '',
            'Rate $': '',
            'Fees $': '',
            'Fee Type': 'Gross Sales',
            'Date': metadata['Date'],
            'Account Holder': metadata['Account Holder'],
            'Account Short ID': '',
            'Gateway': '',
            'Account Type': '',
            'Filename': metadata['Filename'],
            'Processor': '',
            'User Permissions': ''
        })
        # Refunds
        rows.append({
            'Account Name': metadata['Account Name'],
            'Account ID': metadata['Account ID'],
            'Statement Description': 'Refunds',
            'Brand': brand,
            'Count #': clean_numeric_value(s.get("refunds_items", "")),
            'Amount $': clean_numeric_value(s.get("refunds_amount", "")),
            'Rate %': '',
            'Rate $': '',
            'Fees $': '',
            'Fee Type': 'Refunds',
            'Date': metadata['Date'],
            'Account Holder': metadata['Account Holder'],
            'Account Short ID': '',
            'Gateway': '',
            'Account Type': '',
            'Filename': metadata['Filename'],
            'Processor': '',
            'User Permissions': ''
        })
    return rows

def save_extracted_data_to_excel(extracted_data, output_excel):
    if not extracted_data:
        print("No data to save")
        return
    columns = [
        'Account Name', 'Account ID', 'Statement Description', 'Brand', 'Count #',
        'Amount $', 'Rate %', 'Rate $', 'Fees $', 'Fee Type', 'Date', 'Account Holder',
        'Account Short ID', 'Gateway', 'Account Type', 'Filename', 'Processor', 'User Permissions'
    ]
    df = pd.DataFrame(extracted_data)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    df = df[columns]
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)

# --- Optional LLM post-processor (strict, no new rows) ---

def post_process_rows_with_llm(rows, metadata):
    try:
        if not rows:
            return rows
        # Use the global LLM instance

        system_rules = (
            "You are a financial statement table normalizer.\n"
            "ONLY transform the provided rows. NEVER add or delete rows.\n"
            "Keep the SAME ORDER and SAME LENGTH.\n"
            "Return STRICT JSON only.\n\n"
            "Goals: Clean descriptions, infer Brand if missing, clean Rate % (remove 'TIMES'),\n"
            "map Interchange vs Fees correctly, and keep 18-column schema values.\n\n"
            "Exclusions: Do NOT fabricate data. Do NOT include Third Party Transactions,\n"
            "Miscellaneous Adjustment, Chargebacks, Month End Charges, or date-only/admin entries.\n\n"
            "Brand inference: VISA if tokens like 'VI', 'VISA'; MASTERCARD for 'MC', 'MASTERCARD';\n"
            "DISCOVER for 'DS', 'DISCOVER'; American Express for 'AMEX', 'AMERICAN EXPRESS';\n"
            "others default to DEBIT.\n\n"
            "Fee Type categorization: Assessment/Network fees (ASSESSMNT/ASSESSMENT, NTWK/NETWORK, ACCESS, LICENSE,\n"
            "VOLUME FEE, ACQUIRER, AVS, SALES DISCOUNT, DISC RATE) → 'Fees'.\n"
            "Interchange descriptions like prefixes 'VI-', 'MC-', 'DISCOVR', 'AMEX PASS', 'STAR', 'ACCEL', 'MAESTRO' → 'Interchange Charges'.\n\n"
            "Field rules:\n"
            "- 'Fees $' is the actual fee amount (positive values for reporting).\n"
            "- Interchange rows: map Sub Total (if present) to 'Fees $'; 'Sales Total' to 'Amount $'; 'Number of Transactions' to 'Count #'; 'Rate' to 'Rate %'.\n"
            "- Keep commas in formatted numbers, plain strings otherwise.\n"
            "- NEVER invent amounts or counts. Leave unknown fields empty.\n"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_rules),
            ("human", (
                "Metadata (do not change): Account Name={account_name}, Account ID={account_id}, Date={date}, Account Holder={account_holder}, Filename={filename}.\n"
                "Given the JSON array 'rows', return JSON with key 'rows' containing the SAME number of items and order.\n"
                "You may ONLY modify these fields per row: 'Statement Description', 'Brand', 'Amount $', 'Rate %', 'Rate $', 'Fees $', 'Fee Type'.\n"
                "All other fields must be echoed unchanged.\n\n"
                "rows: {rows_json}"
            )),
        ])

        # Attach row_id to preserve order and to help strict mapping
        input_rows = []
        for i, r in enumerate(rows):
            r2 = dict(r)
            r2["_row_id"] = i
            input_rows.append(r2)

        import json as _json
        chain = prompt | llm
        result = chain.invoke({
            "account_name": metadata.get("Account Name", ""),
            "account_id": metadata.get("Account ID", ""),
            "date": metadata.get("Date", ""),
            "account_holder": metadata.get("Account Holder", ""),
            "filename": metadata.get("Filename", ""),
            "rows_json": _json.dumps(input_rows, ensure_ascii=False)
        })

        text = getattr(result, "content", None) or str(result)
        # Extract JSON body
        try:
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                text = text.split('```')[1].split('```')[0]
            text = text.strip()
        except Exception:
            pass

        data = None
        try:
            data = _json.loads(text)
        except Exception:
            # Try to find JSON braces
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                data = _json.loads(text[start:end+1])

        if not isinstance(data, dict) or 'rows' not in data:
            return rows

        cleaned = data['rows']
        if not isinstance(cleaned, list) or len(cleaned) != len(rows):
            return rows

        # Ensure order by _row_id
        cleaned.sort(key=lambda x: x.get('_row_id', 0))
        for c in cleaned:
            if '_row_id' in c:
                c.pop('_row_id', None)
        return cleaned
    except Exception:
        return rows

def extract_rows_with_llm_from_sections(sections_markdown, metadata):
    try:
        content_parts = []
        fees_md = sections_markdown.get("FEES") or ""
        inter_md = sections_markdown.get("INTERCHANGE") or ""
        if fees_md:
            content_parts.append("FEES SECTION (markdown):\n" + fees_md)
        if inter_md:
            content_parts.append("INTERCHANGE SECTION (markdown):\n" + inter_md)
        if not content_parts:
            return []

        # Use the global LLM instance

        system_rules = (
            "You are a financial statement table extractor.\n"
            "Input: Two markdown tables from a credit card statement - FEES table and INTERCHANGE CHARGES/PROGRAM FEES table.\n"
            "Output: Single JSON combining BOTH tables into our 18-column schema, with DESCRIPTION-BASED ENRICHMENT.\n\n"
            "CRITICAL: Extract EVERY ROW from BOTH tables. Do NOT skip any financial data.\n\n"
            "18-COLUMN SCHEMA: ['Account Name','Account ID','Statement Description','Brand','Count #','Amount $','Rate %','Rate $','Fees $','Fee Type','Date','Account Holder','Account Short ID','Gateway','Account Type','Filename','Processor','User Permissions']\n\n"
            "=== FEES TABLE MAPPING ===\n"
            "- Description column → 'Statement Description'\n"
            "- Amount column → 'Fees $' (this is the actual fee amount)\n"
            "- Fee Type = 'Fees' (unless assessment/network fee patterns detected)\n"
            "- Assessment/Network keywords (ASSESSMNT, ASSESSMENT, NTWK, NETWORK, ACCESS, LICENSE, VOLUME FEE, ACQUIRER, AVS, SALES DISCOUNT, DISC RATE) → Fee Type='Fees'\n"
            "- Examples: 'MAESTRO REG | -$0.79' → Statement Description='MAESTRO REG', Fees $='0.79', Fee Type='Fees'\n\n"
            "=== INTERCHANGE TABLE MAPPING ===\n"
            "- Product/Description → 'Statement Description'\n"
            "- Sales Total → 'Amount $' (transaction volume, NOT fees)\n"
            "- Number of Transactions → 'Count #'\n"
            "- Rate → 'Rate %' (clean by removing 'TIMES' token)\n"
            "- Sub Total → 'Fees $' (ONLY if Sub Total > 0; if 0.00, skip row as it's summary)\n"
            "- Fee Type = 'Interchange Charges'\n"
            "- Examples: 'VI-RETAIL | $5,000 | 50 | 0.015 | $75' → Statement Description='VI-RETAIL', Amount $='5,000', Count #='50', Rate %='0.015', Fees $='75'\n\n"
            "=== DESCRIPTION NORMALIZATION & ENRICHMENT (CRITICAL) ===\n"
            "Normalize descriptions so equivalent labels merge, then when a description appears in BOTH sections,\n"
            "ENRICH the FEES row with fields from INTERCHANGE: copy 'Rate %', 'Rate $', 'Count #', and 'Amount $' from the matching INTERCHANGE description.\n"
            "Examples of canonical descriptions:\n"
            "- 'VI-SERVICES TRAD REWARDS'\n"
            "- 'MC-BUS LEVEL 5 DATA RATE II'\n"
            "- 'DISCOVR PSL RTL DB'\n"
            "- 'AMEX PASS THRU'\n"
            "Descriptions may have explanatory parentheses; keep them out of the canonical description field.\n\n"
            "=== BRAND INFERENCE ===\n"
            "- VI/VISA → 'VISA'\n"
            "- MC/MASTERCARD → 'MASTERCARD'\n" 
            "- DS/DISCOVER/DISCOVR → 'DISCOVER'\n"
            "- AX/AMEX/AMERICAN EXPRESS → 'American Express'\n"
            "- All others (STAR, ACCEL, MAESTRO, etc.) → 'DEBIT'\n\n"
            "=== EXCLUSIONS ===\n"
            "- Do NOT include Third Party Transactions, Misc Adjustment, Chargebacks\n"
            "- Do NOT include Month End Charges or admin entries\n"
            "- Do NOT include date-only entries like '07/31/2025'\n"
            "- Skip interchange rows where Sub Total = 0.00 (summaries)\n\n"
            "RETURN: Complete extraction from BOTH tables as single JSON array."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_rules),
            ("human", (
                "Apply this metadata to every row: Account Name={account_name}, Account ID={account_id}, Date={date}, Account Holder={account_holder}, Filename={filename}.\n"
                "Return JSON as {{\"rows\":[...]}} only.\n\n"
                "MARKDOWN INPUT:\n{md}"
            )),
        ])

        chain = prompt | llm
        md_joined = "\n\n---\n\n".join(content_parts)
        import json as _json
        result = chain.invoke({
            "account_name": metadata.get("Account Name", ""),
            "account_id": metadata.get("Account ID", ""),
            "date": metadata.get("Date", ""),
            "account_holder": metadata.get("Account Holder", ""),
            "filename": metadata.get("Filename", ""),
            "md": md_joined,
        })

        text = getattr(result, "content", None) or str(result)
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        text = text.strip()
        data = _json.loads(text)
        if isinstance(data, dict) and isinstance(data.get('rows'), list):
            return data['rows']
        return []
    except Exception:
        return []

# --- Duplicate merging utilities (keep only merge) ---

def _parse_number(value):
    try:
        if value is None:
            return None
        s = str(value).replace(',', '').replace('$', '').strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None

def merge_duplicate_rows(rows):
    print(f"🔄 Starting aggressive duplicate merging on {len(rows)} rows...")
    merged = {}
    
    def key(row):
        # More aggressive key - only use description and brand for interchange charges
        desc = str(row.get('Statement Description', '') or '').upper().strip()
        brand = str(row.get('Brand', '') or '').upper().strip()
        fee_type = str(row.get('Fee Type', '') or '').upper().strip()
        
        # For interchange charges, merge by description + brand only
        if fee_type == 'INTERCHANGE CHARGES':
            return (desc, brand, fee_type)
        else:
            # For other types, use full key
            return (
                desc, brand, fee_type,
                str(row.get('Account Name', '') or '').upper().strip(),
                str(row.get('Account ID', '') or '').upper().strip(),
                str(row.get('Date', '') or '').strip(),
            )
    
    for row in rows:
        k = key(row)
        if k not in merged:
            merged[k] = dict(row)
            print(f"   ➕ New entry: {row.get('Statement Description', '')[:30]}...")
        else:
            base = merged[k]
            print(f"   🔄 Merging: {row.get('Statement Description', '')[:30]}...")
            
            # Count #: sum when numeric
            c1 = _parse_number(base.get('Count #'))
            c2 = _parse_number(row.get('Count #'))
            if c1 is not None or c2 is not None:
                new_count = (c1 or 0) + (c2 or 0)
                base['Count #'] = f"{new_count:,.0f}" if new_count == int(new_count) else f"{new_count:,.2f}"
            
            # Amount $: sum for interchange charges, prefer first non-empty for others
            a1 = _parse_number(base.get('Amount $'))
            a2 = _parse_number(row.get('Amount $'))
            fee_type = str(base.get('Fee Type', '')).upper().strip()
            if fee_type == 'INTERCHANGE CHARGES' and (a1 is not None or a2 is not None):
                new_amount = (a1 or 0) + (a2 or 0)
                base['Amount $'] = f"{new_amount:,.2f}"
            elif not str(base.get('Amount $', '')).strip() and str(row.get('Amount $', '')).strip():
                base['Amount $'] = row.get('Amount $')
            
            # Fees $: sum when numeric
            f1 = _parse_number(base.get('Fees $'))
            f2 = _parse_number(row.get('Fees $'))
            if f1 is not None or f2 is not None:
                new_fees = (f1 or 0) + (f2 or 0)
                base['Fees $'] = f"{new_fees:,.2f}"
            
            # Rate %: keep first non-empty
            if not str(base.get('Rate %', '')).strip() and str(row.get('Rate %', '')).strip():
                base['Rate %'] = row.get('Rate %')
            
            # Rate $: keep first non-empty
            if not str(base.get('Rate $', '')).strip() and str(row.get('Rate $', '')).strip():
                base['Rate $'] = row.get('Rate $')
            
            # Fill other metadata if empty
            for meta in ['Account Holder', 'Account Short ID', 'Gateway', 'Account Type', 'Filename', 'Processor', 'User Permissions']:
                if not str(base.get(meta, '')).strip() and str(row.get(meta, '')).strip():
                    base[meta] = row.get(meta)
    
    final_rows = list(merged.values())
    print(f"✅ Merging complete: {len(rows)} → {len(final_rows)} rows")
    return final_rows

# Display the structured statement (console + markdown) - COMMENTED OUT FOR DEBUGGING
# display_statement(statement_data)

# DEBUG helper moved into process_single_pdf

# New section-by-section LLM processing approach
def process_section_with_llm(section_name, section_markdown, metadata):
    """Process individual section with LLM and return clean rows"""
    if not section_markdown.strip():
        return []
    
    print(f"🔄 Processing {section_name} section with LLM...")
    
    # Use the global LLM instance
    
    if section_name == "FEES":
        system_prompt = """
        Extract FEES table data into clean 18-column JSON format.
        CRITICAL: Extract ONLY data rows, NO headers, NO dashes, NO formatting lines.
        MANDATORY: Every row MUST have a non-empty 'Statement Description'. If unclear, skip the row.
        
        REQUIRED FIELDS - ALL must be populated:
        - Statement Description: REQUIRED (never empty)
        - Brand: REQUIRED (infer from description)
        - Fee Type: REQUIRED (categorize as "Fees", "Interchange Charges", or "Service Charges")
        - Count #: REQUIRED (extract transaction count, use 1 if not specified)
        - Amount $: REQUIRED (extract transaction volume, use 0 if not specified)
        - Rate %: REQUIRED (extract percentage rate, use 0 if not specified)
        - Rate $: REQUIRED (extract fixed rate, use 0 if not specified)
        - Fees $: REQUIRED (extract actual fee amount, use 0 if not specified)
        
        Expected columns from FEES table (handle BOTH formats):
        
        FORMAT 1 (Simple table - Fee Type embedded in description):
        - Description → FIRST extract rate, amount, and fee type, THEN clean to 'Statement Description'
        - Amount column → 'Fees $' (actual fee amount)
        - Fee Type: Extract from end of description (e.g., "Interchange charges", "Fees", "Service charges")
        
        FORMAT 2 (Full table - HAS Fee Type column):
        - Description column → 'Statement Description' (extract rate/amount if embedded)
        - Type column → 'Fee Type' (use exactly as-is from the table)
        - Amount column → 'Fees $' (actual fee amount)
        
        RATE/AMOUNT EXTRACTION (for both formats):
        - Extract rate patterns like "0.0014 TIMES", "0.0003", etc. → put in 'Rate %' (remove "TIMES")
        - Extract dollar amounts like "$80502.99", "$21929.53" → put in 'Amount $' (remove "$")
        - Clean description by removing extracted rate/amount → 'Statement Description'
        - Extract transaction counts from descriptions or use 1 as default
        - Extract fixed rates from descriptions or use 0 as default
        
        Brand inference: VI/VISA→VISA, MC/MASTERCARD→MASTERCARD, DS/DISCOVER→DISCOVER, AMEX→American Express, others→DEBIT
        
        EXCLUDE: Third Party, Misc Adjustment, Chargebacks, headers, dashes, formatting
        
        JSON FORMAT: Return {{"rows": [{{"Statement Description": "...", "Brand": "...", "Fee Type": "...", "Count #": 1, "Amount $": 0, "Rate %": 0, "Rate $": 0, "Fees $": 0, ...}}]}}
        """
    elif section_name == "INTERCHANGE":
        system_prompt = """
        Extract INTERCHANGE CHARGES table data into clean 18-column JSON format.
        CRITICAL: Extract ONLY data rows, NO headers, NO dashes, NO formatting lines.
        MANDATORY: Every row MUST have a non-empty 'Statement Description'. If unclear, skip the row.
        
        REQUIRED FIELDS - ALL must be populated:
        - Statement Description: REQUIRED (never empty)
        - Brand: REQUIRED (infer from description)
        - Fee Type: REQUIRED (categorize as "Fees" or "Interchange Charges")
        - Count #: REQUIRED (extract transaction count, use 1 if not specified)
        - Amount $: REQUIRED (extract transaction volume, use 0 if not specified)
        - Rate %: REQUIRED (extract percentage rate, use 0 if not specified)
        - Rate $: REQUIRED (extract fixed rate, use 0 if not specified)
        - Fees $: REQUIRED (extract actual fee amount, use 0 if not specified)
        
        Expected columns from INTERCHANGE table:
        - Product/Description → FIRST extract embedded rate/amount, THEN clean to 'Statement Description'
          EXTRACTION PROCESS for descriptions with embedded data:
          1. Extract rate patterns like "0.0014 TIMES", "AT .000500" → put in 'Rate %' (remove "TIMES"/"AT")
          2. Extract dollar amounts like "$80502.99", "$64,212.52" → put in 'Amount $' (remove "$")
          3. Clean description by removing extracted rate/amount → 'Statement Description'
          
          Examples:
            - "MASTERCARD ASSESSMENT FEE 0.0014 TIMES $80502.99" → 
              Statement Description: "MASTERCARD ASSESSMENT FEE"
              Rate %: "0.0014"
              Amount $: "80502.99"
            - "VI CEDP COMM ENH DATA PGM FEE $64,212.52 AT .000500" →
              Statement Description: "VI CEDP COMM ENH DATA PGM FEE"
              Rate %: "0.000500"
              Amount $: "64,212.52"
        - Sales Total column → 'Amount $' (transaction volume, if separate column exists)
        - Number of Transactions → 'Count #'
        - Rate column → 'Rate %' (clean by removing 'TIMES', PRESERVE original decimal places: 0.100 stays 0.100, not 0.1)
        - Cost Per Transaction → 'Rate $' (per-transaction cost, PRESERVE original decimal places: 0.100 stays 0.100, not 0.1)
        - Sub Total → 'Fees $' (ONLY if > 0, skip if 0.00)
        - Fee Type:
            - If the description contains assessment-like numeric tails (e.g., contains ASSESSMENT/ASSESSMNT and a phrase like "TIMES $...", "AT ....", or "X TRNS $..."), set Fee Type='Fees'
            - Otherwise, set Fee Type='Interchange Charges'
        
        CRITICAL: Preserve exact decimal formatting from source data. Do NOT normalize decimals (0.100 should remain 0.100, not become 0.1).
        
        Brand inference: VI/VISA→VISA, MC/MASTERCARD→MASTERCARD, DS/DISCOVER→DISCOVER, AMEX→American Express, others→DEBIT
        
        EXCLUDE: headers, dashes, formatting
        
        IMPORTANT: Include ALL data rows, even if Sub Total=0.00, as they contain valid transaction data (Count #, Rate %, Rate $, Amount $)
        
        JSON FORMAT: Return {{"rows": [{{"Statement Description": "...", "Brand": "...", "Fee Type": "...", "Count #": 1, "Amount $": 0, "Rate %": 0, "Rate $": 0, "Fees $": 0, ...}}]}}
        """
    else:  # SUMMARY_CARD
        system_prompt = """
        Extract SUMMARY BY CARD TYPE into Gross Sales and Refunds rows.
        CRITICAL: Extract ONLY data rows, NO headers, NO dashes, NO formatting lines.
        MANDATORY: Every row MUST have a non-empty 'Statement Description'. If unclear, skip the row.
        
        REQUIRED FIELDS - ALL must be populated:
        - Statement Description: REQUIRED (never empty)
        - Brand: REQUIRED (infer from card type)
        - Fee Type: REQUIRED ("Gross Sales" or "Refunds")
        - Count #: REQUIRED (extract transaction count, use 1 if not specified)
        - Amount $: REQUIRED (extract transaction volume, use 0 if not specified)
        - Rate %: REQUIRED (use 0 for summary rows)
        - Rate $: REQUIRED (use 0 for summary rows)
        - Fees $: REQUIRED (use 0 for summary rows)
        
        Expected columns from SUMMARY BY CARD TYPE table:
        - Card Type → 'Brand' (apply brand inference)
        - Items/Count → 'Count #' (for gross sales)
        - Gross Sales → 'Amount $' (for gross sales)
        - Refund Items → 'Count #' (for refunds) 
        - Refund Amount → 'Amount $' (for refunds)
        
        For each card type row, create TWO output rows:
        1. Gross Sales row:
           - Statement Description: "Gross Sales"
           - Brand: [inferred from card type]
           - Count #: [items count]
           - Amount $: [gross sales amount]
           - Fee Type: "Gross Sales"
           
        2. Refunds row:
           - Statement Description: "Refunds"  
           - Brand: [inferred from card type]
           - Count #: [refund items count]
           - Amount $: [refund amount]
           - Fee Type: "Refunds"
        
        Brand inference: 
        - MC/MASTERCARD → "MASTERCARD"
        - VI/VISA → "VISA" 
        - DS/DISCOVER → "DISCOVER"
        - AX/AMEX/AMERICAN EXPRESS → "American Express"
        - All others (DEBIT, etc.) → "DEBIT"
        
        EXCLUDE: Total rows, headers, dashes, formatting lines
        
        JSON FORMAT: Return {{"rows": [{{"Statement Description": "Gross Sales", "Brand": "...", "Fee Type": "Gross Sales", "Count #": 1, "Amount $": 0, "Rate %": 0, "Rate $": 0, "Fees $": 0, ...}}]}}
        """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", (
            "Metadata for ALL rows: Account Name={account_name}, Account ID={account_id}, Date={date}, Account Holder={account_holder}, Filename={filename}\n\n"
            "Return JSON: {{\"rows\":[...]}}\n"
            "18-column schema: Account Name,Account ID,Statement Description,Brand,Count #,Amount $,Rate %,Rate $,Fees $,Fee Type,Date,Account Holder,Account Short ID,Gateway,Account Type,Filename,Processor,User Permissions\n\n"
            "Section markdown:\n{markdown}"
        ))
    ])
    
    try:
        chain = prompt | llm
        result = chain.invoke({
            "account_name": metadata.get("Account Name", ""),
            "account_id": metadata.get("Account ID", ""),
            "date": metadata.get("Date", ""),
            "account_holder": metadata.get("Account Holder", ""),
            "filename": metadata.get("Filename", ""),
            "markdown": section_markdown
        })
        
        text = getattr(result, "content", None) or str(result)
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        
        import json as _json
        data = _json.loads(text.strip())
        
        if isinstance(data, dict) and isinstance(data.get('rows'), list):
            # Drop any rows with empty description
            cleaned_rows = [r for r in data['rows'] if str(r.get('Statement Description','')).strip()]
            dropped = len(data['rows']) - len(cleaned_rows)
            if dropped > 0:
                print(f"   ⚠️ {section_name}: Dropped {dropped} rows with empty descriptions")
            
            # Debug: Show Fee Type mapping for first few rows
            print(f"   🔍 {section_name}: Fee Type mapping debug:")
            for i, row in enumerate(cleaned_rows[:3]):  # Show first 3 rows
                desc = row.get('Statement Description', '')[:30]
                fee_type = row.get('Fee Type', '')
                print(f"     {i+1}. '{desc}...' → Fee Type: '{fee_type}'")
            
            print(f"   ✅ {section_name}: Extracted {len(cleaned_rows)} rows")
            return cleaned_rows
        return []
    except Exception as e:
        print(f"   ❌ {section_name}: Error - {e}")
        return []

def main():
    process_all_pdfs()
    print("\n🎉 All PDFs processed successfully!")
    print("📁 Check the output_excels/ directory for the generated Excel files.")

if __name__ == "__main__":
    main()