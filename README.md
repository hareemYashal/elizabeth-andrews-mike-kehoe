# Document Parser

A Python application for parsing bank statements and financial documents using LlamaParse and Google's Gemini AI.

## Features

- Parse PDF bank statements using LlamaParse
- Extract structured data from markdown tables
- Use Google Gemini AI for intelligent data extraction and processing
- Export results to Excel format
- Support for various financial document types (fees, interchange charges, card type summaries)

## Approach & Methodology

### 1. Document Processing Pipeline

Our document parser follows a sophisticated multi-stage approach:

1. **PDF to Markdown Conversion**: Uses LlamaParse to convert PDF bank statements into structured markdown format
2. **Section Identification**: Automatically identifies and extracts different sections (Fees, Interchange Charges, Summary by Card Type, etc.)
3. **AI-Powered Extraction**: Leverages Google Gemini AI for intelligent data extraction and normalization
4. **Data Enrichment**: Cross-references and enriches data between different sections
5. **Output Generation**: Exports clean, structured data to Excel format

### 2. AI Processing Strategy

**Single LLM Instance**: We use one global Google Gemini AI instance across all functions for:
- Consistency in processing
- Better performance (no repeated initialization)
- Easier maintenance and configuration

**Multi-Function AI Usage**:
- `extract_account_info_from_header()`: Extracts merchant name, account holder, and statement period
- `process_section_with_llm()`: Processes individual sections (Fees, Interchange, Summary)
- `extract_rows_with_llm_from_sections()`: Extracts structured data from markdown tables
- `post_process_rows_with_llm()`: Normalizes and cleans extracted data

### 3. Data Extraction Techniques

**Section-Based Processing**:
- **FEES Section**: Extracts fee descriptions, amounts, and types
- **INTERCHANGE Section**: Processes interchange charges with rates and transaction counts
- **SUMMARY BY CARD TYPE**: Creates gross sales and refunds entries for each card type

**Smart Data Enrichment**:
- Matches descriptions between sections using fuzzy matching
- Enriches fee rows with interchange data (rates, counts, amounts)
- Applies intelligent brand inference (VISA, Mastercard, Discover, American Express)

**Data Normalization**:
- Cleans descriptions by removing embedded rates and amounts
- Standardizes fee types (Fees vs Interchange Charges)
- Applies consistent formatting and sign conventions

### 4. Error Handling & Validation

- **API Key Validation**: Ensures all required API keys are present before processing
- **Data Validation**: Validates extracted data before export
- **Graceful Degradation**: Continues processing even if some sections fail
- **Comprehensive Logging**: Detailed progress tracking and error reporting

### 5. Output Structure

The parser generates Excel files with a standardized 18-column schema:
- Account information (Name, ID, Holder, Date)
- Transaction details (Description, Brand, Count, Amount)
- Financial data (Rates, Fees, Fee Type)
- Metadata (Filename, Processor, Gateway, etc.)

## Setup

### Prerequisites

- Python 3.8+
- Required packages (see requirements.txt)

### Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your API keys:
     ```
     LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here
     GOOGLE_API_KEY=your_google_api_key_here
     ```

### API Keys

You'll need the following API keys:

1. **Llama Cloud API Key**: Get it from [LlamaIndex Cloud](https://cloud.llamaindex.ai/)
2. **Google API Key**: Get it from [Google AI Studio](https://aistudio.google.com/)

## Usage

1. Place your PDF files in the `input_pdfs/` directory
2. Run the parser:
   ```bash
   python parser.py
   ```
3. Find the processed Excel files in the `output_excels/` directory

## Project Structure

```
document_parser/
├── input_pdfs/          # Place your PDF files here
├── output_excels/       # Processed Excel files will be saved here
├── parser.py            # Main parsing script
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (API keys)
└── README.md           # This file
```

## Configuration

The script uses a single global LLM instance for all AI processing, making it more efficient and consistent. You can modify the LLM settings in the `parser.py` file:

```python
# Initialize LLM once for reuse across all functions
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
```

## Features

- **Environment Variable Support**: All API keys are loaded from `.env` file for security
- **Single LLM Instance**: Efficient reuse of the same LLM instance across all functions
- **Comprehensive Data Extraction**: Handles fees, interchange charges, card type summaries, and more
- **Excel Export**: Clean, structured output in Excel format
- **Error Handling**: Robust error handling and validation

## Troubleshooting

- Make sure your `.env` file is in the project root and contains valid API keys
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that your PDF files are in the `input_pdfs/` directory
- Verify your API keys have the necessary permissions

## License

This project is for educational and personal use.
