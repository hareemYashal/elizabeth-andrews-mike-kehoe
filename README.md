# Elizabeth Andrews Mike Kehoe - Bank Statement Parser

A sophisticated Python application that parses bank statements and extracts financial data with intelligent Brand column logic for credit card processing fees.

## Features

- **PDF Parsing**: Uses Llama Cloud API to parse bank statement PDFs
- **AI-Powered Extraction**: Leverages Google Gemini AI for intelligent data extraction
- **Smart Brand Logic**: Automatically populates Brand column based on description patterns
- **Excel Export**: Generates structured Excel files with extracted data
- **Comprehensive Data Processing**: Handles fees, interchange charges, and transaction summaries

## Brand Column Logic

The parser implements intelligent Brand column population based on the following rules:

### For Interchange Charges
- Populates Brand column as usual (maintains existing logic)

### For Fees/Service Charges
- **MC** prefix → MASTERCARD
- **VI** prefix → VISA
- **DC** prefix → DISCOVER
- **AMEX/AMERICAN EXPRESS** prefix → American Express
- **DEBIT** prefix → DEBIT
- **No matching prefix** → Leave Brand column blank

## Prerequisites

- Python 3.8 or higher
- Llama Cloud API key
- Google API key (for Gemini AI)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hareemYashal/elizabeth-andrews-mike-kehoe.git
cd elizabeth-andrews-mike-kehoe
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory:
```env
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

1. Place your bank statement PDF in the `input_pdfs` folder
2. Rename it to `ALPH072025.pdf` (or update the filename in `parser.py`)
3. Run the parser:
```bash
python parser.py
```

4. The processed data will be saved to `output_excels/ALPH072025_tables.xlsx`

## API Keys Setup

### Llama Cloud API
1. Visit [Llama Cloud](https://cloud.llamaindex.ai/)
2. Sign up/Login to your account
3. Get your API key from the dashboard
4. Add it to your `.env` file

### Google API Key
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable the Generative AI API
4. Create credentials (API key)
5. Add it to your `.env` file

## Project Structure

```
elizabeth-andrews-mike-kehoe/
├── input_pdfs/           # Place your PDF files here
├── output_excels/        # Generated Excel files
├── parser.py             # Main parser script
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
├── .gitignore           # Git ignore file
└── README.md            # This file
```

## Output Format

The parser generates Excel files with the following columns:
- Account Name
- Account ID
- Statement Description
- Brand (populated based on logic)
- Count #
- Amount $
- Rate %
- Rate $
- Fees $
- Fee Type
- Date
- Account Holder
- Account Short ID
- Gateway
- Account Type
- Filename
- Processor
- User Permissions

## Error Handling

The parser includes comprehensive error handling for:
- Missing API keys
- Invalid PDF files
- Network connectivity issues
- Data extraction failures

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue in the GitHub repository
- Contact: [Your contact information]

## Changelog

### Version 1.1.0
- Added intelligent Brand column logic
- Improved error handling
- Enhanced documentation
- Added comprehensive logging

### Version 1.0.0
- Initial release
- Basic PDF parsing functionality
- Excel export capability