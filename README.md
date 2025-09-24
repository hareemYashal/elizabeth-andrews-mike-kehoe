# Elizabeth Andrews Bank Statement Parser API

A powerful REST API for parsing bank statement PDFs and extracting financial data using AI-powered processing. This project provides automated extraction of interchange charges, fees, and transaction data from bank statements with comprehensive validation and Excel export capabilities.

## 🚀 Features

- **AI-Powered Processing**: Uses LlamaParse and Google Gemini AI for intelligent document parsing
- **REST API**: FastAPI-based API with automatic OpenAPI/Swagger documentation
- **Async Processing**: Background job processing with real-time status tracking
- **Comprehensive Validation**: Built-in file validation and data integrity checks
- **Excel Export**: Automatic generation of structured Excel reports
- **Interactive Documentation**: Swagger UI and ReDoc for easy API testing
- **CORS Support**: Cross-origin resource sharing enabled
- **Error Handling**: Comprehensive error handling with detailed responses
- **Health Monitoring**: Built-in health checks and monitoring endpoints

## 📋 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information and status |
| `POST` | `/upload` | Upload and process PDF files |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/job/{job_id}` | Check job processing status |
| `GET` | `/download/{job_id}` | Download processed Excel file |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | ReDoc documentation |

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- API keys for Llama Cloud and Google Gemini AI

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/elizabeth-andrews-bank-parser.git
cd elizabeth-andrews-bank-parser
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```env
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

**Get API Keys:**
- **Llama Cloud**: Sign up at [LlamaIndex](https://cloud.llamaindex.ai/)
- **Google Gemini AI**: Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### 5. Run the Application

#### Option 1: Clean Startup (Recommended)
```bash
python start_api_clean.py
```

#### Option 2: Direct Uvicorn
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 📖 Usage

### 1. Access the API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 2. Upload and Process PDFs

1. **Upload a PDF** using the `/upload` endpoint
2. **Check job status** using the `/job/{job_id}` endpoint
3. **Download results** using the `/download/{job_id}` endpoint

### 3. Example API Calls

#### Upload a PDF
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_bank_statement.pdf"
```

#### Check Job Status
```bash
curl "http://localhost:8000/job/{job_id}"
```

#### Download Processed File
```bash
curl "http://localhost:8000/download/{job_id}" \
  -o "processed_statement.xlsx"
```

## 📊 Data Processing

The API extracts and processes the following data from bank statements:

### Interchange Charges
- Mastercard, Visa, Discover, American Express fees
- Rate percentages and fixed amounts
- Transaction counts and total amounts

### Service Charges
- Network fees and processing charges
- Authorization and connectivity fees
- Assessment fees and discounts

### Transaction Data
- Gross sales and refunds
- Card type summaries
- Account information

### Output Format
- **Excel file** with multiple sheets
- **Structured data** with proper categorization
- **Brand classification** for each fee type
- **Rate and amount validation**

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `LLAMA_CLOUD_API_KEY` | Llama Cloud API key for PDF parsing | Yes |
| `GOOGLE_API_KEY` | Google Gemini AI API key | Yes |

### API Configuration

The API can be configured by modifying `main.py`:

- **Port**: Default 8000
- **Host**: Default 0.0.0.0 (all interfaces)
- **Timeout**: 10 minutes for processing
- **Max file size**: 50MB
- **Supported formats**: PDF only

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8000/health
```

### Test File Upload
```bash
python test_api.py
```

## 📁 Project Structure

```
elizabeth-andrews-bank-parser/
├── main.py                 # FastAPI application
├── parser.py              # Core parsing logic
├── start_api_clean.py     # Clean startup script
├── test_api.py            # API testing script
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
├── README.md             # This file
├── API_README.md         # Detailed API documentation
└── docs/                 # Additional documentation
```

## 🚨 Error Handling

The API provides comprehensive error handling:

- **400 Bad Request**: Invalid file format or request
- **404 Not Found**: Job or file not found
- **422 Unprocessable Entity**: Validation errors
- **500 Internal Server Error**: Processing failures

All errors include detailed messages and timestamps.

## 🔒 Security

- **API Key Protection**: Environment variables for sensitive data
- **File Validation**: Strict file type and size validation
- **CORS Configuration**: Configurable cross-origin policies
- **Input Sanitization**: All inputs are validated and sanitized

## 📈 Performance

- **Async Processing**: Non-blocking background jobs
- **Thread Pool**: Configurable worker threads
- **Memory Management**: Automatic cleanup of temporary files
- **Timeout Protection**: 10-minute processing timeout

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:

- **Issues**: [GitHub Issues](https://github.com/yourusername/elizabeth-andrews-bank-parser/issues)
- **Documentation**: [API Documentation](http://localhost:8000/docs)
- **Email**: support@example.com

## 🙏 Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) for PDF parsing capabilities
- [Google Gemini AI](https://ai.google.dev/) for intelligent data extraction
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Pandas](https://pandas.pydata.org/) for data processing

## 📊 Changelog

### Version 1.0.0
- Initial release
- AI-powered PDF parsing
- REST API with FastAPI
- Excel export functionality
- Comprehensive error handling
- Interactive documentation

---

**Made with ❤️ for financial data processing**