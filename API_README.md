# Elizabeth Andrews Bank Statement Parser API

A superior REST API framework with built-in OpenAPI/Swagger documentation, async processing, and automatic data validation for parsing bank statements and extracting financial data.

## 🚀 Features

- **Superior REST API Framework**: Built with FastAPI for high performance and automatic validation
- **OpenAPI/Swagger Documentation**: Interactive API documentation with built-in testing interface
- **Async Processing**: Handles multiple requests concurrently with ThreadPoolExecutor
- **Comprehensive Validation**: Built-in file upload validation and error handling
- **AI-Powered Extraction**: Uses Llama Cloud API and Google Gemini AI for intelligent data extraction
- **Smart Brand Logic**: Automatically populates Brand column based on description patterns
- **Excel Export**: Generates structured Excel files with extracted data
- **CORS Support**: Cross-origin request handling for web applications
- **Health Monitoring**: Built-in health checks and status monitoring
- **Timeout Protection**: 10-minute timeout protection for long-running processes
- **Automatic Cleanup**: Automatic resource cleanup for scalable deployment

## 📋 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information and status |
| `POST` | `/upload` | Upload and process bank statement PDFs |
| `GET` | `/health` | Health check for monitoring |

### Job Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/job/{job_id}` | Get job status and progress |
| `GET` | `/download/{job_id}` | Download processed Excel file |

### Documentation

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | ReDoc API documentation |
| `GET` | `/openapi.json` | OpenAPI specification |

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Llama Cloud API key
- Google API key

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hareemYashal/elizabeth-andrews-mike-kehoe.git
   cd elizabeth-andrews-mike-kehoe
   ```

2. **Install dependencies**:
   ```bash
   pip install -r api_requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. **Start the API server**:
   ```bash
   python start_api.py
   ```

   Or run directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 📖 Usage

### 1. Start the API Server

```bash
python start_api.py
```

The API will be available at `http://localhost:8000`

### 2. Access Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3. Upload and Process a PDF

```bash
curl -X POST "http://localhost:8000/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_bank_statement.pdf"
```

### 4. Check Job Status

```bash
curl -X GET "http://localhost:8000/job/{job_id}" \
     -H "accept: application/json"
```

### 5. Download Results

```bash
curl -X GET "http://localhost:8000/download/{job_id}" \
     -H "accept: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" \
     --output processed_statement.xlsx
```

## 🧪 Testing

Run the test script to verify API functionality:

```bash
python test_api.py
```

## 📊 Response Examples

### Upload Response

```json
{
  "success": true,
  "message": "File uploaded successfully. Processing started.",
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "filename": "bank_statement.pdf",
  "processed_at": "2024-01-15T10:30:00Z"
}
```

### Job Status Response

```json
{
  "status": "completed",
  "filename": "bank_statement.pdf",
  "started_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:32:15Z",
  "progress": 100,
  "row_count": 45,
  "output_file": "/tmp/output/ALPH072025_tables_123e4567.xlsx"
}
```

### Health Check Response

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "uptime": "2:15:30"
}
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `LLAMA_CLOUD_API_KEY` | Llama Cloud API key for PDF parsing | Yes |
| `GOOGLE_API_KEY` | Google API key for Gemini AI | Yes |

### API Configuration

The API can be configured by modifying the FastAPI app settings in `main.py`:

- **Host**: `0.0.0.0` (all interfaces)
- **Port**: `8000`
- **Timeout**: `600` seconds (10 minutes)
- **Max Workers**: `4` (ThreadPoolExecutor)
- **File Size Limit**: `50MB`

## 🚀 Deployment

### Production Deployment

For production deployment, consider:

1. **Use a production ASGI server**:
   ```bash
   pip install gunicorn
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **Set up reverse proxy** (nginx/Apache)

3. **Configure environment variables** securely

4. **Set up monitoring** and logging

5. **Configure CORS** appropriately for your domain

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📈 Performance Features

- **Async Processing**: Non-blocking request handling
- **ThreadPoolExecutor**: Concurrent PDF processing
- **Automatic Cleanup**: Temporary file management
- **Timeout Protection**: Prevents hanging requests
- **Resource Management**: Memory and CPU optimization
- **Error Handling**: Comprehensive error responses

## 🔒 Security Features

- **File Validation**: PDF file type and size validation
- **CORS Configuration**: Configurable cross-origin policies
- **Error Sanitization**: Safe error message handling
- **Resource Limits**: File size and processing time limits
- **Input Validation**: Pydantic model validation

## 📝 API Specification

The complete API specification is available at:
- **OpenAPI JSON**: http://localhost:8000/openapi.json
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- Create an issue in the GitHub repository
- Check the API documentation at `/docs`
- Review the health status at `/health`

---

**Elizabeth Andrews Bank Statement Parser API** - Superior REST API framework with built-in OpenAPI/Swagger documentation, async processing, and automatic data validation.
