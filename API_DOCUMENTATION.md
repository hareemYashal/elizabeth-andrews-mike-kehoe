# API Documentation

## Overview

The Elizabeth Andrews Bank Statement Parser API provides a comprehensive solution for processing bank statement PDFs and extracting structured financial data. The API is built with FastAPI and provides both REST endpoints and interactive documentation.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. However, it requires valid API keys for Llama Cloud and Google Gemini AI to be configured in the environment variables.

## Endpoints

### 1. Root Endpoint

**GET** `/`

Returns basic API information and status.

#### Response

```json
{
  "message": "Elizabeth Andrews Bank Statement Parser API",
  "version": "1.0.0",
  "status": "running",
  "docs_url": "/docs",
  "redoc_url": "/redoc"
}
```

### 2. Health Check

**GET** `/health`

Returns the current health status of the API.

#### Response

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "version": "1.0.0",
  "uptime": "0:05:23"
}
```

### 3. Upload and Process PDF

**POST** `/upload`

Uploads a PDF file for processing.

#### Request

- **Content-Type**: `multipart/form-data`
- **Body**: 
  - `file` (required): PDF file to process

#### Response

```json
{
  "success": true,
  "message": "File uploaded successfully. Processing started.",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "bank_statement.pdf",
  "processed_at": "2024-01-15T10:30:00.000Z"
}
```

#### Error Responses

- **400 Bad Request**: Invalid file format or size
- **422 Unprocessable Entity**: Validation errors
- **500 Internal Server Error**: Processing initialization failed

### 4. Check Job Status

**GET** `/job/{job_id}`

Returns the current status of a processing job.

#### Parameters

- `job_id` (string, required): Unique job identifier

#### Response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "filename": "bank_statement.pdf",
  "created_at": "2024-01-15T10:30:00.000Z",
  "completed_at": "2024-01-15T10:32:15.000Z",
  "row_count": 130,
  "output_file": "/tmp/output/bank_statement_tables_550e8400.xlsx"
}
```

#### Job Statuses

- `pending`: Job is queued for processing
- `processing`: Job is currently being processed
- `completed`: Job completed successfully
- `failed`: Job failed with an error

#### Error Responses

- **404 Not Found**: Job ID not found

### 5. Download Processed File

**GET** `/download/{job_id}`

Downloads the processed Excel file for a completed job.

#### Parameters

- `job_id` (string, required): Unique job identifier

#### Response

- **Content-Type**: `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`
- **Content-Disposition**: `attachment; filename="bank_statement_processed_550e8400.xlsx"`

#### Error Responses

- **400 Bad Request**: Job not completed yet
- **404 Not Found**: Job or file not found

## Data Models

### Upload Response

```json
{
  "success": boolean,
  "message": string,
  "job_id": string,
  "filename": string,
  "processed_at": string
}
```

### Job Status

```json
{
  "job_id": string,
  "status": "pending" | "processing" | "completed" | "failed",
  "progress": number (0-100),
  "filename": string,
  "created_at": string,
  "completed_at": string | null,
  "failed_at": string | null,
  "error": string | null,
  "row_count": number | null,
  "output_file": string | null
}
```

### Error Response

```json
{
  "error": string,
  "detail": string | null,
  "timestamp": string
}
```

## Processing Details

### Supported File Types

- **PDF**: Bank statement PDFs only
- **Maximum file size**: 50MB
- **Minimum file size**: 1KB

### Data Extraction

The API extracts the following data from bank statements:

#### Interchange Charges
- Mastercard fees and rates
- Visa fees and rates
- Discover fees and rates
- American Express fees and rates
- Transaction counts and amounts

#### Service Charges
- Network processing fees
- Authorization fees
- Assessment fees
- Sales discounts

#### Transaction Data
- Gross sales by card type
- Refunds and adjustments
- Account information
- Statement metadata

### Output Format

The processed data is exported to an Excel file with the following structure:

- **Sheet 1**: All extracted data with proper categorization
- **Columns**: Account Name, Statement Description, Brand, Count #, Amount $, Rate %, Rate $, Fees $, Fee Type, Date, Account Holder, etc.
- **Data Types**: Properly formatted numbers, dates, and text
- **Validation**: All data is validated and cleaned

## Error Handling

### HTTP Status Codes

- **200 OK**: Request successful
- **400 Bad Request**: Invalid request parameters
- **404 Not Found**: Resource not found
- **422 Unprocessable Entity**: Validation errors
- **500 Internal Server Error**: Server error

### Error Response Format

All errors follow a consistent format:

```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## Rate Limiting

Currently, there are no rate limits implemented. However, the API processes files sequentially to prevent resource exhaustion.

## CORS

Cross-Origin Resource Sharing (CORS) is enabled for all origins. This can be configured in the `main.py` file if needed.

## Interactive Documentation

The API provides interactive documentation through:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Examples

### Complete Workflow

1. **Upload a PDF**:
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@bank_statement.pdf"
```

2. **Check job status**:
```bash
curl "http://localhost:8000/job/550e8400-e29b-41d4-a716-446655440000"
```

3. **Download results**:
```bash
curl "http://localhost:8000/download/550e8400-e29b-41d4-a716-446655440000" \
  -o "processed_statement.xlsx"
```

### Python Example

```python
import requests
import time

# Upload file
with open('bank_statement.pdf', 'rb') as f:
    response = requests.post('http://localhost:8000/upload', files={'file': f})
    job_data = response.json()
    job_id = job_data['job_id']

# Wait for processing
while True:
    status_response = requests.get(f'http://localhost:8000/job/{job_id}')
    status_data = status_response.json()
    
    if status_data['status'] == 'completed':
        break
    elif status_data['status'] == 'failed':
        print(f"Processing failed: {status_data.get('error')}")
        break
    
    time.sleep(5)  # Wait 5 seconds before checking again

# Download results
download_response = requests.get(f'http://localhost:8000/download/{job_id}')
with open('processed_statement.xlsx', 'wb') as f:
    f.write(download_response.content)
```

## Troubleshooting

### Common Issues

1. **File not found error**: Ensure the job is completed before downloading
2. **Processing timeout**: Large files may take longer to process
3. **API key errors**: Verify your environment variables are set correctly
4. **Memory issues**: Ensure sufficient RAM for processing large PDFs

### Debug Mode

Enable debug logging by setting the log level to DEBUG in the uvicorn configuration:

```bash
uvicorn main:app --log-level debug
```

## Support

For technical support or questions:

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/elizabeth-andrews-bank-parser/issues)
- **Documentation**: Check the interactive docs at `/docs`
- **Email**: support@example.com
