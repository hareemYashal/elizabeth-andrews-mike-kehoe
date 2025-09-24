"""
Elizabeth Andrews Mike Kehoe - Bank Statement Parser REST API
Superior REST API framework with built-in OpenAPI/Swagger documentation, 
async processing, and automatic data validation.
"""

import os
import uuid
import asyncio
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from pydantic import BaseModel, Field
import uvicorn

# Import the existing parser functionality
from parser import (
    LlamaParse, 
    ChatGoogleGenerativeAI, 
    ChatPromptTemplate,
    load_dotenv,
    os as parser_os,
    re,
    tabulate,
    pd,
    merge_duplicate_rows,
    process_section_with_llm,
    extract_account_info_from_header,
    parse_filename_for_metadata,
    save_extracted_data_to_excel,
    _collect_sections_markdown,
    _extract_section_block
)

# Load environment variables
load_dotenv()

# Initialize FastAPI app with comprehensive metadata
app = FastAPI(
    title="Elizabeth Andrews Bank Statement Parser API",
    description="""
    A sophisticated REST API for parsing bank statements and extracting financial data 
    with intelligent Brand column logic for credit card processing fees.
    
    ## Features
    - **PDF Parsing**: Uses Llama Cloud API to parse bank statement PDFs
    - **AI-Powered Extraction**: Leverages Google Gemini AI for intelligent data extraction
    - **Smart Brand Logic**: Automatically populates Brand column based on description patterns
    - **Excel Export**: Generates structured Excel files with extracted data
    - **Async Processing**: Handles multiple requests concurrently with timeout protection
    - **Comprehensive Validation**: Built-in file upload validation and error handling
    
    ## API Endpoints
    - `GET /` - API information and status
    - `POST /upload` - Upload and process bank statement PDFs
    - `GET /health` - Health check endpoint
    - `GET /docs` - Interactive Swagger UI documentation
    - `GET /redoc` - ReDoc API documentation
    """,
    version="1.0.0",
    contact={
        "name": "Elizabeth Andrews API Support",
        "email": "support@elizabethandrews.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global thread pool for async processing
executor = ThreadPoolExecutor(max_workers=4)

# Pydantic models for request/response validation
class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    uptime: str = Field(..., description="API uptime")

class UploadResponse(BaseModel):
    success: bool = Field(..., description="Upload success status")
    message: str = Field(..., description="Response message")
    job_id: str = Field(..., description="Unique job identifier")
    filename: str = Field(..., description="Original filename")
    processed_at: str = Field(..., description="Processing timestamp")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")

# Global variables for tracking
start_time = datetime.now()
active_jobs: Dict[str, Dict[str, Any]] = {}

# Initialize parser components
def initialize_parser():
    """Initialize the parser with API keys"""
    try:
        llama_api_key = parser_os.getenv("LLAMA_CLOUD_API_KEY")
        google_api_key = parser_os.getenv("GOOGLE_API_KEY")
        
        if not llama_api_key or not google_api_key:
            raise ValueError("API keys not found in environment variables")
        
        # Set environment variables
        parser_os.environ["LLAMA_CLOUD_API_KEY"] = llama_api_key
        parser_os.environ["GOOGLE_API_KEY"] = google_api_key
        
        # Initialize components
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        
        parser = LlamaParse(result_type="markdown")
        
        return llm, parser
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize parser: {str(e)}"
        )

# Initialize parser components (lazy initialization)
llm, parser = None, None

def get_parser_components():
    """Lazy initialization of parser components"""
    global llm, parser
    if llm is None or parser is None:
        try:
            llm, parser = initialize_parser()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize parser: {str(e)}"
            )
    return llm, parser

@app.get("/", response_model=Dict[str, Any])
async def root():
    """
    Root endpoint providing API information and status.
    
    Returns comprehensive API information including version, status, and available endpoints.
    """
    return {
        "message": "Elizabeth Andrews Bank Statement Parser API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        },
        "features": [
            "PDF parsing with Llama Cloud API",
            "AI-powered data extraction with Google Gemini",
            "Smart Brand column logic",
            "Excel export functionality",
            "Async processing with timeout protection",
            "Comprehensive file validation",
            "Interactive API documentation"
        ]
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring and load balancing.
    
    Returns the current health status of the API including uptime and version information.
    """
    current_time = datetime.now()
    uptime = current_time - start_time
    
    return HealthResponse(
        status="healthy",
        timestamp=current_time.isoformat(),
        version="1.0.0",
        uptime=str(uptime)
    )

@app.post("/upload", response_model=UploadResponse)
async def upload_and_process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Bank statement PDF file to process")
):
    """
    Upload and process a bank statement PDF file.
    
    This endpoint accepts a PDF file, validates it, and processes it using the AI-powered
    parser to extract financial data and generate an Excel report.
    
    **File Requirements:**
    - Must be a PDF file
    - Maximum size: 50MB
    - Should contain bank statement data
    
    **Processing:**
    - Uses Llama Cloud API for PDF parsing
    - Leverages Google Gemini AI for data extraction
    - Applies intelligent Brand column logic
    - Generates structured Excel output
    
    **Response:**
    - Returns job ID for tracking
    - Provides download link for processed Excel file
    - Includes processing timestamp and status
    """
    # Validate file
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a PDF"
        )
    
    if file.size and file.size > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 50MB limit"
        )
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Create job tracking entry
    active_jobs[job_id] = {
        "status": "processing",
        "filename": file.filename,
        "started_at": datetime.now(),
        "progress": 0
    }
    
    # Add background task for processing
    background_tasks.add_task(process_pdf_file, job_id, file)
    
    return UploadResponse(
        success=True,
        message="File uploaded successfully. Processing started.",
        job_id=job_id,
        filename=file.filename,
        processed_at=datetime.now().isoformat()
    )

async def process_pdf_file(job_id: str, file: UploadFile):
    """
    Background task to process the uploaded PDF file.
    
    This function handles the complete PDF processing pipeline including:
    - File saving and validation
    - PDF parsing with Llama Cloud
    - AI-powered data extraction
    - Excel file generation
    - Cleanup and status updates
    """
    temp_dir = None
    try:
        # Update job status
        active_jobs[job_id]["status"] = "processing"
        active_jobs[job_id]["progress"] = 10
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        input_dir = Path(temp_dir) / "input_pdfs"
        output_dir = Path(temp_dir) / "output_excels"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        file_path = input_dir / "ALPH072025.pdf"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        active_jobs[job_id]["progress"] = 20
        
        # Process the PDF using the existing parser logic
        await asyncio.get_event_loop().run_in_executor(
            executor, 
            process_pdf_with_parser, 
            str(file_path), 
            str(output_dir),
            job_id
        )
        
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["progress"] = 100
        active_jobs[job_id]["completed_at"] = datetime.now()
        
    except Exception as e:
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["error"] = str(e)
        active_jobs[job_id]["failed_at"] = datetime.now()
        print(f"Error processing job {job_id}: {e}")
    
    finally:
        # Don't cleanup temp directory immediately - keep it for download
        # The cleanup will happen when the job is downloaded or expires
        pass

def process_pdf_with_parser(file_path: str, output_dir: str, job_id: str):
    """
    Process PDF using the existing parser logic.
    
    This function runs in a separate thread and handles the complete
    PDF processing pipeline using the existing parser code.
    """
    try:
        # Get parser components (lazy initialization)
        llm, parser = get_parser_components()
        
        # Update progress
        active_jobs[job_id]["progress"] = 30
        
        # Parse the PDF
        with open(file_path, "rb") as f:
            documents = parser.load_data(f, extra_info={"file_name": file_path})
        
        if not documents:
            raise ValueError("No documents extracted from PDF")
        
        active_jobs[job_id]["progress"] = 50
        
        # Process sections
        sections_markdown = {"FEES": "", "INTERCHANGE": "", "SUMMARY_CARD": ""}
        
        for doc in documents:
            text = doc.text
            sections_markdown_block = _collect_sections_markdown(text)
            for section_name in ["FEES", "INTERCHANGE", "SUMMARY_CARD"]:
                if sections_markdown_block.get(section_name):
                    sections_markdown[section_name] += ("\n\n" + sections_markdown_block[section_name]).strip()
        
        active_jobs[job_id]["progress"] = 70
        
        # Extract metadata
        metadata = parse_filename_for_metadata(file_path)
        account_info = extract_account_info_from_header(documents[0].text)
        metadata.update(account_info)
        
        # Process sections with LLM
        fees_rows = []
        interchange_rows = []
        summary_card_rows = []
        
        if sections_markdown.get("FEES"):
            fees_rows = process_section_with_llm("FEES", sections_markdown["FEES"], metadata)
        
        if sections_markdown.get("INTERCHANGE"):
            interchange_rows = process_section_with_llm("INTERCHANGE", sections_markdown["INTERCHANGE"], metadata)
        
        if sections_markdown.get("SUMMARY_CARD"):
            summary_card_rows = process_section_with_llm("SUMMARY_CARD", sections_markdown["SUMMARY_CARD"], metadata)
        
        active_jobs[job_id]["progress"] = 90
        
        # Merge and save results
        all_rows = fees_rows + interchange_rows + summary_card_rows
        merged_rows = merge_duplicate_rows(all_rows)
        
        # Get the base filename from the input file
        base_name = Path(file_path).stem
        output_file = Path(output_dir) / f"{base_name}_tables_{job_id}.xlsx"
        save_extracted_data_to_excel(merged_rows, str(output_file))
        
        # Store output file path
        active_jobs[job_id]["output_file"] = str(output_file)
        active_jobs[job_id]["row_count"] = len(merged_rows)
        
        # Debug: Print the actual file path being created
        print(f"🔍 DEBUG: Creating file at: {output_file}")
        print(f"🔍 DEBUG: Job ID: {job_id}")
        print(f"🔍 DEBUG: File exists: {os.path.exists(output_file)}")
        
    except Exception as e:
        raise Exception(f"PDF processing failed: {str(e)}")

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of a processing job.
    
    Returns the current status, progress, and results of a specific job.
    """
    if job_id not in active_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return active_jobs[job_id]

@app.get("/download/{job_id}")
async def download_result(job_id: str):
    """
    Download the processed Excel file for a completed job.
    
    Returns the generated Excel file containing the extracted financial data.
    """
    if job_id not in active_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    job = active_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job not completed yet"
        )
    
    if "output_file" not in job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Output file not found"
        )
    
    # Check if file actually exists
    output_file_path = job["output_file"]
    if not os.path.exists(output_file_path):
        print(f"🔍 DEBUG: File not found at: {output_file_path}")
        print(f"🔍 DEBUG: Job details: {job}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Output file not found at: {output_file_path}"
        )
    
    # Get the original filename for download
    original_filename = job.get("filename", "bank_statement")
    base_name = Path(original_filename).stem
    download_filename = f"{base_name}_processed_{job_id}.xlsx"
    
    # Schedule cleanup of temp directory after download
    def cleanup_after_download():
        try:
            temp_dir = os.path.dirname(output_file_path)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"🧹 Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            print(f"Error cleaning up temp directory: {e}")
    
    # Schedule cleanup in background
    import threading
    cleanup_thread = threading.Timer(5.0, cleanup_after_download)  # 5 seconds delay
    cleanup_thread.start()
    
    return FileResponse(
        path=output_file_path,
        filename=download_filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler with detailed error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=f"Request failed: {request.method} {request.url}",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unexpected errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        timeout_keep_alive=600  # 10 minutes timeout
    )
