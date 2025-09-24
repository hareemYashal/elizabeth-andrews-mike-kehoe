#!/usr/bin/env python3
"""
Startup script for Elizabeth Andrews Bank Statement Parser API
"""

import uvicorn
import os
from pathlib import Path

if __name__ == "__main__":
    # Ensure we're in the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("🚀 Starting Elizabeth Andrews Bank Statement Parser API...")
    print("📚 API Documentation will be available at:")
    print("   - Swagger UI: http://localhost:8000/docs")
    print("   - ReDoc: http://localhost:8000/redoc")
    print("   - OpenAPI JSON: http://localhost:8000/openapi.json")
    print("\n🔧 API Endpoints:")
    print("   - GET  /          - API information")
    print("   - POST /upload    - Upload and process PDF files")
    print("   - GET  /health    - Health check")
    print("   - GET  /job/{id}  - Check job status")
    print("   - GET  /download/{id} - Download processed Excel file")
    print("\n⚡ Starting server...")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        timeout_keep_alive=600  # 10 minutes timeout
    )
