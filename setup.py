#!/usr/bin/env python3
"""
Setup script for Elizabeth Andrews Bank Statement Parser API
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="elizabeth-andrews-bank-parser",
    version="1.0.0",
    author="Elizabeth Andrews",
    author_email="support@example.com",
    description="AI-powered bank statement parser with REST API",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/elizabeth-andrews-bank-parser",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "black>=23.11.0",
            "flake8>=6.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bank-parser-api=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="bank statement parser pdf ai financial data extraction",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/elizabeth-andrews-bank-parser/issues",
        "Source": "https://github.com/yourusername/elizabeth-andrews-bank-parser",
        "Documentation": "https://github.com/yourusername/elizabeth-andrews-bank-parser#readme",
    },
)
