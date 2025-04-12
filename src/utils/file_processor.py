import io
import os
from typing import Union
from docx import Document
import fitz  # PyMuPDF
import pandas as pd
import camelot
from src.core.core import track_performance, logger

@track_performance
def process_file(file_path: str) -> list:
    """Process different file formats and extract text with tables"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    text_content = []
    
    try:
        if file_path.endswith('.docx'):
            return process_docx(file_path)
        elif file_path.endswith('.pdf'):
            return process_pdf(file_path)
        elif file_path.endswith(('.csv', '.xlsx', '.xls', '.xlsm')):
            return process_spreadsheet(file_path)
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return []

def process_docx(file_path: str) -> list:
    doc = Document(file_path)
    content = []
    
    # Extract paragraphs
    for para in doc.paragraphs:
        if para.text.strip():
            content.append(para.text)
    
    # Extract tables
    for table in doc.tables:
        table_data = []
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            table_data.append(" | ".join(row_data))
        content.append("\n".join(table_data))
    
    return content

def process_pdf(file_path: str) -> list:
    content = []
    with fitz.open(file_path) as doc:
        for page_num, page in enumerate(doc):
            # Extract text
            text = page.get_text()
            if text.strip():
                content.append((page_num + 1, text))
            
            # Extract tables
            tables = camelot.read_pdf(file_path, pages=str(page_num + 1))
            for table in tables:
                content.append((page_num + 1, table.df.to_csv(index=False)))
    
    return content

def process_spreadsheet(file_path: str) -> list:
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path, engine='openpyxl')
    
    output = io.StringIO()
    df.to_csv(output, index=False)
    return [output.getvalue()]