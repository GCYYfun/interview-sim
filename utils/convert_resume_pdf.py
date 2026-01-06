"""
读取简历 PDF 文件
"""
import argparse
import traceback
import os
import json
import datetime
import sys
from pathlib import Path
from components.file_parser import FileParser

def load_transcript(transcript_name, resource_path):
    """Loads transcript from PDF or falls back to text."""
    # 获取项目跟目录
    resource_path = resource_path


    pdf_file = os.path.join(resource_path, f"{transcript_name}.pdf")
    parser = FileParser()
    
    if os.path.exists(pdf_file):
        print(f"Reading {pdf_file}...")
        return parser.read_file(pdf_file)
    
    # Fallback checks (legacy support or specific dev cases)
    txt_path = os.path.join(resource_path, f"{transcript_name}.txt")
    if os.path.exists(txt_path):
        print(f"Reading {txt_path}...")
        return parser.read_file(txt_path)
        
    if os.path.exists("test.txt"):
        print("Falling back to test.txt...")
        return parser.read_file("test.txt")
        
    raise FileNotFoundError(f"Resume not found for {transcript_name} in {resource_path}")


t = load_transcript("xxx_resume","data/resources/candidate_resumes")
print(t)