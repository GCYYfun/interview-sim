import traceback
import os
import json
import pdfplumber
import re
from typing import Optional, Dict, Any, Union

from menglong.models import Model
from menglong.schemas.chat import User, DocumentPart, TextPart

import base64
import httpx

class FileParser:
    """
    Component for parsing various file formats (txt, json, pdf).
    """

    @staticmethod
    def read_file(file_path: str) -> Union[str, Dict[str, Any], None]:
        """
        Reads a file and returns its content based on the extension.
        
        Args:
            file_path: Absolute or relative path to the file.
            
        Returns:
            - str: For .txt and .pdf files (text content).
            - dict/list: For .json files (parsed JSON).
            - None: If file does not exist or format is unsupported.
        """
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return None

        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == '.txt':
                return FileParser._read_txt(file_path)
            elif ext == '.json':
                return FileParser._read_json(file_path)
            elif ext == '.pdf':
                return FileParser._read_pdf(file_path)
            elif ext == '.md':
                return FileParser._read_markdown(file_path)
            else:
                print(f"Error: Unsupported file extension {ext}")
                return None
        except Exception as e:
            traceback.print_exc()
            print(f"Error reading file {file_path}: {e}")
            return None

    @staticmethod
    def _read_txt(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def _read_markdown(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def _read_json(file_path: str) -> Union[Dict, list]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def _read_pdf(file_path: str) -> str:
        if "transcript" in file_path:
            return FileParser._read_pdf_transcript(file_path)
        elif "jd" in file_path:
            return FileParser._read_pdf_jd(file_path)
        elif "resume" in file_path:
            return FileParser._read_pdf_resume(file_path)
        else:
            raise ValueError("未预期的PDF,请遵循命名规范,文件名携带 transcript, jd or resume")
        
    @staticmethod
    def _read_pdf_transcript(file_path):
        text_content = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_content.append(text)
        raw_text = "\n".join(text_content)
        return FileParser._clean_transcript(raw_text)
    
    @staticmethod
    def _read_pdf_jd(file_path):
        pass

    @staticmethod
    def _read_pdf_resume(file_path):


        pdf_data = None
        # First, load and encode the PDF 
        # pdf_url = "https://assets.anthropic.com/m/1cd9d098ac3e6467/original/Claude-3-Model-Card-October-Addendum.pdf"
        # pdf_data = base64.standard_b64encode(httpx.get(pdf_url).content).decode("utf-8")

        # Alternative: Load from a local file
        with open(file_path, "rb") as f:
            pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")

        # Send to Claude using base64 encoding

        client = Model()
        
        # Construct message using schemas
        messages = [
            User([
                DocumentPart(data=pdf_data, media_type="application/pdf"),
                TextPart(text="解析resume文件内容。并markdown格式输出。")
            ])
        ]
        
        response = client.chat(messages=messages)
        content = response.text

        # save
        with open(f"{file_path[:-4]}.md", "w") as f:
            f.write(content)

        return content
        


    @staticmethod
    def _clean_transcript(text: str) -> str:
        """
        Cleans the transcript text.
        Merges lines that do not start with a speaker timestamp pattern into the previous line.
        Ensures strict "Name (Time): Content" format per line.
        """
        import re
        
        # Pattern to match the start of a speaker line: "Name (Time):"
        # Supports:
        # - Brackets: (), （）, [], 【】
        # - Time: MM:SS, HH:MM:SS, H:MM:SS
        # - Separators: :, ：
        # regex: Start -> non-greedy text -> open bracket -> time -> close bracket -> colon
        header_pattern = re.compile(r'^.*?[\(\[\（【]\s*\d{1,2}:\d{2}(?::\d{2})?\s*[\)\]\）】][:：]')
        
        lines = text.split('\n')
        merged_lines = []
        current_line = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if header_pattern.match(line):
                if current_line:
                    merged_lines.append(current_line)
                current_line = line
            else:
                # This is a continuation of the previous line
                if current_line:
                    current_line += " " + line
                else:
                    # Case where text starts without a header, treat as new line
                    current_line = line
                    
        if current_line:
            merged_lines.append(current_line)
            
        return "\n".join(merged_lines)
