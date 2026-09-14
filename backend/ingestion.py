import re
import hashlib
from typing import List, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field

class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    version: str
    section_title: str
    section_order: int
    source_file: str
    content_hash: str
    text: str

MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12"
}

def normalize_date(date_str: str) -> str:
    # E.g. "March 2025" -> "2025-03"
    date_str = date_str.strip().lower()
    parts = date_str.split()
    if len(parts) == 2 and parts[0] in MONTHS and parts[1].isdigit():
        return f"{parts[1]}-{MONTHS[parts[0]]}"
    return date_str

def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s]+', '-', text).strip('-')
    return text

def normalize_text(text: str) -> str:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def parse_markdown_file(file_path: str | Path) -> List[ChunkRecord]:
    path = Path(file_path)
    content = path.read_text(encoding='utf-8')
    
    lines = content.split('\n')
    
    document_title = None
    document_id = None
    version = None
    
    sections = []
    current_section_title = None
    current_section_lines = []
    
    for line in lines:
        if line.startswith('# ') and not document_title:
            document_title = line[2:].strip()
            continue
            
        doc_id_match = re.match(r'^\*\*Document ID:\*\*\s*(.+)', line)
        if doc_id_match:
            document_id = doc_id_match.group(1).strip()
            continue
            
        updated_match = re.match(r'^\*\*Last updated:\*\*\s*(.+)', line)
        if updated_match:
            version = normalize_date(updated_match.group(1))
            continue
            
        if line.startswith('## '):
            if current_section_title is not None:
                sections.append((current_section_title, '\n'.join(current_section_lines)))
            current_section_title = line[3:].strip()
            current_section_lines = []
            continue
            
        if current_section_title is not None:
            current_section_lines.append(line)
            
    if current_section_title is not None:
        sections.append((current_section_title, '\n'.join(current_section_lines)))
        
    if not document_title:
        raise ValueError(f"No document title found in {path.name}")
    if not document_id:
        raise ValueError(f"No document ID found in {path.name}")
    if not version:
        raise ValueError(f"No last updated date found in {path.name}")
    if not sections:
        raise ValueError(f"No sections found in {path.name}")
        
    chunks = []
    for i, (sec_title, sec_content) in enumerate(sections, 1):
        norm_text = normalize_text(sec_content)
        if not norm_text:
            raise ValueError(f"Empty section content in {path.name}: {sec_title}")
            
        slug = slugify(sec_title)
        chunk_id = f"{document_id}@{version}::{slug}"
        content_hash = "sha256:" + hashlib.sha256(norm_text.encode('utf-8')).hexdigest()
        
        chunk = ChunkRecord(
            chunk_id=chunk_id,
            document_id=document_id,
            document_title=document_title,
            version=version,
            section_title=sec_title,
            section_order=i,
            source_file=path.name,
            content_hash=content_hash,
            text=norm_text
        )
        chunks.append(chunk)
        
    return chunks

def ingest_directory(directory: str | Path) -> List[ChunkRecord]:
    path = Path(directory)
    all_chunks = []
    for md_file in sorted(path.glob("*.md")):  # sorted for deterministic ordering
        all_chunks.extend(parse_markdown_file(md_file))
    return all_chunks
