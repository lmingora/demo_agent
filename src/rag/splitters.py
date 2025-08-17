from __future__ import annotations
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

def make_splitter_for(path: Path, chunk_size=1200, overlap=200):
    ext = path.suffix.lower()
    if ext in {".md", ".markdown"}:
        return MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")]
        )
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
