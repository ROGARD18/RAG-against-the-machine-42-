from pathlib import Path
from typing import List, Tuple
from src.models import MinimalSource
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, Language)


def get_source_files(raw_dir: str = "data/raw"
                     ) -> Tuple[List[Path], List[Path]]:
    base_path = Path(raw_dir)
    python_files: List[Path] = []
    markdown_files: List[Path] = []

    if not base_path.exists():
        return (python_files, markdown_files)

    for file_path in base_path.rglob("*"):
        if file_path.is_file():
            if file_path.suffix == '.md':
                markdown_files.append(file_path)
            elif file_path.suffix == '.py':
                python_files.append(file_path)

    return (python_files, markdown_files)


def chunk_markdown(file_path: Path, max_chunk_size: int
                   ) -> List[MinimalSource]:
    chunks: List[MinimalSource] = []

    text = file_path.read_text(encoding="utf-8")

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(text)
    chunk_overlap = 30
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size, chunk_overlap=chunk_overlap,
        add_start_index=True

    )
    documents = text_splitter.split_documents(md_header_splits)

    for doc in documents:
        start_idx: int = doc.metadata.get("start_index", 0)
        end_idx: int = start_idx + len(doc.page_content)

        source = MinimalSource(
            file_path=str(file_path),
            first_character_index=start_idx,
            last_character_index=end_idx
        )
        chunks.append(source)

    return chunks


def chunk_python(file_path: Path, max_chunk_size: int) -> List[MinimalSource]:
    chunks: List[MinimalSource] = []
    text = file_path.read_text()

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=max_chunk_size,
        chunk_overlap=30,
        add_start_index=True
    )

    documents = python_splitter.create_documents([text])

    for doc in documents:
        start_idx: int = doc.metadata.get("start_index", 0)
        end_idx: int = start_idx + len(doc.page_content)

        source = MinimalSource(
            file_path=str(file_path),
            first_character_index=start_idx,
            last_character_index=end_idx
        )
        chunks.append(source)

    return chunks
