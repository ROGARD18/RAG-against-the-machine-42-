import fire
from pathlib import Path
from tqdm import tqdm
from typing import List

from src.models import (
    MinimalSource, MinimalAnswer, MinimalSearchResults,
    AnswerQuestion, UnansweredQuestion, RagDataset,
    StudentSearchResults, StudentSearchResultsAndAnswer
)
from src.indexer import get_source_files, chunk_markdown, chunk_python


class CliError(Exception):
    pass


class Cli:
    """Interface class"""

    def index(self, max_chunk_size: int = 2000) -> None:
        python_files, markdown_files = get_source_files()
        all_files = python_files + markdown_files

        chunks: List[MinimalSource]
        for file in tqdm(all_files):
            if file in markdown_files:
                chunk_markdown(file_path=file, max_chunk_size=max_chunk_size)
            elif file in python_files:
                chunk_python(file_path=file, max_chunk_size=max_chunk_size)

        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int) -> None:
        pass

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str) -> None:
        pass

    def answer(self, query: str, k: int) -> None:
        pass

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        pass

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        pass
