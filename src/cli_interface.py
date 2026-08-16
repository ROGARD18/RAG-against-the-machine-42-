import fire
import pickle
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from pathlib import Path
from tqdm import tqdm
from typing import List

from src.models import (
    MinimalSource, MinimalAnswer, MinimalSearchResults,
    AnswerQuestion, UnansweredQuestion, RagDataset,
    StudentSearchResults, StudentSearchResultsAndAnswer
)
from src.indexer_utils import get_source_files, chunk_markdown, chunk_python


class CliError(Exception):
    pass


class Cli:
    """Interface class"""

    def index(self, max_chunk_size: int = 2000) -> None:
        python_files, markdown_files = get_source_files()
        all_files = python_files + markdown_files

        all_chunks: List[MinimalSource] = []

        for file in tqdm(all_files, desc="Chunking"):
            if file in markdown_files:
                all_chunks.extend(chunk_markdown(file_path=file, max_chunk_size=max_chunk_size))
            elif file in python_files:
                all_chunks.extend(chunk_python(file_path=file, max_chunk_size=max_chunk_size))

        tokenized = []

        for chunk in tqdm(all_chunks, desc="Tokenizing"):
            file_text = Path(chunk.file_path).read_text()
            chunk_text = file_text[chunk.first_character_index:chunk.last_character_index]

            tokens = chunk_text.lower().split()
            tokenized.append(tokens)

        bm25 = BM25Okapi(tokenized)

        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)

        index_path = processed_dir / "index.pkl"
        with open(index_path, "wb") as f:
            pickle.dump({
                "chunks": all_chunks,
                "bm25": bm25
            }, f)

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
