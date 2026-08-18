import fire
import json
import pickle
from src.llm_model import LlmModel
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
import re

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

    def __init__(self):
        self.generator = LlmModel()

    def index(self, max_chunk_size: int = 2000) -> None:
        python_files, markdown_files = get_source_files()
        all_files = python_files + markdown_files

        all_chunks: List[MinimalSource] = []

        for file in tqdm(all_files, desc="Chunking"):
            if file in markdown_files:
                all_chunks.extend(chunk_markdown(file_path=file, max_chunk_size=max_chunk_size))
            elif file in python_files:
                all_chunks.extend(chunk_python(file_path=file, max_chunk_size=max_chunk_size))

        tokens_list = []
        file_cache = {}

        for chunk in tqdm(all_chunks, desc="Tokenizing"):
            if chunk.file_path not in file_cache:
                file_cache[chunk.file_path] = Path(chunk.file_path).read_text(encoding="utf-8")

            file_text = file_cache[chunk.file_path]
            chunk_text = file_text[chunk.first_character_index:chunk.last_character_index]

            tokens = [word for word in re.findall(r'\w+', chunk_text.lower()) if len(word) > 2]
            tokens_list.append(tokens)

        bm25 = BM25Okapi(tokens_list)

        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)

        index_path = processed_dir / "index.pkl"
        with open(index_path, "wb") as f:
            pickle.dump({
                "chunks": all_chunks,
                "bm25": bm25
            }, f)

        print(f"Ingestion complete! Indexed {len(all_chunks)} under data/processed/")

    def _find_k_chunks(self, query: str, k: int = 5) -> List:
        index_path = Path("data/processed/index.pkl")

        if not index_path.exists():
            raise FileNotFoundError("Error: Index does not exist. Start with 'index' command")

        with open(index_path, "rb") as f:
            data = pickle.load(f)

        all_chunks = data["chunks"]
        bm25 = data["bm25"]

        tokenized_query = [word for word in re.findall(r'\w+', query.lower()) if len(word) > 2]
        best_chunks = bm25.get_top_n(tokenized_query, all_chunks, n=k)

        return best_chunks

    def search(self, query: str, k: int = 5) -> None:
        try:
            best_chunks = self._find_k_chunks(query, k)
        except FileNotFoundError as e:
            print(e)
            return

        for chunk in best_chunks:
            print(f"{chunk.file_path} [{chunk.first_character_index}:{chunk.last_character_index}]")

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str) -> None:
        search_results: List[MinimalSearchResults] = []

        with open(dataset_path, "r") as f:
            questions_dict = json.load(f)
        for _, questions in questions_dict.items():
            for question in questions:
                query: str | None = question.get("question", None)
                question_id: str | None = question.get("question_id", None)
                if query:
                    try:
                        result = self._find_k_chunks(query, k)
                    except FileNotFoundError as e:
                        print(e)
                        return
                    search_results.append(MinimalSearchResults(
                            question_id=question_id,
                            question=query,
                            retrieved_sources=result))

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        output_file = save_path / "student_search_results.json"

        with open(output_file, "w") as f:
            json.dump([r.model_dump() for r in search_results], f, indent=4)

        print(f"Saved student_search_results to {save_directory}")

    def answer(self, query: str, k: int) -> None:
        chunks: List = self._find_k_chunks(query=query, k=k)
        print(self.generator.generate(query=query, chunks=chunks))

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        pass

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        pass
