import fire
import json
import pickle
import sys
from src.llm_model import LlmModel
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
import re

from src.models import (
    MinimalSource,
    MinimalAnswer,
    MinimalSearchResults,
    AnswerQuestion,
    UnansweredQuestion,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.indexer_utils import get_source_files, chunk_markdown, chunk_python


class CliError(Exception):
    pass


class Cli:
    """Interface class"""

    def __init__(self):
        self.generator = LlmModel()

    @staticmethod
    def tokenize_for_code(text: str) -> List[str]:
        clean_text = text.replace("_", " ").replace("-", " ")
        return [
            word for word in re.findall(r"\w+", clean_text.lower()) if len(word) > 2
        ]

    def index(self, max_chunk_size: int = 2000) -> None:
        python_files, markdown_files = get_source_files()
        all_files = python_files + markdown_files

        if not all_files:
            print("Error ! : No files found to indexing.", file=sys.stderr)
            return
        all_chunks: List[MinimalSource] = []

        for file in tqdm(all_files, desc="Chunking"):
            if file in markdown_files:
                all_chunks.extend(
                    chunk_markdown(file_path=file, max_chunk_size=max_chunk_size)
                )
            elif file in python_files:
                all_chunks.extend(
                    chunk_python(file_path=file, max_chunk_size=max_chunk_size)
                )

        tokens_list = []
        file_cache = {}

        for chunk in tqdm(all_chunks, desc="Tokenizing"):
            if chunk.file_path not in file_cache:
                file_cache[chunk.file_path] = Path(chunk.file_path).read_text(
                    encoding="utf-8"
                )

            file_text = file_cache[chunk.file_path]
            chunk_text = file_text[
                chunk.first_character_index : chunk.last_character_index
            ]

            tokens = self.tokenize_for_code(chunk_text)
            tokens_list.append(tokens)

        bm25 = BM25Okapi(tokens_list)

        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)

        index_path = processed_dir / "index.pkl"
        with open(index_path, "wb") as f:
            pickle.dump({"chunks": all_chunks, "bm25": bm25}, f)

        print(f"Ingestion complete! Indexed {len(all_chunks)} under data/processed/")

    def _find_k_chunks(self, query: str, k: int = 5) -> List:
        index_path = Path("data/processed/index.pkl")

        if not index_path.exists():
            raise FileNotFoundError(
                "Error: Index does not exist. Start with 'index' command"
            )

        with open(index_path, "rb") as f:
            data = pickle.load(f)

        all_chunks = data["chunks"]
        bm25 = data["bm25"]

        tokenized_query = self.tokenize_for_code(query)
        best_chunks = bm25.get_top_n(tokenized_query, all_chunks, n=k)

        return best_chunks

    def search(self, query: str, k: int = 5) -> None:
        try:
            best_chunks = self._find_k_chunks(query, k)
        except FileNotFoundError as e:
            print(e)
            return

        for chunk in best_chunks:
            print(
                f"{chunk.file_path} [{chunk.first_character_index}:{chunk.last_character_index}]"
            )

    def search_dataset(self, dataset_path: str, k: int, save_directory: str) -> None:
        import json
        from pathlib import Path

        with open(dataset_path, "r", encoding="utf-8") as f:
            questions_dict = json.load(f)

        search_results_list = []

        for key, questions in questions_dict.items():
            if not isinstance(questions, list):
                continue

            for question in questions:
                query = question.get("question", "")
                question_id = question.get("question_id", "")

                if not question_id:
                    continue

                result_chunks = []
                if query:
                    try:
                        result_chunks = self._find_k_chunks(query, k)
                    except Exception as e:
                        print(f"Erreur sur la question {question_id}: {e}")

                search_results_list.append(
                    {
                        "question_id": question_id,
                        "question": query,
                        "retrieved_sources": [
                            chunk.model_dump() for chunk in result_chunks
                        ],
                    }
                )

        final_output = {"search_results": search_results_list, "k": k}

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        output_file = save_path / Path(dataset_path).name

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4)

    def answer(self, query: str, k: int = 5) -> None:
        try:
            chunks: List = self._find_k_chunks(query=query, k=k)
        except FileNotFoundError as e:
            print(e)
            return

        reversed_chunks = chunks[::-1]
        answer = self.generator.generate(query=query, chunks=reversed_chunks)
        print("---\n\n\n\nAnswer:\n")
        print(answer)

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:

        input_path = Path(student_search_results_path)
        if not input_path.exists():
            print(f"Error: File '{student_search_results_path}' does not exist.")
            return

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        search_results_data = data.get("search_results", [])
        k = data.get("k", 5)

        answers_results_list = []

        for item in tqdm(search_results_data, desc="Generating answers"):
            question_id = item.get("question_id")
            question = item.get("question")
            retrieved_sources_raw = item.get("retrieved_sources", [])

            chunks = [
                MinimalSource(
                    file_path=src["file_path"],
                    first_character_index=src["first_character_index"],
                    last_character_index=src["last_character_index"],
                )
                for src in retrieved_sources_raw
            ]

            reversed_chunks = chunks[::-1]

            generated_answer = self.generator.generate(
                query=question, chunks=reversed_chunks
            )

            answers_results_list.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "retrieved_sources": retrieved_sources_raw,
                    "answer": generated_answer,
                }
            )

        final_output = {"search_results": answers_results_list, "k": k}

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        output_file = save_path / "student_answers.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4)

        print(f"Saved student answers to {output_file}")

    def evaluate(self, student_search_results_path: str, dataset_path: str) -> None:
        import json
        import sys

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
        except FileNotFoundError:
            print(f"Erreur : '{dataset_path}' introuvable.", file=sys.stderr)
            return

        ground_truth = {}
        for key, questions in gt_data.items():
            if not isinstance(questions, list):
                continue
            for q in questions:
                q_id = q.get("question_id")
                sources = q.get("retrieved_sources", [])
                ground_truth[q_id] = set(src.get("file_path") for src in sources if "file_path" in src)

        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                student_data = json.load(f)
        except FileNotFoundError:
            print(f"Erreur : '{student_search_results_path}' introuvable.", file=sys.stderr)
            return

        student_results = student_data.get("search_results", [])

        hits = {1: 0, 3: 0, 5: 0, 10: 0}
        total = 0

        for item in student_results:
            q_id = item.get("question_id")

            if not q_id or q_id not in ground_truth or not ground_truth[q_id]:
                continue

            total += 1
            gt_paths = ground_truth[q_id]
            student_sources = item.get("retrieved_sources", [])
            student_paths = [src.get("file_path") for src in student_sources if "file_path" in src]

            for k in [1, 3, 5, 10]:
                if set(student_paths[:k]).intersection(gt_paths):
                    hits[k] += 1

        if total == 0:
            print("Erreur : Aucune question correspondante trouvée pour l'évaluation.", file=sys.stderr)
            return

        print(f"📊 Questions évaluées : {total}")
        for k in [1, 3, 5, 10]:
            recall = hits[k] / total
            print(f"📈 Recall@{k}: {recall:.3f} ({recall * 100:.1f}%)")
