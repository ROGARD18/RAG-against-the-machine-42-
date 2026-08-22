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
            word for word in re.findall(r"\w+", clean_text.lower())
            if len(word) > 2
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
                    chunk_markdown(file_path=file,
                                   max_chunk_size=max_chunk_size)
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
                chunk.first_character_index: chunk.last_character_index
            ]

            tokens = self.tokenize_for_code(chunk_text)
            tokens_list.append(tokens)

        bm25 = BM25Okapi(tokens_list)

        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)

        index_path = processed_dir / "index.pkl"
        with open(index_path, "wb") as f:
            pickle.dump({"chunks": all_chunks, "bm25": bm25}, f)

        print(f"Ingestion complete! Indexed {len(all_chunks)} "
              "under data/processed/")

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
                f"{chunk.file_path} [{chunk.first_character_index}"
                f":{chunk.last_character_index}]"
            )

    def search_dataset(self, dataset_path: str, k: int,
                       save_directory: str) -> None:
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
                        print(f"Error on question {question_id}: {e}")

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
            print(f"Error: File '{student_search_results_path}' "
                  "does not exist.")
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

    def evaluate(self, student_search_results_path: str, dataset_path: str, k: int = 10) -> None:
        import json
        import os

        def normalize_path(path: str) -> str:
            return os.path.normpath(str(path)) if path else ""

        def overlap_process(retrieved: dict, expected: dict) -> float:
            if normalize_path(retrieved.get("file_path")) != normalize_path(expected.get("file_path")):
                return 0.0

            start_inter = max(retrieved.get(
                "first_character_index", 0), expected.get("first_character_index", 0))
            end_inter = min(retrieved.get("last_character_index", 0),
                            expected.get("last_character_index", 0))

            overlap_length = max(0, end_inter - start_inter)
            expected_length = expected.get(
                "last_character_index", 0) - expected.get("first_character_index", 0)

            if expected_length <= 0:
                return 0.0

            return overlap_length / expected_length

        def calculate_question_recall(retrieved_sources: list, expected_sources: list, k_val: int) -> float:
            if not expected_sources:
                return 0.0

            top_k_retrieved = retrieved_sources[:k_val]
            sources_found = 0

            for expected in expected_sources:
                for retrieved in top_k_retrieved:
                    overlap = overlap_process(retrieved, expected)
                    if overlap >= 0.05:
                        sources_found += 1
                        break

            return sources_found / len(expected_sources)

        if not os.path.exists(dataset_path):
            print(f"Error: Dataset path missing: {dataset_path}")
            return
        if not os.path.exists(student_search_results_path):
            print(
                f"Error: Student answer path missing: {student_search_results_path}")
            return

        with open(dataset_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        with open(student_search_results_path, "r", encoding="utf-8") as f:
            student_data = json.load(f)

        print("Student data is valid: True")

        student_search_map = {
            res.get("question_id"): res for res in student_data.get("search_results", [])
        }

        rag_questions = gt_data.get("rag_questions", [])
        if not rag_questions:
            for val in gt_data.values():
                if isinstance(val, list):
                    rag_questions.extend(val)

        valid_gt_questions = [
            q for q in rag_questions
            if q.get("sources") is not None and len(q.get("sources", [])) > 0
        ]

        total_questions = len(rag_questions)
        total_with_sources = len(valid_gt_questions)

        questions_with_student_sources = sum(
            1 for q in valid_gt_questions
            if q.get("question_id") in student_search_map
            and len(student_search_map.get(q.get("question_id"), {}).get("retrieved_sources", [])) > 0
        )

        print(f"Total number of questions: {total_questions}")
        print(f"Total number of questions with sources: {total_with_sources}")
        print(
            f"Total number of questions with student sources: {questions_with_student_sources}\n")

        cutoffs = [1, 3, 5, 10]
        print("🎯 Evaluation Results")
        print("========================================")
        print(f"📊 Questions evaluated: {total_with_sources}")

        final_metrics = {}
        for c in cutoffs:
            total_recall_score = 0.0

            for q in valid_gt_questions:
                expected_list = q.get("sources", [])
                student_res = student_search_map.get(q.get("question_id"))
                retrieved_list = student_res.get(
                    "retrieved_sources", []) if student_res else []

                q_score = calculate_question_recall(
                    retrieved_list, expected_list, c)
                total_recall_score += q_score

            final_macro_recall = total_recall_score / \
                total_with_sources if total_with_sources > 0 else 0.0
            final_metrics[f"recall@{c}"] = final_macro_recall
            print(
                f"📈 Recall@{c}: {final_macro_recall:.3f} ({(final_macro_recall * 100):.1f}%)")

        print(final_metrics)
