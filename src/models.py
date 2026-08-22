"""Pydantic data models for the RAG pipeline."""

import uuid
from typing import List, Union

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Represents a specific chunk of text from a source file."""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represents a question without an answer."""
    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    question: str


class AnswerQuestion(UnansweredQuestion):
    """Represents a question with its retrieved sources and answer."""
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represents a dataset containing a list of RAG questions."""
    rag_questions: List[Union[AnswerQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """Represents the search results for a specific question."""
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represents a generated answer based on search results."""
    answer: str


class StudentSearchResults(BaseModel):
    """Represents the output format for a dataset search operation."""
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Represents the final output containing questions,
    sources, and answers."""
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]
    answer: str
    k: int
