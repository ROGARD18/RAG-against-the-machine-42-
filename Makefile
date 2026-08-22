.PHONY: install run debug clean lint search search_dataset answer answer_dataset moulinette

install:
	@echo "Installing dependencies with uv..."
	uv sync

run:
	@uv run python3 -m src index 2000

search:
	uv run python -m src search 'how to configure openai server' --k 5

search_dataset:
	uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions


answer:
	uv run python -m src answer "How to configure OpenAi server ?" --k 15

answer_dataset:
	uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions

evaluate:
	uv run python -m src evaluate data/output/search_results/UnansweredQuestions/dataset_code_public.json data/datasets/AnsweredQuestions/dataset_code_public.json

moulinette:
	./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_code_public.json data/datasets/AnsweredQuestions/dataset_code_public.json --k 10


debug:
	@echo "Running in debug mode..."
	uv run python3 -m pdb -m src

clean:
	@echo "Cleaning temporary files..."
	rm -rf data/output
	rm -rf data/processed
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "Clean complete."

lint:
	@echo "Running mypy..."
	mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
	@echo "Running flake8..."
	flake8 src
