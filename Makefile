.PHONY: install test lint fmt seed check-llm eval eval-replay

install:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

seed:
	uv run python -m src.db --reset

check-llm:
	uv run python scripts/check_llm.py

eval:
	uv run python -m evals.run

eval-replay:
	uv run python -m evals.run --llm replay
