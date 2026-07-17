.PHONY: install test lint run docker-up docker-down benchmark clean

install:
	pip install -r requirements.txt ruff black pytest

lint:
	ruff check src tests --fix
	black src tests

test:
	pytest tests/ -v

run:
	uvicorn src.serve:app --reload

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

benchmark:
	python -m src.benchmark --requests 16 --batch-size 8 --max-new-tokens 32

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ src/__pycache__ tests/__pycache__
