run:
	uvicorn src.main:app --reload

lint:
	ruff check .

format:
	ruff format .
	