run:
	uvicorn src.main:app --reload

# API + Ollama in containers (Ollama on CPU)
docker-up:
	docker compose up --build

# API in a container, Ollama on the host (uses the Mac GPU)
docker-up-host:
	docker compose -f compose.yaml -f compose.host-ollama.yaml up --build

docker-down:
	docker compose down

lint:
	ruff check .

format:
	ruff format .
	