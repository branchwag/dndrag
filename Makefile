CHAT_MODEL ?= llama3.2
EMBED_MODEL ?= nomic-embed-text

.PHONY: build up setup ingest query down

build:
	docker compose --profile cli build

# Start Qdrant and Ollama in the background
up:
	docker compose up -d qdrant ollama

# Pull the two models into Ollama (only needed once; models persist in the ollama_data volume)
setup: up
	docker compose exec ollama ollama pull $(EMBED_MODEL)
	docker compose exec ollama ollama pull $(CHAT_MODEL)

# Index all PDFs in ./docs
ingest: up
	docker compose --profile cli run --rm dnd_rag ingest

# Usage: make query Q="Who is the main villain?"
query: up
	docker compose --profile cli run --rm dnd_rag query "$(Q)"

down:
	docker compose down
