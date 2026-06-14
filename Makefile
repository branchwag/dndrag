CHAT_MODEL ?= gemma2:9b
EMBED_MODEL ?= nomic-embed-text

# GPU autodetect. When an NVIDIA GPU is usable (driver responds *and* the
# container toolkit is installed), layer in docker-compose.gpu.yml so Ollama
# runs on the GPU. Otherwise everything runs on CPU with zero setup — no file
# renaming needed. Force it either way with `make up GPU=1` or `GPU=0`.
GPU ?= auto
ifeq ($(GPU),auto)
GPU_ON := $(shell nvidia-smi -L >/dev/null 2>&1 && echo 1 || echo 0)
else
GPU_ON := $(GPU)
endif

ifeq ($(GPU_ON),1)
COMPOSE := docker compose -f docker-compose.yml -f docker-compose.gpu.yml
else
COMPOSE := docker compose
endif

.PHONY: build up setup ingest query serve down clean-pdfs

build:
	$(COMPOSE) --profile cli build

# Start Qdrant and Ollama in the background
up:
	@echo "Ollama acceleration: $(if $(filter 1,$(GPU_ON)),GPU,CPU)"
	$(COMPOSE) up -d qdrant ollama

# Pull the two models into Ollama (only needed once; models persist in the ollama_data volume)
setup: up
	$(COMPOSE) exec ollama ollama pull $(EMBED_MODEL)
	$(COMPOSE) exec ollama ollama pull $(CHAT_MODEL)

# Index all PDFs in ./docs. Pass ARGS="--fresh" to wipe and rebuild from scratch.
ingest: up
	$(COMPOSE) --profile cli run --rm dnd_rag ingest $(ARGS)

# Usage: make query Q="Who is the main villain?"
query: up
	$(COMPOSE) --profile cli run --rm dnd_rag query "$(Q)"

# Run labeled Q&A pairs from eval.json and report pass rate
eval: up
	$(COMPOSE) --profile cli run --rm dnd_rag eval $(ARGS)

# Start the browser front-end at http://localhost:3000
serve: up
	$(COMPOSE) --profile serve up dnd_rag_serve

down:
	$(COMPOSE) down

# Redact YouTube links from PDFs in ./docs (overwrites in place).
# After running, re-index with: make ingest ARGS="--fresh"
clean-pdfs:
	$(COMPOSE) --profile tools run --rm pdf_tools python scripts/clean_pdfs.py
