# ── Hardware autodetect ──────────────────────────────────────────────────────
# Pick GPU vs CPU and a model tier that fits the machine, so a fresh clone runs
# well without hand-tuning. All of it is overridable on the command line, e.g.
# `make serve GPU=0 CHAT_MODEL=gemma2:9b LITE_RAM_GB=16`.

# GPU: layer in docker-compose.gpu.yml when an NVIDIA GPU is detected (nvidia-smi
# responds). Otherwise run on CPU. No file renaming needed.
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

# Model tier: gemma2:9b (~5.4 GB) needs ~12 GB to run alongside the rerank model
# on CPU. With a GPU, or enough CPU RAM, use it ("full"). On a smaller CPU-only
# box, fall back to the lightweight llama3.2 everywhere ("lite") and give Ollama
# a longer timeout, since CPU inference is slow. RAM is detected on Linux (free)
# and macOS (sysctl); if it can't be determined, we assume "lite" (always fits).
LITE_RAM_GB ?= 12
RAM_GB := $(shell (free -g 2>/dev/null | awk '/^Mem:/{print $$2}' | grep .) || (sysctl -n hw.memsize 2>/dev/null | awk '{printf "%d", $$1/1073741824}') || echo 0)
ifeq ($(GPU_ON),1)
MODEL_TIER := full
else ifeq ($(shell [ "$(RAM_GB)" -ge $(LITE_RAM_GB) ] 2>/dev/null && echo full || echo lite),full)
MODEL_TIER := full
else
MODEL_TIER := lite
endif

ifeq ($(MODEL_TIER),lite)
CHAT_MODEL ?= llama3.2
OLLAMA_TIMEOUT_SECS ?= 600
else
CHAT_MODEL ?= gemma2:9b
OLLAMA_TIMEOUT_SECS ?= 120
endif
EMBED_MODEL ?= nomic-embed-text
# Entity extraction + reranking model. Must match RERANK_MODEL in
# docker-compose.yml so queries don't call a model that was never pulled.
RERANK_MODEL ?= llama3.2

# Export so `docker compose` substitution (${VAR:-default}) and the container all
# see the same values the `setup` pull step uses.
export CHAT_MODEL EMBED_MODEL RERANK_MODEL OLLAMA_TIMEOUT_SECS

.PHONY: build up setup ingest query serve down clean-pdfs

build:
	$(COMPOSE) --profile cli build

# Start Qdrant and Ollama in the background
up:
	@echo "Ollama: $(if $(filter 1,$(GPU_ON)),GPU,CPU) | tier=$(MODEL_TIER) (RAM=$(RAM_GB)GB) | chat=$(CHAT_MODEL) rerank=$(RERANK_MODEL) timeout=$(OLLAMA_TIMEOUT_SECS)s"
	$(COMPOSE) up -d qdrant ollama

# Pull the two models into Ollama (only needed once; models persist in the ollama_data volume)
setup: up
	$(COMPOSE) exec ollama ollama pull $(EMBED_MODEL)
	$(COMPOSE) exec ollama ollama pull $(CHAT_MODEL)
	@[ "$(RERANK_MODEL)" = "$(CHAT_MODEL)" ] || $(COMPOSE) exec ollama ollama pull $(RERANK_MODEL)

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
