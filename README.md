# dndrag

A fully local RAG pipeline for querying DnD campaign lore, written in Rust. Point it at your DM documents, ask questions, get answers grounded in your world — no API keys, no data leaving your machine.

## How it works

1. **Ingest** — PDFs in `docs/` are extracted, split into 500-word overlapping chunks, and embedded using `nomic-embed-text` via Ollama. Embeddings are stored in Qdrant.
2. **Query** — Your question is embedded with the same model, the closest chunks are retrieved from Qdrant, and `llama3.2` generates an answer grounded in that lore.

Everything runs in Docker. No GPU required, but one is supported.

## Stack

| Layer | Tool |
|---|---|
| Language | Rust (tokio, reqwest, clap) |
| PDF extraction | pdf-extract |
| Embeddings | Ollama + nomic-embed-text (768-dim) |
| Vector store | Qdrant |
| Generation | Ollama + llama3.2 (configurable) |
| Infrastructure | Docker Compose |

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with the WSL2 backend enabled
- An NVIDIA GPU is optional — delete `docker-compose.override.yml` to run CPU-only

## Quick start

```bash
# 1. Clone
git clone https://github.com/branchwag/dndrag.git
cd dndrag

# 2. Add your PDFs
cp your-documents.pdf docs/

# 3. Build the image
make build

# 4. Start Qdrant + Ollama and pull models (one-time, ~2 GB download)
make setup

# 5. Index your documents
make ingest

# 6. Ask a question
make query Q="Who are the main villains and what are their motivations?"
```

## Commands

| Command | What it does |
|---|---|
| `make build` | Build the `dnd_rag` Docker image |
| `make up` | Start Qdrant and Ollama in the background |
| `make setup` | `up` + pull Ollama models (run once) |
| `make ingest` | Index all PDFs in `docs/` |
| `make query Q="..."` | Search the lore and generate an answer |
| `make down` | Stop all services |

Or use Docker Compose directly:

```bash
docker compose --profile cli run --rm dnd_rag query "What happened to Siadiff?"
docker compose --profile cli run --rm dnd_rag ingest --docs-dir /app/docs
```

## Configuration

All settings have defaults and can be overridden via environment variables in `docker-compose.yml`:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://ollama:11434` | Ollama server address |
| `QDRANT_URL` | `http://qdrant:6334` | Qdrant gRPC address |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `CHAT_MODEL` | `llama3.2` | Ollama generation model |

To use a different model (e.g. `mistral` or `qwen2.5`):
```yaml
# docker-compose.yml
environment:
  CHAT_MODEL: mistral
```
Then re-run `make setup` to pull the new model.

## GPU passthrough

GPU support is enabled by default via `docker-compose.override.yml` (NVIDIA only). To verify:

```bash
docker compose exec ollama nvidia-smi
```

To run CPU-only, delete or rename `docker-compose.override.yml`.

## Project structure

```
.
├── src/
│   ├── main.rs       # CLI entry point (ingest / query subcommands)
│   ├── embed.rs      # Ollama embedding client
│   ├── store.rs      # Qdrant vector store client
│   ├── ingest.rs     # PDF → chunks → embed → upsert pipeline
│   └── query.rs      # embed question → search → generate answer
├── docker/
│   └── entrypoint.sh
├── docs/             # Put your PDFs here (gitignored)
├── Dockerfile
├── docker-compose.yml
├── docker-compose.override.yml  # GPU passthrough (delete for CPU-only)
└── Makefile
```

## Notes

- `docs/` is gitignored — your source documents are never committed
- Qdrant data persists in a Docker named volume (`dndrag_qdrant_data`) — re-run `make ingest` after `make down` only if you add new documents
- First query after startup is slow (~10–30s) while the model loads into memory; subsequent queries are faster
