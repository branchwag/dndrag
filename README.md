# dndrag

A fully local RAG pipeline for querying DnD campaign lore, written in Rust. Point it at your DM documents, ask questions, get answers grounded in your world — no API keys, no data leaving your machine.

## How it works

1. **Ingest** — PDFs in `docs/` are extracted, split into sentence-aware overlapping chunks, and embedded using `nomic-embed-text` via Ollama. Embeddings and page numbers are stored in Qdrant.
2. **Query** — Your question is embedded and named entities are extracted (concurrently). Qdrant returns the top candidates via both keyword and semantic search. MMR diversity selection picks the best non-redundant passages, which are assembled into a grounded prompt.
3. **Generate** — The LLM streams a response token-by-token, grounded strictly in the retrieved lore.
4. **Browse** — A DnD-themed browser front-end at `http://localhost:3000` lets anyone query the lore with an ink-reveal animation on a weathered parchment panel.

Everything runs in Docker. No GPU required, but one is supported.

## Stack

| Layer | Tool |
|---|---|
| Language | Rust (tokio, reqwest, clap, axum) |
| PDF extraction | pdf-extract |
| Embeddings | Ollama + nomic-embed-text (768-dim) |
| Vector store | Qdrant (cosine similarity + full-text index) |
| Generation | Ollama + gemma2:9b (configurable) |
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
cp your-campaign.pdf docs/

# 3. Build the images
make build

# 4. Start Qdrant + Ollama and pull models (one-time, ~5 GB download)
make setup

# 5. Index your documents
make ingest

# 6. Open the browser front-end
make serve
# then visit http://localhost:3000

# Or query from the CLI
make query Q="Who is the main villain?"
```

## Commands

| Command | What it does |
|---|---|
| `make build` | Build the Docker image |
| `make up` | Start Qdrant and Ollama in the background |
| `make setup` | `up` + pull Ollama models (run once) |
| `make ingest` | Index all PDFs in `docs/` (incremental — safe to re-run) |
| `make ingest ARGS="--fresh"` | Wipe the collection and re-index from scratch |
| `make query Q="..."` | Query from the CLI, streams response to stdout |
| `make serve` | Start the browser front-end at http://localhost:3000 |
| `make down` | Stop all services |

## Configuration

All settings have defaults and can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://ollama:11434` | Ollama server address |
| `QDRANT_URL` | `http://qdrant:6334` | Qdrant gRPC address |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHAT_MODEL` | `gemma2:9b` | Generation model |

To switch models:
```yaml
# docker-compose.yml
environment:
  CHAT_MODEL: llama3.2
```
Then re-run `make setup` to pull the new model.

## GPU passthrough

GPU support is enabled by default via `docker-compose.override.yml` (NVIDIA only). To verify:

```bash
docker compose exec ollama nvidia-smi
```

To run CPU-only, delete or rename `docker-compose.override.yml`.

## Notes

- `docs/` is gitignored — your source documents are never committed
- Qdrant data persists in a Docker named volume — re-run `make ingest` only when you add new documents
- First query after a cold start is slow (~10–30s) while Ollama loads the model; subsequent queries are faster
- The eval subcommand (`make query` with `eval` instead) runs labeled Q&A pairs from `eval.json` and reports a pass rate — useful for catching regressions when you change models or prompts
