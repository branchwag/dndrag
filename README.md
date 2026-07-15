# dndrag

A fully local RAG pipeline for querying DnD campaign lore, written in Rust. Point it at your DM documents, ask questions, get answers grounded in your world — no API keys, no data leaving your machine.

## Screenshot

![dndrag browser front-end](final.gif)

## How it works

1. **Ingest** — PDFs in `docs/` are extracted, split into sentence-aware overlapping chunks, and embedded using `nomic-embed-text` via Ollama. Embeddings, page numbers and lore-entity tags are written to a single index file in `index/`.
2. **Query** — Your question is embedded and named entities are extracted (concurrently). A purpose-built, in-process vector index returns top candidates via both keyword and semantic search. MMR diversity selection picks the best non-redundant passages. A smaller LLM reranks them by relevance before they are assembled into a grounded prompt.
3. **Generate** — The LLM streams a response token-by-token, grounded strictly in the retrieved lore.
4. **Browse** — A DnD-themed browser front-end at `http://localhost:3000` lets anyone query the lore with an ink-reveal animation on a weathered parchment panel.

Everything runs in Docker. No GPU required; if an NVIDIA GPU is available it's detected and used automatically.

## Stack

| Layer | Tool |
|---|---|
| Language | Rust (tokio, reqwest, clap, axum) |
| PDF extraction | pdf-extract |
| Embeddings | Ollama + nomic-embed-text (768-dim) |
| Vector index | Purpose-built, in-process — exact cosine search + inverted index (no server, no ANN) |
| Generation | Ollama + gemma2:9b (configurable) |
| Infrastructure | Docker Compose |

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with the WSL2 backend enabled
- An NVIDIA GPU is optional — it's auto-detected and used when present; otherwise everything runs on CPU, no changes needed

## Quick start

```bash
# 1. Clone
git clone https://github.com/branchwag/dndrag.git
cd dndrag

# 2. Add your PDFs
cp your-campaign.pdf docs/

# 3. Build the images
make build

# 4. Start Ollama and pull models (one-time, ~5 GB download)
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
| `make up` | Start Ollama in the background |
| `make setup` | `up` + pull Ollama models (run once) |
| `make ingest` | Index all PDFs in `docs/` (incremental — safe to re-run) |
| `make ingest ARGS="--fresh"` | Wipe the index and re-index from scratch |
| `make query Q="..."` | Query from the CLI, streams response to stdout |
| `make eval` | Run eval.json Q&A pairs and report pass rate |
| `make serve` | Start the browser front-end at http://localhost:3000 |
| `make down` | Stop all services |
| `make clean-pdfs` | Strip images from PDFs in `docs/` to reduce noise (then re-ingest) |

## Configuration

All settings have defaults and can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://ollama:11434` | Ollama server address |
| `OLLAMA_HOST_PORT` | `11434` | Host port the Ollama container publishes. Set this (e.g. `11435`) if a native Ollama already uses `11434`. |
| `INDEX_PATH` | `index/dnd_lore.idx` | Where the vector index file lives |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHAT_MODEL` | autodetected | Generation model. `gemma2:9b` on a GPU or a CPU box with enough RAM; `llama3.2` on smaller CPU-only machines (see below). |
| `RERANK_MODEL` | `llama3.2` | Model used for entity extraction and reranking |
| `OLLAMA_TIMEOUT_SECS` | autodetected | Per-request timeout for Ollama calls. `120` normally; `600` on the low-RAM CPU tier, where the rerank step is slow. |

### Model autodetect

When you run `make` targets, the model tier is chosen to fit the machine, so a
fresh clone runs well without hand-tuning:

- **GPU present**, or **CPU with ≥ `LITE_RAM_GB` (default 12) GB RAM** → the full
  `gemma2:9b` for generation ("full" tier).
- **CPU-only with less RAM** → the lightweight `llama3.2` everywhere, plus a
  longer Ollama timeout ("lite" tier). `gemma2:9b` (~5.4 GB) plus the rerank
  model won't fit in memory otherwise, causing swap thrashing and timeouts.

`make up` prints the resolved tier, e.g.
`Ollama: CPU | tier=lite (RAM=7GB) | chat=llama3.2 rerank=llama3.2 timeout=600s`.

Override anything on the command line (it's also exported to the containers):
```bash
make serve CHAT_MODEL=gemma2:9b   # force the full chat model
make serve LITE_RAM_GB=16         # change the lite/full RAM threshold
make serve OLLAMA_TIMEOUT_SECS=300
```
`make setup` pulls whichever models the tier resolves to. (Autodetect runs
through `make`; a raw `docker compose up` uses the compose defaults.)

## GPU passthrough

When you run `make` targets, an NVIDIA GPU is auto-detected: if the driver
responds and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
is installed, `docker-compose.gpu.yml` is layered in automatically so Ollama
uses the GPU. If not, everything runs on CPU — no changes required.

Override the autodetect explicitly:

```bash
make up GPU=1   # force GPU (fails if no usable GPU)
make up GPU=0   # force CPU even if a GPU is present
```

To verify the GPU is in use:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec ollama nvidia-smi
```

Not using `make`? The GPU overlay is opt-in via `-f`:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d ollama
```

## Notes

- `docs/` is gitignored — your source documents are never committed
- The index is a single file in `index/` (also gitignored), written by `make ingest` and read by `make query`/`make serve`. Re-run `make ingest` only when you add or change documents — `serve` notices the new index on its next question, no restart needed
- There is no database to run. The whole corpus (~2.3k chunks, ~10 MB) is held in memory and searched exactly, so every query scans all of it rather than approximating. At this scale a full scan is well under a millisecond, so an approximate index (HNSW/IVF) would add complexity and cost recall for no speed benefit — it's the right tradeoff below roughly 10⁵ vectors, not the lazy one
- First query after a cold start is slow (~10–30s) while Ollama loads the model; subsequent queries are faster
- `make eval` runs labeled Q&A pairs from `eval.json` and reports a pass rate — useful for catching regressions when you change models or prompts
- `make serve` runs in the foreground — logs stream to your terminal and Ctrl+C stops the server. Add `-d` to the compose call in the Makefile if you want it to stay up after closing the terminal

## Credits

Parchment background photo by [Divya M](https://unsplash.com/@divya66) on [Unsplash](https://unsplash.com/photos/person-in-black-shoes-standing-on-brown-floor-1LVIgG629Do).
