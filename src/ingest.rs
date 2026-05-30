use anyhow::Result;
use std::path::{Path, PathBuf};

use crate::embed::Embedder;
use crate::store::VectorStore;

const CHUNK_SIZE: usize = 500;    // words per chunk
const CHUNK_OVERLAP: usize = 50;  // words shared between adjacent chunks
const EMBED_BATCH: usize = 32;    // chunks per embedding request

pub async fn run(docs_dir: &Path) -> Result<()> {
    let embedder = Embedder::new();
    let store = VectorStore::new().await?;
    store.ensure_collection().await?;

    let pdfs = find_pdfs(docs_dir)?;
    if pdfs.is_empty() {
        println!("No PDFs found in {:?}", docs_dir);
        return Ok(());
    }

    for pdf in &pdfs {
        let filename = pdf.file_name().unwrap().to_string_lossy().to_string();
        println!("Processing {filename}...");

        let text = extract_pdf_text(pdf)?;
        let chunks = chunk_text(&text, CHUNK_SIZE, CHUNK_OVERLAP);
        let total = chunks.len();

        for (batch_idx, batch) in chunks.chunks(EMBED_BATCH).enumerate() {
            let sources: Vec<String> = vec![filename.clone(); batch.len()];
            let embeddings = embedder.embed(batch.to_vec()).await?;
            store.upsert(batch, &sources, embeddings).await?;
            print!("\r  {filename}: batch {}/{} ({} chunks)        ",
                batch_idx + 1, (total + EMBED_BATCH - 1) / EMBED_BATCH, total);
        }
        println!("\r  Indexed {total} chunks from {filename}                ");
    }

    println!("Ingestion complete.");
    Ok(())
}

fn find_pdfs(dir: &Path) -> Result<Vec<PathBuf>> {
    let paths = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "pdf").unwrap_or(false))
        .collect();
    Ok(paths)
}

fn extract_pdf_text(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    let text = pdf_extract::extract_text_from_mem(&bytes)?;
    Ok(text)
}

// Sliding window over words. Overlap keeps context across chunk boundaries,
// which improves retrieval for sentences that straddle a split.
fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    // Drop tokens longer than 40 chars — PDF artifacts (hex blobs, base64, etc.)
    // that blow up the embedding model's context window.
    let words: Vec<&str> = text
        .split_whitespace()
        .filter(|w| w.len() <= 40)
        .collect();
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < words.len() {
        let end = (start + chunk_size).min(words.len());
        chunks.push(words[start..end].join(" "));
        if end == words.len() {
            break;
        }
        start += chunk_size - overlap;
    }

    chunks
}
