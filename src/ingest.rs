use anyhow::Result;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use uuid::Uuid;

use crate::embed::Embedder;
use crate::store::VectorStore;

const TARGET_WORDS: usize = 200;
const OVERLAP_SENTENCES: usize = 2;
const EMBED_BATCH: usize = 32;

struct Chunk {
    text: String,
    page: u32,
}

pub async fn run(docs_dir: &Path, fresh: bool) -> Result<()> {
    let embedder = Embedder::new();
    let store = VectorStore::new().await?;

    if fresh {
        store.reset_collection().await?;
    } else {
        store.ensure_collection().await?;
    }

    let docs = find_docs(docs_dir)?;
    if docs.is_empty() {
        println!("No documents found in {:?}", docs_dir);
        return Ok(());
    }

    for pdf in &docs {
        let filename = pdf.file_name().unwrap().to_string_lossy().to_string();
        println!("Processing {filename}...");

        let text = extract_text(pdf)?;
        let chunks = chunk_text_semantic(&text, TARGET_WORDS, OVERLAP_SENTENCES);
        let total = chunks.len();
        let lore_entity = lore_entity_from_filename(&filename);

        for (batch_idx, batch) in chunks.chunks(EMBED_BATCH).enumerate() {
            let chunk_start = batch_idx * EMBED_BATCH;
            let ids: Vec<String> = (chunk_start..chunk_start + batch.len())
                .map(|i| chunk_id(&filename, i))
                .collect();
            let texts: Vec<String> = batch.iter().map(|c| c.text.clone()).collect();
            let pages: Vec<u32> = batch.iter().map(|c| c.page).collect();
            let sources: Vec<String> = vec![filename.clone(); batch.len()];
            let lore_entities: Vec<Option<String>> = vec![lore_entity.clone(); batch.len()];
            let embeddings = embedder.embed(texts.clone()).await?;
            store.upsert(&ids, &texts, &sources, &pages, &lore_entities, embeddings).await?;
            print!(
                "\r  {filename}: batch {}/{} ({total} chunks)        ",
                batch_idx + 1,
                (total + EMBED_BATCH - 1) / EMBED_BATCH,
            );
            let _ = std::io::stdout().flush();
        }
        println!("\r  Indexed {total} chunks from {filename}                ");
    }

    println!("Ingestion complete.");
    Ok(())
}

// For lore files named "lore_<slug>.txt", returns the slug with underscores
// replaced by spaces (e.g. "lore_alora_venyette.txt" -> Some("alora venyette")).
// Qdrant's word tokenizer splits on spaces, so storing the entity as space-separated
// tokens lets a Text match for "alora" find "alora venyette" reliably.
fn lore_entity_from_filename(filename: &str) -> Option<String> {
    let stem = filename.strip_prefix("lore_")?.strip_suffix(".txt")?;
    Some(stem.replace('_', " "))
}

fn find_docs(dir: &Path) -> Result<Vec<PathBuf>> {
    let paths = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| matches!(
            p.extension().and_then(|e| e.to_str()),
            Some("pdf") | Some("txt")
        ))
        .collect();
    Ok(paths)
}

fn extract_text(path: &Path) -> Result<String> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("txt") => Ok(std::fs::read_to_string(path)?),
        _ => {
            let bytes = std::fs::read(path)?;
            let text = pdf_extract::extract_text_from_mem(&bytes)?;
            Ok(clean_text(&text))
        }
    }
}

// Strips content that should never reach the vector store: YouTube URLs and
// the channel name "beardedboggan". Operates on the raw extracted text before
// any chunking, so nothing leaks through even in mid-sentence references.
// Page-break markers (\x0c) are preserved so page tracking still works.
fn clean_text(text: &str) -> String {
    text.split('\x0c')
        .map(|page| {
            page.lines()
                .map(|line| {
                    // Strip any token that looks like a YouTube URL.
                    line.split_whitespace()
                        .filter(|w| {
                            let l = w.to_lowercase();
                            !l.contains("youtube") && !l.contains("youtu.be")
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                // Drop entire lines that name the channel.
                .filter(|line| !line.to_lowercase().contains("beardedboggan"))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect::<Vec<_>>()
        .join("\x0c")
}

// Stable chunk ID: same file + same chunk position always produces the same UUID,
// so re-running ingest upserts rather than duplicating.
fn chunk_id(source: &str, chunk_idx: usize) -> String {
    let name = format!("{source}:{chunk_idx}");
    Uuid::new_v5(&Uuid::NAMESPACE_DNS, name.as_bytes()).to_string()
}

// Sentence-aware sliding-window chunker. Splits on paragraph and sentence
// boundaries so chunks don't cut mid-thought. Tracks PDF page numbers via
// the \x0c form-feed markers that pdf_extract inserts between pages.
fn chunk_text_semantic(text: &str, target_words: usize, overlap: usize) -> Vec<Chunk> {
    // Collect all sentences paired with their source page number.
    let mut indexed: Vec<(String, u32)> = Vec::new();
    let mut page = 1u32;

    for page_text in text.split('\x0c') {
        for sent in extract_sentences(page_text) {
            let filtered: String = sent
                .split_whitespace()
                .filter(|w| w.len() <= 40) // drop PDF artifacts (hex blobs, base64, etc.)
                .collect::<Vec<_>>()
                .join(" ");
            if filtered.split_whitespace().count() > 2 {
                indexed.push((filtered, page));
            }
        }
        page += 1;
    }

    let mut chunks: Vec<Chunk> = Vec::new();
    let mut start = 0usize;

    while start < indexed.len() {
        let mut word_count = 0usize;
        let mut end = start;

        while end < indexed.len() && word_count < target_words {
            word_count += indexed[end].0.split_whitespace().count();
            end += 1;
        }

        if end == start {
            break;
        }

        let text = indexed[start..end]
            .iter()
            .map(|(s, _)| s.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let chunk_page = indexed[start].1;
        chunks.push(Chunk { text, page: chunk_page });

        // Overlap: back up `overlap` sentences so adjacent chunks share context.
        // Guard ensures we always make forward progress.
        let next_start = end.saturating_sub(overlap);
        start = if next_start > start { next_start } else { end };
    }

    chunks
}

// Splits text into sentences, respecting paragraph boundaries (\n\n) and
// sentence-ending punctuation followed by an uppercase letter.
fn extract_sentences(text: &str) -> Vec<String> {
    text.split("\n\n")
        .flat_map(|para| sentences_from_para(para.trim()))
        .collect()
}

fn sentences_from_para(para: &str) -> Vec<String> {
    if para.is_empty() {
        return vec![];
    }

    // Collapse soft line-wraps (single \n) to spaces.
    let para = para.replace('\n', " ");
    let chars: Vec<char> = para.chars().collect();
    let n = chars.len();
    let mut sentences: Vec<String> = Vec::new();
    let mut buf = String::new();
    let mut i = 0;

    while i < n {
        let ch = chars[i];
        buf.push(ch);

        if matches!(ch, '.' | '!' | '?') {
            // Treat short words before '.' as abbreviations (Dr., Mr., vs., etc.)
            let last_word: String = buf
                .trim_end_matches(|c: char| matches!(c, '.' | '!' | '?'))
                .split_whitespace()
                .last()
                .unwrap_or("")
                .chars()
                .filter(|c| c.is_alphabetic())
                .collect();
            let is_abbr = last_word.len() <= 2;

            let next_is_upper = chars[i + 1..]
                .iter()
                .skip_while(|&&c| c == ' ')
                .next()
                .map(|&c| c.is_uppercase())
                .unwrap_or(true); // end of string = sentence boundary

            if next_is_upper && !is_abbr {
                let s = buf.trim().to_string();
                if !s.is_empty() {
                    sentences.push(s);
                }
                buf.clear();
            }
        }

        i += 1;
    }

    let s = buf.trim().to_string();
    if !s.is_empty() {
        sentences.push(s);
    }

    sentences
}
