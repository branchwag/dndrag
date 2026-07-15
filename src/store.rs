use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::time::SystemTime;

use crate::embed::EMBEDDING_DIM;

const SCORE_THRESHOLD: f32 = 0.45;
const DEFAULT_INDEX_PATH: &str = "index/dnd_lore.idx";
const MAGIC: &[u8; 8] = b"DNDVEC01";

#[derive(Clone, PartialEq)]
pub enum RetrievalType {
    Semantic,
    Keyword,
    LoreFile,
}

#[derive(Clone)]
pub struct SearchResult {
    pub text: String,
    pub source: String,
    pub page: u32,
    pub score: f32,
    pub retrieval_type: RetrievalType,
}

/// One indexed passage. Vectors live separately in a flat arena, addressed by
/// this chunk's position in `Index::chunks`.
#[derive(Clone, Serialize, Deserialize)]
struct Chunk {
    id: String,
    text: String,
    source: String,
    page: u32,
    lore_entity: Option<String>,
}

/// The whole corpus, held in memory.
///
/// At ~2.5k chunks x 768 dims this is about 8 MB of vectors, so every search is
/// an exact scan over the full arena rather than an approximate graph walk. That
/// costs well under a millisecond and, unlike HNSW, cannot miss a hit.
struct Index {
    /// Row-major, `chunks.len() * EMBEDDING_DIM` floats. L2-normalized on insert,
    /// so cosine similarity against a normalized query is a plain dot product.
    vectors: Vec<f32>,
    chunks: Vec<Chunk>,
    /// Chunk id -> row, so re-ingesting a document overwrites in place.
    by_id: HashMap<String, usize>,
    /// token -> rows whose `text` contains it.
    postings: HashMap<String, Vec<u32>>,
    /// token -> rows whose `lore_entity` contains it.
    entity_postings: HashMap<String, Vec<u32>>,
    /// Set by writes; the lookup maps are rebuilt from `chunks` on next read.
    dirty: bool,
    /// mtime of the file this was loaded from, for detecting a re-ingest.
    loaded_at: Option<SystemTime>,
}

impl Index {
    fn empty() -> Self {
        Self {
            vectors: Vec::new(),
            chunks: Vec::new(),
            by_id: HashMap::new(),
            postings: HashMap::new(),
            entity_postings: HashMap::new(),
            dirty: false,
            loaded_at: None,
        }
    }

    fn row(&self, i: usize) -> &[f32] {
        let dim = EMBEDDING_DIM as usize;
        &self.vectors[i * dim..(i + 1) * dim]
    }

    /// Rebuilds the id map and both posting lists from `chunks`. Cheap enough
    /// (~100 ms over the full corpus) that writes just invalidate and let the
    /// next read redo it, instead of trying to patch postings incrementally.
    fn rebuild(&mut self) {
        self.by_id = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (c.id.clone(), i))
            .collect();

        self.postings.clear();
        self.entity_postings.clear();
        for (i, chunk) in self.chunks.iter().enumerate() {
            for token in tokenize(&chunk.text) {
                push_unique(self.postings.entry(token).or_default(), i as u32);
            }
            if let Some(entity) = &chunk.lore_entity {
                for token in tokenize(entity) {
                    push_unique(self.entity_postings.entry(token).or_default(), i as u32);
                }
            }
        }
        self.dirty = false;
    }

    /// Rows matching every token in `query` — the same all-tokens-must-be-present
    /// rule Qdrant's full-text `Match` applied. Intersects shortest-first.
    fn matching_rows(&self, index: &HashMap<String, Vec<u32>>, query: &str) -> Vec<u32> {
        let tokens = tokenize(query);
        if tokens.is_empty() {
            return Vec::new();
        }

        let mut lists: Vec<&Vec<u32>> = Vec::with_capacity(tokens.len());
        for token in &tokens {
            match index.get(token) {
                Some(list) => lists.push(list),
                None => return Vec::new(), // a token nothing contains -> no match
            }
        }
        lists.sort_by_key(|l| l.len());

        let (first, rest) = lists.split_first().expect("tokens is non-empty");
        first
            .iter()
            .copied()
            .filter(|row| rest.iter().all(|list| list.binary_search(row).is_ok()))
            .collect()
    }

    /// Scores `rows` against `query` and returns the best `limit`, highest first.
    fn rank(&self, rows: &[u32], query: &[f32], limit: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = rows
            .iter()
            .map(|&row| (row as usize, dot(self.row(row as usize), query)))
            .collect();
        // Tie-break on row so repeated queries return a stable order.
        scored.sort_unstable_by(|(i, a), (j, b)| {
            b.partial_cmp(a)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(i.cmp(j))
        });
        scored.truncate(limit);
        scored
    }

    fn result(&self, row: usize, score: f32, retrieval_type: RetrievalType) -> SearchResult {
        let chunk = &self.chunks[row];
        SearchResult {
            text: chunk.text.clone(),
            source: chunk.source.clone(),
            page: chunk.page,
            score,
            retrieval_type,
        }
    }
}

pub struct VectorStore {
    path: PathBuf,
    inner: RwLock<Index>,
}

impl VectorStore {
    /// Loads the index from disk, or starts empty if it doesn't exist yet
    /// (the first `ingest` on a fresh checkout).
    pub fn open() -> Result<Self> {
        let path = PathBuf::from(
            std::env::var("INDEX_PATH").unwrap_or_else(|_| DEFAULT_INDEX_PATH.to_string()),
        );
        let index = if path.exists() {
            let mut index = load(&path)
                .with_context(|| format!("loading index from {}", path.display()))?;
            index.rebuild();
            println!("Loaded {} chunks from {}", index.chunks.len(), path.display());
            index
        } else {
            Index::empty()
        };
        Ok(Self { path, inner: RwLock::new(index) })
    }

    /// Drops every chunk, for `ingest --fresh`. The file is only overwritten
    /// once `save` runs, so a crash mid-ingest leaves the old index intact.
    pub fn reset(&self) {
        let mut index = self.inner.write().expect("index lock poisoned");
        *index = Index::empty();
        index.dirty = true;
        println!("Index reset.");
    }

    /// Adds or replaces chunks, keyed by id — re-ingesting a document overwrites
    /// its chunks rather than duplicating them. In-memory only until `save`.
    pub fn upsert(
        &self,
        ids: &[String],
        texts: &[String],
        sources: &[String],
        pages: &[u32],
        lore_entities: &[Option<String>],
        embeddings: Vec<Vec<f32>>,
    ) -> Result<()> {
        anyhow::ensure!(
            ids.len() == texts.len()
                && ids.len() == sources.len()
                && ids.len() == pages.len()
                && ids.len() == lore_entities.len()
                && ids.len() == embeddings.len(),
            "upsert: slice length mismatch (ids={}, texts={}, sources={}, pages={}, entities={}, embeddings={})",
            ids.len(), texts.len(), sources.len(), pages.len(), lore_entities.len(), embeddings.len()
        );

        let dim = EMBEDDING_DIM as usize;
        let mut index = self.inner.write().expect("index lock poisoned");
        if index.dirty {
            // by_id must be current to detect overwrites.
            index.rebuild();
        }

        for (i, embedding) in embeddings.into_iter().enumerate() {
            anyhow::ensure!(
                embedding.len() == dim,
                "upsert: embedding for '{}' has {} dims, expected {dim}",
                ids[i],
                embedding.len()
            );
            let vector = normalized(embedding);
            let chunk = Chunk {
                id: ids[i].clone(),
                text: texts[i].clone(),
                source: sources[i].clone(),
                page: pages[i],
                lore_entity: lore_entities[i].clone(),
            };

            match index.by_id.get(&chunk.id).copied() {
                Some(row) => {
                    index.vectors[row * dim..(row + 1) * dim].copy_from_slice(&vector);
                    index.chunks[row] = chunk;
                }
                None => {
                    let row = index.chunks.len();
                    index.vectors.extend_from_slice(&vector);
                    index.chunks.push(chunk);
                    index.by_id.insert(ids[i].clone(), row);
                }
            }
        }

        index.dirty = true;
        Ok(())
    }

    /// Writes the index to disk. Renamed into place, so a reader either sees the
    /// complete old file or the complete new one, never a half-written index.
    pub fn save(&self) -> Result<()> {
        let mut index = self.inner.write().expect("index lock poisoned");

        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
        }
        let tmp = self.path.with_extension("tmp");
        write_to(&tmp, &index)
            .with_context(|| format!("writing index to {}", tmp.display()))?;
        std::fs::rename(&tmp, &self.path)
            .with_context(|| format!("replacing {}", self.path.display()))?;

        // Adopt the file we just wrote, so our own save doesn't look like someone
        // else's re-ingest to `refresh` below.
        index.loaded_at = mtime(&self.path);
        println!("Saved {} chunks to {}", index.chunks.len(), self.path.display());
        Ok(())
    }

    /// Picks up a re-ingest by another process and rebuilds stale lookup maps.
    /// `ingest` runs in its own container, so `serve` would otherwise answer from
    /// whatever index existed when it started.
    fn refresh(&self) -> Result<()> {
        let on_disk = mtime(&self.path);
        let stale = {
            let index = self.inner.read().expect("index lock poisoned");
            let reloadable = on_disk.is_some() && on_disk != index.loaded_at;
            if !reloadable && !index.dirty {
                return Ok(());
            }
            reloadable
        };

        let mut index = self.inner.write().expect("index lock poisoned");
        if stale {
            *index = load(&self.path)
                .with_context(|| format!("reloading index from {}", self.path.display()))?;
            tracing::info!(chunks = index.chunks.len(), "index reloaded after re-ingest");
        }
        if index.dirty {
            index.rebuild();
        }
        Ok(())
    }

    /// Top `top_k` by cosine similarity, paired with their vectors for MMR.
    /// Scoring the full arena means the top-k is exact, not approximate.
    pub async fn search_with_vectors(
        &self,
        query_embedding: &[f32],
        top_k: u64,
    ) -> Result<Vec<(SearchResult, Vec<f32>)>> {
        self.refresh()?;
        let index = self.inner.read().expect("index lock poisoned");
        let query = normalized(query_embedding.to_vec());

        let all: Vec<u32> = (0..index.chunks.len() as u32).collect();
        Ok(index
            .rank(&all, &query, top_k as usize)
            .into_iter()
            // Threshold after the cut, exactly as the Qdrant path did: take the
            // best top_k, then drop any that are too weak to be worth showing.
            .filter(|&(_, score)| score >= SCORE_THRESHOLD)
            .map(|(row, score)| {
                let result = index.result(row, score, RetrievalType::Semantic);
                (result, index.row(row).to_vec())
            })
            .collect())
    }

    /// Chunks from a character's own lore file, ranked by similarity to the query
    /// (e.g. slug "alora" matches the file whose lore_entity is "alora venyette").
    /// Guarantees the lore file surfaces regardless of how narrative chunks score.
    pub async fn search_lore_file(
        &self,
        slug: &str,
        query_vec: &[f32],
        limit: u32,
    ) -> Result<Vec<SearchResult>> {
        self.refresh()?;
        let index = self.inner.read().expect("index lock poisoned");
        let query = normalized(query_vec.to_vec());

        let rows = index.matching_rows(&index.entity_postings, slug);
        Ok(index
            .rank(&rows, &query, limit as usize)
            .into_iter()
            .map(|(row, score)| index.result(row, score, RetrievalType::LoreFile))
            .collect())
    }

    /// Chunks containing `term`, ranked by similarity to the query vector.
    /// Catches proper nouns that semantic search alone misses.
    pub async fn keyword_search(
        &self,
        term: &str,
        query_vec: &[f32],
        limit: u32,
    ) -> Result<Vec<SearchResult>> {
        self.refresh()?;
        let index = self.inner.read().expect("index lock poisoned");
        let query = normalized(query_vec.to_vec());

        let rows = index.matching_rows(&index.postings, term);
        Ok(index
            .rank(&rows, &query, limit as usize)
            .into_iter()
            .map(|(row, score)| index.result(row, score, RetrievalType::Keyword))
            .collect())
    }
}

/// Lowercase, split on anything non-alphanumeric.
///
/// Splitting on apostrophes means a search for "Florian" also matches "Florian's"
/// — matching the tokenizer the Qdrant text index used, and which the pipeline's
/// `has_non_possessive_mention` filter is written to clean up afterwards.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// Posting lists stay sorted and duplicate-free so `matching_rows` can binary search.
fn push_unique(list: &mut Vec<u32>, row: u32) {
    if list.last() != Some(&row) {
        list.push(row);
    }
}

/// Scales to unit length, so a dot product against another normalized vector is
/// the cosine similarity. A zero vector is left alone (it scores 0 against
/// everything either way).
fn normalized(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut vector {
            *x /= norm;
        }
    }
    vector
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn mtime(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).ok()?.modified().ok()
}

/// Layout: MAGIC | dim: u32 | count: u32 | json_len: u64 | chunks as JSON | vectors as LE f32.
/// Only the chunks and vectors are persisted; the posting lists are rebuilt on load.
fn write_to(path: &Path, index: &Index) -> Result<()> {
    let json = serde_json::to_vec(&index.chunks)?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);

    out.write_all(MAGIC)?;
    out.write_all(&(EMBEDDING_DIM as u32).to_le_bytes())?;
    out.write_all(&(index.chunks.len() as u32).to_le_bytes())?;
    out.write_all(&(json.len() as u64).to_le_bytes())?;
    out.write_all(&json)?;
    for value in &index.vectors {
        out.write_all(&value.to_le_bytes())?;
    }
    out.flush()?;
    Ok(())
}

fn load(path: &Path) -> Result<Index> {
    let bytes = std::fs::read(path)?;
    anyhow::ensure!(
        bytes.len() >= 24 && &bytes[..8] == MAGIC,
        "not a dndrag index file (bad magic): {}",
        path.display()
    );

    let dim = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
    let count = u32::from_le_bytes(bytes[12..16].try_into()?) as usize;
    let json_len = u64::from_le_bytes(bytes[16..24].try_into()?) as usize;
    anyhow::ensure!(
        dim == EMBEDDING_DIM as usize,
        "index has {dim}-dim vectors but this build expects {EMBEDDING_DIM}. \
         Re-run `make ingest ARGS=\"--fresh\"` after changing EMBED_MODEL.",
    );

    let json_end = 24 + json_len;
    let vectors_end = json_end + count * dim * 4;
    anyhow::ensure!(
        bytes.len() >= vectors_end,
        "index file is truncated: expected {vectors_end} bytes, found {}",
        bytes.len()
    );

    let chunks: Vec<Chunk> = serde_json::from_slice(&bytes[24..json_end])?;
    anyhow::ensure!(
        chunks.len() == count,
        "index header claims {count} chunks but holds {}",
        chunks.len()
    );
    let vectors: Vec<f32> = bytes[json_end..vectors_end]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("chunks_exact(4)")))
        .collect();

    Ok(Index {
        vectors,
        chunks,
        by_id: HashMap::new(),
        postings: HashMap::new(),
        entity_postings: HashMap::new(),
        dirty: true, // lookup maps are rebuilt by the caller
        loaded_at: mtime(path),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vector pointing mostly along axis `axis`, so similarity is predictable.
    fn vec_along(axis: usize) -> Vec<f32> {
        let mut v = vec![0.01f32; EMBEDDING_DIM as usize];
        v[axis] = 1.0;
        v
    }

    struct Doc {
        id: &'static str,
        text: &'static str,
        source: &'static str,
        page: u32,
        entity: Option<&'static str>,
        axis: usize,
    }

    /// Builds a store backed by a temp file, seeded with `docs`.
    fn store_with(name: &str, docs: &[Doc]) -> VectorStore {
        let path = std::env::temp_dir().join(format!("dndrag-test-{name}.idx"));
        let _ = std::fs::remove_file(&path);

        let store = VectorStore { path, inner: RwLock::new(Index::empty()) };
        let ids: Vec<String> = docs.iter().map(|d| d.id.to_string()).collect();
        let texts: Vec<String> = docs.iter().map(|d| d.text.to_string()).collect();
        let sources: Vec<String> = docs.iter().map(|d| d.source.to_string()).collect();
        let pages: Vec<u32> = docs.iter().map(|d| d.page).collect();
        let entities: Vec<Option<String>> =
            docs.iter().map(|d| d.entity.map(String::from)).collect();
        let embeddings: Vec<Vec<f32>> = docs.iter().map(|d| vec_along(d.axis)).collect();

        store
            .upsert(&ids, &texts, &sources, &pages, &entities, embeddings)
            .expect("seed upsert");
        store
    }

    /// Real passages from docs/, kept faithful to the lore so a reader can trust
    /// what they say. Between them they cover every tokenizer shape the pipeline
    /// depends on: an entity with its own lore file, a possessive mention in
    /// campaign prose, a plain mention of that same name, and a two-word entity.
    fn corpus() -> Vec<Doc> {
        vec![
            Doc {
                id: "a",
                text: "Alora Venyette is an ancient arcane mage and vampire, \
                       originally turned by a succubus from Avernus.",
                source: "lore_alora_venyette.txt",
                page: 0,
                entity: Some("alora venyette"),
                axis: 0,
            },
            Doc {
                id: "b",
                text: "She could have heard about that through the bards singing \
                       Florian's praises.",
                source: "campaign_2.txt",
                page: 12,
                entity: None,
                axis: 1,
            },
            Doc {
                id: "c",
                text: "Rose and Florian fought their way through this guarded \
                       location against the paladins of Torm.",
                source: "campaign_2.txt",
                page: 13,
                entity: None,
                axis: 2,
            },
            Doc {
                id: "d",
                text: "This story spans three continents: Crevalon, Ikovia, and Anearios.",
                source: "lore_story_overview.txt",
                page: 0,
                entity: Some("story overview"),
                axis: 3,
            },
        ]
    }

    #[tokio::test]
    async fn keyword_search_matches_possessives_like_the_qdrant_tokenizer_did() {
        // "Florian's" tokenizes to ["florian", "s"], so a search for "Florian"
        // must still return it. The pipeline's has_non_possessive_mention filter
        // depends on seeing these and rejecting them itself.
        let store = store_with("possessive", &corpus());
        let hits = store
            .keyword_search("Florian", &vec_along(1), 10)
            .await
            .unwrap();
        let texts: Vec<&str> = hits.iter().map(|h| h.text.as_str()).collect();
        assert!(texts.iter().any(|t| t.contains("Florian's praises")));
        assert!(texts.iter().any(|t| t.contains("paladins of Torm")));
    }

    #[tokio::test]
    async fn keyword_search_requires_every_token() {
        let store = store_with("all-tokens", &corpus());
        // "Alora" alone hits; "Alora Florian" does not, since the two names appear
        // in different chunks and never together in one.
        assert_eq!(
            store.keyword_search("Alora", &vec_along(0), 10).await.unwrap().len(),
            1
        );
        assert!(store
            .keyword_search("Alora Florian", &vec_along(0), 10)
            .await
            .unwrap()
            .is_empty());
        // A token in no chunk at all yields nothing rather than everything.
        assert!(store
            .keyword_search("Nonexistent", &vec_along(0), 10)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn lore_file_search_matches_a_single_name_token() {
        // query.rs relies on slug "alora" reaching lore_entity "alora venyette",
        // and on multi-token slugs like "story overview" resolving too.
        let store = store_with("lore", &corpus());

        let hits = store.search_lore_file("alora", &vec_along(0), 10).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].source, "lore_alora_venyette.txt");
        assert!(hits[0].retrieval_type == RetrievalType::LoreFile);

        let hits = store
            .search_lore_file("story overview", &vec_along(3), 10)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].source, "lore_story_overview.txt");

        // A name that only ever appears in prose — Torm has no lore file — must not
        // match the entity index, which indexes lore_entity and not text.
        assert!(store
            .search_lore_file("torm", &vec_along(2), 10)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn semantic_search_ranks_by_cosine_and_drops_weak_hits() {
        let store = store_with("semantic", &corpus());
        let hits = store.search_with_vectors(&vec_along(2), 30).await.unwrap();

        // Only the chunk on the query's axis clears SCORE_THRESHOLD; the others
        // are near-orthogonal and get filtered out.
        assert_eq!(hits.len(), 1);
        let (result, vector) = &hits[0];
        assert!(result.text.contains("paladins of Torm"));
        assert!(result.score >= SCORE_THRESHOLD);
        assert!(result.retrieval_type == RetrievalType::Semantic);

        // MMR needs the vector back, unit length after normalization. The bound is
        // loose because summing 768 f32 squares accumulates rounding error.
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "expected unit vector, got {norm}");
    }

    #[tokio::test]
    async fn upsert_replaces_a_chunk_rather_than_duplicating_it() {
        // Re-running ingest reuses chunk ids, so it must overwrite in place.
        let store = store_with("upsert", &corpus());
        store
            .upsert(
                &["c".to_string()],
                &["Yuri helped Florian and Rose by finding a location he believed \
                   the paladins had taken Caeda to."
                    .to_string()],
                &["campaign_2.txt".to_string()],
                &[99],
                &[None],
                vec![vec_along(2)],
            )
            .unwrap();

        let hits = store.keyword_search("Florian", &vec_along(2), 10).await.unwrap();
        assert_eq!(hits.len(), 2, "expected 2 Florian chunks, not a duplicate");

        let updated = hits.iter().find(|h| h.page == 99).expect("updated chunk");
        assert!(updated.text.contains("Caeda"));
        // "Torm" appeared only in the replaced text, so it must be gone from the
        // inverted index too — not just from the stored payload.
        assert!(store
            .keyword_search("Torm", &vec_along(2), 10)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn save_and_reopen_round_trips_the_index() {
        let store = store_with("roundtrip", &corpus());
        store.save().unwrap();

        let reopened = load(&store.path).unwrap();
        let reopened = VectorStore {
            path: store.path.clone(),
            inner: RwLock::new(reopened),
        };

        let hits = reopened.search_lore_file("alora", &vec_along(0), 10).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].page, 0);
        assert_eq!(hits[0].source, "lore_alora_venyette.txt");

        let semantic = reopened.search_with_vectors(&vec_along(2), 30).await.unwrap();
        assert!(semantic[0].0.text.contains("paladins of Torm"));

        std::fs::remove_file(&store.path).ok();
    }

    #[tokio::test]
    async fn search_picks_up_a_re_ingest_by_another_process() {
        // serve holds an open store while ingest rewrites the file underneath it.
        let store = store_with("reload", &corpus());
        store.save().unwrap();
        assert_eq!(
            store.keyword_search("Alora", &vec_along(0), 10).await.unwrap().len(),
            1
        );

        // A separate writer replaces the file, as `make ingest` would.
        let writer = VectorStore {
            path: store.path.clone(),
            inner: RwLock::new(Index::empty()),
        };
        writer
            .upsert(
                &["z".to_string()],
                &["Alora Venyette was court mage of Crevalon, a title she held \
                   before King Titus's death and the Siadiff takeover."
                    .to_string()],
                &["lore_alora_venyette.txt".to_string()],
                &[7],
                &[Some("alora venyette".to_string())],
                vec![vec_along(0)],
            )
            .unwrap();
        // mtime has 1s granularity on some filesystems; make the change detectable.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        writer.save().unwrap();

        let hits = store.keyword_search("Alora", &vec_along(0), 10).await.unwrap();
        assert_eq!(hits.len(), 1, "reader should see the rewritten index");
        assert_eq!(hits[0].page, 7);
        assert!(hits[0].text.contains("court mage"));

        std::fs::remove_file(&store.path).ok();
    }
}
