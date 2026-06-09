use anyhow::Result;
use qdrant_client::qdrant::{
    r#match::MatchValue, Condition, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder,
    Distance, FieldType, Filter, PointStruct, SearchPointsBuilder,
    UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::{Payload, Qdrant};
use std::collections::HashMap;

use crate::embed::EMBEDDING_DIM;

const COLLECTION: &str = "dnd_lore";
const SCORE_THRESHOLD: f32 = 0.45;

#[derive(Clone)]
pub struct SearchResult {
    pub text: String,
    pub source: String,
    pub page: u32,
    pub score: f32,
    pub is_keyword_match: bool,
}

pub struct VectorStore {
    client: Qdrant,
}

impl VectorStore {
    pub async fn new() -> Result<Self> {
        let url = std::env::var("QDRANT_URL")
            .unwrap_or_else(|_| "http://localhost:6334".to_string());
        let client = Qdrant::from_url(&url).build()?;
        Ok(Self { client })
    }

    // Creates the collection only if it doesn't already exist (safe for incremental ingest).
    pub async fn ensure_collection(&self) -> Result<()> {
        if !self.client.collection_exists(COLLECTION).await? {
            self.create_collection_internal().await?;
            println!("Collection '{COLLECTION}' created.");
        }
        Ok(())
    }

    // Wipes and recreates the collection from scratch.
    pub async fn reset_collection(&self) -> Result<()> {
        if self.client.collection_exists(COLLECTION).await? {
            self.client.delete_collection(COLLECTION).await?;
        }
        self.create_collection_internal().await?;
        println!("Collection '{COLLECTION}' reset.");
        Ok(())
    }

    async fn create_collection_internal(&self) -> Result<()> {
        self.client
            .create_collection(
                CreateCollectionBuilder::new(COLLECTION).vectors_config(
                    VectorParamsBuilder::new(EMBEDDING_DIM, Distance::Cosine),
                ),
            )
            .await?;
        // Full-text index on the text field enables keyword search alongside vector search.
        self.client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(COLLECTION, "text", FieldType::Text),
            )
            .await?;
        // Text index on lore_entity enables fast lookup by character name token.
        // The lore_entity field stores the space-separated entity name extracted
        // from the filename (e.g. "lore_alora_venyette.txt" -> "alora venyette"),
        // so a Text match for "alora" reliably finds all chunks from that file.
        self.client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(COLLECTION, "lore_entity", FieldType::Text),
            )
            .await?;
        Ok(())
    }

    pub async fn upsert(
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
        let points: Vec<PointStruct> = ids
            .iter()
            .zip(texts.iter())
            .zip(sources.iter())
            .zip(pages.iter())
            .zip(lore_entities.iter())
            .zip(embeddings.into_iter())
            .map(|(((((id, text), source), page), lore_entity), embedding)| {
                let mut payload_json = serde_json::json!({
                    "text": text,
                    "source": source,
                    "page": page,
                });
                if let Some(entity) = lore_entity {
                    payload_json["lore_entity"] = serde_json::Value::String(entity.clone());
                }
                let payload: Payload = payload_json.try_into().expect("valid payload JSON");
                PointStruct::new(id.clone(), embedding, payload)
            })
            .collect();

        self.client
            .upsert_points(UpsertPointsBuilder::new(COLLECTION, points))
            .await?;
        Ok(())
    }

    // Returns results paired with their embedding vectors for MMR diversity selection.
    pub async fn search_with_vectors(
        &self,
        query_embedding: Vec<f32>,
        top_k: u64,
    ) -> Result<Vec<(SearchResult, Vec<f32>)>> {
        use qdrant_client::qdrant::vector_output::Vector;

        let response = self
            .client
            .search_points(
                SearchPointsBuilder::new(COLLECTION, query_embedding, top_k)
                    .with_payload(true)
                    .with_vectors(true),
            )
            .await?;

        Ok(response
            .result
            .into_iter()
            .filter(|r| r.score >= SCORE_THRESHOLD)
            .map(|r| {
                let vector = r
                    .vectors
                    .as_ref()
                    .and_then(|v| v.get_vector())
                    .and_then(|v| match v {
                        Vector::Dense(dv) => Some(dv.data),
                        _ => None,
                    })
                    .unwrap_or_default();

                let text = extract_string(&r.payload, "text");
                let source = extract_string(&r.payload, "source");
                let page = extract_u32(&r.payload, "page");
                let score = r.score;
                let result = SearchResult { text, source, page, score, is_keyword_match: false };
                (result, vector)
            })
            .collect())
    }

    // Lore-file search: returns chunks whose lore_entity field contains `slug` as a
    // text token (e.g. slug "alora" matches lore_entity "alora venyette"), ranked by
    // cosine similarity to the query vector. Guarantees the character's own lore
    // file surfaces regardless of how narrative PDF chunks score.
    pub async fn search_lore_file(
        &self,
        slug: &str,
        query_vec: Vec<f32>,
        limit: u32,
    ) -> Result<Vec<SearchResult>> {
        let filter = Filter::must([Condition::matches(
            "lore_entity",
            MatchValue::Text(slug.to_string()),
        )]);

        let response = self
            .client
            .search_points(
                SearchPointsBuilder::new(COLLECTION, query_vec, limit as u64)
                    .filter(filter)
                    .with_payload(true),
            )
            .await?;

        Ok(response
            .result
            .into_iter()
            .map(|r| {
                let text = extract_string(&r.payload, "text");
                let source = extract_string(&r.payload, "source");
                let page = extract_u32(&r.payload, "page");
                let score = r.score;
                SearchResult { text, source, page, score, is_keyword_match: true }
            })
            .collect())
    }

    // Keyword-filtered semantic search: returns chunks that contain `term` ranked
    // by cosine similarity to the query vector. Beats a plain scroll (which returns
    // arbitrary insertion order) for common names that appear in hundreds of chunks.
    pub async fn keyword_search(
        &self,
        term: &str,
        query_vec: Vec<f32>,
        limit: u32,
    ) -> Result<Vec<SearchResult>> {
        let filter = Filter::must([Condition::matches(
            "text",
            MatchValue::Text(term.to_string()),
        )]);

        let response = self
            .client
            .search_points(
                SearchPointsBuilder::new(COLLECTION, query_vec, limit as u64)
                    .filter(filter)
                    .with_payload(true),
            )
            .await?;

        Ok(response
            .result
            .into_iter()
            .map(|r| {
                let text = extract_string(&r.payload, "text");
                let source = extract_string(&r.payload, "source");
                let page = extract_u32(&r.payload, "page");
                let score = r.score;
                SearchResult { text, source, page, score, is_keyword_match: true }
            })
            .collect())
    }
}

fn extract_string(
    payload: &HashMap<String, qdrant_client::qdrant::Value>,
    key: &str,
) -> String {
    use qdrant_client::qdrant::value::Kind;
    payload
        .get(key)
        .and_then(|v| match v.kind.as_ref() {
            Some(Kind::StringValue(s)) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

fn extract_u32(
    payload: &HashMap<String, qdrant_client::qdrant::Value>,
    key: &str,
) -> u32 {
    use qdrant_client::qdrant::value::Kind;
    payload
        .get(key)
        .and_then(|v| match v.kind.as_ref() {
            Some(Kind::IntegerValue(n)) if *n >= 0 => Some(*n as u32),
            _ => None,
        })
        .unwrap_or(0)
}
