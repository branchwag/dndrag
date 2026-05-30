use anyhow::Result;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, PointStruct, SearchPointsBuilder, UpsertPointsBuilder,
    VectorParamsBuilder,
};
use qdrant_client::{Payload, Qdrant};
use std::collections::HashMap;
use uuid::Uuid;

use crate::embed::EMBEDDING_DIM;

const COLLECTION: &str = "dnd_lore";

pub struct SearchResult {
    pub text: String,
    pub source: String,
    pub score: f32,
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

    pub async fn ensure_collection(&self) -> Result<()> {
        if !self.client.collection_exists(COLLECTION).await? {
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(COLLECTION).vectors_config(
                        VectorParamsBuilder::new(EMBEDDING_DIM, Distance::Cosine),
                    ),
                )
                .await?;
            println!("Created Qdrant collection '{COLLECTION}'");
        }
        Ok(())
    }

    pub async fn upsert(
        &self,
        texts: &[String],
        sources: &[String],
        embeddings: Vec<Vec<f32>>,
    ) -> Result<()> {
        let points: Vec<PointStruct> = texts
            .iter()
            .zip(sources.iter())
            .zip(embeddings.into_iter())
            .map(|((text, source), embedding)| {
                let payload: Payload = serde_json::json!({
                    "text": text,
                    "source": source,
                })
                .try_into()
                .expect("valid payload JSON");
                PointStruct::new(Uuid::new_v4().to_string(), embedding, payload)
            })
            .collect();

        self.client
            .upsert_points(UpsertPointsBuilder::new(COLLECTION, points))
            .await?;
        Ok(())
    }

    pub async fn search(&self, query_embedding: Vec<f32>, top_k: u64) -> Result<Vec<SearchResult>> {
        let response = self
            .client
            .search_points(
                SearchPointsBuilder::new(COLLECTION, query_embedding, top_k).with_payload(true),
            )
            .await?;

        Ok(response
            .result
            .into_iter()
            .map(|r| {
                let text = extract_string(&r.payload, "text");
                let source = extract_string(&r.payload, "source");
                SearchResult { text, source, score: r.score }
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
