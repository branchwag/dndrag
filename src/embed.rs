use anyhow::{anyhow, Result};
use reqwest::Client;
use serde_json::json;

// nomic-embed-text produces 768-dim vectors; must match the Qdrant collection config
pub const EMBEDDING_DIM: u64 = 768;

pub struct Embedder {
    client: Client,
    url: String,
    model: String,
}

impl Embedder {
    pub fn new() -> Self {
        let url = std::env::var("OLLAMA_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        let model = std::env::var("EMBED_MODEL")
            .unwrap_or_else(|_| "nomic-embed-text".to_string());
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .expect("failed to build HTTP client");
        Self { client, url, model }
    }

    // Uses Ollama's native /api/embed endpoint (supports array input)
    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let response = self
            .client
            .post(format!("{}/api/embed", self.url))
            .json(&json!({ "model": self.model, "input": texts }))
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let embeddings = response["embeddings"]
            .as_array()
            .ok_or_else(|| anyhow!("Unexpected Ollama response: {response}"))?
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect()
            })
            .collect();

        Ok(embeddings)
    }
}
