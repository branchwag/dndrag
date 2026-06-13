use anyhow::{anyhow, Result};
use reqwest::Client;
use serde_json::json;

// nomic-embed-text produces 768-dim vectors; must match the Qdrant collection config
pub const EMBEDDING_DIM: u64 = 768;

#[derive(Clone)]
pub struct Embedder {
    client: Client,
    url: String,
    model: String,
}

impl Embedder {
    // Accepts a shared Client so the caller controls the connection pool and timeout.
    pub fn new(client: Client) -> Self {
        let url = std::env::var("OLLAMA_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        let model = std::env::var("EMBED_MODEL")
            .unwrap_or_else(|_| "nomic-embed-text".to_string());
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

        response["embeddings"]
            .as_array()
            .ok_or_else(|| anyhow!("Unexpected Ollama response: {response}"))?
            .iter()
            .map(|row| {
                row.as_array()
                    .ok_or_else(|| anyhow!("embedding row is not an array"))?
                    .iter()
                    .map(|v| {
                        v.as_f64()
                            .ok_or_else(|| anyhow!("non-numeric value in embedding"))
                            .map(|f| f as f32)
                    })
                    .collect::<Result<Vec<f32>>>()
            })
            .collect()
    }
}
