use anyhow::{anyhow, Result};
use reqwest::Client;
use serde_json::json;

use crate::embed::Embedder;
use crate::store::VectorStore;

const TOP_K: u64 = 5;

pub async fn run(question: &str) -> Result<String> {
    let ollama_url = std::env::var("OLLAMA_URL")
        .unwrap_or_else(|_| "http://localhost:11434".to_string());
    let chat_model = std::env::var("CHAT_MODEL")
        .unwrap_or_else(|_| "llama3.2".to_string());

    let embedder = Embedder::new();
    let store = VectorStore::new().await?;

    // Embed the question and find the closest chunks
    let query_vec = embedder.embed(vec![question.to_string()]).await?.remove(0);
    let results = store.search(query_vec, TOP_K).await?;

    if results.is_empty() {
        return Ok("No relevant lore found for that query.".to_string());
    }

    let context = results
        .iter()
        .enumerate()
        .map(|(i, r)| format!("[{}] (from {}, score {:.2})\n{}", i + 1, r.source, r.score, r.text))
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");

    let prompt = format!(
        "You are a DnD world assistant with access to campaign lore. \
Answer the question using only the provided lore excerpts. \
If the answer isn't in the lore, say so clearly.\n\
\n\
Lore excerpts:\n{context}\n\
\nQuestion: {question}"
    );

    // Ollama OpenAI-compatible chat completions endpoint
    let client = Client::new();
    let response = client
        .post(format!("{ollama_url}/v1/chat/completions"))
        .json(&json!({
            "model": chat_model,
            "messages": [{"role": "user", "content": prompt}]
        }))
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;

    let answer = response["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| anyhow!("Unexpected Ollama response: {response}"))?
        .to_string();

    Ok(answer)
}
