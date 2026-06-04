use anyhow::{anyhow, Result};
use reqwest::Client;
use serde_json::json;

use crate::embed::Embedder;
use crate::store::{SearchResult, VectorStore};

const TOP_K: u64 = 10;
const KEYWORD_K: u32 = 8;
const MODEL: &str = "llama3.1";

pub async fn run(question: &str) -> Result<String> {
    let ollama_url = std::env::var("OLLAMA_URL")
        .unwrap_or_else(|_| "http://localhost:11434".to_string());
    let chat_model = std::env::var("CHAT_MODEL")
        .unwrap_or_else(|_| MODEL.to_string());

    let embedder = Embedder::new();
    let store = VectorStore::new().await?;

    let names = extract_names(question);

    let mut keyword_results: Vec<SearchResult> = Vec::new();
    for name in &names {
        for hit in store.keyword_search(name, KEYWORD_K).await? {
            if !keyword_results.iter().any(|r| r.text == hit.text) {
                keyword_results.push(hit);
            }
        }
    }

    let semantic_k = if keyword_results.len() >= 4 { 3 } else { TOP_K };
    let query_vec = embedder.embed(vec![question.to_string()]).await?.remove(0);
    let semantic_results = store.search(query_vec, semantic_k).await?;

    let mut results = keyword_results;
    for hit in semantic_results {
        if !results.iter().any(|r| r.text == hit.text) {
            results.push(hit);
        }
    }

    if results.is_empty() {
        return Ok("No relevant lore found for that query.".to_string());
    }

    let keyword_cutoff = results.iter().filter(|r| r.score == 0.6).count();
    let context = results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let label = if i < keyword_cutoff { "DIRECT MATCH" } else { "related" };
            format!("[{}] [{label}] from {}\n{}", i + 1, r.source, r.text)
        })
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");

    let subject_hint = if !names.is_empty() {
        format!(
            "The question is specifically about: {}. Prioritise excerpts marked [DIRECT MATCH] — those directly contain the subject. Do not attribute details from other characters or places to {}.\n\n",
            names.join(", "),
            names.join("/")
        )
    } else {
        String::new()
    };

    let prompt = format!(
        "You are an assistant for a specific DnD campaign. \
Your ONLY source of truth is the lore excerpts below. \
Rules you must follow without exception:\n\
- Only report what is written in the excerpts. You may read them naturally and summarise, but do not invent facts.\n\
- Do NOT use any general DnD or fantasy knowledge. This is a fully custom world.\n\
- If something is not mentioned anywhere in the excerpts, say \"The lore doesn't mention this.\"\n\
- Do not confuse separate characters or locations with each other.\n\
\n\
{subject_hint}\
Lore excerpts:\n{context}\n\
\nQuestion: {question}"
    );

    let client = Client::new();
    let response = client
        .post(format!("{ollama_url}/v1/chat/completions"))
        .json(&json!({
            "model": chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "num_ctx": 8192,
            "num_predict": 1024
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

fn extract_names(query: &str) -> Vec<String> {
    const SKIP: &[&str] = &[
        "Who", "What", "Where", "When", "Why", "How", "Tell", "Is", "Are", "Was",
        "The", "A", "An", "In", "On", "At", "For", "Of", "With", "Me", "About",
        "Her", "His", "Their", "Your", "My", "Do", "Did", "Can", "Could", "Would",
        "Show", "Give", "List", "Describe", "Explain",
    ];
    query
        .split_whitespace()
        .filter_map(|w| {
            let clean: String = w.chars().filter(|c| c.is_alphanumeric()).collect();
            if clean.len() > 2
                && clean.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                && !SKIP.contains(&clean.as_str())
            {
                Some(clean)
            } else {
                None
            }
        })
        .collect()
}
