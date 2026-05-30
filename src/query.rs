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

    // Keyword retrieval first — for named characters/places this is the most
    // reliable signal. Collect all direct hits before running semantic search.
    let mut keyword_results: Vec<SearchResult> = Vec::new();
    for name in &names {
        for hit in store.keyword_search(name, KEYWORD_K).await? {
            if !keyword_results.iter().any(|r| r.text == hit.text) {
                keyword_results.push(hit);
            }
        }
    }

    // Semantic retrieval — use a smaller budget when keyword hits are plentiful
    // so they don't get drowned out by loosely related chunks.
    let semantic_k = if keyword_results.len() >= 4 { 3 } else { TOP_K };
    let query_vec = embedder.embed(vec![question.to_string()]).await?.remove(0);
    let semantic_results = store.search(query_vec, semantic_k).await?;

    // Merge: keyword hits first (most directly relevant), then any unique semantic hits
    let mut results = keyword_results;
    for hit in semantic_results {
        if !results.iter().any(|r| r.text == hit.text) {
            results.push(hit);
        }
    }

    if results.is_empty() {
        return Ok("No relevant lore found for that query.".to_string());
    }

    // Label keyword hits so the LLM knows which excerpts directly name the subject
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
            "The question is specifically about: {}. Only report facts that are explicitly stated about {} in the excerpts marked [DIRECT MATCH].\n\n",
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
- State ONLY what is explicitly written in the excerpts. Do not infer, extrapolate, or fill gaps.\n\
- Do NOT use any general DnD or fantasy knowledge. This is a custom world.\n\
- If a fact is not in the excerpts, say \"The lore doesn't specify this\" — never guess.\n\
- Do not attribute details from one character or location to another.\n\
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

// Extract capitalised words that look like character/place names — used to
// supplement semantic search with exact keyword hits.
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
