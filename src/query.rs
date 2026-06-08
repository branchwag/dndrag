use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::json;
use std::io::Write as _;
use std::time::Instant;
use tracing::info;

use crate::embed::Embedder;
use crate::store::{SearchResult, VectorStore};

const TOP_K_CANDIDATES: u64 = 30; // more candidates for MMR to choose from
const MMR_K: usize = 8;           // diverse results to keep after MMR
const KEYWORD_K: u32 = 12;
const MMR_LAMBDA: f32 = 0.6;      // 0 = max diversity, 1 = max relevance
// Semantic results above this score bypass the entity-name filter.
// Lets high-confidence topically-relevant chunks through even when they
// don't repeat the entity name (e.g. city chunks for "major cities of X").
const ENTITY_FILTER_BYPASS_SCORE: f32 = 0.60;
const MODEL: &str = "llama3.1";

struct PipelineOutput {
    system_prompt: String,
    user_content: String,
    client: Client,
    ollama_url: String,
    chat_model: String,
}

// Shared retrieval pipeline: NER, keyword search, semantic search + MMR, rerank.
// Returns None if no relevant lore was found.
async fn pipeline(question: &str) -> Result<Option<PipelineOutput>> {
    let ollama_url = std::env::var("OLLAMA_URL")
        .unwrap_or_else(|_| "http://localhost:11434".to_string());
    let chat_model = std::env::var("CHAT_MODEL")
        .unwrap_or_else(|_| MODEL.to_string());

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;
    let embedder = Embedder::new();
    let store = VectorStore::new().await?;

    // NER and query embedding run concurrently — they're independent.
    let t_start = Instant::now();
    let (names, query_vec_result) = tokio::join!(
        extract_entities(&client, &ollama_url, &chat_model, question),
        embedder.embed(vec![question.to_string()])
    );
    let query_vec = query_vec_result?.remove(0);
    info!(
        entities = ?names,
        elapsed_ms = t_start.elapsed().as_millis(),
        "entity extraction + embedding"
    );

    // Keyword retrieval — catches proper nouns semantic search misses.
    let t_kw = Instant::now();
    let mut keyword_results: Vec<SearchResult> = Vec::new();
    for name in &names {
        for hit in store.keyword_search(name, KEYWORD_K).await? {
            if !keyword_results.iter().any(|r| r.text == hit.text) {
                keyword_results.push(hit);
            }
        }
    }
    info!(
        hits = keyword_results.len(),
        elapsed_ms = t_kw.elapsed().as_millis(),
        "keyword search"
    );

    // Semantic retrieval with MMR diversity selection.
    let t_sem = Instant::now();
    let candidates = store.search_with_vectors(query_vec.clone(), TOP_K_CANDIDATES).await?;
    info!(
        candidates = candidates.len(),
        elapsed_ms = t_sem.elapsed().as_millis(),
        "semantic search"
    );

    let t_mmr = Instant::now();
    let semantic_results = mmr_select(&query_vec, candidates, MMR_K, MMR_LAMBDA);
    info!(
        selected = semantic_results.len(),
        elapsed_ms = t_mmr.elapsed().as_millis(),
        "MMR diversity"
    );

    // Merge, deduplicating on exact text.
    let mut results = keyword_results;
    for hit in semantic_results {
        if !results.iter().any(|r| r.text == hit.text) {
            results.push(hit);
        }
    }

    // For named-entity queries, drop low-scoring semantic chunks that don't
    // mention any of the extracted entity names. High-scoring semantic results
    // (>= ENTITY_FILTER_BYPASS_SCORE) are kept unconditionally — they're
    // topically on-point even when the entity name doesn't appear verbatim
    // (e.g. individual city chunks for a "major cities of X" question).
    if !names.is_empty() {
        results.retain(|r| {
            r.is_keyword_match
                || r.score >= ENTITY_FILTER_BYPASS_SCORE
                || names
                    .iter()
                    .any(|n| r.text.to_lowercase().contains(&n.to_lowercase()))
        });
    }

    if results.is_empty() {
        return Ok(None);
    }

    // Build the prompt.
    let context = results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let label = if r.is_keyword_match { "DIRECT MATCH" } else { "related" };
            let page_str = if r.page > 0 { format!(" p.{}", r.page) } else { String::new() };
            format!("[{}] [{label}] from {}{}\n{}", i + 1, r.source, page_str, r.text)
        })
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");

    let subject_hint = if !names.is_empty() {
        format!(
            "The question is specifically about: {}. Prioritise excerpts marked [DIRECT MATCH]. Do not attribute details from other characters or places to {}.\n\n",
            names.join(", "),
            names.join("/")
        )
    } else {
        String::new()
    };

    let system_prompt = "\
You are the authoritative narrator of a specific, fully original DnD campaign world. \
A visitor with no prior knowledge of this world is asking you about it.\n\
\n\
RULES — follow without exception:\n\
- Your sole source of truth is the numbered lore passages provided by the user. \
Do NOT draw on any general DnD lore, fantasy tropes, or outside knowledge. \
This is a custom world with its own history, people, and places.\n\
- Speak as a narrator who simply *knows* this world — authoritative, measured, and literary. \
Never reveal that you are consulting documents. \
Forbidden phrases: \"based on the excerpts\", \"according to the passages\", \
\"the lore mentions\", \"from the information provided\", \"these notes\", \
or any phrasing that implies you are reading a source.\n\
- Write in an eloquent, formal register. \
No casual language, no hedging, no filler. \
Your response ends when the information is delivered — nothing more. \
Never append any closing phrase that invites further questions, acknowledges the exchange, \
or addresses the reader directly. \
Forbidden closings include any form of: \"Let me know if...\", \"Feel free to ask...\", \
\"I hope this helps\", \"Is there anything else...\", \"If you have more questions...\", \
or any other conversational sign-off. The tome does not solicit queries — it simply speaks.\n\
- Vary sentence construction. Do not open consecutive sentences with the same word or phrase. \
Weave related details into compound and complex sentences; \
do not list facts as a series of isolated simple clauses all beginning with the same subject.\n\
- State only what is explicitly written in the provided passages. \
Do not infer, embellish, invent atmosphere, or fill gaps with plausible-sounding detail.\n\
- If a detail is absent from the passages, say exactly: \"The lore does not speak of this.\" \
Do not speculate or approximate.\n\
- Never conflate separate characters, places, or factions with one another.\n\
- Every claim must be traceable to a specific numbered passage. If you cannot trace it, omit it.\n\
- Some passages may contain out-of-game player instructions: references to D&D Beyond, \
character creation, campaign links, dice mechanics, or directions addressed to players. \
These are not lore. Disregard them entirely and do not include their content in your answer."
        .to_string();

    let user_content = format!(
        "{subject_hint}\
Lore passages:\n{context}\n\
\nQuestion: {question}\n\
\nAnswer using only what is stated in the numbered passages above. \
Omit anything not explicitly present there."
    );

    Ok(Some(PipelineOutput { system_prompt, user_content, client, ollama_url, chat_model }))
}

/// Interactive query: streams tokens to stdout as they arrive.
/// With show_context=true, prints the retrieved prompt and exits without generating.
pub async fn run(question: &str, show_context: bool) -> Result<()> {
    match pipeline(question).await? {
        None => println!("No relevant lore found for that query."),
        Some(ctx) => {
            if show_context {
                println!("=== SYSTEM ===\n{}\n\n=== USER ===\n{}", ctx.system_prompt, ctx.user_content);
            } else {
                stream_generation(&ctx).await?;
            }
        }
    }
    Ok(())
}

/// Batch query: returns the full answer as a string. Used by the eval subcommand.
pub async fn answer(question: &str) -> Result<String> {
    match pipeline(question).await? {
        None => Ok("No relevant lore found for that query.".to_string()),
        Some(ctx) => generate(&ctx).await,
    }
}

/// SSE bridge: runs the pipeline and forwards tokens through an mpsc channel.
/// The serve subcommand spawns this in a task and bridges the channel to axum SSE.
pub async fn stream_to_sender(
    question: &str,
    tx: tokio::sync::mpsc::Sender<String>,
) -> Result<()> {
    let ctx = match pipeline(question).await {
        Err(e) => {
            let _ = tx.send(format!("⚠ Error: {e}")).await;
            return Ok(());
        }
        Ok(None) => {
            let _ = tx.send("The lore does not speak of this.".to_string()).await;
            return Ok(());
        }
        Ok(Some(ctx)) => ctx,
    };

    let response = ctx
        .client
        .post(format!("{}/v1/chat/completions", ctx.ollama_url))
        .json(&json!({
            "model": ctx.chat_model,
            "messages": [
                {"role": "system", "content": ctx.system_prompt},
                {"role": "user",   "content": ctx.user_content}
            ],
            "num_ctx": 8192,
            "num_predict": 1024,
            "temperature": 0,
            "stream": true
        }))
        .send()
        .await?;

    let mut stream = response.bytes_stream();
    let mut line_buf = String::new();

    while let Some(chunk) = stream.next().await {
        line_buf.push_str(std::str::from_utf8(&chunk?).unwrap_or(""));

        while let Some(pos) = line_buf.find('\n') {
            let line = line_buf[..pos].trim_end_matches('\r').to_string();
            line_buf.drain(..pos + 1);

            if let Some(data) = line.strip_prefix("data: ") {
                if data.trim() == "[DONE]" {
                    return Ok(());
                }
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(token) = val["choices"][0]["delta"]["content"].as_str() {
                        if tx.send(token.to_string()).await.is_err() {
                            return Ok(()); // client disconnected
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

// Streams the LLM response token-by-token to stdout via SSE.
async fn stream_generation(ctx: &PipelineOutput) -> Result<()> {
    let response = ctx
        .client
        .post(format!("{}/v1/chat/completions", ctx.ollama_url))
        .json(&json!({
            "model": ctx.chat_model,
            "messages": [
                {"role": "system", "content": ctx.system_prompt},
                {"role": "user",   "content": ctx.user_content}
            ],
            "num_ctx": 8192,
            "num_predict": 1024,
            "temperature": 0,
            "stream": true
        }))
        .send()
        .await?;

    let mut stream = response.bytes_stream();
    let mut line_buf = String::new();

    while let Some(chunk) = stream.next().await {
        line_buf.push_str(std::str::from_utf8(&chunk?).unwrap_or(""));

        while let Some(pos) = line_buf.find('\n') {
            let line = line_buf[..pos].trim_end_matches('\r').to_string();
            line_buf.drain(..pos + 1);

            if let Some(data) = line.strip_prefix("data: ") {
                if data.trim() == "[DONE]" {
                    println!();
                    return Ok(());
                }
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(token) = val["choices"][0]["delta"]["content"].as_str() {
                        print!("{token}");
                        std::io::stdout().flush()?;
                    }
                }
            }
        }
    }

    println!();
    Ok(())
}

// Non-streaming generation for eval: collects the full answer.
async fn generate(ctx: &PipelineOutput) -> Result<String> {
    let response = ctx
        .client
        .post(format!("{}/v1/chat/completions", ctx.ollama_url))
        .json(&json!({
            "model": ctx.chat_model,
            "messages": [
                {"role": "system", "content": ctx.system_prompt},
                {"role": "user",   "content": ctx.user_content}
            ],
            "num_ctx": 8192,
            "num_predict": 1024,
            "temperature": 0,
            "stream": false
        }))
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;

    response["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| anyhow!("Unexpected Ollama response: {response}"))
        .map(str::to_string)
}

// Asks the LLM to extract named entities from the query (character names, place
// names, item names). Runs concurrently with query embedding. Returns an empty
// vec on any failure so the pipeline degrades gracefully.
async fn extract_entities(
    client: &Client,
    ollama_url: &str,
    chat_model: &str,
    question: &str,
) -> Vec<String> {
    let prompt = format!(
        "Extract all named entities (character names, place names, item names, \
        organization names) from this question. Reply with ONLY a JSON array of strings. \
        If there are none, reply with []. Question: \"{question}\""
    );

    let Ok(response) = client
        .post(format!("{ollama_url}/v1/chat/completions"))
        .json(&json!({
            "model": chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "num_ctx": 1024,
            "num_predict": 64,
            "temperature": 0
        }))
        .send()
        .await
    else {
        return vec![];
    };

    let Ok(body) = response.json::<serde_json::Value>().await else {
        return vec![];
    };

    let content = body["choices"][0]["message"]["content"].as_str().unwrap_or("[]");

    if let (Some(start), Some(end)) = (content.find('['), content.rfind(']')) {
        if let Ok(names) = serde_json::from_str::<Vec<String>>(&content[start..=end]) {
            return names;
        }
    }

    vec![]
}


// Maximal Marginal Relevance: selects k results that balance relevance to the
// query with diversity from already-selected results. lambda=1 is pure relevance,
// lambda=0 is pure diversity. Operates on the embedding vectors returned by
// search_with_vectors so no re-embedding is needed.
fn mmr_select(
    query_vec: &[f32],
    mut candidates: Vec<(SearchResult, Vec<f32>)>,
    k: usize,
    lambda: f32,
) -> Vec<SearchResult> {
    let mut selected_vecs: Vec<Vec<f32>> = Vec::new();
    let mut output: Vec<SearchResult> = Vec::new();

    while output.len() < k && !candidates.is_empty() {
        let best_idx = candidates
            .iter()
            .enumerate()
            .max_by(|(_, (_, va)), (_, (_, vb))| {
                mmr_score(query_vec, va, &selected_vecs, lambda)
                    .partial_cmp(&mmr_score(query_vec, vb, &selected_vecs, lambda))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let (result, vec) = candidates.remove(best_idx);
        selected_vecs.push(vec);
        output.push(result);
    }

    output
}

fn mmr_score(query_vec: &[f32], candidate_vec: &[f32], selected: &[Vec<f32>], lambda: f32) -> f32 {
    let relevance = cosine_sim(query_vec, candidate_vec);
    let max_redundancy = selected
        .iter()
        .map(|sv| cosine_sim(candidate_vec, sv))
        .fold(0.0f32, f32::max);
    lambda * relevance - (1.0 - lambda) * max_redundancy
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
}
