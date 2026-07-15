use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::json;
use std::collections::HashSet;
use std::io::Write as _;
use std::sync::Arc;
use std::time::Instant;
use tracing::info;

use crate::config::RagConfig;
use crate::embed::Embedder;
use crate::store::{SearchResult, VectorStore};

const TOP_K_CANDIDATES: u64 = 30; // more candidates for MMR to choose from
const MMR_K: usize = 8;           // diverse results to keep after MMR
const KEYWORD_K: u32 = 16;
const MMR_LAMBDA: f32 = 0.6;      // 0 = max diversity, 1 = max relevance
const RERANK_K: usize = 8;        // passages kept after LLM reranking
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

/// Shared resources created once and reused across queries.
pub struct QueryResources {
    pub client: Client,
    pub embedder: Embedder,
    pub store: VectorStore,
    pub ollama_url: String,
    pub chat_model: String,
    pub rerank_model: String,
}

impl QueryResources {
    pub async fn new() -> Result<Self> {
        let ollama_url = std::env::var("OLLAMA_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        let chat_model = std::env::var("CHAT_MODEL")
            .unwrap_or_else(|_| MODEL.to_string());
        let rerank_model = std::env::var("RERANK_MODEL")
            .unwrap_or_else(|_| chat_model.clone());
        // Per-request timeout. The rerank step sends a large multi-passage prompt;
        // on a slow CPU that single call can exceed the default, so allow tuning
        // via OLLAMA_TIMEOUT_SECS (raise it on CPU-only / low-RAM machines).
        let timeout_secs = std::env::var("OLLAMA_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(120);
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()?;
        // Share the same client so embed calls use the same connection pool and timeout.
        let embedder = Embedder::new(client.clone());
        let store = VectorStore::open()?;
        Ok(Self { client, embedder, store, ollama_url, chat_model, rerank_model })
    }
}

// Shared retrieval pipeline: NER, keyword search, semantic search + MMR, rerank.
// Returns None if no relevant lore was found.
async fn pipeline(question: &str, res: &QueryResources) -> Result<Option<PipelineOutput>> {

    // NER and query embedding run concurrently — they're independent.
    let t_start = Instant::now();
    let (names, query_vec_result) = tokio::join!(
        extract_entities(&res.client, &res.ollama_url, &res.rerank_model, question),
        res.embedder.embed(vec![question.to_string()])
    );
    let query_vec = query_vec_result?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("embedder returned empty result for query"))?;
    info!(
        entities = ?names,
        elapsed_ms = t_start.elapsed().as_millis(),
        "entity extraction + embedding"
    );

    let RagConfig { scene_markers, prompt_extra_rules } = RagConfig::load();

    // Keyword retrieval — catches proper nouns semantic search misses.
    // Uses the query vector so results are ranked by relevance, not insertion order.
    // A second biography-focused pass uses a different query vector to surface
    // backstory/origin chunks that rank lower against the literal question phrasing.
    let t_kw = Instant::now();
    let mut seen: HashSet<String> = HashSet::new();
    let mut keyword_results: Vec<SearchResult> = Vec::new();

    // Compute bio/events query strings (depend on entity names, not query_vec).
    let bio_query = if !names.is_empty() {
        Some(format!(
            "{} paladin knight origin became history background turned",
            names.join(" ")
        ))
    } else {
        None
    };
    let events_query = if !names.is_empty() {
        Some(format!(
            "{} defended kingdom battle fought dragon king served protected victory threat defeated",
            names.join(" ")
        ))
    } else {
        None
    };

    // Semantic search, bio embedding, and events embedding are all independent —
    // run them concurrently with each other and with the keyword loop below.
    let t_sem = Instant::now();
    let (candidates, bio_vec, events_vec) = tokio::join!(
        res.store.search_with_vectors(&query_vec, TOP_K_CANDIDATES),
        async {
            if let Some(q) = bio_query {
                res.embedder.embed(vec![q]).await.ok()
                    .and_then(|mut v| if v.is_empty() { None } else { Some(v.remove(0)) })
            } else {
                None
            }
        },
        async {
            if let Some(q) = events_query {
                res.embedder.embed(vec![q]).await.ok()
                    .and_then(|mut v| if v.is_empty() { None } else { Some(v.remove(0)) })
            } else {
                None
            }
        },
    );
    let candidates = candidates?;
    info!(
        candidates = candidates.len(),
        elapsed_ms = t_sem.elapsed().as_millis(),
        "semantic search (concurrent with bio/events embedding)"
    );

    for name in &names {
        let slug = name.to_lowercase();
        // All 4 searches per entity are independent Qdrant calls — run them concurrently.
        let (kw_hits, bio_hits, ev_hits, lore_hits) = tokio::join!(
            res.store.keyword_search(name, &query_vec, KEYWORD_K),
            async {
                if let Some(bv) = bio_vec.as_ref() {
                    res.store.keyword_search(name, bv, KEYWORD_K / 2).await
                } else {
                    Ok(vec![])
                }
            },
            async {
                if let Some(ev) = events_vec.as_ref() {
                    res.store.keyword_search(name, ev, KEYWORD_K / 2).await
                } else {
                    Ok(vec![])
                }
            },
            // Lore-file pass: always include chunks from the character's own lore file.
            // Structured lore docs embed differently from natural questions, so they can
            // fall outside the top-K semantic candidates despite being the best source.
            // Use space-separated name so Text match finds lore_entity values like "alora venyette".
            res.store.search_lore_file(&slug, &query_vec, KEYWORD_K),
        );
        for hit in kw_hits?.into_iter().chain(bio_hits?).chain(ev_hits?) {
            if !seen.contains(&hit.text)
                && has_non_possessive_mention(&hit.text, name)
                && !is_scene_filtered(&hit.text, &scene_markers)
            {
                seen.insert(hit.text.clone());
                keyword_results.push(hit);
            }
        }
        for hit in lore_hits? {
            if !seen.contains(&hit.text)
                && !is_scene_filtered(&hit.text, &scene_markers)
            {
                seen.insert(hit.text.clone());
                keyword_results.push(hit);
            }
        }
    }
    // When NER returns nothing, try a lore-file search on each word in the question
    // that is long enough to be a proper noun (>=5 chars). Catches cases where the
    // entity extractor misses place names like "FrostLands" or "Ikovia".
    if names.is_empty() {
        let q_lower = question.to_lowercase();
        for word in q_lower.split(|c: char| !c.is_alphanumeric()) {
            if word.len() >= 5 {
                for hit in res.store.search_lore_file(word, &query_vec, 5).await? {
                    if !seen.contains(&hit.text)
                        && !is_scene_filtered(&hit.text, &scene_markers)
                    {
                        seen.insert(hit.text.clone());
                        keyword_results.push(hit);
                    }
                }
            }
        }
    }

    // For broad narrative questions with no named entities, force-include relevant
    // lore files so the LLM has coherent context rather than unrelated semantic results.
    if names.is_empty() {
        let q = question.to_lowercase();
        if q.contains("story") || q.contains("plot") || q.contains("overview")
            || q.contains("what happened") || q.contains("history")
            || q.contains("tell me about") || q.contains("what is going on")
        {
            for hit in res.store.search_lore_file("story overview", &query_vec, KEYWORD_K).await? {
                if !seen.contains(&hit.text)
                    && !is_scene_filtered(&hit.text, &scene_markers)
                {
                    seen.insert(hit.text.clone());
                    keyword_results.push(hit);
                }
            }
        }
        // Villain/antagonist questions: surface Virion's lore file directly, since
        // semantic search alone tends to find whichever villain appears most in PDF chunks.
        if q.contains("villain") || q.contains("antagonist") || q.contains("main enemy") {
            for hit in res.store.search_lore_file("virion", &query_vec, KEYWORD_K).await? {
                if !seen.contains(&hit.text)
                    && !is_scene_filtered(&hit.text, &scene_markers)
                {
                    seen.insert(hit.text.clone());
                    keyword_results.push(hit);
                }
            }
        }
    }

    info!(
        hits = keyword_results.len(),
        elapsed_ms = t_kw.elapsed().as_millis(),
        "keyword search"
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
        if !seen.contains(&hit.text) {
            seen.insert(hit.text.clone());
            results.push(hit);
        }
    }

    // For named-entity queries, filter results based on how the entity name appears:
    // - Keyword results: entity must appear in non-possessive form (already enforced above).
    // - Semantic results that contain the entity name: same non-possessive requirement.
    //   This catches chunks like "Florian's group went missing" (Virion's story) that
    //   score high semantically but are not primarily about the entity.
    // - Semantic results that don't mention the entity at all: kept only if score >= bypass
    //   threshold (topically relevant even without the name, e.g. city list chunks).
    if !names.is_empty() {
        results.retain(|r| {
            if r.retrieval_type != crate::store::RetrievalType::Semantic {
                return true; // already filtered at collection time
            }
            let name_in_text = names
                .iter()
                .any(|n| r.text.to_lowercase().contains(&n.to_lowercase()));
            if name_in_text {
                names.iter().any(|n| has_non_possessive_mention(&r.text, n))
            } else {
                r.score >= ENTITY_FILTER_BYPASS_SCORE
            }
        });
    }

    // Drop any scene-specific chunks that slipped through semantic search.
    results.retain(|r| !is_scene_filtered(&r.text, &scene_markers));

    if results.is_empty() {
        return Ok(None);
    }

    // Rerank: score each candidate for relevance to the question with a single
    // LLM call, then keep only the top RERANK_K. Fixes cases where a relevant
    // chunk is retrieved but buried under lower-quality results.
    let t_rerank = Instant::now();
    results = rerank(&res.client, &res.ollama_url, &res.rerank_model, question, &names, results, RERANK_K).await;
    info!(
        kept = results.len(),
        elapsed_ms = t_rerank.elapsed().as_millis(),
        "rerank"
    );

    // Build the prompt.
    let context = results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let label = if r.retrieval_type != crate::store::RetrievalType::Semantic { "DIRECT MATCH" } else { "related" };
            let page_str = if r.page > 0 { format!(" p.{}", r.page) } else { String::new() };
            format!("[{}] [{label}] from {}{}\n{}", i + 1, r.source, page_str, r.text)
        })
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");

    let subject_hint = if !names.is_empty() {
        format!(
            "The question is specifically about: {}. \
             Answer the question asked directly — if it asks for a specific name, detail, or fact, \
             state that clearly first. Then draw on [DIRECT MATCH] passages as your authoritative source \
             and weave in supporting detail from the remaining passages. \
             Do not attribute details from other characters or places to {}.\n\n",
            names.join(", "),
            names.join("/")
        )
    } else {
        String::new()
    };

    let system_prompt = "\
You are an ancient keeper of lore — a chronicler who has lived among the kingdoms, \
peoples, and histories of this world and speaks of them as living truth. \
A seeker has come to you with a question. Answer as one who simply knows, \
not as one who describes from without. There is no \"fantasy\", no \"realm\" — \
only history, people, and places that are real.\n\
\n\
RULES — follow without exception:\n\
- Your sole source of truth is the numbered lore passages provided by the user. \
Do NOT draw on outside knowledge, invent detail, or use genre conventions. \
This world has its own history, people, and places — speak only of those.\n\
- Speak as a narrator who simply *knows* this world — authoritative, measured, and precise. \
Never reveal that you are consulting documents. \
Forbidden phrases: \"based on the excerpts\", \"according to the passages\", \
\"the lore mentions\", \"from the information provided\", \"these notes\", \
or any phrasing that implies you are reading a source.\n\
- Write in a formal, precise register. \
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
- When a passage explains the cause or origin of something — how a character came to be in \
their current state, why an event occurred, what brought something about — include that \
causal detail. Do not describe only the outcome while omitting how it came to pass.\n\
- When describing a character, prioritise facts that define who they are: their nature, \
history, role, abilities, and relationships. Do not lead with or emphasise incidental \
scene details — a costume element worn at one event, a single minor action — when \
more significant defining facts are available in the passages.\n\
- Do not narrate romantic or intimate moments (kissing, physical affection, passionate exchanges). \
If two characters have a romantic relationship, state that the relationship exists; \
do not describe specific intimate scenes.\n\
- If a detail is absent from the passages, say exactly: \"The lore does not speak of this.\" \
Do not speculate or approximate.\n\
- Never conflate separate characters, places, or factions with one another. \
A passage may mention multiple characters; attribute each action or trait only to the character \
who performed or possesses it — never to the subject of the question simply because they appear \
in the same passage.\n\
- When a passage uses a pronoun (he, she, they) without a clear referent, \
do not assume that referent is the character being asked about. If you cannot confirm \
who a pronoun refers to, omit the claim entirely.\n\
- Every claim must be traceable to a specific numbered passage. If you cannot trace it, omit it.\n\
- Some passages may contain out-of-game player instructions: references to D&D Beyond, \
character creation, campaign links, dice mechanics, or directions addressed to players. \
These are not lore. Disregard them entirely and do not include their content in your answer."
        .to_string();

    let system_prompt = prompt_extra_rules
        .iter()
        .fold(system_prompt, |mut s, rule| {
            s.push_str(&format!("\n- {rule}"));
            s
        });

    let user_content = format!(
        "{subject_hint}\
Lore passages:\n{context}\n\
\nQuestion: {question}\n\
\nWrite a full, flowing answer in prose paragraphs. \
Answer the specific question asked directly — if it asks for a name, location, or particular fact, state that clearly before elaborating. \
Weave all relevant details from the passages into a coherent narrative. \
Never use bullet points, dashes, or list formatting."
    );

    Ok(Some(PipelineOutput {
        system_prompt,
        user_content,
        client: res.client.clone(),
        ollama_url: res.ollama_url.clone(),
        chat_model: res.chat_model.clone(),
    }))
}

const INJECTION_RESPONSE: &str = "The lore does not speak of this.";

fn is_injection_attempt(q: &str) -> bool {
    let q = q.to_lowercase();
    let triggers = [
        ("ignore",     "instruction"),
        ("ignore",     "previous"),
        ("ignore",     "above"),
        ("forget",     "instruction"),
        ("forget",     "previous"),
        ("disregard",  "instruction"),
        ("disregard",  "previous"),
        ("bypass",     "instruction"),
        ("override",   "instruction"),
    ];
    triggers.iter().any(|(a, b)| q.contains(a) && q.contains(b))
}

/// Interactive query: streams tokens to stdout as they arrive.
/// With show_context=true, prints the retrieved prompt and exits without generating.
pub async fn run(question: &str, show_context: bool) -> Result<()> {
    if is_injection_attempt(question) {
        println!("{INJECTION_RESPONSE}");
        return Ok(());
    }
    let res = QueryResources::new().await?;
    match pipeline(question, &res).await? {
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
pub async fn answer(question: &str, res: &QueryResources) -> Result<String> {
    if is_injection_attempt(question) {
        return Ok(INJECTION_RESPONSE.to_string());
    }
    match pipeline(question, res).await? {
        None => Ok("No relevant lore found for that query.".to_string()),
        Some(ctx) => generate(&ctx).await,
    }
}

/// SSE bridge: runs the pipeline and forwards tokens through an mpsc channel.
/// The serve subcommand spawns this in a task and bridges the channel to axum SSE.
fn is_connection_error(e: &anyhow::Error) -> bool {
    let msg = format!("{e:#}");
    msg.contains("error sending request")
        || msg.contains("connection refused")
        || msg.contains("tcp connect error")
        || msg.contains("os error 111")
}

pub async fn stream_to_sender(
    question: &str,
    tx: tokio::sync::mpsc::Sender<String>,
    res: Arc<QueryResources>,
) -> Result<()> {
    if is_injection_attempt(question) {
        let _ = tx.send(INJECTION_RESPONSE.to_string()).await;
        return Ok(());
    }
    let ctx = match pipeline(question, &res).await {
        Err(e) => {
            let msg = if is_connection_error(&e) {
                "The arcane conduit is dark. The Oracle does not stir — the wellspring cannot be reached.".to_string()
            } else {
                format!("⚠ Error: {e}")
            };
            let _ = tx.send(msg).await;
            return Ok(());
        }
        Ok(None) => {
            let _ = tx.send("The lore does not speak of this.".to_string()).await;
            return Ok(());
        }
        Ok(Some(ctx)) => ctx,
    };

    let response = match ctx
        .client
        .post(format!("{}/v1/chat/completions", ctx.ollama_url))
        .json(&json!({
            "model": ctx.chat_model,
            "messages": [
                {"role": "system", "content": ctx.system_prompt},
                {"role": "user",   "content": ctx.user_content}
            ],
            "num_ctx": 8192,
            "num_predict": 1500,
            "temperature": 0,
            "stream": true
        }))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            let msg = if is_connection_error(&anyhow::Error::new(e)) {
                "The arcane conduit is dark. The Oracle does not stir — the wellspring cannot be reached.".to_string()
            } else {
                "The Oracle fell silent mid-breath. Something went wrong with the telling.".to_string()
            };
            let _ = tx.send(msg).await;
            return Ok(());
        }
    };

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
            "num_predict": 1500,
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
            "num_predict": 1500,
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


fn is_scene_filtered(text: &str, markers: &[String]) -> bool {
    let t = text.to_lowercase();
    markers.iter().any(|m| t.contains(m.as_str()))
}

// Returns true if `name` appears at least once in `text` in non-possessive form
// (i.e. not immediately followed by "'s"). Filters out chunks where the entity
// is only a passing possessive reference ("Florian's group") in a passage
// otherwise about a different character.
fn has_non_possessive_mention(text: &str, name: &str) -> bool {
    let text_lower = text.to_lowercase();
    let name_lower = name.to_lowercase();
    let mut start = 0;
    while let Some(pos) = text_lower[start..].find(&name_lower) {
        let abs_end = start + pos + name_lower.len();
        let after = &text_lower[abs_end..];
        if !after.starts_with("'s") && !after.starts_with("\u{2019}s") {
            return true;
        }
        start = abs_end;
    }
    false
}

// Scores each candidate chunk against the question with a single LLM call and
// returns the top `keep` results in descending relevance order. Falls back to
// returning the input truncated to `keep` on any parse failure so the pipeline
// always continues. `entity_names` is passed so known aliases inside passages
// (e.g. "Adrastea (Lady Orvir)") are normalised before scoring.
async fn rerank(
    client: &Client,
    ollama_url: &str,
    chat_model: &str,
    question: &str,
    entity_names: &[String],
    results: Vec<SearchResult>,
    keep: usize,
) -> Vec<SearchResult> {
    if results.len() <= keep {
        return results;
    }

    // Normalise alias annotations of the form "OtherName (EntityName)" → "EntityName"
    // so the scorer recognises the character regardless of how the PDF labels them.
    let normalise = |text: &str| -> String {
        let mut out = text.to_string();
        for name in entity_names {
            // Match "Anything (Name)" → replace the whole token with just "Name"
            let pattern = format!("({name})");
            while let Some(start) = out.find(&pattern) {
                // Walk back to find the preceding alias word(s) and drop them.
                let before = &out[..start];
                let trim_end = before.trim_end();
                // Drop everything from the last space before the alias up to and
                // including the closing parenthesis of the alias annotation.
                let word_start = trim_end.rfind(' ').map(|p| p + 1).unwrap_or(0);
                let end = start + pattern.len();
                out = format!("{}{}{}", &out[..word_start], name, &out[end..]);
            }
        }
        out
    };

    // Build numbered passage list (400-char snippets to stay within token budget).
    let passages = results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let normalised = normalise(&r.text);
            let snippet: String = normalised.chars().take(400).collect();
            format!("[{}] {}", i + 1, snippet)
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let entity_list = entity_names.join(", ");
    let prompt = format!(
        "Question: {question}\n\n\
         Score each passage by how DIRECTLY it answers the question asked.\n\
         Score HIGH (8-10) for: passages that directly contain the answer — specific names, facts, \
         events, abilities, or descriptions that respond to exactly what was asked.\n\
         Score MEDIUM (4-7) for: passages with relevant supporting context about {entity_list}.\n\
         Score LOW (0-3) for: general background that does not address the question, scene dialogue, \
         planning notes, or passages primarily about other subjects.\n\n\
         Output ONLY one line per passage in this exact format: [N]: score\n\
         No explanation. No other text.\n\n\
         {passages}"
    );

    let Ok(response) = client
        .post(format!("{ollama_url}/v1/chat/completions"))
        .json(&json!({
            "model": chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "num_ctx": 8192,
            "num_predict": 256,
            "temperature": 0,
            "stream": false
        }))
        .send()
        .await
    else {
        let mut fallback = results;
        fallback.truncate(keep);
        return fallback;
    };

    let Ok(body) = response.json::<serde_json::Value>().await else {
        let mut fallback = results;
        fallback.truncate(keep);
        return fallback;
    };

    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");

    // Use original similarity score as baseline so unscored chunks aren't penalised to 0.
    let mut scores: Vec<f32> = results.iter().map(|r| r.score).collect();

    for line in content.lines() {
        let line = line.trim();
        // Accept "[N]: score", "[N] score", "N: score", "N. score"
        let (idx_str, score_str) = if let Some(rest) = line.strip_prefix('[') {
            let Some(end) = rest.find(']') else { continue };
            let after = rest[end + 1..].trim_start_matches([' ', ':']);
            (&rest[..end], after)
        } else {
            let Some(sep) = line.find(['.', ':']) else { continue };
            (&line[..sep], line[sep + 1..].trim())
        };

        let Ok(idx) = idx_str.trim().parse::<usize>() else { continue };
        let score_tok = score_str.split_whitespace().next().unwrap_or("0");
        let Ok(score) = score_tok.parse::<f32>() else { continue };

        if idx >= 1 && idx <= results.len() {
            // Normalise LLM 0-10 score to 0-1 range so it's comparable to cosine scores.
            scores[idx - 1] = score / 10.0;
        }
    }

    // Boost chunks that come from the character's own lore file.
    // e.g. "lore_alora_venyette.txt" gets a bonus when asking about "Alora Venyette".
    // Also handles multi-word names where a middle name breaks an exact slug match,
    // e.g. "Lady Orvir" → "lore_lady_adrastea_orvir.txt" (all words present individually).
    for (i, result) in results.iter().enumerate() {
        let src = result.source.to_lowercase();
        if src.starts_with("lore_") {
            for name in entity_names {
                let slug = name.to_lowercase().replace(' ', "_");
                let words: Vec<&str> = name.split_whitespace().collect();
                let all_words_match = words.len() > 1
                    && words.iter().all(|w| src.contains(&w.to_lowercase()));
                if src.contains(&slug) || all_words_match {
                    scores[i] = (scores[i] + 0.55).min(1.0);
                    break;
                }
            }
        }
    }

    // Sort indices by score descending.
    let mut order: Vec<usize> = (0..results.len()).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));

    // For character queries: drop lore files about OTHER characters before
    // truncating to `keep`. This prevents cross-character contamination
    // (e.g. Lady Vesemir's dragon gem appearing in a Lady Orvir response).
    //
    // The filter ONLY activates when a matching lore file is already present in
    // the ranked results — this distinguishes "who is Florian?" (his own lore
    // file scores into the set) from "who are the instructors at Taelreth?"
    // (no lore_taelreth.txt exists, so character lore files like lore_ali_hassan.txt
    // should be kept rather than stripped).
    //
    // Place / world lore files (lore_city_, lore_continent_, lore_region_,
    // lore_plane_) are always kept regardless.
    if !entity_names.is_empty() {
        let entity_has_own_lore = order.iter().any(|&i| {
            let src = results[i].source.to_lowercase();
            if !src.starts_with("lore_") { return false; }
            entity_names.iter().any(|name| {
                let slug = name.to_lowercase().replace(' ', "_");
                let words: Vec<&str> = name.split_whitespace().collect();
                let all_words = words.len() > 1
                    && words.iter().all(|w| src.contains(&w.to_lowercase()));
                src.contains(&slug) || all_words
            })
        });

        if entity_has_own_lore {
            order.retain(|&i| {
                let src = results[i].source.to_lowercase();
                if !src.starts_with("lore_") { return true; }
                // Place / world lore files are never filtered
                if src.starts_with("lore_city_")
                    || src.starts_with("lore_continent_")
                    || src.starts_with("lore_region_")
                    || src.starts_with("lore_plane_")
                {
                    return true;
                }
                // Keep only the queried character's own lore file
                entity_names.iter().any(|name| {
                    let slug = name.to_lowercase().replace(' ', "_");
                    let words: Vec<&str> = name.split_whitespace().collect();
                    let all_words = words.len() > 1
                        && words.iter().all(|w| src.contains(&w.to_lowercase()));
                    src.contains(&slug) || all_words
                })
            });
        }
    }

    order.truncate(keep);
    order.into_iter().map(|i| results[i].clone()).collect()
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
            .map(|(_, v)| mmr_score(query_vec, v, &selected_vecs, lambda))
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
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
