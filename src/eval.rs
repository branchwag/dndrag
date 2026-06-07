use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct EvalCase {
    question: String,
    must_mention: Vec<String>,
    #[serde(default)]
    must_not_mention: Vec<String>,
}

/// Loads eval.json, runs each question through the RAG pipeline, and reports
/// pass/fail based on must_mention / must_not_mention keyword checks.
pub async fn run(eval_file: &Path) -> Result<()> {
    let content = std::fs::read_to_string(eval_file)
        .map_err(|e| anyhow::anyhow!("Could not read {}: {e}", eval_file.display()))?;
    let cases: Vec<EvalCase> = serde_json::from_str(&content)
        .map_err(|e| anyhow::anyhow!("eval.json parse error: {e}"))?;

    if cases.is_empty() {
        println!("No eval cases found in {}", eval_file.display());
        return Ok(());
    }

    println!("Running eval: {} questions\n", cases.len());

    let mut passed = 0usize;
    let total = cases.len();

    for case in &cases {
        let answer = match crate::query::answer(&case.question).await {
            Ok(a) => a,
            Err(e) => {
                println!("[ERROR] {}\n        {e}\n", case.question);
                continue;
            }
        };
        let answer_lower = answer.to_lowercase();

        let missing: Vec<&str> = case
            .must_mention
            .iter()
            .filter(|m| !answer_lower.contains(m.to_lowercase().as_str()))
            .map(String::as_str)
            .collect();

        let hallucinated: Vec<&str> = case
            .must_not_mention
            .iter()
            .filter(|m| answer_lower.contains(m.to_lowercase().as_str()))
            .map(String::as_str)
            .collect();

        let ok = missing.is_empty() && hallucinated.is_empty();
        if ok {
            passed += 1;
        }

        let status = if ok { "PASS" } else { "FAIL" };
        println!("[{status}] {}", case.question);
        if !missing.is_empty() {
            println!("       missing:      {}", missing.join(", "));
        }
        if !hallucinated.is_empty() {
            println!("       hallucinated: {}", hallucinated.join(", "));
        }
    }

    let pct = if total > 0 { 100.0 * passed as f64 / total as f64 } else { 0.0 };
    println!("\nScore: {passed}/{total} ({pct:.0}%)");
    Ok(())
}
