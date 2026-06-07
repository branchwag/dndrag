use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod embed;
mod eval;
mod ingest;
mod query;
mod store;

#[derive(Parser)]
#[command(name = "dnd_rag", about = "RAG pipeline for DnD campaign lore")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Ingest PDFs from a directory into the vector store
    Ingest {
        #[arg(short, long, default_value = "docs")]
        docs_dir: PathBuf,
        /// Wipe and rebuild the collection from scratch (default: incremental upsert)
        #[arg(long)]
        fresh: bool,
    },
    /// Ask a question about your DnD world (streams response to stdout)
    Query {
        question: String,
    },
    /// Run labeled Q&A pairs and report how many the system answers correctly
    Eval {
        #[arg(short, long, default_value = "eval.json")]
        eval_file: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Traces go to stderr; control verbosity with RUST_LOG=info (or =debug, =warn).
    tracing_subscriber::fmt()
        .with_target(false)
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    dotenvy::dotenv().ok();
    let cli = Cli::parse();

    match cli.command {
        Command::Ingest { docs_dir, fresh } => ingest::run(&docs_dir, fresh).await?,
        Command::Query { question } => query::run(&question).await?,
        Command::Eval { eval_file } => eval::run(&eval_file).await?,
    }

    Ok(())
}
