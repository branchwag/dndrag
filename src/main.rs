use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod config;
mod embed;
mod eval;
mod ingest;
mod query;
mod serve;
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
        /// Wrap multi-word questions in quotes: dnd_rag query "Who is Alora?"
        question: String,
        /// Print the retrieved context sent to the model instead of generating a response
        #[arg(long)]
        show_context: bool,
    },
    /// Run labeled Q&A pairs and report how many the system answers correctly
    Eval {
        #[arg(short, long, default_value = "eval.json")]
        eval_file: PathBuf,
    },
    /// Start the HTTP server and serve the browser front-end on the given port
    Serve {
        #[arg(short, long, default_value_t = 3000u16)]
        port: u16,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env before tracing so RUST_LOG from .env takes effect.
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_target(false)
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Ingest { docs_dir, fresh } => {
            if !docs_dir.exists() {
                anyhow::bail!("docs directory not found: {}", docs_dir.display());
            }
            ingest::run(&docs_dir, fresh).await?
        }
        Command::Query { question, show_context } => query::run(&question, show_context).await?,
        Command::Eval { eval_file } => eval::run(&eval_file).await?,
        Command::Serve { port } => serve::run(port).await?,
    }

    Ok(())
}
