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
        #[arg(short, long, default_value = "3000")]
        port: u16,
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
        Command::Query { question, show_context } => query::run(&question, show_context).await?,
        Command::Eval { eval_file } => eval::run(&eval_file).await?,
        Command::Serve { port } => serve::run(port).await?,
    }

    Ok(())
}
