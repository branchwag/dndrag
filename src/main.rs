use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod embed;
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
    },
    /// Ask a question about your DnD world
    Query {
        question: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let cli = Cli::parse();

    match cli.command {
        Command::Ingest { docs_dir } => ingest::run(&docs_dir).await?,
        Command::Query { question } => {
            let answer = query::run(&question).await?;
            println!("{}", answer);
        }
    }

    Ok(())
}
