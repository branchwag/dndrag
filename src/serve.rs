use anyhow::Result;
use axum::{
    extract::Json,
    response::{
        sse::{Event, KeepAlive, Sse},
        Html,
    },
    routing::{get, post},
    Router,
};
use futures_util::StreamExt as _;
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

#[derive(Deserialize)]
struct QueryRequest {
    question: String,
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

async fn health() -> &'static str {
    "ok"
}

async fn query_sse(
    Json(req): Json<QueryRequest>,
) -> Sse<impl futures_util::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, rx) = mpsc::channel::<String>(64);

    tokio::spawn(async move {
        let _ = crate::query::stream_to_sender(&req.question, tx).await;
    });

    let stream = ReceiverStream::new(rx).map(|token| Ok(Event::default().data(token)));
    Sse::new(stream).keep_alive(KeepAlive::default())
}

pub async fn run(port: u16) -> Result<()> {
    let app = Router::new()
        .route("/", get(index))
        .route("/query", post(query_sse))
        .route("/health", get(health));

    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("Arcane Tome listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
