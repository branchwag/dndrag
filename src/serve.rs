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

async fn cedarville_font() -> impl axum::response::IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "font/truetype")],
        include_bytes!("../static/fonts/cedarville-cursive.ttf").as_slice(),
    )
}

async fn homemade_apple_font() -> impl axum::response::IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "font/truetype")],
        include_bytes!("../static/fonts/homemade-apple.ttf").as_slice(),
    )
}

async fn parchment_image() -> impl axum::response::IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "image/jpeg")],
        include_bytes!("../static/images/parchment.jpg").as_slice(),
    )
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
        .route("/health", get(health))
        .route("/fonts/cedarville-cursive.ttf", get(cedarville_font))
        .route("/fonts/homemade-apple.ttf", get(homemade_apple_font))
        .route("/images/parchment.jpg", get(parchment_image));

    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("Arcane Tome listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
