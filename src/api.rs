use std::pin::pin;
use std::sync::{Arc, OnceLock};

use aha::models::{GenerateModel, ModelInstance, WhichModel, load_model};
use aha::utils::string_to_static_str;
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use rocket::futures::StreamExt;
use rocket::serde::json::Json;
use rocket::{
    Request,
    futures::Stream,
    http::{ContentType, Status},
    post,
    response::{Responder, stream::TextStream},
};
use tokio::sync::RwLock;

static MODEL: OnceLock<Arc<RwLock<ModelInstance<'static>>>> = OnceLock::new();

pub fn init(model_type: WhichModel, path: String) -> anyhow::Result<()> {
    let model_path = string_to_static_str(path);
    let model = load_model(model_type, model_path)?;
    MODEL.get_or_init(|| Arc::new(RwLock::new(model)));
    Ok(())
}

pub(crate) enum Response<R: Stream<Item = String> + Send> {
    Stream(TextStream<R>),
    Text(String),
    Error(String),
}

impl<'r, 'o: 'r, R> Responder<'r, 'o> for Response<R>
where
    R: Stream<Item = String> + Send + 'o,
    'r: 'o,
{
    fn respond_to(self, req: &'r Request<'_>) -> rocket::response::Result<'o> {
        match self {
            Response::Stream(stream) => stream.respond_to(req),
            Response::Text(text) => text.respond_to(req),
            Response::Error(e) => {
                let mut res = rocket::response::Response::new();
                res.set_status(Status::InternalServerError);
                res.set_header(ContentType::JSON);
                res.set_sized_body(e.len(), std::io::Cursor::new(e));
                Ok(res)
            }
        }
    }
}

#[post("/completions", data = "<req>")]
pub(crate) async fn chat(
    req: Json<ChatCompletionParameters>,
) -> (ContentType, Response<impl Stream<Item = String> + Send>) {
    match req.stream {
        Some(false) => {
            let response = {
                let model_ref = MODEL
                    .get()
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("model not init"))
                    .unwrap();
                model_ref.write().await.generate(req.into_inner())
            };
            match response {
                Ok(res) => {
                    let response_str = serde_json::to_string(&res).unwrap();
                    (ContentType::Text, Response::Text(response_str))
                }
                Err(e) => (ContentType::Text, Response::Error(e.to_string())),
            }
        }
        _ => {
            let text_stream = TextStream! {
                let model_ref = MODEL.get().cloned().ok_or_else(|| anyhow::anyhow!("model not init")).unwrap();
                let mut guard = model_ref.write().await;
                let stream_result = guard.generate_stream(req.into_inner());
                match stream_result {
                    Ok(stream) => {
                        let mut stream = pin!(stream);
                        while let Some(result) = stream.next().await {
                            match result {
                                Ok(chunk) => {
                                    if let Ok(json_str) = serde_json::to_string(&chunk) {
                                        yield format!("data: {}\n\n", json_str);
                                    }
                                }
                                Err(e) => {
                                    yield format!("data: {{\"error\": \"{}\"}}\n\n", e);
                                    break;
                                }
                            }
                        }
                        yield "data: [DONE]\n\n".to_string();
                    },
                    Err(e) => {
                        yield format!("event: error\ndata: {}\n\n", e.to_string());
                    }
                }
            };
            (ContentType::EventStream, Response::Stream(text_stream))
        }
    }
}
