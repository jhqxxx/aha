use anyhow::{Result, anyhow};

pub trait TextEmbeddingBackend {
    fn embed_texts(&mut self, input: &[String]) -> Result<Vec<Vec<f32>>>;
}

pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

pub fn mean_pool(embeddings: &[Vec<f32>]) -> Result<Vec<f32>> {
    let first = embeddings
        .first()
        .ok_or_else(|| anyhow!("embedding hidden state is empty"))?;
    let mut pooled = vec![0f32; first.len()];
    for row in embeddings {
        if row.len() != first.len() {
            return Err(anyhow!("inconsistent embedding width in hidden state"));
        }
        for (idx, value) in row.iter().enumerate() {
            pooled[idx] += *value;
        }
    }
    let inv = 1.0f32 / embeddings.len() as f32;
    for value in &mut pooled {
        *value *= inv;
    }
    Ok(pooled)
}

pub fn cosine_similarity(lhs: &[f32], rhs: &[f32]) -> Result<f32> {
    if lhs.len() != rhs.len() {
        return Err(anyhow!("embedding dimension mismatch"));
    }
    Ok(lhs.iter().zip(rhs.iter()).map(|(l, r)| l * r).sum::<f32>())
}
