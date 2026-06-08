use anyhow::{Result, anyhow};
use candle_core::{IndexOp, Tensor};
use candle_nn::ops::softmax;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use rand::distr::Distribution;

pub fn get_logit_processor(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: u64,
) -> LogitsProcessor {
    let temperature = temperature.and_then(|v| if v < 1e-7 { None } else { Some(v) });
    match top_k {
        None => LogitsProcessor::new(
            seed,
            temperature.map(|temp| temp as f64),
            top_p.map(|tp| tp as f64),
        ),
        Some(k) => {
            let sampling = match temperature {
                None => Sampling::ArgMax,
                Some(temperature) => match top_p {
                    None => Sampling::TopK {
                        k,
                        temperature: temperature as f64,
                    },
                    Some(p) => Sampling::TopKThenTopP {
                        k,
                        p: p as f64,
                        temperature: temperature as f64,
                    },
                },
            };
            LogitsProcessor::from_sampling(seed, sampling)
        }
    }
}

pub fn use_repeat_penalty(
    repeat_penalty: f32,
    repeat_last_n: Option<usize>,
    logits: &Tensor,
    context: &[u32],
) -> Result<Tensor> {
    if repeat_penalty == 1.0 || repeat_last_n == Some(0) {
        Ok(logits.clone())
    } else {
        let start_at = if let Some(last_n) = repeat_last_n {
            context.len().saturating_sub(last_n)
        } else {
            0
        };
        Ok(candle_transformers::utils::apply_repeat_penalty(
            logits,
            repeat_penalty,
            &context[start_at..],
        )?)
    }
}

pub fn sample_weighted(prs: &[f32]) -> Result<u32> {
    let mut rng = rand::rng();
    let dist = rand::distr::weighted::WeightedIndex::new(prs).map_err(|e| {
        anyhow!(format!(
            "simple_sampel new  rand::distr::weighted::WeightedIndex Failed: {}",
            e
        ))
    })?;
    Ok(dist.sample(&mut rng) as u32)
}

/// logits shape: (dim)
pub fn simple_sample(
    logits: &Tensor,
    do_sample: bool,
    temperature: Option<f64>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    previous_token_ids: Option<&[u32]>,
    repeat_penalty: f32,
) -> Result<u32> {
    if logits.rank() != 1 {
        return Err(anyhow!("simple_sample logits need rank = 1"));
    }
    let mut logits = if repeat_penalty != 1.0
        && let Some(tokens) = previous_token_ids
    {
        use_repeat_penalty(repeat_penalty, None, logits, tokens)?
    } else {
        logits.clone()
    };
    if !do_sample {
        Ok(logits.argmax(0)?.to_scalar::<u32>()?)
    } else {
        if let Some(temp) = temperature
            && temp > 0.0
        {
            logits = logits.affine(1.0 / temp, 0.0)?;
        }
        if let Some(top_k) = top_k
            && top_k > 0
            && top_k < logits.dim(0)?
        {
            let sorted_indices = logits.arg_sort_last_dim(false)?;
            let top_k_indices = sorted_indices.narrow(0, 0, top_k)?;
            let top_k_logits = logits.gather(&top_k_indices, 0)?;
            let threshold = top_k_logits.min_all()?;
            let mask = logits.broadcast_lt(&threshold)?;
            let on_true = Tensor::new(f32::NEG_INFINITY, logits.device())?
                .to_dtype(logits.dtype())?
                .broadcast_as(mask.shape())?;
            logits = mask.where_cond(&on_true, &logits)?;
        }
        if let Some(top_p) = top_p
            && top_p > 0.0
            && top_p < 1.0
        {
            let sorted_indices = logits.arg_sort_last_dim(false)?;
            let sorted_logits = logits.gather(&sorted_indices, 0)?;
            let sorted_probs = softmax(&sorted_logits, 0)?;
            let sorted_cumsum = sorted_probs.cumsum(0)?;
            let mut mask = sorted_cumsum
                .broadcast_gt(&Tensor::new(top_p, logits.device())?.to_dtype(logits.dtype())?)?;
            // 保证数据不会被全部置为-inf
            if mask.i(0)?.to_scalar::<u8>()? == 1 {
                mask = mask.slice_scatter(&Tensor::new(&[0u8], logits.device())?, 0, 0)?;
            }
            let on_true = Tensor::new(f32::NEG_INFINITY, logits.device())?
                .to_dtype(logits.dtype())?
                .broadcast_as(mask.shape())?;
            let new_logits = mask.where_cond(&on_true, &sorted_logits)?;
            logits = logits.scatter(&sorted_indices, &new_logits, 0)?;
        }
        let probs = softmax(&logits, 0)?;

        let probs = probs.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?;
        sample_weighted(&probs)
    }
}
