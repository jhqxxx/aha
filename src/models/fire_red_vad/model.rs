use anyhow::Result;
use candle_core::{D, Tensor};
use candle_nn::{Conv1d, Linear, Module, VarBuilder, linear, linear_no_bias, ops::sigmoid};

use crate::{
    models::{
        common::modules::{conv1d_depthwise, get_conv1d},
        fire_red_vad::config::DetectModelConfig,
    },
    utils::tensor_utils::{get_mask_from_lengths, masked_fill_zeros},
};

pub struct FSMN {
    lookback_padding: usize,
    lookback_filter: Conv1d,
    lookahead_filter: Option<Conv1d>,
    n1: usize,
    s1: usize,
    n2: usize,
    s2: usize,
}

impl FSMN {
    pub fn new(
        vb: VarBuilder,
        p: usize,
        n1: usize,
        s1: usize,
        n2: usize,
        s2: usize,
    ) -> Result<Self> {
        let lookback_padding = (n1 - 1) * s1;
        let lookback_filter = get_conv1d(
            vb.pp("lookback_filter"),
            p,
            p,
            n1,
            lookback_padding,
            1,
            s1,
            p,
            false,
        )?;
        let lookahead_filter = if n2 > 0 {
            Some(get_conv1d(
                vb.pp("lookahead_filter"),
                p,
                p,
                n2,
                (n2 - 1) * s2,
                1,
                s2,
                p,
                false,
            )?)
        } else {
            None
        };
        Ok(Self {
            lookback_padding,
            lookback_filter,
            lookahead_filter,
            n1,
            s1,
            n2,
            s2,
        })
    }

    pub fn forward(
        &self,
        inputs: &Tensor,
        mask: Option<&Tensor>,
        cache: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let t = inputs.dim(1)?;
        let inputs = if let Some(mask) = mask {
            masked_fill_zeros(inputs, mask)?
        } else {
            inputs.clone()
        };
        // [N, T, P] -> [N, P, T]
        let residual = inputs.permute((0, 2, 1))?.contiguous()?;
        let inputs = if let Some(cache) = cache {
            Tensor::cat(&[cache, &residual], 2)?
        } else {
            residual.clone()
        };
        let len = inputs.dim(D::Minus1)?;
        let new_cache = if len > self.lookback_padding {
            let start = len - self.lookback_padding;
            inputs.narrow(D::Minus1, start, self.lookback_padding)?
        } else {
            inputs.clone()
        };
        // conv1d_depthwise仅支持dilation=1的情况
        let lookback = if self.s1 == 1 {
            let inputs =
                inputs.pad_with_zeros(D::Minus1, self.lookback_padding, self.lookback_padding)?;
            conv1d_depthwise(
                &inputs,
                self.lookback_filter.weight(),
                self.lookback_filter.bias(),
            )?
        } else {
            self.lookback_filter.forward(&inputs)?
        };
        let mut memory = if self.n1 > 1 {
            let len = lookback.dim(D::Minus1)? - (self.n1 - 1) * self.s1;
            let mut lookback = lookback.narrow(D::Minus1, 0, len)?;
            if let Some(cache) = cache {
                let start = cache.dim(2)?;
                let len = lookback.dim(D::Minus1)? - start;
                lookback = lookback.narrow(D::Minus1, start, len)?;
            }
            residual.add(&lookback)?
        } else {
            residual.add(&lookback)?
        };
        if self.n2 > 0
            && t > 1
            && let Some(ahead_filter) = &self.lookahead_filter
        {
            let lookahead = if self.s2 == 1 {
                let inputs = inputs.pad_with_zeros(
                    D::Minus1,
                    self.lookback_padding,
                    self.lookback_padding,
                )?;
                conv1d_depthwise(&inputs, ahead_filter.weight(), ahead_filter.bias())?
            } else {
                ahead_filter.forward(&inputs)?
            };
            let start = self.n2 * self.s2;
            let len = lookahead.dim(D::Minus1)? - start;
            let lookahead = lookahead.narrow(D::Minus1, start, len)?;
            let lookahead = lookahead.pad_with_zeros(D::Minus1, 0, self.s2)?;
            memory = memory.add(&lookahead)?;
        }
        memory = memory.permute((0, 2, 1))?.contiguous()?;
        if let Some(mask) = mask {
            memory = masked_fill_zeros(&memory, mask)?;
        }
        Ok((memory, new_cache))
    }
}

struct DFSMNBlock {
    fc1: Linear, // linear + relu
    fc2: Linear,
    fsmn: FSMN,
}

impl DFSMNBlock {
    pub fn new(
        vb: VarBuilder,
        h: usize,
        p: usize,
        n1: usize,
        s1: usize,
        n2: usize,
        s2: usize,
    ) -> Result<Self> {
        let fc1 = linear(p, h, vb.pp("fc1.0"))?;
        let fc2 = linear_no_bias(h, p, vb.pp("fc2"))?;
        let fsmn = FSMN::new(vb.pp("fsmn"), p, n1, s1, n2, s2)?;
        Ok(Self { fc1, fc2, fsmn })
    }

    pub fn forward(
        &self,
        inputs: &Tensor,
        mask: Option<&Tensor>,
        cache: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let residual = inputs.clone();
        let h = self.fc1.forward(inputs)?.relu()?;
        let p = self.fc2.forward(&h)?;
        let (memory, new_cache) = self.fsmn.forward(&p, mask, cache)?;
        let output = memory.add(&residual)?;
        Ok((output, new_cache))
    }
}

#[allow(clippy::upper_case_acronyms)]
struct DFSMN {
    fc1: Linear, // linear + relu
    fc2: Linear, // linear + relu
    fsmn1: FSMN,
    fsmns: Vec<DFSMNBlock>,
    dnns: Vec<Linear>, // linear + relu
}

impl DFSMN {
    pub fn new(
        vb: VarBuilder,
        d: usize,
        r: usize,
        m: usize,
        h: usize,
        p: usize,
        n1: usize,
        s1: usize,
        n2: usize,
        s2: usize,
    ) -> Result<Self> {
        let fc1 = linear(d, h, vb.pp("fc1.0"))?;
        let fc2 = linear(h, p, vb.pp("fc2.0"))?;
        let fsmn1 = FSMN::new(vb.pp("fsmn1"), p, n1, s1, n2, s2)?;
        let mut fsmns = vec![];
        let vb_fsmns = vb.pp("fsmns");
        for i in 0..(r - 1) {
            let block = DFSMNBlock::new(vb_fsmns.pp(i), h, p, n1, s1, n2, s2)?;
            fsmns.push(block);
        }
        let vb_dnns = vb.pp("dnns");
        let mut dnns = vec![];
        for i in 0..m {
            let in_dim = if i == 0 { p } else { h };
            let dnn = linear(in_dim, h, vb_dnns.pp(i))?;
            dnns.push(dnn);
        }
        Ok(Self {
            fc1,
            fc2,
            fsmn1,
            fsmns,
            dnns,
        })
    }

    pub fn forward(
        &self,
        inputs: &Tensor,
        input_lengths: Option<&Tensor>,
        caches: Option<&Vec<Tensor>>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mask = if let Some(input_lengths) = input_lengths {
            Some(get_mask_from_lengths(input_lengths)?)
        } else {
            None
        };
        let h = self.fc1.forward(inputs)?.relu()?;
        let p = self.fc2.forward(&h)?.relu()?;
        let mut new_caches = vec![];
        let cache = caches.map(|caches| &caches[0]);
        let (mut memory, mut new_cache) = self.fsmn1.forward(&p, mask.as_ref(), cache)?;
        new_caches.push(new_cache);
        for (i, fsmn) in (1..).zip(self.fsmns.iter()) {
            let cache = caches.map(|caches| &caches[i]);
            (memory, new_cache) = fsmn.forward(&memory, mask.as_ref(), cache)?;
            new_caches.push(new_cache);
        }
        for dnn in &self.dnns {
            memory = dnn.forward(&memory)?.relu()?;
        }
        Ok((memory, new_caches))
    }
}

pub struct DetectModel {
    dfsmn: DFSMN,
    out: Linear,
}

impl DetectModel {
    pub fn new(vb: VarBuilder, cfg: DetectModelConfig) -> Result<Self> {
        let dfsmn = DFSMN::new(
            vb.pp("dfsmn"),
            cfg.idim,
            cfg.r,
            cfg.m,
            cfg.h,
            cfg.p,
            cfg.n1,
            cfg.s1,
            cfg.n2,
            cfg.s2,
        )?;
        let out = linear(cfg.h, cfg.odim, vb.pp("out"))?;
        Ok(Self { dfsmn, out })
    }

    pub fn forward(
        &self,
        feat: &Tensor,
        caches: Option<&Vec<Tensor>>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let (x, new_caches) = self.dfsmn.forward(feat, None, caches)?;
        let logits = self.out.forward(&x)?;
        let probs = sigmoid(&logits)?;
        Ok((probs, new_caches))
    }
}
