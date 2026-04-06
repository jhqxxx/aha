use anyhow::Result;
use candle_core::Tensor;
pub mod embedding;
pub mod generate;
pub mod gguf;
pub mod model_mapping;
pub mod modules;

/// 多模态模型的特征数据
/// 每个模型数据不一样
/// 需按顺序存放与取用
#[derive(Clone, Debug)]
pub struct MultiModalData {
    pub data_vec: Vec<Option<Tensor>>,
}
impl MultiModalData {
    pub fn new(data_vec: Vec<Option<Tensor>>) -> Self {
        Self { data_vec }
    }
}

#[allow(unused)]
pub trait InferenceModel {
    /// 初始前向传播（考虑多模态输入）
    /// 默认实现无特殊数据
    fn forward_initial(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        data: MultiModalData,
    ) -> Result<Tensor> {
        Self::forward_step(self, input_ids, seqlen_offset)
    }

    /// 后续前向传播（自回归步骤）
    fn forward_step(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor>;

    /// 清理 KV cache
    fn clear_cache(&mut self);

    /// 获取结束 token IDs
    fn stop_token_ids(&self) -> Vec<u32>;
}
