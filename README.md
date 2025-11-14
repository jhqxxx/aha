# aha 
一个基于 Candle 框架的 Rust 模型推理库，提供高效、易用的多模态模型推理能力。

## 特性
* 🚀 高性能推理 - 基于 Candle 框架，提供高效的张量计算和模型推理
* 🎯 多模型支持 - 集成视觉、语言和语音多模态模型
* 🔧 易于使用 - 简洁的 API 设计，快速上手
* 🛡️ 内存安全 - 得益于 Rust 的所有权系统，确保内存安全
* 📦 轻量级 - 最小化依赖，编译产物小巧
* ⚡ GPU 加速 - 可选 CUDA 支持
* 🧠 注意力优化 - 可选 Flash Attention 支持，优化长序列处理

## 支持的模型
### 当前已实现
* Qwen2.5VL - 阿里通义千问 2.5 多模态大语言模型
* MiniCPM4 - 面壁智能 MiniCPM 系列语言模型
* VoxCPM - 面壁智能语音生成模型
* Qwen3VL - 阿里通义千问 3 多模态大语言模型

## 计划支持
我们持续扩展支持的模型列表，欢迎贡献！

## 环境依赖
1. ffmpeg: 
* ubuntu/WSL
```bash
sudo apt-get update
sudo apt-get install -y pkg-config ffmpeg libavutil-dev libavcodec-dev libavformat-dev libavfilter-dev libavdevice-dev libswresample-dev libswscale-dev
```
* windows参考： https://github.com/zmwangx/rust-ffmpeg/wiki/Notes-on-building

## 安装
### 作为库使用
* cargo add aha
* 或者在Cargo.toml中添加
```toml
[dependencies]
aha = { git = "https://github.com/jhqxxx/aha.git" }

# 启用 CUDA 支持（可选）
aha = { git = "https://github.com/jhqxxx/aha.git", features = ["cuda"] }

# 启用Flash Attention 支持（可选）
aha = { git = "https://github.com/jhqxxx/aha.git", features = ["cuda", "flash-attn"] }
```

#### VoxCPM使用示例
```rust
use aha::models::voxcpm::generate::VoxCPMGenerate;
use aha::utils::audio_utils::save_wav;
use anyhow::Result;

fn main() -> Result<()> {
    let model_path = "xxx/openbmb/VoxCPM-0.5B/";
    
    let mut voxcpm_generate = VoxCPMGenerate::init(model_path, None, None)?;
    
    let generate = voxcpm_generate.generate(
        "太阳当空照，花儿对我笑，小鸟说早早早".to_string(),
        None,
        None,
        2,
        100,
        10,
        2.0,
        false,
        6.0,
    )?;

    let _ = save_wav(&generate, "voxcpm.wav")?;
    Ok(())
}
```

### 从源码构建运行测试
```bash
git clone https://github.com/jhqxxx/aha.git
cd aha
# 修改测试用例中模型路径
# 运行 Qwen3VL 示例
cargo test -F cuda qwen3vl_generate -- --nocapture

# 运行 MiniCPM4 示例  
cargo test -F cuda minicpm_generate -- --nocapture

# 运行 VoxCPM 示例
cargo test -F cuda voxcpm_generate -- --nocapture
```

### 从源码构建部署
```bash
git clone https://github.com/jhqxxx/aha.git
cd aha
git checkout deploy
```

#### cargo run 运行参数说明
##### 基本用法
```bash
cargo run -F cuda -- [参数]
```
##### 参数详解
1. 端口设置
-----
    -p, --port <PORT>
* 设置HTTP服务监听的端口号
* 默认值：10100
* 示例：--port 8080 或 -p 8080

2. 模型选择（必选）
-----
    -m, --model <MODEL>
* 指定要加载的模型类型
* 可选值：
    * minicpm4-0.5b：MiniCPM4-0.5B 模型
    * qwen2.5vl-3b：Qwen2.5-VL-3B 模型
    * qwen3vl-2b：Qwen3-VL-2B 模型
* 示例：--model minicpm4-0.5b 或 -m qwen3vl-2b

3. 权重路径
-----
    --weight-path <WEIGHT_PATH>
* 指定本地模型权重文件路径
* 如果指定此参数，则跳过模型下载步骤
* 示例：--weight-path /path/to/model/dir

4. 保存路径
-----
    --save-dir <SAVE_DIR>
* 指定模型下载保存的目录
* 默认保存在用户主目录下的 .aha 文件夹中
* 示例：--save-dir /custom/model/path

5. 下载重试次数
----- 
    --download-retries <DOWNLOAD_RETRIES>
* 设置模型下载失败时的最大重试次数
* 默认值：3次
* 示例：--download-retries 5

##### 注意事项
* 参数前需要使用双横线 -- 分隔 cargo 命令和应用程序参数
* 模型参数 (--model 或 -m) 是必需的
* 如果未指定 --weight-path，程序会自动下载指定模型
* 下载的模型默认保存在 ~/.aha/ 目录下（除非指定了 --save-dir）

## 开发
### 项目结构
```text
.
├── Cargo.toml
├── README.md
├── src
│   ├── chat_template
│   ├── models
│   │   ├── common
│   │   ├── minicpm4
│   │   ├── qwen2_5vl
│   │   ├── qwen3vl
│   │   ├── voxcpm
│   │   └── mod.rs
│   ├── position_embed
│   ├── tokenizer
│   ├── utils
│   └── lib.rs
└── tests
    ├── test_minicpm4.rs
    ├── test_qwen2_5vl.rs
    └── test_voxcpm.rs
```

### 添加新模型
* 在src/models/创建新模型文件
* 在src/models/mod.rs中导出
* 在tests/中添加测试和示例

## 许可证
本项目采用 Apache License, Version 2.0 许可证 - 查看 [LICENSE](./LICENSE) 文件了解详情。

## 致谢
* [Candle](https://github.com/huggingface/candle) - 优秀的 Rust 机器学习框架
* 所有模型的原作者和贡献者

## 支持
#### 如果你遇到问题：
1. 查看 Issues 是否已有解决方案
2. 提交新的 Issue，包含详细描述和复现步骤

## 更新日志
### v0.1.1
* 添加 Qwen3VL 模型

### v0.1.0
* 初始版本发布
* 支持 Qwen2.5VL, MiniCPM4, VoxCPM 模型


⭐ 如果这个项目对你有帮助，请给我们一个 Star！