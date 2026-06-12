# 测试数据生成脚本 (PowerShell)
# 用于生成多模型测试所需的各种测试数据

param(
    [string]$OutputDir = ".\test-data",
    [switch]$Force
)

# 颜色定义
$ColorSuccess = "Green"
$ColorError = "Red"
$ColorInfo = "Cyan"

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor $ColorSuccess
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor $ColorInfo
}

Write-Host "`n========================================" -ForegroundColor $ColorInfo
Write-Host "  aha 测试数据生成器" -ForegroundColor $ColorInfo
Write-Host "========================================`n" -ForegroundColor $ColorInfo

# 创建输出目录
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
    Write-Info "创建输出目录: $OutputDir"
} elseif ($Force) {
    Write-Info "清理现有目录: $OutputDir"
    Remove-Item -Recurse -Force "$OutputDir\*"
} else {
    Write-Info "使用现有目录: $OutputDir"
}

# ========================================
# 1. 生成测试文本数据
# ========================================
Write-Host "`n[1/5] 生成测试文本数据..." -ForegroundColor $ColorInfo

$textData = @{
    short_prompts = @(
        "你好",
        "介绍一下 Rust 编程语言",
        "计算 1+1",
        "今天天气怎么样？",
        "写一首短诗"
    )
    medium_prompts = @(
        "请解释什么是人工智能，以及它在日常生活中的应用。",
        "如何用 Rust 编写一个高性能的 Web 服务器？请给出关键步骤。",
        "分析一下气候变化对全球经济的影响。",
        "请比较 Python、JavaScript 和 Rust 三种编程语言的优缺点。",
        "描述一下量子计算的基本原理和潜在应用场景。"
    )
    long_prompts = @(
        "请详细阐述深度学习在自然语言处理领域的最新进展，包括 Transformer 架构、BERT 模型、GPT 系列等技术的发展脉络，以及它们在实际应用中的表现和挑战。",
        "从历史、技术、经济和社会多个角度，全面分析区块链技术对金融行业的颠覆性影响，包括加密货币、DeFi、NFT 等创新应用。",
        "探讨未来 10 年人工智能可能带来的社会变革，包括就业结构变化、教育体系改革、医疗诊断进步等方面，并提出应对策略。"
    )
    code_examples = @(
        "fn main() { println!(`"Hello, world!`"); }",
        "let x = vec![1, 2, 3, 4, 5]; let sum: i32 = x.iter().sum();",
        "async fn fetch_data(url: &str) -> Result<String, Error> { reqwest::get(url).await?.text().await }"
    )
}

# 保存文本数据
$textData | ConvertTo-Json -Depth 10 | Out-File "$OutputDir\text_prompts.json" -Encoding UTF8
Write-Success "文本提示数据已保存"

# 生成单独的文本文件
for ($i = 0; $i -lt $textData.short_prompts.Count; $i++) {
    $textData.short_prompts[$i] | Out-File "$OutputDir\prompt_short_$($i+1).txt" -Encoding UTF8
}

for ($i = 0; $i -lt $textData.medium_prompts.Count; $i++) {
    $textData.medium_prompts[$i] | Out-File "$OutputDir\prompt_medium_$($i+1).txt" -Encoding UTF8
}

Write-Success "生成了 $($textData.short_prompts.Count) 个短提示文件"
Write-Success "生成了 $($textData.medium_prompts.Count) 中等长度提示文件"

# ========================================
# 2. 生成测试配置模板
# ========================================
Write-Host "`n[2/5] 生成测试配置模板..." -ForegroundColor $ColorInfo

# API 请求配置
$apiConfigs = @{
    chat_completion = @{
        model = "Qwen3"
        messages = @(
            @{ role = "user"; content = "Hello" }
        )
        temperature = 0.7
        max_tokens = 100
        stream = $false
    }
    chat_streaming = @{
        model = "Qwen3"
        messages = @(
            @{ role = "user"; content = "Write a poem" }
        )
        temperature = 0.8
        max_tokens = 200
        stream = $true
    }
    embedding = @{
        model = "Qwen3-Embedding"
        input = "This is a test sentence for embedding."
    }
    asr = @{
        model = "GlmAsrNano"
        language = "zh"
        response_format = "json"
    }
}

$apiConfigs | ConvertTo-Json -Depth 10 | Out-File "$OutputDir\api_configs.json" -Encoding UTF8
Write-Success "API 配置模板已保存"

# ========================================
# 3. 生成性能测试场景
# ========================================
Write-Host "`n[3/5] 生成性能测试场景..." -ForegroundColor $ColorInfo

$performanceScenarios = @(
    @{
        name = "低负载"
        concurrent_users = 1
        requests_per_user = 10
        description = "单用户顺序请求，测试基础延迟"
    },
    @{
        name = "中负载"
        concurrent_users = 5
        requests_per_user = 10
        description = "适度并发，模拟正常使用场景"
    },
    @{
        name = "高负载"
        concurrent_users = 10
        requests_per_user = 20
        description = "高并发压力测试，评估系统极限性能"
    },
    @{
        name = "超高负载"
        concurrent_users = 20
        requests_per_user = 50
        description = "极端压力测试，检测系统瓶颈"
    }
)

$performanceScenarios | ConvertTo-Json -Depth 10 | Out-File "$OutputDir\performance_scenarios.json" -Encoding UTF8
Write-Success "性能测试场景已保存"

# ========================================
# 4. 生成测试报告模板
# ========================================
Write-Host "`n[4/5] 生成测试报告模板..." -ForegroundColor $ColorInfo

$reportTemplate = @{
    test_info = @{
        date = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        hostname = $env:COMPUTERNAME
        os = "$($env:OS) $($env:PROCESSOR_ARCHITECTURE)"
        cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
        memory_gb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)
    }
    service_config = @{
        models_loaded = @()
        port = 10100
        gpu_enabled = $false
        flash_attention = $false
    }
    results = @{
        latency = @{
            min_ms = 0
            max_ms = 0
            avg_ms = 0
            p50_ms = 0
            p90_ms = 0
            p95_ms = 0
            p99_ms = 0
        }
        throughput = @{
            total_requests = 0
            successful_requests = 0
            failed_requests = 0
            requests_per_second = 0
        }
        resource_usage = @{
            memory_mb = 0
            cpu_percent = 0
            gpu_memory_mb = 0
        }
    }
}

$reportTemplate | ConvertTo-Json -Depth 10 | Out-File "$OutputDir\report_template.json" -Encoding UTF8
Write-Success "测试报告模板已保存"

# ========================================
# 5. 生成测试清单
# ========================================
Write-Host "`n[5/5] 生成测试清单..." -ForegroundColor $ColorInfo

$testChecklist = @"
# aha 多模型测试清单

## 前置条件
- [ ] Rust 工具链已安装
- [ ] aha 已编译 (cargo build --release)
- [ ] 至少一个模型已下载
- [ ] PowerShell 执行策略允许运行脚本

## 单元测试
- [ ] cargo test test_multi_model_manager_init
- [ ] cargo test test_shared_tokenizer_cache
- [ ] cargo test test_concurrent_model_access
- [ ] cargo test test_stress_concurrent_requests

## 功能测试
- [ ] 启动多模型服务
- [ ] 运行 test_multi_model.ps1
- [ ] 验证所有 API 端点正常
- [ ] 测试流式输出

## 性能测试
- [ ] 运行 benchmark_multi_model.ps1
- [ ] 记录延迟统计数据
- [ ] 记录吞吐量数据
- [ ] 监控系统资源使用

## 集成测试
- [ ] 测试两个模型并行运行
- [ ] 测试三个模型并行运行
- [ ] 验证模型间无干扰
- [ ] 测试长时间运行稳定性

## 回归测试
- [ ] 确保向后兼容性
- [ ] 验证单模型模式仍正常工作
- [ ] 检查内存泄漏
- [ ] 验证错误处理

## 文档
- [ ] 更新测试结果到 README
- [ ] 记录性能基准数据
- [ ] 标注已知问题和限制
"@

$testChecklist | Out-File "$OutputDir\TEST_CHECKLIST.md" -Encoding UTF8
Write-Success "测试清单已保存"

# ========================================
# 生成总结
# ========================================
Write-Host "`n========================================" -ForegroundColor $ColorInfo
Write-Host "  测试数据生成完成！" -ForegroundColor $ColorInfo
Write-Host "========================================`n" -ForegroundColor $ColorInfo

Write-Host "生成的文件:" -ForegroundColor $ColorInfo
Get-ChildItem $OutputDir | ForEach-Object {
    $size = if ($_.Length -gt 1KB) {
        "$([math]::Round($_.Length / 1KB, 2)) KB"
    } else {
        "$($_.Length) B"
    }
    Write-Host "  - $($_.Name) ($size)" -ForegroundColor Gray
}

Write-Host "`n下一步:" -ForegroundColor $ColorInfo
Write-Host "  1. 查看测试清单: $OutputDir\TEST_CHECKLIST.md" -ForegroundColor Gray
Write-Host "  2. 启动多模型服务" -ForegroundColor Gray
Write-Host "  3. 运行功能测试: .\tests\test_multi_model.ps1" -ForegroundColor Gray
Write-Host "  4. 运行性能测试: .\tests\benchmark_multi_model.ps1`n" -ForegroundColor Gray

Write-Success "所有测试数据准备就绪！`n"
