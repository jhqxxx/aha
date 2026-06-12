# aha 多模型并行测试套件

本目录包含 aha 多模型并行运行功能的完整测试套件。

## 📁 文件说明

### Rust 集成测试
- **`test_multi_model.rs`** - Rust 单元测试和集成测试
  - 多模型管理器初始化测试
  - Tokenizer 缓存功能测试
  - 模型注册和查询测试
  - 并发访问测试
  - 性能基准测试
  - 压力测试

### PowerShell 功能测试
- **`test_multi_model.ps1`** - 端到端功能测试脚本
  - 服务状态检查
  - 模型列表 API 验证
  - 文本生成测试
  - 流式输出测试
  - 并发请求测试
  - 健康检查
  - 模型信息查询

### PowerShell 性能测试
- **`benchmark_multi_model.ps1`** - 性能基准测试脚本
  - 预热阶段
  - 单请求延迟测试（P50/P90/P95/P99）
  - 吞吐量测试
  - 资源使用监控
  - 性能分析报告

## 🚀 快速开始

### 1. 运行 Rust 集成测试

```powershell
# 运行所有多模型测试
cargo test test_multi_model

# 运行特定测试
cargo test test_multi_model_manager_init

# 显示详细输出
cargo test test_multi_model -- --nocapture

# 运行性能相关测试
cargo test test_stress_concurrent_requests
```

### 2. 启动多模型服务

在运行 PowerShell 测试之前，需要先启动多模型服务：

```powershell
# 基本用法：加载两个模型
aha serv -m Qwen/Qwen3-0.6B -m GLM-ASR-Nano -p 10100

# 加载三个模型
aha serv `
  -m Qwen/Qwen3-0.6B `
  -m GLM-ASR-Nano `
  -m PaddleOCR-VL `
  -p 10100

# GPU 加速（需要预先编译）
./target/release/aha serv -m Qwen/Qwen3-0.6B -m GLM-ASR-Nano -p 10100
```

### 3. 运行功能测试

```powershell
# 使用默认配置
.\tests\test_multi_model.ps1

# 指定端口
.\tests\test_multi_model.ps1 -Port 10100

# 自定义超时时间
.\tests\test_multi_model.ps1 -TimeoutSeconds 60
```

### 4. 运行性能基准测试

```powershell
# 使用默认配置
.\tests\benchmark_multi_model.ps1

# 自定义参数
.\tests\benchmark_multi_model.ps1 `
  -Port 10100 `
  -WarmupRequests 5 `
  -BenchmarkRequests 20 `
  -ConcurrentUsers 10
```

## 📊 测试覆盖范围

### 单元测试
- ✅ 多模型管理器初始化
- ✅ 共享资源创建
- ✅ Tokenizer 缓存机制
- ✅ 模型注册表操作
- ✅ 并发安全性验证

### 集成测试
- ✅ 完整工作流测试
- ✅ 多模型并发访问
- ✅ 内存共享优化验证
- ✅ 压力测试（50+ 并发请求）

### 功能测试
- ✅ HTTP API 端点验证
- ✅ 文本生成（chat completions）
- ✅ 流式输出（SSE）
- ✅ 并发请求处理
- ✅ 错误处理和恢复

### 性能测试
- ✅ 延迟统计（平均、P50、P90、P95、P99）
- ✅ 吞吐量测量（requests/s）
- ✅ 资源监控（内存、CPU、GPU）
- ✅ 预热效果评估

## 🔧 测试配置

### 环境变量

```powershell
# 设置日志级别
$env:RUST_LOG = "debug"

# 设置测试超时
$env:TEST_TIMEOUT = "60"
```

### 推荐硬件配置

| 测试类型 | 最低配置 | 推荐配置 |
|---------|---------|---------|
| 单元测试 | 4GB RAM | 8GB RAM |
| 功能测试 | 8GB RAM | 16GB RAM |
| 性能测试 | 16GB RAM | 32GB RAM + GPU |
| 压力测试 | 32GB RAM | 64GB RAM + GPU |

## 📈 解读测试结果

### Rust 测试输出示例

```
running 10 tests
test test_multi_model_manager_init ... ok
test test_shared_tokenizer_cache ... ok
test test_concurrent_model_access ... ok
test test_stress_concurrent_requests ... ok

test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured
```

### PowerShell 功能测试输出

```
========================================
  aha 多模型并行功能测试
========================================

✓ 服务正在运行 (端口: 10100)
✓ 文本生成成功
✓ 流式输出完成 (共 25 个数据块)
✓ 并发测试完成: 5/5 成功
```

### 性能测试报告

```
延迟统计 (毫秒):
  最小值: 120.45ms
  最大值: 350.78ms
  平均值: 185.32ms
  P50: 175.20ms
  P90: 280.50ms
  P95: 310.25ms
  P99: 345.60ms

吞吐量统计:
  总请求数: 50
  成功请求: 50
  失败请求: 0
  总耗时: 12.35s
  吞吐量: 4.05 requests/s
```

## 🐛 故障排查

### 问题 1: Rust 测试编译失败

**症状**: `cargo test` 报错

**解决**:
```powershell
# 清理构建缓存
cargo clean

# 重新构建
cargo build --release

# 再次运行测试
cargo test test_multi_model
```

### 问题 2: PowerShell 脚本执行策略限制

**症状**: `无法加载文件，因为在此系统上禁止运行脚本`

**解决**:
```powershell
# 以管理员身份运行 PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 或者临时绕过
powershell -ExecutionPolicy Bypass -File .\tests\test_multi_model.ps1
```

### 问题 3: 服务连接失败

**症状**: `无法连接到远程服务器`

**解决**:
```powershell
# 检查服务是否运行
Get-Process | Where-Object { $_.ProcessName -like "*aha*" }

# 检查端口占用
netstat -ano | findstr "10100"

# 重新启动服务
aha serv -m Qwen/Qwen3-0.6B -m GLM-ASR-Nano -p 10100
```

### 问题 4: 测试超时

**症状**: 请求超时或响应缓慢

**解决**:
```powershell
# 增加超时时间
.\tests\test_multi_model.ps1 -TimeoutSeconds 120

# 减少并发数量
.\tests\benchmark_multi_model.ps1 -ConcurrentUsers 2

# 检查系统资源
Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 10
```

### 问题 5: 内存不足

**症状**: OOM (Out of Memory) 错误

**解决**:
```powershell
# 减少同时加载的模型数量
aha serv -m Qwen/Qwen3-0.6B -p 10100

# 使用量化模型
aha serv -m Qwen/Qwen3-0.6B-GGUF --gguf-path path/to/q4_k_m.gguf -p 10100

# 增加虚拟内存（Windows）
# 系统属性 -> 高级 -> 性能 -> 高级 -> 虚拟内存
```

## 📝 添加新测试

### Rust 测试模板

```rust
#[tokio::test]
async fn test_your_feature() -> Result<()> {
    // 测试逻辑
    
    assert!(condition);
    
    println!("✓ 测试通过");
    Ok(())
}
```

### PowerShell 测试模板

```powershell
Write-Section "测试 X: 测试名称"

try {
    # 测试逻辑
    $response = Invoke-RestMethod -Uri "$BaseUrl/endpoint" -Method GET
    
    if ($response.success) {
        Write-Success "测试通过"
    } else {
        Write-Error-Custom "测试失败"
    }
} catch {
    Write-Error-Custom "测试异常: $_"
}
```

## 🔄 持续集成

### GitHub Actions 示例

```yaml
name: Multi-Model Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: windows-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
    
    - name: Run Rust tests
      run: cargo test test_multi_model
    
    - name: Build release
      run: cargo build --release
    
    - name: Start service
      run: Start-Process -FilePath "./target/release/aha.exe" -ArgumentList "serv", "-m", "Qwen/Qwen3-0.6B", "-p", "10100"
    
    - name: Wait for service
      run: Start-Sleep -Seconds 10
    
    - name: Run PowerShell tests
      run: .\tests\test_multi_model.ps1
```

## 📚 相关文档

- [多模型使用指南](../docs/multi-model-guide.md)
- [架构设计文档](../docs/multi-model-architecture.md)
- [快速参考卡片](../docs/MULTI_MODEL_QUICK_REFERENCE.md)
- [API 文档](../docs/api.md)

## 💬 反馈和支持

如有问题或建议，请：
1. 提交 GitHub Issue
2. 查看现有文档
3. 加入微信群讨论

---

**最后更新**: 2026-05-30  
**维护者**: aha Team
