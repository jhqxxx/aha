# aha 多模型测试 - 快速开始指南

## 📦 新增的测试文件

本次更新在 `tests/` 目录中添加了以下文件：

### Rust 测试
- ✅ **`test_multi_model.rs`** - 多模型并行功能的 Rust 集成测试（10+ 测试用例）

### PowerShell 脚本
- ✅ **`test_multi_model.ps1`** - 端到端功能测试脚本
- ✅ **`benchmark_multi_model.ps1`** - 性能基准测试脚本
- ✅ **`generate_test_data.ps1`** - 测试数据生成工具
- ✅ **`run_all_tests.ps1`** - 统一测试运行器

### 文档
- ✅ **`README_MULTI_MODEL.md`** - 完整的测试文档和故障排查指南

---

## 🚀 5 分钟快速开始

### 1️⃣ 运行 Rust 单元测试

```powershell
# 运行所有多模型测试
cargo test test_multi_model

# 查看测试结果
cargo test test_multi_model -- --nocapture
```

**预期输出：**
```
running 10 tests
test test_multi_model_manager_init ... ok
test test_shared_tokenizer_cache ... ok
test test_concurrent_model_access ... ok
...
test result: ok. 10 passed; 0 failed
```

### 2️⃣ 启动多模型服务

```powershell
# 编译项目（如果还未编译）
cargo build --release

# 启动服务，加载两个模型
.\target\release\aha.exe serv -m Qwen/Qwen3-0.6B -m GLM-ASR-Nano -p 10100
```

等待服务启动完成，看到类似输出：
```
Loading 2 model(s)...
[1/2] Loading model: Qwen/Qwen3-0.6B
[2/2] Loading model: GLM-ASR-Nano
All models loaded successfully!
Starting HTTP server on 127.0.0.1:10100
```

### 3️⃣ 运行功能测试

打开**新的 PowerShell 窗口**，执行：

```powershell
# 进入项目根目录
cd l:\Documents\GitHub\aha

# 运行功能测试
.\tests\test_multi_model.ps1
```

**预期输出：**
```
========================================
  aha 多模型并行功能测试
========================================

✓ 服务正在运行 (端口: 10100)
✓ 文本生成成功
✓ 流式输出完成
✓ 并发测试完成: 5/5 成功
```

### 4️⃣ 运行性能测试

```powershell
# 运行性能基准测试
.\tests\benchmark_multi_model.ps1
```

**预期输出：**
```
延迟统计 (毫秒):
  最小值: 120.45ms
  平均值: 185.32ms
  P95: 310.25ms
  
吞吐量: 4.05 requests/s
```

### 5️⃣ 一键运行所有测试

```powershell
# 使用测试运行器（推荐）
.\tests\run_all_tests.ps1 -TestType all

# 只运行单元测试
.\tests\run_all_tests.ps1 -TestType unit

# 只运行功能测试
.\tests\run_all_tests.ps1 -TestType functional

# 只运行性能测试
.\tests\run_all_tests.ps1 -TestType benchmark

# 生成测试报告
.\tests\run_all_tests.ps1 -GenerateReport
```

---

## 📊 测试覆盖范围

| 测试类型 | 测试数量 | 说明 |
|---------|---------|------|
| **单元测试** | 10+ | 管理器初始化、并发安全、缓存机制等 |
| **功能测试** | 7 | API 端点、文本生成、流式输出、并发请求等 |
| **性能测试** | 4 | 延迟、吞吐量、资源监控、压力测试 |
| **集成测试** | 3 | 完整工作流、内存共享、多模型协作 |

---

## 🔧 常见问题

### Q1: PowerShell 脚本无法执行？

**错误信息：** `无法加载文件，因为在此系统上禁止运行脚本`

**解决方案：**
```powershell
# 以管理员身份运行 PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 或临时绕过
powershell -ExecutionPolicy Bypass -File .\tests\test_multi_model.ps1
```

### Q2: Rust 测试找不到？

**错误信息：** `no tests found`

**解决方案：**
```powershell
# 确保在项目根目录
cd l:\Documents\GitHub\aha

# 清理并重新构建
cargo clean
cargo build

# 再次运行测试
cargo test test_multi_model
```

### Q3: 服务连接失败？

**错误信息：** `无法连接到远程服务器`

**解决方案：**
```powershell
# 检查服务是否运行
Get-Process | Where-Object { $_.ProcessName -like "*aha*" }

# 检查端口占用
netstat -ano | findstr "10100"

# 重新启动服务
.\target\release\aha.exe serv -m Qwen/Qwen3-0.6B -p 10100
```

### Q4: 测试超时？

**解决方案：**
```powershell
# 增加超时时间
.\tests\test_multi_model.ps1 -TimeoutSeconds 120

# 减少并发数量
.\tests\benchmark_multi_model.ps1 -ConcurrentUsers 2
```

---

## 📈 解读测试结果

### Rust 测试通过标准
```
test result: ok. X passed; 0 failed
```

### 功能测试通过标准
- ✓ 所有 API 端点返回 200
- ✓ 文本生成成功
- ✓ 流式输出正常
- ✓ 并发请求成功率 100%

### 性能测试参考指标

| 指标 | CPU 模式 | GPU 模式 |
|------|---------|---------|
| 平均延迟 | < 300ms | < 150ms |
| P95 延迟 | < 500ms | < 250ms |
| 吞吐量 | > 2 req/s | > 5 req/s |

---

## 🎯 下一步

1. **查看详细文档**: [README_MULTI_MODEL.md](README_MULTI_MODEL.md)
2. **自定义测试**: 修改 `.ps1` 脚本中的参数
3. **持续集成**: 配置 GitHub Actions 自动测试
4. **性能优化**: 根据测试结果调整模型配置

---

## 💡 提示

- ✅ 首次运行时建议先生成测试数据：`.\tests\generate_test_data.ps1`
- ✅ 性能测试前确保关闭其他占用资源的程序
- ✅ 定期运行测试以确保代码质量
- ✅ 将测试结果保存到版本控制系统

---

**祝测试顺利！** 🎉

如有问题，请查看 [README_MULTI_MODEL.md](README_MULTI_MODEL.md) 获取详细的故障排查指南。
