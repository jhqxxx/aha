# 多模型性能基准测试 (PowerShell)
# 用于评估多模型并行运行的性能表现

param(
    [string]$Port = "10100",
    [string]$BaseUrl = "http://localhost:10100",
    [int]$WarmupRequests = 3,
    [int]$BenchmarkRequests = 10,
    [int]$ConcurrentUsers = 5
)

# 颜色定义
$ColorSuccess = "Green"
$ColorError = "Red"
$ColorWarning = "Yellow"
$ColorInfo = "Cyan"

function Write-Section {
    param([string]$Title)
    Write-Host "`n========================================" -ForegroundColor $ColorInfo
    Write-Host "  $Title" -ForegroundColor $ColorInfo
    Write-Host "========================================`n" -ForegroundColor $ColorInfo
}

Write-Section "aha 多模型性能基准测试"

# ========================================
# 配置测试参数
# ========================================
$TestConfig = @{
    BaseUrl           = $BaseUrl
    WarmupRequests    = $WarmupRequests
    BenchmarkRequests = $BenchmarkRequests
    ConcurrentUsers   = $ConcurrentUsers
}

Write-Host "测试配置:" -ForegroundColor $ColorInfo
Write-Host "  预热请求数: $WarmupRequests" -ForegroundColor Gray
Write-Host "  基准测试请求数: $BenchmarkRequests" -ForegroundColor Gray
Write-Host "  并发用户数: $ConcurrentUsers" -ForegroundColor Gray
Write-Host ""

# ========================================
# 检查服务可用性
# ========================================
Write-Section "步骤 1: 检查服务状态"

try {
    $response = Invoke-WebRequest -Uri "$BaseUrl/admin/models/list" -Method GET -TimeoutSec 5 -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ 服务正在运行" -ForegroundColor $ColorSuccess
        $modelsData = $response.Content | ConvertFrom-Json
        Write-Host "  已加载模型: $($modelsData.count) 个" -ForegroundColor Gray
    }
}
catch {
    Write-Host "✗ 服务未运行，请先启动服务" -ForegroundColor $ColorError
    exit 1
}

# ========================================
# 预热阶段
# ========================================
Write-Section "步骤 2: 预热阶段"

$warmupPayload = @{
    model      = "Qwen3"
    messages   = @(@{ role = "user"; content = "Hi" })
    stream     = $false
    max_tokens = 10
} | ConvertTo-Json -Depth 10

Write-Host "执行 $WarmupRequests 个预热请求..." -ForegroundColor $ColorInfo

for ($i = 1; $i -le $WarmupRequests; $i++) {
    try {
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        Invoke-RestMethod -Uri "$BaseUrl/v1/chat/completions" `
            -Method POST `
            -Body $warmupPayload `
            -ContentType "application/json" `
            -TimeoutSec 30 | Out-Null
        $stopwatch.Stop()
        Write-Host "  预热请求 $i`: $($stopwatch.Elapsed.TotalSeconds)s" -ForegroundColor Gray
    }
    catch {
        Write-Host "  预热请求 $i 失败: $_" -ForegroundColor $ColorWarning
    }
}

Write-Host "✓ 预热完成`n" -ForegroundColor $ColorSuccess

# ========================================
# 单请求延迟测试
# ========================================
Write-Section "步骤 3: 单请求延迟测试"

$latencies = @()

for ($i = 1; $i -le $BenchmarkRequests; $i++) {
    try {
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $response = Invoke-RestMethod -Uri "$BaseUrl/v1/chat/completions" `
            -Method POST `
            -Body $warmupPayload `
            -ContentType "application/json" `
            -TimeoutSec 30
        $stopwatch.Stop()
        
        $latency = $stopwatch.Elapsed.TotalMilliseconds
        $latencies += $latency
        
        $progress = [math]::Round(($i / $BenchmarkRequests) * 100, 2)
        Write-Host "  请求 $i/$BenchmarkRequests`: ${latency}ms ($progress%)" -ForegroundColor Gray
    }
    catch {
        Write-Host "  请求 $i 失败: $_" -ForegroundColor $ColorWarning
    }
}

# 计算统计数据
if ($latencies.Count -gt 0) {
    $sortedLatencies = $latencies | Sort-Object
    $minLatency = $sortedLatencies[0]
    $maxLatency = $sortedLatencies[-1]
    $avgLatency = ($latencies | Measure-Object -Average).Average
    $p50Index = [math]::Floor($sortedLatencies.Count * 0.5)
    $p90Index = [math]::Floor($sortedLatencies.Count * 0.9)
    $p95Index = [math]::Floor($sortedLatencies.Count * 0.95)
    $p99Index = [math]::Floor($sortedLatencies.Count * 0.99)
    
    Write-Host "`n延迟统计 (毫秒):" -ForegroundColor $ColorInfo
    Write-Host "  最小值: $([math]::Round($minLatency, 2))ms" -ForegroundColor Gray
    Write-Host "  最大值: $([math]::Round($maxLatency, 2))ms" -ForegroundColor Gray
    Write-Host "  平均值: $([math]::Round($avgLatency, 2))ms" -ForegroundColor Gray
    Write-Host "  P50: $([math]::Round($sortedLatencies[$p50Index], 2))ms" -ForegroundColor Gray
    Write-Host "  P90: $([math]::Round($sortedLatencies[$p90Index], 2))ms" -ForegroundColor Gray
    Write-Host "  P95: $([math]::Round($sortedLatencies[$p95Index], 2))ms" -ForegroundColor Gray
    Write-Host "  P99: $([math]::Round($sortedLatencies[$p99Index], 2))ms" -ForegroundColor Gray
}

# ========================================
# 吞吐量测试
# ========================================
Write-Section "步骤 4: 吞吐量测试"

Write-Host "发起 $ConcurrentUsers 个并发用户，每个用户发送 $( [math]::Ceiling($BenchmarkRequests / $ConcurrentUsers) ) 个请求..." -ForegroundColor $ColorInfo

$throughputPayload = @{
    model      = "Qwen3"
    messages   = @(@{ role = "user"; content = "Calculate 1+1" })
    stream     = $false
    max_tokens = 5
} | ConvertTo-Json -Depth 10

$startTime = Get-Date
$completedRequests = 0
$failedRequests = 0

# 创建并发作业
$jobs = 1..$ConcurrentUsers | ForEach-Object {
    Start-Job -ScriptBlock {
        param($url, $payload, $requestCount, $userId)
        
        $successCount = 0
        $failCount = 0
        
        for ($i = 1; $i -le $requestCount; $i++) {
            try {
                Invoke-RestMethod -Uri $url `
                    -Method POST `
                    -Body $payload `
                    -ContentType "application/json" `
                    -TimeoutSec 30 | Out-Null
                $successCount++
            }
            catch {
                $failCount++
            }
        }
        
        return @{
            UserId  = $userId
            Success = $successCount
            Failed  = $failCount
        }
    } -ArgumentList @(
        "$BaseUrl/v1/chat/completions",
        $throughputPayload,
        [math]::Ceiling($BenchmarkRequests / $ConcurrentUsers),
        $_
    )
}

# 等待所有作业完成
$results = $jobs | Wait-Job | Receive-Job

# 统计结果
foreach ($result in $results) {
    $completedRequests += $result.Success
    $failedRequests += $result.Failed
}

$endTime = Get-Date
$totalTime = ($endTime - $startTime).TotalSeconds

# 清理作业
$jobs | Remove-Job

# 计算吞吐量
$requestsPerSecond = if ($totalTime -gt 0) { $completedRequests / $totalTime } else { 0 }

Write-Host "`n吞吐量统计:" -ForegroundColor $ColorInfo
Write-Host "  总请求数: $BenchmarkRequests" -ForegroundColor Gray
Write-Host "  成功请求: $completedRequests" -ForegroundColor Gray
Write-Host "  失败请求: $failedRequests" -ForegroundColor Gray
Write-Host "  总耗时: $([math]::Round($totalTime, 2))s" -ForegroundColor Gray
Write-Host "  吞吐量: $([math]::Round($requestsPerSecond, 2)) requests/s" -ForegroundColor Gray
Write-Host "  并发用户: $ConcurrentUsers" -ForegroundColor Gray

# ========================================
# 内存使用监控（Windows）
# ========================================
Write-Section "步骤 5: 资源使用情况"

try {
    # 获取 aha 进程信息
    $process = Get-Process | Where-Object { $_.ProcessName -like "*aha*" } | Select-Object -First 1
    
    if ($process) {
        $memoryMB = [math]::Round($process.WorkingSet64 / 1MB, 2)
        $cpuPercent = $process.CPU
        
        Write-Host "进程信息:" -ForegroundColor $ColorInfo
        Write-Host "  进程名: $($process.ProcessName)" -ForegroundColor Gray
        Write-Host "  PID: $($process.Id)" -ForegroundColor Gray
        Write-Host "  内存使用: ${memoryMB} MB" -ForegroundColor Gray
        Write-Host "  CPU 时间: $([math]::Round($cpuPercent, 2))s" -ForegroundColor Gray
        
        # GPU 信息（如果有 NVIDIA GPU）
        try {
            $gpuInfo = nvidia-smi --query-gpu=memory.used, memory.total, utilization.gpu --format=csv, noheader, nounits 2>$null
            if ($gpuInfo) {
                Write-Host "`nGPU 使用情况:" -ForegroundColor $ColorInfo
                Write-Host "  $gpuInfo" -ForegroundColor Gray
            }
        }
        catch {
            # 忽略 GPU 查询错误
        }
    }
    else {
        Write-Host "未找到 aha 进程" -ForegroundColor $ColorWarning
    }
}
catch {
    Write-Host "无法获取进程信息: $_" -ForegroundColor $ColorWarning
}

# ========================================
# 测试总结
# ========================================
Write-Section "性能测试总结"

Write-Host "`n关键指标:" -ForegroundColor $ColorInfo
if ($latencies.Count -gt 0) {
    Write-Host "  平均延迟: $([math]::Round($avgLatency, 2))ms" -ForegroundColor Gray
    Write-Host "  P95 延迟: $([math]::Round($sortedLatencies[$p95Index], 2))ms" -ForegroundColor Gray
}
Write-Host "  吞吐量: $([math]::Round($requestsPerSecond, 2)) req/s" -ForegroundColor Gray
Write-Host "  成功率: $([math]::Round(($completedRequests / $BenchmarkRequests) * 100, 2))%" -ForegroundColor Gray

Write-Host "`n性能建议:" -ForegroundColor $ColorInfo
Write-Host "  1. 如果延迟过高，考虑启用 GPU 加速 (--features cuda/metal)" -ForegroundColor Gray
Write-Host "  2. 如果吞吐量不足，增加并发用户数或优化模型大小" -ForegroundColor Gray
Write-Host "  3. 监控内存使用，避免 OOM" -ForegroundColor Gray
Write-Host "  4. 使用 Flash Attention 优化长序列处理 (--features flash-attn)`n" -ForegroundColor Gray

Write-Host "✓ 性能基准测试完成！`n" -ForegroundColor $ColorSuccess
