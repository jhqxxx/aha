# 多模型并行功能测试脚本 (PowerShell)
# 用于测试 aha 的多模型并行运行功能

param(
    [string]$Port = "10100",
    [string]$BaseUrl = "http://localhost:10100",
    [int]$TimeoutSeconds = 30
)

# 颜色定义
$ColorSuccess = "Green"
$ColorError = "Red"
$ColorWarning = "Yellow"
$ColorInfo = "Cyan"

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor $ColorSuccess
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor $ColorError
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor $ColorInfo
}

function Write-Section {
    param([string]$Title)
    Write-Host "`n========================================" -ForegroundColor $ColorInfo
    Write-Host "  $Title" -ForegroundColor $ColorInfo
    Write-Host "========================================`n" -ForegroundColor $ColorInfo
}

# 测试配置
$TestConfig = @{
    Port = $Port
    BaseUrl = $BaseUrl
    Timeout = $TimeoutSeconds
}

Write-Section "aha 多模型并行功能测试"

# ========================================
# 测试 1: 检查服务是否运行
# ========================================
Write-Section "测试 1: 检查服务状态"

try {
    $response = Invoke-WebRequest -Uri "$BaseUrl/admin/models/list" -Method GET -TimeoutSec 5 -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Success "服务正在运行 (端口: $Port)"
        $modelsData = $response.Content | ConvertFrom-Json
        Write-Info "已加载模型数量: $($modelsData.count)"
        if ($modelsData.models.Count -gt 0) {
            Write-Info "模型列表: $($modelsData.models -join ', ')"
        }
    }
} catch {
    Write-Error-Custom "服务未运行或无法访问"
    Write-Info "请先启动多模型服务:"
    Write-Host "  aha serv -m Qwen/Qwen3-0.6B -m GLM-ASR-Nano -p $Port" -ForegroundColor Gray
    exit 1
}

# ========================================
# 测试 2: 验证模型列表 API
# ========================================
Write-Section "测试 2: 模型列表 API"

try {
    $response = Invoke-RestMethod -Uri "$BaseUrl/admin/models/list" -Method GET -TimeoutSec 5
    
    if ($response.models -and $response.count -ge 0) {
        Write-Success "模型列表 API 响应正常"
        Write-Info "返回格式正确"
        
        # 显示详细信息
        Write-Host "`n已加载的模型:" -ForegroundColor $ColorInfo
        foreach ($model in $response.models) {
            Write-Host "  - $model" -ForegroundColor Gray
        }
        Write-Host "总计: $($response.count) 个模型`n" -ForegroundColor $ColorInfo
    } else {
        Write-Error-Custom "模型列表 API 返回格式错误"
    }
} catch {
    Write-Error-Custom "模型列表 API 调用失败: $_"
}

# ========================================
# 测试 3: 文本生成模型测试
# ========================================
Write-Section "测试 3: 文本生成模型测试"

$chatPayload = @{
    model = "Qwen3"
    messages = @(
        @{
            role = "user"
            content = "你好，请简单介绍一下自己"
        }
    )
    stream = $false
    max_tokens = 50
} | ConvertTo-Json -Depth 10

try {
    Write-Info "发送文本生成请求..."
    $response = Invoke-RestMethod -Uri "$BaseUrl/v1/chat/completions" `
        -Method POST `
        -Body $chatPayload `
        -ContentType "application/json" `
        -TimeoutSec $TimeoutSeconds
    
    if ($response.choices -and $response.choices.Count -gt 0) {
        Write-Success "文本生成成功"
        $content = $response.choices[0].message.content
        Write-Host "回复内容: $content`n" -ForegroundColor Gray
        
        # 检查响应结构
        if ($response.usage) {
            Write-Info "Token 使用统计:"
            Write-Host "  - Prompt tokens: $($response.usage.prompt_tokens)" -ForegroundColor Gray
            Write-Host "  - Completion tokens: $($response.usage.completion_tokens)" -ForegroundColor Gray
            Write-Host "  - Total tokens: $($response.usage.total_tokens)" -ForegroundColor Gray
        }
    } else {
        Write-Error-Custom "文本生成响应格式错误"
    }
} catch {
    Write-Error-Custom "文本生成请求失败: $_"
}

# ========================================
# 测试 4: 流式输出测试
# ========================================
Write-Section "测试 4: 流式输出测试"

$streamPayload = @{
    model = "Qwen3"
    messages = @(
        @{
            role = "user"
            content = "写一首关于春天的短诗"
        }
    )
    stream = $true
    max_tokens = 100
} | ConvertTo-Json -Depth 10

try {
    Write-Info "发送流式请求..."
    
    $request = [System.Net.WebRequest]::Create("$BaseUrl/v1/chat/completions")
    $request.Method = "POST"
    $request.ContentType = "application/json"
    $request.Timeout = $TimeoutSeconds * 1000
    
    $bodyBytes = [System.Text.Encoding]::UTF8.GetBytes($streamPayload)
    $request.ContentLength = $bodyBytes.Length
    
    $requestStream = $request.GetRequestStream()
    $requestStream.Write($bodyBytes, 0, $bodyBytes.Length)
    $requestStream.Close()
    
    $response = $request.GetResponse()
    $responseStream = $response.GetResponseStream()
    $reader = New-Object System.IO.StreamReader($responseStream)
    
    Write-Host "`n流式输出:" -ForegroundColor $ColorInfo
    $chunkCount = 0
    while (-not $reader.EndOfStream) {
        $line = $reader.ReadLine()
        if ($line -and $line.StartsWith("data: ")) {
            $data = $line.Substring(6)
            if ($data -ne "[DONE]") {
                try {
                    $chunk = $data | ConvertFrom-Json
                    if ($chunk.choices -and $chunk.choices[0].delta.content) {
                        Write-Host $chunk.choices[0].delta.content -NoNewline -ForegroundColor Gray
                        $chunkCount++
                    }
                } catch {
                    # 忽略解析错误
                }
            }
        }
    }
    
    Write-Host "`n`n✓ 流式输出完成 (共 $chunkCount 个数据块)" -ForegroundColor $ColorSuccess
    $reader.Close()
    $responseStream.Close()
    $response.Close()
    
} catch {
    Write-Error-Custom "流式输出测试失败: $_"
}

# ========================================
# 测试 5: 并发请求测试
# ========================================
Write-Section "测试 5: 并发请求性能测试"

$ConcurrentCount = 5
Write-Info "发起 $ConcurrentCount 个并发请求..."

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

$jobs = 1..$ConcurrentCount | ForEach-Object {
    Start-Job -ScriptBlock {
        param($url, $payload, $index)
        
        try {
            $response = Invoke-RestMethod -Uri $url `
                -Method POST `
                -Body $payload `
                -ContentType "application/json" `
                -TimeoutSec 30
            
            return @{
                Index = $index
                Success = $true
                Duration = 0
            }
        } catch {
            return @{
                Index = $index
                Success = $false
                Error = $_.Exception.Message
            }
        }
    } -ArgumentList @("$BaseUrl/v1/chat/completions", $chatPayload, $_)
}

# 等待所有任务完成
$jobs | Wait-Job | Receive-Job

$stopwatch.Stop()

$successCount = ($jobs | Where-Object { $_.State -eq "Completed" }).Count
Write-Success "并发测试完成: $successCount/$ConcurrentCount 成功"
Write-Info "总耗时: $($stopwatch.Elapsed.TotalSeconds) 秒"
Write-Info "平均每个请求: $($stopwatch.Elapsed.TotalSeconds / $ConcurrentCount) 秒"

# 清理作业
$jobs | Remove-Job

# ========================================
# 测试 6: 健康检查端点
# ========================================
Write-Section "测试 6: 健康检查"

try {
    $response = Invoke-RestMethod -Uri "$BaseUrl/health" -Method GET -TimeoutSec 5
    
    if ($response.status -eq "healthy" -or $response.ok) {
        Write-Success "健康检查通过"
        Write-Info "服务状态: 正常"
    } else {
        Write-Warning "健康检查响应异常"
    }
} catch {
    Write-Error-Custom "健康检查失败: $_"
}

# ========================================
# 测试 7: 模型信息端点
# ========================================
Write-Section "测试 7: 模型信息查询"

try {
    $response = Invoke-RestMethod -Uri "$BaseUrl/v1/models" -Method GET -TimeoutSec 5
    
    if ($response.data -or $response.models) {
        Write-Success "模型信息 API 正常"
        Write-Info "API 返回模型元数据"
    } else {
        Write-Warning "模型信息 API 返回格式异常"
    }
} catch {
    Write-Error-Custom "模型信息查询失败: $_"
}

# ========================================
# 测试总结
# ========================================
Write-Section "测试总结"

Write-Host "`n测试环境:" -ForegroundColor $ColorInfo
Write-Host "  服务地址: $BaseUrl" -ForegroundColor Gray
Write-Host "  超时设置: ${TimeoutSeconds}s" -ForegroundColor Gray
Write-Host "  并发数量: $ConcurrentCount" -ForegroundColor Gray

Write-Host "`n建议:" -ForegroundColor $ColorInfo
Write-Host "  1. 如果某些测试失败，检查模型是否正确加载" -ForegroundColor Gray
Write-Host "  2. 查看服务日志以获取更多调试信息" -ForegroundColor Gray
Write-Host "  3. 确保有足够的内存运行多个模型" -ForegroundColor Gray
Write-Host "  4. GPU 加速可显著提升性能 (--features cuda/metal)`n" -ForegroundColor Gray

Write-Success "所有测试完成！`n"
