# aha 多模型测试运行器 (PowerShell)
# 一键执行所有测试套件

param(
    [ValidateSet("all", "unit", "functional", "benchmark", "generate-data")]
    [string]$TestType = "all",
    
    [string]$Port = "10100",
    
    [switch]$SkipServiceCheck,
    
    [switch]$GenerateReport
)

# 颜色定义
$ColorSuccess = "Green"
$ColorError = "Red"
$ColorWarning = "Yellow"
$ColorInfo = "Cyan"
$ColorHeader = "Magenta"

function Write-Header {
    param([string]$Title)
    Write-Host "`n" -NoNewline
    Write-Host ("=" * 60) -ForegroundColor $ColorHeader
    Write-Host "  $Title" -ForegroundColor $ColorHeader
    Write-Host ("=" * 60) -ForegroundColor $ColorHeader
    Write-Host ""
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor $ColorSuccess
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor $ColorError
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor $ColorWarning
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor $ColorInfo
}

# 测试统计
$TestStats = @{
    Total = 0
    Passed = 0
    Failed = 0
    Skipped = 0
    StartTime = Get-Date
}

function Update-Stats {
    param([string]$Status)
    $TestStats.Total++
    switch ($Status) {
        "Passed" { $TestStats.Passed++ }
        "Failed" { $TestStats.Failed++ }
        "Skipped" { $TestStats.Skipped++ }
    }
}

Write-Header "aha 多模型测试运行器"

Write-Host "测试类型: $TestType" -ForegroundColor $ColorInfo
Write-Host "端口: $Port" -ForegroundColor $ColorInfo
Write-Host "开始时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n" -ForegroundColor $ColorInfo

# ========================================
# 步骤 0: 生成测试数据（如果需要）
# ========================================
if ($TestType -eq "generate-data" -or $TestType -eq "all") {
    Write-Header "步骤 0: 生成测试数据"
    
    try {
        & ".\tests\generate_test_data.ps1" -OutputDir ".\tests\test-data" -Force
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "测试数据生成成功"
            Update-Stats -Status "Passed"
        } else {
            Write-Error-Custom "测试数据生成失败"
            Update-Stats -Status "Failed"
        }
    } catch {
        Write-Error-Custom "测试数据生成异常: $_"
        Update-Stats -Status "Failed"
    }
    
    if ($TestType -eq "generate-data") {
        exit $TestStats.Failed -gt 0 ? 1 : 0
    }
}

# ========================================
# 步骤 1: 运行 Rust 单元测试
# ========================================
if ($TestType -eq "unit" -or $TestType -eq "all") {
    Write-Header "步骤 1: Rust 单元测试"
    
    Write-Info "运行 cargo test..."
    
    try {
        $testOutput = cargo test test_multi_model --lib 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Rust 单元测试全部通过"
            Update-Stats -Status "Passed"
            
            # 显示测试摘要
            if ($testOutput -match "(\d+) passed") {
                Write-Info "通过的测试数: $($matches[1])"
            }
        } else {
            Write-Error-Custom "Rust 单元测试失败"
            Write-Host $testOutput -ForegroundColor Gray
            Update-Stats -Status "Failed"
        }
    } catch {
        Write-Error-Custom "Rust 测试执行异常: $_"
        Update-Stats -Status "Failed"
    }
    
    if ($TestType -eq "unit") {
        exit $TestStats.Failed -gt 0 ? 1 : 0
    }
}

# ========================================
# 步骤 2: 检查服务状态
# ========================================
if (-not $SkipServiceCheck) {
    Write-Header "步骤 2: 检查服务状态"
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$Port/admin/models/list" `
            -Method GET -TimeoutSec 5 -ErrorAction Stop
        
        if ($response.StatusCode -eq 200) {
            Write-Success "服务正在运行 (端口: $Port)"
            $modelsData = $response.Content | ConvertFrom-Json
            Write-Info "已加载模型: $($modelsData.count) 个"
            Update-Stats -Status "Passed"
        }
    } catch {
        Write-Warning-Custom "服务未运行或无法访问"
        Write-Host ""
        Write-Host "请先启动多模型服务:" -ForegroundColor $ColorWarning
        Write-Host "  aha serv -m Qwen/Qwen3-0.6B -m GLM-ASR-Nano -p $Port" -ForegroundColor Gray
        Write-Host ""
        Write-Host "或者使用 -SkipServiceCheck 参数跳过此检查" -ForegroundColor $ColorWarning
        Write-Host ""
        
        if (-not $Force) {
            Update-Stats -Status "Skipped"
            Write-Info "跳过功能测试和性能测试"
            exit 0
        }
    }
}

# ========================================
# 步骤 3: 运行功能测试
# ========================================
if ($TestType -eq "functional" -or $TestType -eq "all") {
    Write-Header "步骤 3: 功能测试"
    
    if (Test-Path ".\tests\test_multi_model.ps1") {
        try {
            & ".\tests\test_multi_model.ps1" -Port $Port
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "功能测试完成"
                Update-Stats -Status "Passed"
            } else {
                Write-Warning-Custom "功能测试部分失败（查看上面输出）"
                Update-Stats -Status "Failed"
            }
        } catch {
            Write-Error-Custom "功能测试执行异常: $_"
            Update-Stats -Status "Failed"
        }
    } else {
        Write-Warning-Custom "功能测试脚本不存在: .\tests\test_multi_model.ps1"
        Update-Stats -Status "Skipped"
    }
    
    if ($TestType -eq "functional") {
        exit $TestStats.Failed -gt 0 ? 1 : 0
    }
}

# ========================================
# 步骤 4: 运行性能基准测试
# ========================================
if ($TestType -eq "benchmark" -or $TestType -eq "all") {
    Write-Header "步骤 4: 性能基准测试"
    
    if (Test-Path ".\tests\benchmark_multi_model.ps1") {
        try {
            & ".\tests\benchmark_multi_model.ps1" -Port $Port
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "性能基准测试完成"
                Update-Stats -Status "Passed"
            } else {
                Write-Warning-Custom "性能测试部分失败（查看上面输出）"
                Update-Stats -Status "Failed"
            }
        } catch {
            Write-Error-Custom "性能测试执行异常: $_"
            Update-Stats -Status "Failed"
        }
    } else {
        Write-Warning-Custom "性能测试脚本不存在: .\tests\benchmark_multi_model.ps1"
        Update-Stats -Status "Skipped"
    }
    
    if ($TestType -eq "benchmark") {
        exit $TestStats.Failed -gt 0 ? 1 : 0
    }
}

# ========================================
# 生成测试报告
# ========================================
if ($GenerateReport) {
    Write-Header "生成测试报告"
    
    $TestStats.EndTime = Get-Date
    $TestStats.Duration = $TestStats.EndTime - $TestStats.StartTime
    
    $report = @{
        timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        duration_seconds = [math]::Round($TestStats.Duration.TotalSeconds, 2)
        summary = @{
            total = $TestStats.Total
            passed = $TestStats.Passed
            failed = $TestStats.Failed
            skipped = $TestStats.Skipped
            success_rate = if ($TestStats.Total -gt 0) {
                [math]::Round(($TestStats.Passed / $TestStats.Total) * 100, 2)
            } else { 0 }
        }
    }
    
    $reportFile = ".\tests\test-report-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
    $report | ConvertTo-Json -Depth 10 | Out-File $reportFile -Encoding UTF8
    
    Write-Success "测试报告已保存: $reportFile"
}

# ========================================
# 测试总结
# ========================================
Write-Header "测试总结"

$TestStats.EndTime = Get-Date
$duration = $TestStats.EndTime - $TestStats.StartTime

Write-Host "测试统计:" -ForegroundColor $ColorInfo
Write-Host "  总测试数: $($TestStats.Total)" -ForegroundColor Gray
Write-Host "  通过: $($TestStats.Passed)" -ForegroundColor $ColorSuccess
Write-Host "  失败: $($TestStats.Failed)" -ForegroundColor $ColorError
Write-Host "  跳过: $($TestStats.Skipped)" -ForegroundColor $ColorWarning
Write-Host "  成功率: $(if ($TestStats.Total -gt 0) { [math]::Round(($TestStats.Passed / $TestStats.Total) * 100, 2) } else { 0 })%" -ForegroundColor Gray
Write-Host "  总耗时: $([math]::Round($duration.TotalSeconds, 2)) 秒`n" -ForegroundColor Gray

if ($TestStats.Failed -eq 0) {
    Write-Success "所有测试通过！🎉`n"
    exit 0
} else {
    Write-Error-Custom "部分测试失败，请检查上面的输出`n"
    exit 1
}
