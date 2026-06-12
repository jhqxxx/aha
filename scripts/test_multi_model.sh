#!/bin/bash

# 多模型并行运行测试脚本

echo "=== aha 多模型功能测试 ==="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 测试配置
PORT=10100
BASE_URL="http://localhost:${PORT}"

echo -e "${YELLOW}步骤 1: 启动多模型服务${NC}"
echo "启动 Qwen3-0.6B + GLM-ASR-Nano..."
echo ""

# 在后台启动服务（需要预先下载模型）
# aha serv -m Qwen/Qwen3-0.6B -m GLM-ASR-Nano -p $PORT &
# SERVICE_PID=$!
# sleep 5

echo -e "${YELLOW}步骤 2: 检查已加载的模型${NC}"
echo ""
curl -s "${BASE_URL}/admin/models/list" | jq . || echo "服务未启动，跳过此测试"
echo ""

echo -e "${YELLOW}步骤 3: 测试文本生成模型${NC}"
echo ""
curl -s -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3",
    "messages": [{"role": "user", "content": "你好，请介绍一下自己"}],
    "stream": false,
    "max_tokens": 50
  }' | jq '.choices[0].message.content' || echo "测试失败"
echo ""

echo -e "${YELLOW}步骤 4: 测试语音识别模型${NC}"
echo ""
# 需要准备一个测试音频文件
if [ -f "test_audio.wav" ]; then
  curl -s -X POST "${BASE_URL}/v1/audio/transcriptions" \
    -F "model=GlmAsrNano" \
    -F "file=@test_audio.wav" | jq . || echo "音频测试失败"
else
  echo "未找到 test_audio.wav，跳过音频测试"
fi
echo ""

echo -e "${YELLOW}步骤 5: 性能对比测试${NC}"
echo ""
echo "单请求延迟测试："
time curl -s -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3",
    "messages": [{"role": "user", "content": "计算 1+1"}],
    "stream": false
  }' > /dev/null
echo ""

echo -e "${YELLOW}清理: 停止服务${NC}"
# kill $SERVICE_PID 2>/dev/null || true
echo "测试完成！"
