#!/bin/bash
# 启动医疗诊断系统 Web UI

echo "========================================"
echo "启动医疗诊断系统 Web UI"
echo "========================================"
echo ""

# 检查是否在虚拟环境中
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  警告：未检测到虚拟环境"
    echo "建议先激活虚拟环境："
    echo "  source venv/bin/activate"
    echo ""
fi

# 启动 Streamlit
echo "正在启动 Web UI..."
echo "访问地址: http://localhost:8501"
echo ""

cd "$(dirname "$0")"
streamlit run src/web_ui.py --server.port 8501 --server.address 0.0.0.0


