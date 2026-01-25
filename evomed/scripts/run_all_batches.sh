#!/bin/bash
# 全量模块化进化脚本 - 运行所有批次

set -e

# 获取脚本所在目录的上一级目录（即项目根目录）
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 检查虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
fi

export OPENAI_API_KEY="sk-mZ1tJ8giPu2WqauY5SivguiTVJmFolWNAkBQ4i5Y3Lh2jxVL"
export OPENAI_BASE_URL="https://yunwu.ai/v1"
export HF_ENDPOINT="https://hf-mirror.com"

EXCEL_FILE="guilin_inpatient_extracted_10000.xlsx"
BATCH_SIZE=500
TOTAL_SAMPLES=10000
TOTAL_BATCHES=$((TOTAL_SAMPLES / BATCH_SIZE))

echo "=============================================="
echo "EvoMed Modular Experience Evolution - Full Run"
echo "=============================================="
echo "Project Root: $ROOT_DIR"
echo "Total Samples: $TOTAL_SAMPLES"
echo "Batch Size: $BATCH_SIZE"
echo "Total Batches: $TOTAL_BATCHES"
echo "Start Time: $(date)"
echo "=============================================="

PREV_PLAYBOOK=""

# 确保 reports 目录存在
mkdir -p outputs/reports

for BATCH_ID in $(seq 1 $TOTAL_BATCHES); do
    echo ""
    echo ">>> Batch $BATCH_ID / $TOTAL_BATCHES Start - $(date)"
    
    if [ -n "$PREV_PLAYBOOK" ] && [ -f "$PREV_PLAYBOOK" ]; then
        echo "    Using previous playbook: $PREV_PLAYBOOK"
        python scripts/run_modular_evolution.py \
            --batch-id $BATCH_ID \
            --batch-size $BATCH_SIZE \
            --model-path gpt-4o-mini \
            --backend openai \
            --excel "$EXCEL_FILE" \
            --previous-playbook "$PREV_PLAYBOOK"
    else
        echo "    First batch, no previous playbook"
        python scripts/run_modular_evolution.py \
            --batch-id $BATCH_ID \
            --batch-size $BATCH_SIZE \
            --model-path gpt-4o-mini \
            --backend openai \
            --excel "$EXCEL_FILE"
    fi
    
    # 找到最新生成的 playbook 文件 (在 outputs/reports 目录下)
    PREV_PLAYBOOK=$(ls -t outputs/reports/modular_playbook_batch${BATCH_ID}_*.json 2>/dev/null | head -1)
    
    if [ -z "$PREV_PLAYBOOK" ]; then
        echo "    [ERROR] No playbook output found for batch $BATCH_ID!"
        exit 1
    fi
    
    echo "    Batch $BATCH_ID Complete, Output: $PREV_PLAYBOOK"
    echo ""
done

echo "=============================================="
echo "Full Run Complete!"
echo "End Time: $(date)"
echo "Final Playbook: $PREV_PLAYBOOK"
echo "=============================================="

# 复制最终结果
FINAL_PLAYBOOK="outputs/reports/modular_playbook_final_$(date +%Y%m%d_%H%M%S).json"
cp "$PREV_PLAYBOOK" "$FINAL_PLAYBOOK"
echo "Final result saved: $FINAL_PLAYBOOK"



