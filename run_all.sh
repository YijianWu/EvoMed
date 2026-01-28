#!/bin/bash
# Full modular evolution script - run all batches

set -e

# Get parent directory of script location (project root)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Check virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

export OPENAI_API_KEY="${OPENAI_API_KEY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
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

# Ensure reports directory exists
mkdir -p outputs/reports

for BATCH_ID in $(seq 1 $TOTAL_BATCHES); do
    echo ""
    echo ">>> Batch $BATCH_ID / $TOTAL_BATCHES Start - $(date)"
    
    if [ -n "$PREV_PLAYBOOK" ] && [ -f "$PREV_PLAYBOOK" ]; then
        echo "    Using previous playbook: $PREV_PLAYBOOK"
        python run_evo.py \
            --batch-id $BATCH_ID \
            --batch-size $BATCH_SIZE \
            --model-path gpt-4o-mini \
            --backend openai \
            --excel "$EXCEL_FILE" \
            --previous-playbook "$PREV_PLAYBOOK"
    else
        echo "    First batch, no previous playbook"
        python run_evo.py \
            --batch-id $BATCH_ID \
            --batch-size $BATCH_SIZE \
            --model-path gpt-4o-mini \
            --backend openai \
            --excel "$EXCEL_FILE"
    fi
    
    # Find latest generated playbook file (in outputs/reports directory)
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

# Copy final result
FINAL_PLAYBOOK="outputs/reports/modular_playbook_final_$(date +%Y%m%d_%H%M%S).json"
cp "$PREV_PLAYBOOK" "$FINAL_PLAYBOOK"
echo "Final result saved: $FINAL_PLAYBOOK"
