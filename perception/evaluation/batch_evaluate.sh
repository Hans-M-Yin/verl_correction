#!/bin/bash

# ==========================================================
# 批量评测与打分脚本
# 用法: bash batch_evaluate.sh [TARGET_ROOT] [OUT_DIR_ROOT]
# ==========================================================

# 1. 基础配置
TARGET_ROOT="${1:-./merged_checkpoints}"    # 转换完的模型目录（刚才脚本的保存目录）
OUT_DIR_ROOT="${2:-./eval_results}"         # 所有结果的根输出目录

# API 配置 (可以直接在此修改，或执行前 export)
export CORRECTION_PROJECT_API_KEY="${CORRECTION_PROJECT_API_KEY:-your_api_key_here}"
export CORRECTION_PROJECT_API_URL="${CORRECTION_PROJECT_API_URL:-your_api_url_here}"

# 2. 检查模型目录
if [ ! -d "$TARGET_ROOT" ]; then
    echo "错误: 目标路径 '$TARGET_ROOT' 不存在！"
    exit 1
fi

mkdir -p "$OUT_DIR_ROOT"

echo "--------------------------------------------------------"
echo "开始批量运行评测和打分程序..."
echo "模型根目录: $TARGET_ROOT"
echo "评测输出根目录: $OUT_DIR_ROOT"
echo "--------------------------------------------------------"

# 3. 遍历每个模型权重文件夹
for MODEL_PATH in "$TARGET_ROOT"/*; do
    if [ -d "$MODEL_PATH" ]; then
        MODEL_NAME=$(basename "$MODEL_PATH")
        
        # 为当前模型创建独立的输出文件夹
        # 例如: ./eval_results/qwen2_5_vl_3b_data1_3k_global_step_300
        MODEL_OUT_DIR="$OUT_DIR_ROOT/$MODEL_NAME"
        mkdir -p "$MODEL_OUT_DIR"

        echo ">>> 处理模型: $MODEL_NAME"
        echo "    路径: $MODEL_PATH"

        # --- 第一步: 运行评测 (Evaluation) ---
        echo "    [Step 1/2] 运行基础评测..."

        python evaluation_mathvista.py --checkpoint "$MODEL_PATH" --out-dir "$MODEL_OUT_DIR"
        python evaluation_mathverse.py --checkpoint "$MODEL_PATH" --out-dir "$MODEL_OUT_DIR"
        python evaluation_mathvision.py --checkpoint "$MODEL_PATH" --out-dir "$MODEL_OUT_DIR"
        python evaluation_wemath.py --checkpoint "$MODEL_PATH" --out-dir "$MODEL_OUT_DIR"

        # --- 第二步: 运行打分 (Judge) ---
        echo "    [Step 2/2] 运行模型打分 (Judge)..."

        python judge_mathvista.py \
            --api_key "$CORRECTION_PROJECT_API_KEY" \
            --api_url "$CORRECTION_PROJECT_API_URL" \
            --eval_basedir "$MODEL_OUT_DIR" \
            --eval_file "mathvista.json"

        python judge_mathverse.py \
            --api_key "$CORRECTION_PROJECT_API_KEY" \
            --api_url "$CORRECTION_PROJECT_API_URL" \
            --eval_basedir "$MODEL_OUT_DIR" \
            --eval_file "mathverse_test.json"

        python judge_mathvision.py \
            --api_key "$CORRECTION_PROJECT_API_KEY" \
            --api_url "$CORRECTION_PROJECT_API_URL" \
            --eval_basedir "$MODEL_OUT_DIR" \
            --eval_file "mathvision_test.json"

        # 检查 judge_wemath.py 是否存在 (当前目录下未发现该文件)
        if [ -f "judge_wemath.py" ]; then
            python judge_wemath.py \
                --api_key "$CORRECTION_PROJECT_API_KEY" \
                --api_url "$CORRECTION_PROJECT_API_URL" \
                --eval_basedir "$MODEL_OUT_DIR" \
                --eval_file "wemath_test.json"
        else
            echo "    [Notice] 跳过 judge_wemath.py (文件不存在)"
        fi

        echo "<<< 完成模型: $MODEL_NAME"
        echo "--------------------------------------------------------"
    fi
done

echo "批量任务全部完成。"
echo "所有结果保存在: $OUT_DIR_ROOT"
