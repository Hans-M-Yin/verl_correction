#!/bin/bash

# ==========================================================
# 批量合并子文件夹下各模型 Checkpoints 的脚本 (V2: 嵌套实验结构)
# ==========================================================

# 1. 基础配置
CHECKPOINTS_BASE="/home/yzh/perception/verl/checkpoints" # 修改为你的固定基础路径
PROJECT_NAME="${1:-verl_correction}"                    # 默认项目名
TARGET_ROOT="${2:-./merged_checkpoints}"                # 默认保存根目录

# 处理 ~ 路径 (如果在脚本中使用变量，建议用绝对路径)
# 如果上面的路径包含 ~，我们需要展开它
CHECKPOINTS_BASE="${CHECKPOINTS_BASE/#\~/$HOME}"
TARGET_ROOT="${TARGET_ROOT/#\~/$HOME}"

# 2. 检查项目路径
PROJECT_PATH="$CHECKPOINTS_BASE/$PROJECT_NAME"
if [ ! -d "$PROJECT_PATH" ]; then
    echo "错误: 项目路径 '$PROJECT_PATH' 不存在！"
    exit 1
fi

echo "--------------------------------------------------------"
echo "正在扫描项目: $PROJECT_NAME"
echo "源地址: $PROJECT_PATH"
echo "--------------------------------------------------------"

# 3. 遍历所有实验 (Experiment)
for EXP_PATH in "$PROJECT_PATH"/*; do
    if [ -d "$EXP_PATH" ]; then
        EXP_NAME=$(basename "$EXP_PATH")
        
        echo "发现实验: $EXP_NAME"

        # 4. 遍历实验下的所有步骤 (Step/Checkpoint)
        for STEP_PATH in "$EXP_PATH"/*; do
            if [ -d "$STEP_PATH" ]; then
                STEP_NAME=$(basename "$STEP_PATH")
                
                # 构造目标文件夹名: {实验名}_{步数}
                # 例如: qwen2_5_vl_3b_data1_3k_global_step_300
                TARGET_DIR_NAME="${EXP_NAME}_${STEP_NAME}"
                FULL_TARGET_PATH="$TARGET_ROOT/$TARGET_DIR_NAME"

                echo "  [处理中] Step: $STEP_NAME"
                echo "  [目的地] $FULL_TARGET_PATH"

                # 5. 执行合并命令
                python -m verl.model_merger merge \
                    --backend fsdp \
                    --local_dir "$STEP_PATH" \
                    --target_dir "$FULL_TARGET_PATH"

                if [ $? -eq 0 ]; then
                    echo "  [成功] $STEP_NAME 已完成。"
                else
                    echo "  [失败] $STEP_NAME 发生错误，跳过..."
                fi
                echo "  ---"
            fi
        done
    fi
done

echo "--------------------------------------------------------"
echo "所有实验的所有 Checkpoints 已处理完毕。"
