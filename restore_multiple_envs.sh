#!/bin/bash
# 批量恢复Conda环境的脚本

BACKUP_DIR="${1:-/caizhenyang/panghuaiwen/legal_LLM/env_backup}"
RESTORE_DIR="${2:-/root/miniconda3/envs}"
shift 2
ENVIRONMENTS=("$@")

# 如果没有指定环境，使用备份目录中的所有tar文件
if [ ${#ENVIRONMENTS[@]} -eq 0 ]; then
    echo "未指定环境，将恢复备份目录中的所有环境..."
    mapfile -t ENVIRONMENTS < <(find "$BACKUP_DIR" -name "*.tar.gz" -exec basename {} .tar.gz \;)
fi

mkdir -p "$RESTORE_DIR"

for env in "${ENVIRONMENTS[@]}"; do
    TAR_FILE="$BACKUP_DIR/${env}.tar.gz"
    
    if [ ! -f "$TAR_FILE" ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✗ 备份文件不存在: $TAR_FILE"
        continue
    fi
    
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 开始恢复环境: $env"
    TARGET_PATH="$RESTORE_DIR/$env"
    
    # 1. 创建目标目录
    rm -rf "$TARGET_PATH"  # 确保干净的环境
    mkdir -p "$TARGET_PATH"
    
    # 2. 解压
    echo "  → 解压环境文件..."
    tar -xzf "$TAR_FILE" -C "$TARGET_PATH"
    
    # 3. 修复路径
    echo "  → 修复环境路径..."
    if [ -f "$TARGET_PATH/bin/conda-unpack" ]; then
        "$TARGET_PATH/bin/conda-unpack" > /dev/null 2>&1
        echo "  ✓ 路径修复完成"
    else
        echo "  ✗ conda-unpack脚本不存在，请检查打包过程"
        continue
    fi
    
    # 4. 清理临时文件
    if [ -d "$TARGET_PATH/__pypackages__" ]; then
        rm -rf "$TARGET_PATH/__pypackages__"
    fi
    
    # 5. 验证环境
    echo "  → 验证环境..."
    if source "$TARGET_PATH/bin/activate" "$env" && conda list > /dev/null 2>&1; then
        echo "  ✓ 环境 $env 恢复成功"
        
        # 检查CUDA可用性（如果环境包含GPU相关包）
        if python -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)" 2>/dev/null; then
            echo "  ✓ GPU支持已验证"
        else
            echo "  → 注意: GPU支持可能未正确配置"
        fi
    else
        echo "  ✗ 环境 $env 恢复失败，请检查日志"
    fi
    
    echo "----------------------------------------"
done

echo "[$(date +'%Y-%m-%d %H:%M:%S')] 所有环境恢复完成！"