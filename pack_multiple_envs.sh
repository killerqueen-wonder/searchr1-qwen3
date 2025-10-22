#!/bin/bash
# 修复后的打包脚本

BACKUP_DIR="${1:-/linzhihang/huaiwenpang/legal_LLM/env_backup}"
#默认地址
shift
ENVIRONMENTS=("$@")

# 如果没有指定环境，获取所有非base环境
if [ ${#ENVIRONMENTS[@]} -eq 0 ]; then
    echo "未指定环境，将打包所有非base环境..."
    # 可靠地获取环境列表
    # mapfile -t ENVIRONMENTS < <(conda env list | grep -vE "^#|^$|^\s*$" | awk 'NR>2 && $1 != "base" && $1 != "*" {print $1}' | sort -u)
    mapfile -t ENVIRONMENTS < <(conda env list | grep -vE "^#|^$|^\s*$" | awk '$1 != "base" && $1 != "*" {print $1}' | sort -u)
    
    # 过滤空行和无效环境
    # ENVIRONMENTS=($(printf '%s\n' "${ENVIRONMENTS[@]}" | grep -v "^$"))
    
    if [ ${#ENVIRONMENTS[@]} -eq 0 ]; then
        echo "未找到有效的Conda环境！"
        exit 1
    fi
fi

mkdir -p "$BACKUP_DIR"

for env in "${ENVIRONMENTS[@]}"; do
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 开始打包环境: $env"
    
    # 双重检查环境是否存在
    if ! conda info --envs | grep -q -E "[[:space:]]${env}[[:space:]]|/${env}$"; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✗ 环境不存在或名称不匹配: $env"
        echo "      可用环境列表:"
        conda env list | grep -v "^#" | grep -v "^\s*$" | awk 'NR>2 {print "      - " $1}'
        continue
    fi
    
    # 检查备份文件是否已存在，如果存在则跳过
    OUTPUT_FILE="$BACKUP_DIR/${env}.tar.gz"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️ 备份文件已存在，跳过环境: $env ($OUTPUT_FILE)"
        continue
    fi
    conda pack -n "$env" -o "$BACKUP_DIR/${env}.tar.gz" --format tar.gz
    
    # 检查打包结果
    if [ $? -eq 0 ] && [ -f "$BACKUP_DIR/${env}.tar.gz" ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✓ 打包成功: $BACKUP_DIR/${env}.tar.gz ($(du -sh "$BACKUP_DIR/${env}.tar.gz" | cut -f1))"
    else
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✗ 打包失败: $env"
    fi
done

echo "[$(date +'%Y-%m-%d %H:%M:%S')] 所有环境打包完成！备份目录: $BACKUP_DIR"
echo "总计尝试打包: ${#ENVIRONMENTS[@]} 个环境"