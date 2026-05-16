import json
import argparse
import os

# 严格按照提供的 1-23 顺序进行大类映射
CATEGORY_MAPPING = [
    "generation",     # 1 根据案件，预测罪名和法条
    "reasoning",      # 2 寻找相关的法律条文
    "knowledge",      # 3 法律条文原文记忆
    "reasoning",      # 4 根据案件，预测罪名
    "reasoning",      # 5 涉案金额计算
    "generation",     # 6 给出证据质证意见
    "understanding",  # 7 确定争议焦点，多选题
    "generation",     # 8 根据案情，给出判决书结果
    "understanding",  # 9 法律文书错别字修改
    "reasoning",      # 10 主观题回答判分
    "knowledge",      # 11 法律解释
    "understanding",  # 12 提取法律文书摘要
    "generation",     # 13 案例分析，综合性较强
    "reasoning",      # 14 法律概念理解，选择题
    "generation",     # 15 法律文章写作
    "consultation",   # 16 法律领域咨询
    "generation",     # 17 政策策略制定
    "reasoning",      # 18 根据案件，预测刑期
    "generation",     # 19 起诉书生成
    "understanding",  # 20 涉法舆情文章摘要
    "understanding",  # 21 选择更相似的案件
    "generation",     # 22 合同生成
    "consultation"    # 23 真实案例咨询
]

def analyze_results(score_path, inference_path, output_path):
    try:
        with open(score_path, 'r', encoding='utf-8') as f:
            score_data = json.load(f)
        
        with open(inference_path, 'r', encoding='utf-8') as f:
            inference_data = json.load(f)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    print("\n" + "="*85)
    print(" 📊 UCL-Bench 深度性能与成本评测报告 (含大类微平均分析)")
    print("="*85)

    # 全局统计变量
    g_base_score = 0; g_local_score = 0; g_items = 0
    g_total_time = 0.0; g_tool_latency = 0.0; g_rag_count = 0
    g_total_tokens = 0; g_user_tokens = 0; g_inter_tokens = 0; g_comp_tokens = 0

    # 大类(Category)统计容器：用于计算微平均
    category_totals = {
        cat: {
            "local_score": 0, "base_score": 0, "total_time": 0.0, "tool_latency": 0.0,
            "rag_count": 0, "total_tokens": 0, "user_tokens": 0, "inter_tokens": 0,
            "comp_tokens": 0, "sample_size": 0
        } for cat in set(CATEGORY_MAPPING)
    }

    if not isinstance(score_data, dict):
        print("错误: 预期的 JSON 结构为按任务分类的字典。")
        return

    task_results = {}

    # 依据字典遍历顺序(保证Python3.7+下与插入顺序一致)与CATEGORY_MAPPING强绑定
    for task_idx, (task_name, score_items) in enumerate(score_data.items()):
        item_count = len(score_items)
        if item_count == 0: continue

        # 确定该任务所属大类
        task_category = CATEGORY_MAPPING[task_idx] if task_idx < len(CATEGORY_MAPPING) else "unknown"

        # 任务内统计变量
        t_base = 0; t_local = 0
        t_time = 0.0; t_tool = 0.0; t_rag = 0
        t_total_tok = 0; t_user = 0; t_inter = 0; t_comp = 0

        inf_task_items = inference_data.get(task_name, [])
        inf_lookup = {str(item.get("id")): item for item in inf_task_items}

        for s_item in score_items:
            # 得分统计
            scores = s_item.get("evaluation_score", [0, 0])
            t_base += scores[0] if len(scores) > 0 else 0
            t_local += scores[1] if len(scores) > 1 else 0

            # 性能与 Token 读取 (使用旧键名兼容底层推理脚本)
            item_id = str(s_item.get("id"))
            inf_item = inf_lookup.get(item_id, {})
            
            t_time += inf_item.get("total_time_sec", 0.0)
            t_tool += inf_item.get("tool_latency_sec", 0.0)
            t_rag += inf_item.get("rag_count", 0)
            
            t_total_tok += inf_item.get("total_tokens", 0)
            t_user += inf_item.get("user_prompt_tokens", 0)
            t_inter += inf_item.get("inter_agent_tokens", 0)
            t_comp += inf_item.get("completion_tokens", 0)

        # 累加到全局
        g_base_score += t_base; g_local_score += t_local; g_items += item_count
        g_total_time += t_time; g_tool_latency += t_tool; g_rag_count += t_rag
        g_total_tokens += t_total_tok; g_user_tokens += t_user; g_inter_tokens += t_inter; g_comp_tokens += t_comp

        # 累加到大类(用于后续计算微平均)
        if task_category in category_totals:
            cat_stats = category_totals[task_category]
            cat_stats["local_score"] += t_local
            cat_stats["base_score"] += t_base
            cat_stats["total_time"] += t_time
            cat_stats["tool_latency"] += t_tool
            cat_stats["rag_count"] += t_rag
            cat_stats["total_tokens"] += t_total_tok
            cat_stats["user_tokens"] += t_user
            cat_stats["inter_tokens"] += t_inter
            cat_stats["comp_tokens"] += t_comp
            cat_stats["sample_size"] += item_count

        # 保存单个任务的统计
        task_results[task_name] = {
            "category": task_category,
            "score_ratio": t_local/t_base if t_base>0 else 0,
            "local_score": t_local,
            "base_score": t_base,
            "avg_time": t_time/item_count,
            "avg_tool_latency": t_tool/item_count,
            "avg_rag_count": t_rag/item_count,
            "avg_total_tokens": t_total_tok/item_count,
            "avg_user_tokens": t_user/item_count,
            "avg_inter_tokens": t_inter/item_count,
            "avg_comp_tokens": t_comp/item_count,
            "sample_size": item_count
        }

        print(f"【任务】: {task_name} | 所属大类: [{task_category}] (样本数: {item_count})")
        print(f"  [质量] 得分比例(Local/Base): {(t_local/t_base if t_base>0 else 0):.4f}  ({t_local}/{t_base})")
        print(f"  [耗时] 均次总时长: {t_time/item_count:.2f}s | 检索耗时: {t_tool/item_count:.2f}s | 检索频次: {t_rag/item_count:.1f}次")
        print(f"  [Token] 均次总计: {int(t_total_tok/item_count)} (User: {int(t_user/item_count)} | Inter: {int(t_inter/item_count)} | Comp: {int(t_comp/item_count)})")
        print("-" * 85)

    # 计算大类微平均输出
    category_breakdown = {}
    print("\n📈 【分类微平均 (Micro-Average) 指标总览】")
    for cat, stats in category_totals.items():
        sz = stats["sample_size"]
        if sz == 0: continue
        
        category_breakdown[cat] = {
            "score_ratio": stats["local_score"]/stats["base_score"] if stats["base_score"]>0 else 0,
            "local_score": stats["local_score"],
            "base_score": stats["base_score"],
            "avg_time": stats["total_time"]/sz,
            "avg_tool_latency": stats["tool_latency"]/sz,
            "avg_rag_count": stats["rag_count"]/sz,
            "avg_total_tokens": stats["total_tokens"]/sz,
            "avg_user_tokens": stats["user_tokens"]/sz,
            "avg_inter_tokens": stats["inter_tokens"]/sz,
            "avg_comp_tokens": stats["comp_tokens"]/sz,
            "sample_size": sz
        }
        
        print(f"  [{cat.upper()}] - 样本数: {sz}")
        print(f"    得分比例: {(stats['local_score']/stats['base_score'] if stats['base_score']>0 else 0):.4f} | "
              f"均次时长: {stats['total_time']/sz:.2f}s | "
              f"均次检索: {stats['rag_count']/sz:.1f}次 | "
              f"均次Total Token: {int(stats['total_tokens']/sz)}")

    # 打印全局汇总
    print("\n🏆 【全局指标总览】")
    print(f"  ➤ 总评测样本数: {g_items}")
    print(f"  ➤ 全局得分比例 (Local/Base): {(g_local_score/g_base_score if g_base_score>0 else 0):.4f} ({g_local_score}/{g_base_score})")
    print("\n⏳ 【时间与交互延迟】 (平均每条 Query)")
    print(f"  - 整体端到端时长 (Total Time): {g_total_time/g_items if g_items else 0:.2f} 秒")
    print(f"  - 外部检索阻塞时长 (Tool Latency): {g_tool_latency/g_items if g_items else 0:.2f} 秒")
    print(f"  - 检索动作发起次数 (RAG Rounds): {g_rag_count/g_items if g_items else 0:.2f} 次")
    print("\n💰 【Token 消耗清单】 (平均每条 Query)")
    print(f"  - Total Tokens: {int(g_total_tokens/g_items if g_items else 0)}")
    print("="*85 + "\n")

    # ================= 持久化保存统计结果 =================
    final_report = {
        "bench_name": "UCL-Bench",
        "global_score_ratio": g_local_score / g_base_score if g_base_score > 0 else 0,
        "global_local_score": g_local_score,
        "global_base_score": g_base_score,
        "efficiency": {
            "avg_end_to_end_time": g_total_time / g_items if g_items else 0,
            "avg_tool_latency": g_tool_latency / g_items if g_items else 0,
            "avg_rag_rounds": g_rag_count / g_items if g_items else 0,
            "avg_total_tokens": g_total_tokens / g_items if g_items else 0,
            "avg_user_tokens": g_user_tokens / g_items if g_items else 0,
            "avg_inter_tokens": g_inter_tokens / g_items if g_items else 0,
            "avg_comp_tokens": g_comp_tokens / g_items if g_items else 0,
        },
        "category_breakdown": category_breakdown,
        "task_breakdown": task_results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
    print(f"✅ UCL 最终统计报告已生成并保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析UCL非流式架构下的各项指标')
    parser.add_argument('--score_path', type=str, required=True, help='打分结果 JSON 路径')
    parser.add_argument('--inference_path', type=str, required=True, help='推理阶段生成的 JSON 路径(含Token统计)')
    parser.add_argument('--output_path', type=str, required=True, help='最终汇总结果 JSON 存放路径')
    
    args = parser.parse_args()
    analyze_results(args.score_path, args.inference_path, args.output_path)