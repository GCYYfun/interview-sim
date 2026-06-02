#!/usr/bin/env python3
"""
对话评估模块 - 快速演示

演示如何使用 ConversationEvaluator 评估面试对话
"""

from modules import ConversationEvaluator


def demo_single_evaluation():
    """演示单条记录评估"""
    print("\n" + "=" * 60)
    print("📝 演示：评估单条面试记录")
    print("=" * 60)

    # 创建评估器
    evaluator = ConversationEvaluator("interview_data.csv")

    # 评估第一条记录的第一轮面试
    print("\n正在评估 Record ID: 20, First Round...")
    result = evaluator.evaluate_record_by_id(record_id=20, round_name="First Round")

    if "error" in result:
        print(f"\n❌ 评估失败: {result['error']}")
        return

    # 显示结果
    print("\n✅ 评估完成！")
    print("\n" + "-" * 60)
    print(result.get("summary", "N/A"))
    print("-" * 60)

    # 保存报告
    output_path = evaluator.export_evaluation_report(result)
    print(f"\n💾 报告已保存: {output_path}")


def demo_batch_evaluation():
    """演示批量评估"""
    print("\n" + "=" * 60)
    print("🔄 演示：批量评估面试记录")
    print("=" * 60)

    # 创建评估器
    evaluator = ConversationEvaluator("interview_data.csv")

    # 批量评估前3条记录
    print("\n正在批量评估 Record IDs: [0, 1, 2]...")
    results = evaluator.batch_evaluate(
        record_ids=[0, 1, 2], round_name="First Round", max_records=3
    )

    # 统计结果
    success_count = sum(1 for r in results if "error" not in r)
    print("\n✅ 批量评估完成！")
    print(f"   成功: {success_count}/{len(results)}")

    # 显示每条记录的平均分
    print("\n评估摘要：")
    print("-" * 60)
    for result in results:
        if "error" not in result:
            record_id = result.get("record_id", "N/A")
            evaluation = result.get("evaluation", {})
            scores = evaluation.get("scores", {})
            avg_score = sum(scores.values()) / len(scores) if scores else 0

            print(f"  Record {record_id}: 平均分 {avg_score:.1f}/100")
        else:
            record_id = result.get("record_id", "N/A")
            print(f"  Record {record_id}: 评估失败")
    print("-" * 60)


def demo_all_rounds_evaluation():
    """演示所有轮次评估"""
    print("\n" + "=" * 60)
    print("🎯 演示：评估单条记录的所有轮次")
    print("=" * 60)

    # 创建评估器
    evaluator = ConversationEvaluator("interview_data.csv")

    # 评估第一条记录的所有轮次
    print("\n正在评估 Record ID: 0 的所有轮次...")
    result = evaluator.evaluate_all_rounds(record_id=0)

    if "error" in result:
        print(f"\n❌ 评估失败: {result['error']}")
        return

    # 显示结果
    print("\n✅ 所有轮次评估完成！")
    print("\n" + "-" * 60)

    summary = result.get("summary", {})
    print(f"岗位: {result.get('job_title', 'N/A')}")
    print(
        f"有效轮次: {summary.get('valid_rounds', 0)}/{summary.get('total_rounds', 3)}"
    )

    if summary.get("average_scores"):
        print("\n各维度平均分:")
        for dimension, score in summary["average_scores"].items():
            print(f"  • {dimension}: {score:.1f}/100")
        print(f"\n总体平均分: {summary.get('overall_average', 0):.1f}/100")

    if summary.get("performance_trend"):
        print("\n表现趋势:")
        print(f"  {summary['performance_trend']}")

    print("\n各轮次详情:")
    for round_name, round_result in result.get("rounds", {}).items():
        status = round_result.get("status", "unknown")
        if status == "success":
            eval_data = round_result.get("evaluation", {})
            scores = eval_data.get("scores", {})
            avg = sum(scores.values()) / len(scores) if scores else 0
            print(f"  [{round_name}] ✓ 平均分: {avg:.1f}/100")
        elif status == "no_data":
            print(f"  [{round_name}] ⚠️ 无数据")
        else:
            print(f"  [{round_name}] ✗ 失败")

    print("-" * 60)

    # 保存报告
    output_path = evaluator.export_evaluation_report(result)
    print(f"\n💾 报告已保存: {output_path}")


def demo_conversation_cleaning():
    """演示对话清洗功能"""
    print("\n" + "=" * 60)
    print("🔧 演示：对话数据清洗")
    print("=" * 60)

    # 模拟原始对话
    raw_dialogue = """
Emma Wang(00:02:19): 老同学能听得见吗？
马智慧(00:02:24): 可以可以听得见我要开摄像头吗？
Emma Wang(00:02:27): 我们都开一下摄像头吧，然后面对面去交流。
马智慧(00:02:30): 好好好。好的好的。
Emma Wang(00:02:38): 那我们开始吧，请做个自我介绍。
马智慧(00:02:46): 好的，我是马智慧，毕业于西南民族大学...
"""

    print("\n原始对话：")
    print("-" * 60)
    print(raw_dialogue)
    print("-" * 60)

    # 创建评估器
    evaluator = ConversationEvaluator("interview_data.csv")

    # 清洗对话
    print("\n正在清洗对话数据...")
    cleaned = evaluator.clean_conversation(raw_dialogue)

    # 显示清洗结果
    print("\n清洗后的标准格式：")
    print("-" * 60)
    import json

    print(json.dumps(cleaned, ensure_ascii=False, indent=2))
    print("-" * 60)


def main():
    """运行所有演示"""
    print("\n" + "=" * 60)
    print("🎉 ConversationEvaluator 模块演示")
    print("=" * 60)

    try:
        # 演示 1: 对话清洗
        # demo_conversation_cleaning()

        # # 演示 2: 单条评估
        demo_single_evaluation()

        # # 演示 3: 所有轮次评估 ⭐ NEW
        # demo_all_rounds_evaluation()

        # # 演示 4: 批量评估
        # demo_batch_evaluation()

        print("\n" + "=" * 60)
        print("✅ 演示完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 演示出错: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
