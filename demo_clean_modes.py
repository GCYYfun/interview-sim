#!/usr/bin/env python3
"""
演示对话清洗的两种模式：QA Pair 和 Topic
"""

from modules import ConversationEvaluator
import json


def demo_qa_pair_mode():
    """演示 QA Pair 模式"""
    print("\n" + "=" * 80)
    print("📝 演示：QA Pair 模式 - 提取问答对")
    print("=" * 80)

    #     # 示例对话
    #     sample_dialogue = """
    # Emma Wang(00:02:19): 老同学能听得见吗？
    # 马智慧(00:02:24): 可以可以听得见我要开摄像头吗？
    # Emma Wang(00:02:27): 我们都开一下摄像头吧，然后面对面去交流。
    # 马智慧(00:02:30): 好好好。好的好的。
    # Emma Wang(00:02:38): 那我们开始吧，请做个自我介绍。
    # 马智慧(00:02:46): 好的，我是马智慧，毕业于西南民族大学，学的是工商管理专业。
    # Emma Wang(00:03:15): 你为什么想加入我们公司？
    # 马智慧(00:03:25): 我觉得贵公司的企业文化很好，而且发展前景广阔，我希望能在这里发挥我的能力。
    # Emma Wang(00:04:10): 你有什么问题想问我吗？
    # 马智慧(00:04:15): 请问公司对新人有什么培训计划吗？
    # """

    # 创建评估器
    evaluator = ConversationEvaluator("interview_data.csv")

    # 加载第一条记录
    query_result = evaluator.manager.query(limit=1)
    if not query_result.records:
        print("❌ 没有找到记录")
        return

    record = query_result.records[0]
    raw_dialogue = record.conversation

    # QA Pair 模式清洗
    print("\n🔧 使用 QA Pair 模式清洗...")
    qa_pairs = evaluator.clean_conversation(raw_dialogue, mode="qa_pair")

    # 显示结果
    print("\n✅ 清洗结果（QA Pair 模式）:")
    print("-" * 80)
    print(json.dumps(qa_pairs, ensure_ascii=False, indent=2))
    print("-" * 80)
    print(f"\n共提取 {len(qa_pairs)} 轮对话")

    # 保存结果
    output_file = f"qa_pair_mode_result_{record.id}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存到: {output_file}")

    # 统计问答对数量
    interviewer_count = sum(1 for item in qa_pairs if item.get("role") == "interviewer")
    candidate_count = sum(1 for item in qa_pairs if item.get("role") == "candidate")
    print(f"面试官问题: {interviewer_count} 个")
    print(f"候选人回答: {candidate_count} 个")


def demo_topic_mode():
    """演示 Topic 模式"""
    print("\n" + "=" * 80)
    print("🎯 演示：Topic 模式 - 按主题划分")
    print("=" * 80)

    # 使用真实数据
    evaluator = ConversationEvaluator("interview_data.csv")

    # 加载第一条记录
    query_result = evaluator.manager.query(limit=21)
    if not query_result.records:
        print("❌ 没有找到记录")
        return

    record = query_result.records[20]
    raw_dialogue = record.conversation

    print(f"\n📋 使用 Record ID: {record.id} 的对话")
    print(f"对话长度: {len(raw_dialogue)} 字符")

    # Topic 模式清洗
    print("\n🔧 使用 Topic 模式清洗...")
    topics = evaluator.clean_conversation(raw_dialogue, mode="topic")

    # 保存结果
    output_file = f"topic_mode_result_{record.id}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存到: {output_file}")

    # 显示结果
    print("\n✅ 清洗结果（Topic 模式）:")
    print("-" * 80)

    for i, topic_data in enumerate(topics, 1):
        topic_name = topic_data.get("topic", "未命名主题")
        dialogue = topic_data.get("dialogue", [])

        print(f"\n【主题 {i}】: {topic_name}")
        print(f"包含 {len(dialogue)} 轮对话")

        # 显示前3轮对话作为示例
        # for j, turn in enumerate(dialogue[:3], 1):
        #     role = "面试官" if turn.get("role") == "interviewer" else "候选人"
        #     content = turn.get("content", "")[:100]  # 只显示前100字符
        #     print(f"  {j}. [{role}]: {content}...")

        if len(dialogue) > 3:
            print(f"  ... (还有 {len(dialogue) - 3} 轮对话)")

    print("-" * 80)
    print(f"\n共划分 {len(topics)} 个主题")


def demo_real_evaluation():
    """演示：使用 QA Pair 模式评估真实数据"""
    print("\n" + "=" * 80)
    print("🔍 演示：评估真实面试记录（QA Pair 模式）")
    print("=" * 80)

    evaluator = ConversationEvaluator("interview_data.csv")

    # 评估第一条记录
    print("\n正在评估 Record ID: 0...")
    result = evaluator.evaluate_record_by_id(record_id=0, round_name="First Round")

    if "error" in result:
        print(f"❌ 评估失败: {result['error']}")
        return

    # 显示摘要
    print("\n✅ 评估完成！")
    print("\n" + result.get("summary", "无摘要"))


def main():
    """运行所有演示"""
    try:
        # 演示 1: QA Pair 模式
        # demo_qa_pair_mode()

        # 演示 2: Topic 模式
        demo_topic_mode()

        print("\n" + "=" * 80)
        print("✅ 所有演示完成！")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 演示出错: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
