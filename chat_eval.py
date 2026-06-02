#!/usr/bin/env python3
"""
主题评估脚本 - 读取清洗过的主题数据进行评估

功能：
- 读取 debug_topic_response_1.json 和 debug_topic_response_2.json 文件
- 使用 EvalAgent 进行主题评估
- 输出详细的评估结果
"""

import datetime
import json
import os
from pathlib import Path
from agents.eval_agent import EvalAgent


def load_topic_data(file_path: str) -> list:
    """
    加载主题数据文件

    Args:
        file_path: JSON文件路径

    Returns:
        list: 主题数据列表
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ 成功加载 {file_path}，包含 {len(data)} 个主题")
        return data
    except FileNotFoundError:
        print(f"❌ 文件不存在: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误 {file_path}: {e}")
        return []
    except Exception as e:
        print(f"❌ 加载文件出错 {file_path}: {e}")
        return []


def save_evaluation_result(result: dict, output_file: str):
    """
    保存评估结果到文件

    Args:
        result: 评估结果字典
        output_file: 输出文件路径
    """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ 评估结果已保存到: {output_file}")
    except Exception as e:
        print(f"❌ 保存结果出错: {e}")


def print_evaluation_summary(result: dict, file_name: str):
    """
    打印评估结果摘要

    Args:
        result: 评估结果字典
        file_name: 文件名称
    """
    print(f"\n" + "=" * 60)
    print(f"📊 {file_name} 评估结果摘要")
    print("=" * 60)

    print(f"主题数量: {result.get('topic_count', 0)}")
    print(f"评估时间: {result.get('evaluation_time', 'N/A')}")

    # 总体评分
    overall_scores = result.get("overall_scores", {})
    print(f"\n🎯 总体评分:")
    print(f"  聪明度: {overall_scores.get('聪明度', 0)}/100")
    print(f"  皮实:   {overall_scores.get('皮实', 0)}/100")
    print(f"  勤奋:   {overall_scores.get('勤奋', 0)}/100")

    # 各主题评分
    topic_results = result.get("topic_results", [])
    if topic_results:
        print(f"\n📝 各主题评分详情:")
        for i, topic_result in enumerate(topic_results, 1):
            topic_name = topic_result.get("topic", f"主题{i}")
            scores = topic_result.get("scores", {})
            dialogue_count = topic_result.get("dialogue_count", 0)

            print(f"\n  {i}. {topic_name}")
            print(f"     对话轮数: {dialogue_count}")
            print(f"     聪明度: {scores.get('聪明度', 0)}/100")
            print(f"     皮实:   {scores.get('皮实', 0)}/100")
            print(f"     勤奋:   {scores.get('勤奋', 0)}/100")

            # 显示评估内容的前200字符
            evaluation = topic_result.get("evaluation", "")
            if evaluation:
                preview = (
                    evaluation[:200] + "..." if len(evaluation) > 200 else evaluation
                )
                print(f"     评估预览: {preview}")


def main():
    """主函数"""
    print("🚀 开始主题评估任务...")

    # 初始化评估Agent
    eval_agent = EvalAgent()

    # 候选人信息（可以根据实际情况调整）
    candidate_info = {
        "position": "产品经理",
        "intelligence_requirement": 80,
        "name": "辛媛媛",
    }

    # 岗位描述
    jd = """
    岗位职责：
    1、开展需求调研与分析，深入挖掘用户场景与痛点，输出产品需求文档（PRD）及原型设计。
    2、协调研发、设计、测试团队资源，推动功能开发与落地，确保项目按计划交付。
    3、跟踪开发进度，识别项目风险并协调跨部门协作，保障产品质量与上线时效。
    4、基于用户反馈及数据表现，持续优化功能体验，制定迭代计划并推动执行。
    任职要求：
    1、统招本科及以上学历，接受应届生。
    2、掌握需求分析方法与工具，擅长跨团队沟通与项目管理。
    3、逻辑严谨，数据敏感，具备从用户场景到产品落地的全流程推动能力。
    """

    # 要处理的文件列表
    files_to_process = [
        # "debug_test.json",
        "debug_topic_response_1.json",
        "debug_topic_response_2.json",
        "debug_topic_response_3.json",
    ]

    # 检查文件是否存在
    existing_files = []
    for file_path in files_to_process:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            print(f"⚠️  文件不存在，跳过: {file_path}")

    if not existing_files:
        print("❌ 没有找到任何可处理的文件")
        return

    # 处理每个文件
    for file_path in existing_files:
        print(f"\n🔄 处理文件: {file_path}")

        # 加载主题数据
        topics = load_topic_data(file_path)
        if not topics:
            print(f"跳过文件: {file_path}")
            continue

        # 执行评估
        print(f"📊 开始评估 {len(topics)} 个主题...")
        try:
            result = eval_agent.evaluate_topics(
                topics=topics, candidate_info=candidate_info, jd=jd
            )

            # 生成输出文件名
            base_name = Path(file_path).stem
            output_file = f"{base_name}_evaluation_md_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # 保存结果
            save_evaluation_result(result, output_file)

            # 打印摘要
            print_evaluation_summary(result, file_path)

        except Exception as e:
            print(f"❌ 评估过程出错: {e}")
            import traceback

            traceback.print_exc()

    print("\n🎉 主题评估任务完成！")


if __name__ == "__main__":
    main()
