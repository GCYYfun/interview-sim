"""
面试模拟模块

功能：
- 基于真实简历数据的面试模拟
- 使用 InterviewAgent 生成问题和分析匹配度
- 使用 CandidateAgent 模拟候选人回答
- 使用 EvalAgent 实时评估
- 支持手动和自动两种模式
"""

from typing import Dict, Optional
from datetime import datetime
import json
from pathlib import Path

from agents import InterviewAgent, CandidateAgent, EvalAgent
from menglong.utils.log import print_message


class InterviewSimulator:
    """面试模拟器 - 完整的面试流程模拟"""

    def __init__(self):
        """初始化面试模拟器"""
        self.interview_agent = InterviewAgent()
        self.eval_agent = EvalAgent()
        self.current_interview = None

    def run_interview(
        self,
        candidate_data: Dict,
        mode: str = "auto",
        max_rounds: int = 5,
        save_result: bool = True,
    ) -> Dict:
        """
        运行完整的面试流程

        Args:
            candidate_data: 候选人数据
                {
                    "resume": str,
                    "jd": str,
                    "position": str,
                    "intelligence_requirement": int
                }
            mode: 面试模式
                - "auto": AI候选人自动回答
                - "manual": 人工手动回答
            max_rounds: 最大问答轮次
            save_result: 是否保存面试结果

        Returns:
            面试结果字典
        """
        print_message("=" * 60)
        print_message(f"🎯 面试模拟系统 - {mode.upper()}模式")
        print_message("=" * 60)

        # 1. 准备阶段 - 生成面试问题
        print_message("\n📋 阶段1: 准备面试问题...")
        questions_result = self.interview_agent.generate_interview_questions(
            resume=candidate_data["resume"],
            jd=candidate_data["jd"],
            focus_areas=["聪明度", "皮实", "勤奋"],
            position=candidate_data.get("position", "未指定"),
        )

        if "error" in questions_result:
            return {
                "error": f"生成面试问题失败: {questions_result['error']}",
                "status": "failed",
            }

        print_message("✓ 面试问题已准备")

        # 2. 匹配度分析
        print_message("\n🔍 阶段2: 分析候选人匹配度...")
        match_result = self.interview_agent.analyze_candidate_match(
            resume=candidate_data["resume"],
            jd=candidate_data["jd"],
            position=candidate_data.get("position", "未指定"),
        )

        if "error" in match_result:
            print_message(f"⚠️ 匹配度分析失败: {match_result['error']}")
        else:
            print_message("✓ 匹配度分析完成")

        # 3. 面试执行阶段
        print_message(f"\n🎭 阶段3: 开始面试 ({mode}模式, {max_rounds}轮)")
        print_message("=" * 60)

        if mode == "auto":
            interview_result = self._run_auto_interview(
                candidate_data, questions_result, max_rounds
            )
        else:
            interview_result = self._run_manual_interview(
                candidate_data, questions_result, max_rounds
            )

        # 4. 最终评估
        print_message("\n📊 阶段4: 生成最终评估...")
        final_eval = self.eval_agent.generate_final_evaluation(
            candidate_info=candidate_data,
            conversation_history=interview_result["conversation_history"],
        )

        print_message("✓ 最终评估完成")

        # 5. 整合结果
        complete_result = {
            "candidate_info": {
                "position": candidate_data.get("position", "未指定"),
                "intelligence_requirement": candidate_data.get(
                    "intelligence_requirement", 0
                ),
            },
            "interview_mode": mode,
            "questions_plan": questions_result,
            "match_analysis": match_result,
            "conversation_history": interview_result["conversation_history"],
            "round_evaluations": interview_result["round_evaluations"],
            "final_evaluation": final_eval,
            "interview_time": datetime.now().isoformat(),
            "total_rounds": len(interview_result["round_evaluations"]),
        }

        # 6. 显示总结
        self._display_summary(complete_result)

        # 7. 保存结果
        if save_result:
            filepath = self._save_interview_result(complete_result, candidate_data)
            print_message(f"\n💾 面试结果已保存: {filepath}")

        self.current_interview = complete_result
        return complete_result

    def _run_auto_interview(
        self, candidate_data: Dict, questions_result: Dict, max_rounds: int
    ) -> Dict:
        """运行自动面试（AI候选人）"""
        # 创建AI候选人
        candidate = CandidateAgent(candidate_data)

        print_message(f"\n🤖 AI候选人: {candidate}")
        print_message(f"📝 岗位: {candidate_data.get('position', '未指定')}")
        print_message("")

        conversation_history = []
        round_evaluations = []

        # 从生成的问题中提取（简化版，实际可以更智能地解析）
        # 这里我们模拟一些问题
        sample_questions = [
            "请介绍一下您最具挑战性的项目经验，以及您是如何解决遇到的困难的？",
            "在团队合作中，您遇到过什么分歧或冲突？您是如何处理的？",
            "当工作压力很大、任务紧急时，您会如何平衡工作质量和交付时间？",
            "请分享一个您主动发现问题并推动解决的案例。",
            "您如何保持技术学习和自我提升？请举例说明。",
        ]

        # 限制问题数量
        questions = sample_questions[:max_rounds]

        for i, question in enumerate(questions, 1):
            print_message(f"\n{'─' * 60}")
            print_message(f"第 {i}/{len(questions)} 轮")
            print_message(f"{'─' * 60}")

            # InterviewAgent 提问
            print_message(f"\n[面试官]: {question}")
            conversation_history.append({"role": "面试官", "content": question})

            # CandidateAgent 回答
            answer = candidate.answer_question(question)
            print_message(f"\n[候选人]: {answer[:300]}...")
            conversation_history.append({"role": "候选人", "content": answer})

            # EvalAgent 评估
            print_message("\n[评估中...]")
            eval_result = self.eval_agent.evaluate_single_response(
                question=question,
                answer=answer,
                candidate_info=candidate_data,
                question_intent=f"第{i}轮考察",
            )

            round_evaluations.append(eval_result)

            # 显示评分
            scores = eval_result.get("scores", {})
            print_message(
                f"✓ 评分: 聪明度={scores.get('聪明度', 'N/A')}/100, "
                f"皮实={scores.get('皮实', 'N/A')}/100, "
                f"勤奋={scores.get('勤奋', 'N/A')}/100"
            )

        return {
            "conversation_history": conversation_history,
            "round_evaluations": round_evaluations,
        }

    def _run_manual_interview(
        self, candidate_data: Dict, questions_result: Dict, max_rounds: int
    ) -> Dict:
        """运行手动面试（人工回答）"""
        print_message("\n👤 手动面试模式")
        print_message(f"📝 岗位: {candidate_data.get('position', '未指定')}")
        print_message("💡 提示: 请根据问题输入您的回答\n")

        conversation_history = []
        round_evaluations = []

        # 模拟问题
        sample_questions = [
            "请介绍一下您最具挑战性的项目经验。",
            "在团队合作中，您如何处理分歧？",
            "工作压力大时，您如何应对？",
            "请分享一个您主动解决问题的案例。",
            "您如何保持学习和成长？",
        ]

        questions = sample_questions[:max_rounds]

        for i, question in enumerate(questions, 1):
            print_message(f"\n{'─' * 60}")
            print_message(f"第 {i}/{len(questions)} 轮")
            print_message(f"{'─' * 60}")

            # 面试官提问
            print_message(f"\n[面试官]: {question}")
            conversation_history.append({"role": "面试官", "content": question})

            # 候选人手动回答
            try:
                answer = input("\n[您的回答]: ")
                if not answer.strip():
                    answer = "（候选人暂未回答）"
            except (EOFError, KeyboardInterrupt):
                answer = "（面试中断）"
                print_message("\n⚠️ 面试已中断")
                break

            conversation_history.append({"role": "候选人", "content": answer})

            # 评估
            print_message("\n[评估中...]")
            eval_result = self.eval_agent.evaluate_single_response(
                question=question,
                answer=answer,
                candidate_info=candidate_data,
                question_intent=f"第{i}轮考察",
            )

            round_evaluations.append(eval_result)

            # 显示评分
            scores = eval_result.get("scores", {})
            print_message(
                f"✓ 评分: 聪明度={scores.get('聪明度', 'N/A')}/100, "
                f"皮实={scores.get('皮实', 'N/A')}/100, "
                f"勤奋={scores.get('勤奋', 'N/A')}/100"
            )

        return {
            "conversation_history": conversation_history,
            "round_evaluations": round_evaluations,
        }

    def _display_summary(self, result: Dict):
        """显示面试总结"""
        print_message("\n" + "=" * 60)
        print_message("📊 面试总结")
        print_message("=" * 60)

        final_eval = result.get("final_evaluation", {})
        avg_scores = final_eval.get("average_scores", {})

        print_message(f"\n岗位: {result['candidate_info']['position']}")
        print_message(f"模式: {result['interview_mode']}")
        print_message(f"轮次: {result['total_rounds']}")

        print_message("\n平均评分:")
        print_message(f"  • 聪明度: {avg_scores.get('聪明度', 0)}/100")
        print_message(f"  • 皮实: {avg_scores.get('皮实', 0)}/100")
        print_message(f"  • 勤奋: {avg_scores.get('勤奋', 0)}/100")

        # 简单的总评
        avg_total = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0
        if avg_total >= 85:
            recommendation = "强烈推荐 ✅"
        elif avg_total >= 75:
            recommendation = "推荐 👍"
        elif avg_total >= 60:
            recommendation = "可以考虑 🤔"
        else:
            recommendation = "不推荐 ❌"

        print_message(f"\n综合建议: {recommendation}")
        print_message("=" * 60)

    def _save_interview_result(self, result: Dict, candidate_data: Dict) -> str:
        """保存面试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        position = candidate_data.get("position", "unknown").replace(" ", "_")
        filename = f"interview_result_{position}_{timestamp}.json"
        filepath = Path("temp") / filename

        # 确保目录存在
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # 保存
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return str(filepath)

    def get_current_interview(self) -> Optional[Dict]:
        """获取当前面试结果"""
        return self.current_interview

    def __repr__(self):
        return f"InterviewSimulator(interview_agent={self.interview_agent}, eval_agent={self.eval_agent})"
