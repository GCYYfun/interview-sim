"""
HR Agent - 负责根据简历和面试回答问HR指标相关问题

功能：
- 基于候选人简历和岗位要求，提出HR评估相关问题
- 考察候选人的聪明度、皮实、勤奋等指标
- 评估候选人回答是否符合岗位要求
- 判断面试进度和是否需要继续
"""

from menglong.agents.role_play import RoleAgent
from typing import Dict, List


class HRAgent(RoleAgent):
    """HR评估Agent - 负责提问和评估候选人"""

    def __init__(self, role_config: dict = None):
        """
        初始化HR Agent

        Args:
            role_config: RoleAgent配置字典，包含角色设定、目标等
        """
        if role_config is None:
            # 默认HR配置
            role_config = {
                "name": "HR面试官",
                "role": "资深HR面试官",
                "goal": "通过问题评估候选人的聪明度、皮实、勤奋等核心素质",
            }
        super().__init__(role_config=role_config)

        # HR评估指标
        self.evaluation_dimensions = {
            "聪明度": "逻辑思维、问题理解、学习能力、创新思维",
            "皮实": "抗压能力、挫折恢复力、情绪管理、韧性",
            "勤奋": "自驱力、主动性、持续投入、结果导向",
        }

    def opening_statement(self, topic: str = None) -> str:
        """
        HR开场白

        Args:
            topic: 面试岗位名称

        Returns:
            str: 开场白内容
        """
        if topic:
            return f"[HR]: 欢迎参加我们公司的面试！今天我们会通过提问交流的方式来了解您的背景和能力，尤其对{topic}岗位的适配情况。请放轻松，展示您最好的一面。"
        return "[HR]: 欢迎参加我们公司的面试！今天我们会通过一些问题来了解您的背景和能力。请放轻松，展示您最好的一面。"

    def generate_question(
        self,
        candidate_resume: str,
        jd: str,
        dimension: str = None,
        conversation_history: List[Dict] = None,
    ) -> str:
        """
        基于简历和岗位要求生成面试问题

        Args:
            candidate_resume: 候选人简历
            jd: 岗位描述
            dimension: 想要考察的维度（聪明度/皮实/勤奋），None则自动选择
            conversation_history: 之前的对话历史

        Returns:
            str: 生成的面试问题
        """
        # 构建提示
        context = f"""
候选人简历：
{candidate_resume}

岗位要求：
{jd}

评估维度说明：
{self._format_dimensions()}
"""

        if dimension:
            context += f"\n重点考察维度：{dimension} - {self.evaluation_dimensions.get(dimension, '')}"

        if conversation_history:
            history_text = "\n".join(
                [
                    f"{msg['role']}: {msg['content'][:100]}..."
                    for msg in conversation_history[-3:]  # 最近3轮
                ]
            )
            context += f"\n\n前序对话：\n{history_text}"

        prompt = f"{context}\n\n请基于以上信息，提出一个针对性的面试问题："

        question = self.chat(prompt)
        return question

    def evaluate_response(
        self, question: str, answer: str, candidate_info: Dict = None
    ) -> Dict:
        """
        评估候选人回答

        Args:
            question: 面试问题
            answer: 候选人回答
            candidate_info: 候选人背景信息（可选）

        Returns:
            dict: 评估结果
                {
                    "is_satisfactory": bool,  # 是否满意
                    "feedback": str,          # 评估反馈
                    "dimension_score": dict,  # 各维度评分
                    "should_record": bool,    # 是否记录为关键问答
                }
        """
        context = f"""
面试问题：{question}
候选人回答：{answer}
"""

        if candidate_info:
            context += f"\n候选人背景：{candidate_info}"

        prompt = f"""{context}

请从HR角度评估这个回答：
1. 回答是否符合我们的评估标准（聪明度/皮实/勤奋）？
2. 回答中体现了哪些能力？
3. 是否需要记录为关键有效问答？

请输出评估结果，如果符合要求，请在最后输出[OK]。
"""

        evaluation_text = self.chat(prompt)

        # 解析评估结果
        is_ok = "[OK]" in evaluation_text or "符合要求" in evaluation_text

        return {
            "is_satisfactory": is_ok,
            "feedback": evaluation_text,
            "dimension_score": {},  # TODO: 可以进一步解析分数
            "should_record": is_ok,
        }

    def should_end_interview(
        self, conversation_history: List[Dict], max_rounds: int = 6
    ) -> bool:
        """
        判断是否应该结束面试

        Args:
            conversation_history: 对话历史
            max_rounds: 最大轮次（默认6轮，即3组问答）

        Returns:
            bool: 是否结束面试
        """
        # 统计有效问答轮次
        dialogue_count = len(
            [
                msg
                for msg in conversation_history
                if msg.get("role") in ["HR", "HR面试官", "候选人"]
            ]
        )

        return dialogue_count >= max_rounds

    def generate_final_summary(
        self, conversation_history: List[Dict], evaluations: List[Dict]
    ) -> Dict:
        """
        生成最终面试总结

        Args:
            conversation_history: 完整对话历史
            evaluations: 所有评估记录

        Returns:
            dict: 面试总结
        """
        # 构建总结上下文
        history_text = "\n".join(
            [f"[{msg['role']}]: {msg['content']}" for msg in conversation_history]
        )

        eval_text = "\n".join(
            [
                f"评估{i + 1}: {eval_item['feedback']}"
                for i, eval_item in enumerate(evaluations)
            ]
        )

        prompt = f"""
基于以下面试记录，请提供综合评估：

【对话记录】
{history_text}

【各轮评估】
{eval_text}

请给出：
1. 候选人整体表现总结
2. 三维能力评分（聪明度/皮实/勤奋，各100分）
3. 录用建议（推荐录用/可以考虑/不推荐）
4. 理由说明
"""

        summary = self.chat(prompt)

        return {
            "summary": summary,
            "timestamp": self._get_timestamp(),
            "total_rounds": len(conversation_history) // 2,  # 问答对数
        }

    def _format_dimensions(self) -> str:
        """格式化评估维度说明"""
        return "\n".join(
            [f"- {dim}: {desc}" for dim, desc in self.evaluation_dimensions.items()]
        )

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def __repr__(self):
        return f"HRAgent(role={self.id})"
