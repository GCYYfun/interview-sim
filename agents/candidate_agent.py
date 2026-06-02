"""
候选人Agent - 基于简历自动对话，用于数据增强/蒸馏

功能：
- 根据候选人简历信息，自动与HR Agent对话
- 模拟真实候选人的回答风格和内容
- 用于生成训练数据（数据增强）
- 支持模型蒸馏场景
"""

from menglong import Model
from menglong.ml_model.schema.ml_request import UserMessage as user


class CandidateAgent:
    """基于真实简历数据的候选人Agent"""

    def __init__(self, candidate_data: dict):
        """
        初始化候选人Agent

        Args:
            candidate_data: 包含简历、JD、岗位信息等的字典
                {
                    "resume": str,           # 候选人简历
                    "jd": str,              # 岗位描述
                    "position": str,        # 岗位名称
                    "intelligence_requirement": int,  # 聪明度要求(0-100)
                }
        """
        self.candidate_data = candidate_data
        self.resume = candidate_data.get("resume", "")
        self.jd = candidate_data.get("jd", "")
        self.position = candidate_data.get("position", "")
        self.intelligence_requirement = candidate_data.get(
            "intelligence_requirement", 0
        )
        self.model = Model()

        # 构建候选人人设提示
        self.persona_prompt = self._build_persona_prompt()

    def _build_persona_prompt(self):
        """构建候选人人设提示"""
        return f"""
你是一位求职者，正在参加{self.position}岗位的面试。

你的简历背景：
{self.resume}

应聘的岗位要求：
{self.jd}

岗位对聪明度的要求：{self.intelligence_requirement}/100

请根据你的简历背景，以第一人称回答面试官的问题。要求：
1. 回答要符合简历中的经历和背景
2. 体现出适合该岗位的能力和特质
3. 回答要真实可信，不要夸大
4. 语气要自然、诚恳，体现求职者的谨慎和积极
5. 如果问题涉及简历中没有的经历，要诚实说明并展示学习意愿
6. 回答长度适中，既要详细又不要过于冗长

现在请准备回答面试官的问题。
"""

    def answer_question(self, question: str) -> str:
        """
        根据简历背景回答面试问题

        Args:
            question: 面试官的问题

        Returns:
            str: 候选人的回答
        """
        try:
            # 构建完整的对话提示
            full_prompt = f"""
{self.persona_prompt}

面试官问：{question}

请以候选人身份回答这个问题：
"""

            response = self.model.chat([user(content=full_prompt)])
            answer = response.text

            # 清理格式，确保回答自然
            answer = self._clean_answer(answer)

            return answer

        except Exception as e:
            return f"抱歉，我需要一点时间思考这个问题。能否请您再详细说明一下？(Error: {str(e)})"

    def _clean_answer(self, answer: str) -> str:
        """清理回答格式，移除可能的角色标识和多余符号"""
        # 移除可能的角色标识
        role_prefixes = ["候选人：", "求职者：", "我：", "候选人:", "求职者:", "我:"]
        for prefix in role_prefixes:
            if answer.startswith(prefix):
                answer = answer.split("：", 1)[-1].split(":", 1)[-1].strip()
                break

        # 移除多余的引号
        answer = answer.strip("\"'")

        return answer.strip()

    def get_candidate_info_summary(self) -> dict:
        """获取候选人信息摘要"""
        return {
            "position": self.position,
            "intelligence_requirement": self.intelligence_requirement,
            "resume_preview": (
                self.resume[:200] + "..." if len(self.resume) > 200 else self.resume
            ),
            "jd_preview": self.jd[:200] + "..." if len(self.jd) > 200 else self.jd,
        }

    def __repr__(self):
        return f"CandidateAgent(position='{self.position}', intelligence_req={self.intelligence_requirement})"
