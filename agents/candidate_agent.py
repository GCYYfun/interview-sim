from typing import List, Dict, Any, Optional
from .base_agent import BaseAgent

class CandidateAgent(BaseAgent):
    """
    候选人 Agent，负责回答面试官的问题。
    支持两种模式：
    1. 无 Transcript 模式：根据简历自然回答
    2. 有 Transcript 模式：参考 Transcript 优化回答
    """
    
    def __init__(self, model: str = "anthropic/global.anthropic.claude-sonnet-4-5-20250929-v1:0"):
        super().__init__(name="Candidate", role="candidate", model=model)
    
    def _build_system_prompt(self, resume: str, jd: str = "", transcript: Optional[str] = None) -> str:
        """构建系统提示词"""
        
        base_prompt = f"""你是一位正在参加面试的候选人。你的背景和经历如下：

## 你的简历
{resume}

## 你正在应聘的岗位
{jd if jd else "未提供岗位信息"}

## 回答要求
1. **真实自然**：基于你的简历内容回答，不要编造不存在的经历
2. **展示能力**：在回答中体现你的专业能力和经验
3. **具体详实**：用具体的例子和数据来支撑你的回答
4. **保持谦逊**：既要自信展示能力，也要表现出学习和成长的意愿
5. **适当发挥**：在简历基础上，可以合理补充细节，使回答更加完整

## 输出格式
直接回答问题即可，像真实面试一样自然地表达。
"""
        
        if transcript:
            transcript_prompt = f"""
## 参考面试记录（Transcript）
以下是一份参考面试记录，你可以借鉴其中的回答思路和表达方式：

{transcript}

注意：
- 参考 Transcript 中的优秀表达和回答结构
- 但要根据你自己的简历内容来回答
- 保持回答的一致性和真实性
"""
            return base_prompt + transcript_prompt
        
        return base_prompt

    def generate_answer(self, context: Dict[str, Any], history: List[Dict[str, str]]) -> str:
        """
        生成对面试官问题的回答。
        
        Args:
            context: 包含 'resume', 可选 'jd', 'transcript'
            history: 对话历史，格式为 [{"role": "interviewer/candidate", "content": "..."}]
        
        Returns:
            候选人的回答
        """
        resume = context.get('resume', '未提供简历')
        jd = context.get('jd', '')
        transcript = context.get('transcript', None)
        
        system_prompt = self._build_system_prompt(resume, jd, transcript)
        
        # 转换历史记录格式：candidate -> assistant, interviewer -> user
        mapped_messages = []
        for msg in history:
            role = "assistant" if msg["role"] == "candidate" else "user"
            mapped_messages.append({"role": role, "content": msg["content"]})
            
        return self.generate_response(mapped_messages, system_prompt=system_prompt)

    def run(self, context: Dict[str, Any], history: List[Dict[str, str]]) -> str:
        return self.generate_answer(context, history)
