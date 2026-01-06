from typing import List, Dict, Any, Optional
from .base_agent import BaseAgent
import json

class InterviewerAgent(BaseAgent):
    """
    面试官 Agent，负责提问、追问和控制面试节奏。
    支持两种模式：
    1. 无 Transcript 模式：自主规划面试内容
    2. 有 Transcript 模式：参考 Transcript 进行面试
    """
    
    END_SIGNAL = "[END_INTERVIEW]"
    
    def __init__(self, model: str = "anthropic/global.anthropic.claude-sonnet-4-5-20250929-v1:0"):
        super().__init__(name="Interviewer", role="interviewer", model=model)
    
    def _build_system_prompt(self, jd: str, resume: str, transcript: Optional[str] = None) -> str:
        """构建系统提示词"""
        
        base_prompt = f"""你是一位资深的技术面试官，正在根据岗位要求（JD）和候选人简历进行面试。

## 岗位要求（JD）
{jd}

## 候选人简历
{resume}

## 你的职责
1. **系统性考察**：根据 JD 要求，全面评估候选人的能力
2. **深入追问**：当候选人回答模糊或不充分时，追问细节
3. **灵活调整**：根据候选人的回答调整后续问题
4. **控制节奏**：合理分配时间，确保重点内容被充分考察

## 面试规则
- 每次只问一个问题
- 问题要具体、有针对性
- 追问要有深度，探索候选人的真实能力
- 当你认为已充分了解候选人时，输出 {self.END_SIGNAL} 结束面试

## 输出格式
直接输出你要问的问题，不需要额外格式。当面试结束时，先给出简短的结束语，然后输出 {self.END_SIGNAL}。
"""
        
        if transcript:
            transcript_prompt = f"""
## 参考面试记录（Transcript）
以下是一份参考面试记录，你可以借鉴其中的提问思路和方向，但需要根据候选人的实际回答灵活调整：

{transcript}

注意：
- Transcript 仅作参考，不要完全照搬
- 根据候选人的回答灵活调整提问
- 如果候选人的回答与 Transcript 中不同，要针对性追问
"""
            return base_prompt + transcript_prompt
        
        return base_prompt
    
    def generate_question(self, context: Dict[str, Any], history: List[Dict[str, str]]) -> str:
        """
        生成下一个面试问题。
        
        Args:
            context: 包含 'jd', 'resume', 可选 'transcript'
            history: 对话历史，格式为 [{"role": "interviewer/candidate", "content": "..."}]
        
        Returns:
            面试官的问题或结束信号
        """
        jd = context.get('jd', '未提供岗位要求')
        resume = context.get('resume', '未提供简历')
        transcript = context.get('transcript', None)
        
        system_prompt = self._build_system_prompt(jd, resume, transcript)
        
        # 转换历史记录格式：interviewer -> assistant, candidate -> user
        mapped_messages = []
        for msg in history:
            role = "assistant" if msg["role"] == "interviewer" else "user"
            mapped_messages.append({"role": role, "content": msg["content"]})
        
        # 如果历史为空，添加一个开始信号
        if not mapped_messages:
            mapped_messages.append({"role": "user", "content": "面试开始，请提出第一个问题。"})
            
        return self.generate_response(mapped_messages, system_prompt=system_prompt)
    
    def is_end_signal(self, response: str) -> bool:
        """检测响应中是否包含结束信号"""
        return self.END_SIGNAL in response
    
    def run(self, context: Dict[str, Any], history: List[Dict[str, str]]) -> str:
        return self.generate_question(context, history)
