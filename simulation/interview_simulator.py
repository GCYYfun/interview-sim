"""
面试模拟环境

负责协调面试官和候选人 Agent 的交互，控制面试流程，
检测结束信号，保存对话记录和上下文信息。
"""

import os
import json
import datetime
from typing import Dict, Any, List, Optional

from agents.interviewer_agent import InterviewerAgent
from agents.candidate_agent import CandidateAgent


class InterviewSimulator:
    """
    面试模拟器，协调两个 Agent 进行模拟面试。
    """
    
    def __init__(
        self, 
        jd: str, 
        resume: str, 
        transcript: Optional[str] = None,
        interviewer_model: str = "anthropic/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        candidate_model: str = "anthropic/global.anthropic.claude-sonnet-4-5-20250929-v1:0"
    ):
        """
        初始化模拟器。
        
        Args:
            jd: 岗位要求文本
            resume: 候选人简历文本
            transcript: 可选的参考面试记录
            interviewer_model: 面试官使用的模型
            candidate_model: 候选人使用的模型
        """
        self.jd = jd
        self.resume = resume
        self.transcript = transcript
        
        # 构建上下文
        self.context = {
            "jd": jd,
            "resume": resume,
        }
        if transcript:
            self.context["transcript"] = transcript
        
        # 初始化 Agents
        self.interviewer = InterviewerAgent(model=interviewer_model)
        self.candidate = CandidateAgent(model=candidate_model)
        
        # 对话历史（统一格式）
        self.conversation: List[Dict[str, str]] = []
        
        # 分别存储两个 Agent 视角的消息历史（用于 debug）
        self.interviewer_messages: List[Dict[str, str]] = []
        self.candidate_messages: List[Dict[str, str]] = []
        
        # 元数据
        self.metadata: Dict[str, Any] = {
            "start_time": None,
            "end_time": None,
            "total_turns": 0,
            "ended_by_interviewer": False,
            "has_transcript": transcript is not None
        }
    
    def run(self, max_turns: int = 20, verbose: bool = True) -> Dict[str, Any]:
        """
        执行模拟面试。
        
        Args:
            max_turns: 最大对话轮数（一问一答算一轮）
            verbose: 是否打印过程
        
        Returns:
            包含完整对话记录和元数据的字典
        """
        self.metadata["start_time"] = datetime.datetime.now().isoformat()
        
        if verbose:
            print("=" * 60)
            print("面试模拟开始")
            print("=" * 60)
            if self.transcript:
                print("[模式] 有 Transcript 参考")
            else:
                print("[模式] 自主面试")
            print("-" * 60)
        
        for turn in range(max_turns):
            self.metadata["total_turns"] = turn + 1
            
            # 面试官提问
            if verbose:
                print(f"\n[第 {turn + 1} 轮 - 面试官提问]")
            
            interviewer_response = self.interviewer.run(self.context, self.conversation)
            
            # 检查是否结束
            if self.interviewer.is_end_signal(interviewer_response):
                self.metadata["ended_by_interviewer"] = True
                # 移除结束信号，保留结束语
                clean_response = interviewer_response.replace(
                    InterviewerAgent.END_SIGNAL, ""
                ).strip()
                if clean_response:
                    self._add_message("interviewer", clean_response)
                    # if verbose:
                    #     print(f"\n{clean_response}")
                
                if verbose:
                    print("\n" + "=" * 60)
                    print("面试官主动结束面试")
                    print("=" * 60)
                break
            
            self._add_message("interviewer", interviewer_response)
            
            # if verbose:
            #     print(f"\n{interviewer_response}")
            
            # 候选人回答
            if verbose:
                print(f"\n[第 {turn + 1} 轮 - 候选人回答]")
            
            candidate_response = self.candidate.run(self.context, self.conversation)
            self._add_message("candidate", candidate_response)
            
            # if verbose:
            #     print(f"\n{candidate_response}")
        
        else:
            # 达到最大轮数
            if verbose:
                print("\n" + "=" * 60)
                print(f"达到最大轮数 ({max_turns})，面试结束")
                print("=" * 60)
        
        self.metadata["end_time"] = datetime.datetime.now().isoformat()
        
        return self._build_result()
    
    def _add_message(self, role: str, content: str):
        """添加消息到历史记录"""
        msg = {"role": role, "content": content}
        self.conversation.append(msg)
        
        # 同时记录到各自的视角历史
        if role == "interviewer":
            self.interviewer_messages.append({
                "role": "assistant",
                "content": content
            })
            self.candidate_messages.append({
                "role": "user", 
                "content": content
            })
        else:
            self.interviewer_messages.append({
                "role": "user",
                "content": content
            })
            self.candidate_messages.append({
                "role": "assistant",
                "content": content
            })
    
    def _build_result(self) -> Dict[str, Any]:
        """构建返回结果"""
        return {
            "conversation": self.conversation,
            "interviewer_context": {
                "system_prompt_info": {
                    "jd": self.jd[:200] + "..." if len(self.jd) > 200 else self.jd,
                    "resume": self.resume[:200] + "..." if len(self.resume) > 200 else self.resume,
                    "has_transcript": self.transcript is not None
                },
                "messages": self.interviewer_messages
            },
            "candidate_context": {
                "system_prompt_info": {
                    "resume": self.resume[:200] + "..." if len(self.resume) > 200 else self.resume,
                    "has_transcript": self.transcript is not None
                },
                "messages": self.candidate_messages
            },
            "metadata": self.metadata
        }
    
    def _conversation_to_txt(self, conversation: List[Dict[str, str]], name: str = "候选人") -> str:
        """
        将对话记录转换为 TXT 格式。
        
        格式: name(time):content
        例如: 面试官(00:01:30): 请先自我介绍一下
        
        Args:
            conversation: 对话记录列表
            name: 候选人名称
        
        Returns:
            格式化的文本字符串
        """
        lines = []
        # 假设每条消息间隔约 30 秒（模拟时间）
        time_offset = 0
        
        for msg in conversation:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # 格式化时间 HH:MM:SS
            hours = time_offset // 3600
            minutes = (time_offset % 3600) // 60
            seconds = time_offset % 60
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # 确定显示名称
            if role == "interviewer":
                display_name = "面试官"
            else:
                display_name = name
            
            lines.append(f"{display_name}({time_str}): {content.replace(chr(10), ' ').replace(chr(13), '')}")
            
            # 根据内容长度估算时间偏移（每 100 字约 30 秒）
            time_offset += max(30, len(content) // 100 * 30 + 30)
        
        return "\n".join(lines)
    
    def _conversation_to_markdown(
        self, 
        conversation: List[Dict[str, str]], 
        name: str = "候选人",
        jd_name: str = ""
    ) -> str:
        """
        将对话记录转换为 Markdown 格式（可视化友好）。
        """
        lines = []
        
        # 标题
        title = f"模拟面试记录 - {name}"
        if jd_name:
            title += f" ({jd_name})"
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"**面试官**: 面试官")
        lines.append(f"**候选人**: {name}")
        lines.append(f"**对话轮次**: {len(conversation)}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        turn = 0
        for msg in conversation:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if role == "interviewer":
                turn += 1
                lines.append(f"## 第 {turn} 轮")
                lines.append("")
                lines.append(f"🎤 **面试官**")
                lines.append("")
                lines.append(content)
            elif role == "candidate":
                lines.append(f"💬 **{name}**")
                lines.append("")
                quoted_content = "\n".join([f"> {line}" for line in content.split("\n")])
                lines.append(quoted_content)
            
            lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append("*面试记录结束*")
        
        return "\n".join(lines)
    
    def save(self, output_dir: str, name: str, jd_name: str):
        """
        保存模拟结果。
        
        Args:
            output_dir: 输出根目录
            name: 候选人名称
            jd_name: JD 名称
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"simulation_{name}_{jd_name}_{timestamp}"
        save_path = os.path.join(output_dir, folder_name)
        
        os.makedirs(save_path, exist_ok=True)
        
        result = self._build_result()
        
        # 保存完整对话 (JSON)
        with open(os.path.join(save_path, "conversation.json"), "w", encoding="utf-8") as f:
            json.dump(result["conversation"], f, ensure_ascii=False, indent=2)
        
        # 保存完整对话 (TXT)
        txt_content = self._conversation_to_txt(result["conversation"], name)
        with open(os.path.join(save_path, "conversation.txt"), "w", encoding="utf-8") as f:
            f.write(txt_content)
        
        # 保存完整对话 (Markdown)
        md_content = self._conversation_to_markdown(result["conversation"], name, jd_name)
        with open(os.path.join(save_path, "conversation.md"), "w", encoding="utf-8") as f:
            f.write(md_content)
        
        # 保存面试官上下文
        with open(os.path.join(save_path, "interviewer_context.json"), "w", encoding="utf-8") as f:
            json.dump(result["interviewer_context"], f, ensure_ascii=False, indent=2)
        
        # 保存候选人上下文
        with open(os.path.join(save_path, "candidate_context.json"), "w", encoding="utf-8") as f:
            json.dump(result["candidate_context"], f, ensure_ascii=False, indent=2)
        
        # 保存元数据
        with open(os.path.join(save_path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(result["metadata"], f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {save_path}")
        return save_path


