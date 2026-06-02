"""
面试Agent - 负责生成面试问题和分析候选人匹配度

功能：
- 基于简历和JD生成针对性的面试问题
- 分析候选人与岗位的匹配度
- 提供深度追问策略
- 基于经验库优化问题设计

这是一个面试辅助Agent，帮助HR准备和优化面试流程
"""

import json
import glob
from pathlib import Path
from datetime import datetime
from menglong import Model
from menglong.ml_model.schema.ml_request import UserMessage as user
from typing import Dict, List


class InterviewAgent:
    """面试辅助Agent - 基于简历和JD生成针对性的面试问题"""

    def __init__(self, experience_pattern: str = None):
        """
        初始化面试Agent

        Args:
            experience_pattern: 经验文件匹配模式，默认查找general_interview_guidelines_*.json
        """
        self.model = Model()
        self.experiences = {}
        self.experience_pattern = (
            experience_pattern or "general_interview_guidelines_*.json"
        )
        self.load_latest_experience()

    def load_latest_experience(self):
        """加载最新的面试经验库"""
        try:
            # 优先查找新的通用面试提问经验文件
            experience_files = glob.glob(self.experience_pattern)

            # 如果没有新格式文件，查找旧格式文件
            if not experience_files:
                experience_files = glob.glob("interview_experience_report_*.json")

            if experience_files:
                latest_file = max(
                    experience_files, key=lambda x: Path(x).stat().st_mtime
                )
                with open(latest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.experiences["integrated"] = data.get(
                        "integrated_experience", ""
                    )
                print(f"✅ 已加载经验库: {latest_file}")
            else:
                print("⚠️ 未找到经验报告，将使用基础面试技巧")
                self.experiences["integrated"] = self._get_default_experience()
        except Exception as e:
            print(f"❌ 加载经验失败: {str(e)}")
            self.experiences["integrated"] = self._get_default_experience()

    def _get_default_experience(self) -> str:
        """获取默认的面试经验"""
        return """
# 基础面试技巧

## 聪明度评估
- 通过具体场景测试逻辑思维
- 递进式追问验证学习能力
- 观察举一反三的能力

## 皮实评估
- 挖掘挫折经历和应对方式
- 适度压力测试
- 观察情绪稳定性

## 勤奋评估
- 验证具体行为和时间投入
- 区分主动性和被动完成
- 考察持续性表现
"""

    def generate_interview_questions(
        self,
        resume: str,
        jd: str,
        focus_areas: List[str] = None,
        position: str = None,
    ) -> Dict:
        """
        根据简历和JD生成针对性的面试问题

        Args:
            resume: 候选人简历
            jd: 岗位描述
            focus_areas: 重点关注领域，如['聪明度', '皮实', '勤奋']
            position: 岗位名称（可选）

        Returns:
            包含问题和追问策略的字典
                {
                    "questions": List[Dict],  # 问题列表
                    "analysis": str,          # 候选人分析
                    "strategy": str,          # 面试策略
                    "token_usage": Dict,      # Token使用情况
                }
        """
        if focus_areas is None:
            focus_areas = ["聪明度", "皮实", "勤奋"]

        focus_str = "、".join(focus_areas)
        position_context = f"岗位：{position}\n\n" if position else ""

        prompt = f"""你是一位资深的HR面试专家，请根据以下信息为即将到来的面试生成针对性的问题和追问策略。

{position_context}## 候选人简历：
{resume}

## 岗位描述：
{jd}

## 重点评估维度：
{focus_str}

## 参考面试经验库：
{self.experiences.get("integrated", "")}

## 任务要求：
请基于上述信息，为这位候选人设计一套完整的面试问题方案，包括：

### 1. 候选人分析
- 简历亮点分析
- 潜在风险点识别
- 与岗位的匹配度评估

### 2. 开场破冰问题（2-3个）
- 让候选人放松的开场问题
- 基于简历的温和询问

### 3. 核心能力测试问题
针对{focus_str}，设计具体的测试问题：

#### 聪明度测试问题
- 基于岗位要求的场景假设题
- 逻辑思维测试问题
- 学习能力验证问题

#### 皮实测试问题  
- 针对简历中可能的挫折点的深度挖掘
- 压力测试问题
- 抗压能力验证

#### 勤奋测试问题
- 基于简历经历的具体行为验证
- 主动性和持续性考察
- 时间投入和成果验证

### 4. 深度追问策略
对于每个核心问题，提供：
- 如果候选人回答优秀，如何进一步验证
- 如果候选人回答一般，如何深度挖掘
- 如果候选人回避问题，如何巧妙追问

### 5. 风险预警
- 需要特别关注的回答模式
- 可能的夸大或虚假信息识别点
- 不匹配的危险信号

### 6. 结尾问题（1-2个）
- 了解候选人期望和动机
- 给候选人提问的机会

## 输出格式要求：
- 结构清晰，便于面试官快速查阅
- 每个问题都要有明确的评估目的
- 提供具体的观察要点和评分标准
- 给出预期的优秀回答示例

请确保问题设计具有针对性，能够有效识别候选人的真实能力水平。"""

        try:
            print("🤖 正在根据简历和JD生成面试问题...")
            response = self.model.chat([user(content=prompt)])
            questions_text = self._extract_response_text(response)

            # 构建返回结果
            result = {
                "position": position or "未指定",
                "candidate_resume": (
                    resume[:200] + "..." if len(resume) > 200 else resume
                ),
                "job_description": jd[:200] + "..." if len(jd) > 200 else jd,
                "focus_areas": focus_areas,
                "generated_questions": questions_text,
                "generation_time": datetime.now().isoformat(),
                "token_usage": self._extract_token_usage(response),
            }

            print("✅ 面试问题生成完成!")
            return result

        except Exception as e:
            print(f"❌ 生成面试问题时出错: {str(e)}")
            return {"error": str(e), "generation_time": datetime.now().isoformat()}

    def analyze_candidate_match(
        self, resume: str, jd: str, position: str = None
    ) -> Dict:
        """
        分析候选人与岗位的匹配度

        Args:
            resume: 候选人简历
            jd: 岗位描述
            position: 岗位名称（可选）

        Returns:
            匹配度分析结果
                {
                    "match_score": int,       # 总体匹配度（1-10）
                    "analysis": str,          # 详细分析
                    "strengths": List[str],   # 主要优势
                    "risks": List[str],       # 风险点
                    "recommendations": str,   # 面试建议
                }
        """
        position_context = f"岗位：{position}\n\n" if position else ""

        prompt = f"""你是一位资深的HR专家，请分析以下候选人与岗位的匹配情况。

{position_context}## 候选人简历：
{resume}

## 岗位描述：
{jd}

## 分析要求：
请从以下维度进行详细分析：

### 1. 硬技能匹配度（1-10分）
- 技术能力匹配情况
- 行业经验相关性
- 学历背景适配性

### 2. 软技能匹配度（1-10分）
- 沟通协调能力
- 团队合作精神
- 抗压和适应能力

### 3. 经验匹配度（1-10分）
- 相关工作经验
- 项目经验匹配
- 成长轨迹合理性

### 4. 文化匹配度（1-10分）
- 价值观匹配
- 工作风格适配
- 发展意愿契合度

### 5. 综合评估
- 总体匹配度评分（1-10分）
- 主要优势（TOP 3）
- 主要风险点（TOP 3）
- 面试重点关注事项

### 6. 面试建议
- 需要重点验证的能力
- 可能的加分项挖掘
- 风险点探查策略

请提供具体的评分理由和建议。"""

        try:
            print("🔍 正在分析候选人匹配度...")
            response = self.model.chat([user(content=prompt)])
            analysis_text = self._extract_response_text(response)

            result = {
                "position": position or "未指定",
                "candidate_resume": (
                    resume[:200] + "..." if len(resume) > 200 else resume
                ),
                "job_description": jd[:200] + "..." if len(jd) > 200 else jd,
                "match_analysis": analysis_text,
                "analysis_time": datetime.now().isoformat(),
                "token_usage": self._extract_token_usage(response),
            }

            print("✅ 匹配度分析完成!")
            return result

        except Exception as e:
            print(f"❌ 分析匹配度时出错: {str(e)}")
            return {"error": str(e), "analysis_time": datetime.now().isoformat()}

    def save_interview_plan(
        self, plan_result: Dict, output_path: str = None, candidate_id: str = None
    ) -> str:
        """
        保存面试方案到文件

        Args:
            plan_result: 面试方案结果
            output_path: 输出路径
            candidate_id: 候选人ID

        Returns:
            保存的文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if candidate_id:
                output_path = f"temp/interview_plan_{candidate_id}_{timestamp}.json"
            else:
                output_path = f"temp/interview_plan_{timestamp}.json"

        try:
            # 确保目录存在
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(plan_result, f, ensure_ascii=False, indent=2)

            print(f"📋 面试方案已保存到: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ 保存面试方案失败: {str(e)}")
            return ""

    def _extract_token_usage(self, response) -> Dict:
        """从响应中提取token使用情况"""
        try:
            if hasattr(response, "usage"):
                usage = response.usage
                return {
                    "input_tokens": getattr(usage, "input_tokens", 0),
                    "output_tokens": getattr(usage, "output_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }
            else:
                return {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }
        except Exception as e:
            print(f"⚠️ 提取token使用情况时出错: {str(e)}")
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

    def _extract_response_text(self, response) -> str:
        """从LLM响应中提取文本内容"""
        try:
            if hasattr(response, "message") and hasattr(response.message, "content"):
                if hasattr(response.message.content, "text"):
                    return response.message.content.text
                else:
                    return str(response.message.content)
            elif hasattr(response, "content"):
                return str(response.content)
            elif hasattr(response, "text"):
                return response.text
            else:
                return str(response)
        except Exception as e:
            print(f"⚠️ 提取响应文本时出错: {str(e)}")
            return str(response)

    def __repr__(self):
        has_exp = "已加载" if self.experiences.get("integrated") else "未加载"
        return f"InterviewAgent(experience={has_exp})"
