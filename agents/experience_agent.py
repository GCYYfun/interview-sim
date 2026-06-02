"""
经验Agent - 根据面试数据抽取通用面试经验，形成文档

功能：
- 基于简历、岗位描述、面试对话、评价结果抽取通用经验
- 形成结构化的面试经验文档
- 供HR Agent参考使用
- 后续将集成到manager模块中

注意：这是一个过渡性的Agent，后续会整合到manager/experience_extractor.py中
"""

from menglong import Model
from menglong.ml_model.schema.ml_request import UserMessage as user
from typing import Dict, List
from datetime import datetime
import json


class ExperienceAgent:
    """经验提取Agent - 从面试数据中抽取通用经验"""

    def __init__(self):
        """初始化经验Agent"""
        self.model = Model()
        self.experiences = []  # 存储提取的经验

    def extract_experience(
        self,
        resume: str,
        jd: str,
        conversation_history: List[Dict],
        evaluation_result: Dict,
    ) -> Dict:
        """
        从单次面试中抽取经验

        Args:
            resume: 候选人简历
            jd: 岗位描述
            conversation_history: 面试对话记录
            evaluation_result: 评估结果

        Returns:
            dict: 提取的经验
        """
        # 构建对话文本
        conversation_text = self._format_conversation(conversation_history)

        # 构建评估摘要
        eval_summary = self._format_evaluation(evaluation_result)

        extraction_prompt = f"""
作为资深HR专家，请从以下面试案例中提取通用的面试经验和洞察：

【候选人简历】
{resume[:500]}...

【岗位要求】
{jd[:500]}...

【面试对话】
{conversation_text}

【评估结果】
{eval_summary}

请提取以下方面的经验：

## 1. 问题设计经验
- 哪些问题有效地考察了候选人的能力？
- 问题的提问方式有什么值得借鉴的地方？
- 如何通过追问深入了解候选人？

## 2. 回答评估要点
- 优秀回答的特征是什么？
- 需要警惕的回答模式有哪些？
- 如何识别候选人的真实水平？

## 3. 三维能力识别技巧
- **聪明度**：如何通过对话判断候选人的思维能力？
- **皮实**：什么样的回答体现了抗压能力？
- **勤奋**：如何识别候选人的自驱力？

## 4. 岗位匹配洞察
- 这个岗位需要重点关注哪些能力？
- 简历和实际表现的匹配度如何验证？
- 有哪些易被忽略但重要的考察点？

## 5. 面试技巧总结
- 本次面试的成功之处
- 可以改进的地方
- 对类似岗位面试的建议

请提供结构化的经验总结，重点突出可复用的模式和方法。
"""

        try:
            response = self.model.chat([user(content=extraction_prompt)])
            experience_text = self._extract_response_text(response)

            experience = {
                "id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "position": evaluation_result.get("candidate_info", {}).get(
                    "position", "未知岗位"
                ),
                "content": experience_text,
                "source": {
                    "resume_length": len(resume),
                    "jd_length": len(jd),
                    "conversation_rounds": len(conversation_history) // 2,
                    "evaluation_scores": evaluation_result.get("average_scores", {}),
                },
                "extracted_at": datetime.now().isoformat(),
            }

            self.experiences.append(experience)
            return experience

        except Exception as e:
            return {
                "id": f"exp_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "error": str(e),
                "content": f"提取经验时出错: {str(e)}",
                "extracted_at": datetime.now().isoformat(),
            }

    def consolidate_experiences(self, experience_ids: List[str] = None) -> Dict:
        """
        整合多个经验，形成通用的面试指南

        Args:
            experience_ids: 要整合的经验ID列表，None则整合所有

        Returns:
            dict: 整合后的通用经验
        """
        # 筛选要整合的经验
        if experience_ids:
            target_experiences = [
                exp for exp in self.experiences if exp["id"] in experience_ids
            ]
        else:
            target_experiences = self.experiences

        if not target_experiences:
            return {
                "error": "没有可整合的经验",
                "content": "请先提取一些面试经验",
            }

        # 准备整合文本
        experiences_text = "\n\n---\n\n".join(
            [
                f"## 经验 {i + 1} ({exp['position']})\n{exp['content']}"
                for i, exp in enumerate(target_experiences)
            ]
        )

        consolidation_prompt = f"""
基于以下多个面试案例的经验，请整合出一份通用的面试指南文档：

{experiences_text}

请整合出：

# 通用面试经验指南

## 一、问题库设计

### 1.1 聪明度考察问题
（整合各案例中有效的聪明度考察问题）

### 1.2 皮实考察问题
（整合各案例中有效的皮实考察问题）

### 1.3 勤奋考察问题
（整合各案例中有效的勤奋考察问题）

## 二、评估标准

### 2.1 优秀回答特征
- 聪明度维度：
- 皮实维度：
- 勤奋维度：

### 2.2 风险回答特征
- 需要警惕的表述
- 常见的夸大模式
- 逻辑不一致的信号

## 三、面试技巧

### 3.1 开场与氛围营造
### 3.2 追问与深挖技巧
### 3.3 压力测试方法
### 3.4 真实性验证技巧

## 四、岗位适配要点

按岗位类型总结：
（基于案例中的不同岗位，总结各类岗位的关键考察点）

## 五、常见陷阱与误区

### 5.1 面试官容易忽略的点
### 5.2 候选人常见的包装手段
### 5.3 评估偏差的避免

---
请提供完整、实用、可操作的面试指南。
"""

        try:
            response = self.model.chat([user(content=consolidation_prompt)])
            guide_text = self._extract_response_text(response)

            consolidated = {
                "id": f"guide_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "title": "通用面试经验指南",
                "content": guide_text,
                "source_count": len(target_experiences),
                "source_positions": list(
                    set([exp["position"] for exp in target_experiences])
                ),
                "created_at": datetime.now().isoformat(),
            }

            return consolidated

        except Exception as e:
            return {
                "id": f"guide_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "error": str(e),
                "content": f"整合经验时出错: {str(e)}",
                "created_at": datetime.now().isoformat(),
            }

    def save_experience(self, experience: Dict, filepath: str = None):
        """
        保存经验到文件

        Args:
            experience: 经验字典
            filepath: 保存路径，默认为experiences/目录
        """
        if filepath is None:
            filename = f"{experience['id']}.json"
            filepath = f"experiences/{filename}"

        try:
            import os

            os.makedirs("experiences", exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(experience, f, ensure_ascii=False, indent=2)

            return {"success": True, "filepath": filepath}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _format_conversation(self, conversation_history: List[Dict]) -> str:
        """格式化对话历史"""
        lines = []
        for msg in conversation_history[:20]:  # 最多20条
            role = msg.get("role", "未知")
            content = msg.get("content", "")
            lines.append(f"[{role}]: {content[:200]}...")  # 每条最多200字符

        return "\n".join(lines)

    def _format_evaluation(self, evaluation_result: Dict) -> str:
        """格式化评估结果"""
        avg_scores = evaluation_result.get("average_scores", {})
        return f"""
- 聪明度评分: {avg_scores.get("聪明度", "N/A")}/100
- 皮实评分: {avg_scores.get("皮实", "N/A")}/100
- 勤奋评分: {avg_scores.get("勤奋", "N/A")}/100
- 评估轮次: {evaluation_result.get("round_count", 0)}
"""

    def _extract_response_text(self, response):
        """提取响应文本"""
        if hasattr(response, "text"):
            return response.text
        elif hasattr(response, "content"):
            return response.content
        elif isinstance(response, dict) and "content" in response:
            return response["content"]
        elif isinstance(response, str):
            return response
        else:
            return str(response)

    def get_all_experiences(self) -> List[Dict]:
        """获取所有已提取的经验"""
        return self.experiences

    def __repr__(self):
        return f"ExperienceAgent(experiences={len(self.experiences)})"


# TODO: 后续将此Agent整合到manager/experience_extractor.py中
# 整合时需要：
# 1. 保持manager模块的数据模型和结构
# 2. 复用manager的存储机制
# 3. 统一token计费和成本统计
# 4. 支持增量提取和批量处理
