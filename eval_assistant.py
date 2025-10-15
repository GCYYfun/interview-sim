"""
面试评估助手模块
基于eval.md的评估标准，对候选人进行专业评估
"""

from menglong import Model
from menglong.ml_model.schema.ml_request import UserMessage as user
from datetime import datetime
from pathlib import Path


class EvalAssistant:
    """面试评估助手 - 基于专业评估标准进行候选人评估"""

    def __init__(self):
        """初始化评估助手"""
        self.model = Model()
        self.evaluation_criteria = self._load_evaluation_criteria()
        self.round_evaluations = []  # 存储每轮评估结果

    def _load_evaluation_criteria(self):
        """加载评估标准"""
        criteria_path = Path("eval.md")
        if criteria_path.exists():
            with open(criteria_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            # 如果文件不存在，返回默认评估标准
            return """
## 评估维度

### 聪明度（智商+情商）
- 85分以上：反应很快，思考能力强，有独到见解
- 75-85分：逻辑思维清晰，沟通顺畅，迅速抓住重点
- 75分以下：思维逻辑不清晰，难以理解复杂问题

### 勤奋（持续投入的自我驱动力）
- 85分以上：有较强自驱力和主动性，全力以赴
- 75-85分：能按时高质量完成任务，主动性较强
- 75分以下：做事拖拉，低质量完成

### 皮实（抗压能力和恢复力）
- 85分以上：面对挫折迎难而上，乐观自信
- 75-85分：能应对压力，不被逆境打败
- 75分以下：抗压能力差，容易情绪崩溃
"""

    def evaluate_single_response(
        self,
        question: str,
        answer: str,
        candidate_info: dict,
        question_intent: str = "",
    ) -> dict:
        """
        评估候选人对单个问题的回答

        Args:
            question: 面试问题
            answer: 候选人回答
            candidate_info: 候选人背景信息
            question_intent: 问题想要考察的能力（可选）

        Returns:
            dict: 包含评估结果的字典
        """
        evaluation_prompt = f"""
作为资深HR面试官，请根据以下评估标准对候选人的回答进行专业评估：

【评估标准】
{self.evaluation_criteria}

【候选人背景】
- 岗位: {candidate_info.get("position", "未知")}
- 聪明度要求: {candidate_info.get("intelligence_requirement", "N/A")}/100

【面试问题】
{question}

{f"【问题考察意图】{question_intent}" if question_intent else ""}

【候选人回答】
{answer}

请从以下几个方面进行评估：

1. **问题考察维度识别**：这个问题主要想考察候选人的哪个能力维度（聪明度/皮实/勤奋/目标感/价值观）？

2. **回答质量分析**：
   - 候选人的回答是否充分展现了该维度的能力？
   - 回答的具体性和真实性如何？
   - 是否有值得关注的亮点或风险点？

3. **三维评分**（按评估标准打分,只给出问题考察维度相关的分数，无关维度不评分）：
   - 聪明度：_/100（需说明评分依据）
   - 皮实：_/100（需说明评分依据）
   - 勤奋：_/100（需说明评分依据）

4. **追问建议**：基于这个回答，建议下一轮如何深入追问？

5. **关键观察点**：有哪些需要在后续面试中重点验证的地方？

请以结构化的方式输出评估结果。
"""

        try:
            response = self.model.chat([user(content=evaluation_prompt)])
            evaluation_text = self._extract_response_text(response)

            # 解析评分（尝试从评估文本中提取）
            scores = self._parse_scores(evaluation_text)

            evaluation_result = {
                "question": question,
                "answer": answer,
                "evaluation": evaluation_text,
                "scores": scores,
                "timestamp": datetime.now().isoformat(),
            }

            # 保存到轮次评估列表
            self.round_evaluations.append(evaluation_result)

            return evaluation_result

        except Exception as e:
            return {
                "question": question,
                "answer": answer,
                "evaluation": f"评估出错: {str(e)}",
                "scores": {"聪明度": 0, "皮实": 0, "勤奋": 0},
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }

    def generate_final_evaluation(
        self, candidate_info: dict, conversation_history: list = None
    ) -> dict:
        """
        基于所有轮次的评估，生成最终综合评估

        Args:
            candidate_info: 候选人信息
            conversation_history: 完整对话历史（可选）

        Returns:
            dict: 最终评估结果
        """
        # 准备每轮评估摘要
        round_summaries = []
        for i, eval_data in enumerate(self.round_evaluations, 1):
            summary = f"""
第{i}轮评估：
- 问题：{eval_data["question"][:100]}...
- 评估：{eval_data["evaluation"][:200]}...
- 评分：聪明度{eval_data["scores"].get("聪明度", "N/A")}/100, 皮实{eval_data["scores"].get("皮实", "N/A")}/100, 勤奋{eval_data["scores"].get("勤奋", "N/A")}/100
"""
            round_summaries.append(summary)

        rounds_text = "\n".join(round_summaries)

        final_prompt = f"""
作为资深HR面试官，请对这次面试进行全面的综合评估和复盘：

【评估标准】
{self.evaluation_criteria}

【候选人信息】
- 岗位: {candidate_info.get("position", "未知")}
- 岗位聪明度要求: {candidate_info.get("intelligence_requirement", "N/A")}/100

【各轮评估摘要】
{rounds_text}

请提供以下内容的综合评估：

## 一、三维能力综合评分

### 1. 聪明度评分：_/100 ⭐⭐⭐⭐⭐
**评分依据**：
- 逻辑思维能力表现
- 问题理解和抓重点能力
- 创新思维和独到见解
- 学习迁移能力

**具体表现**：
（列举面试中的具体案例）

### 2. 皮实评分：_/100 ⭐⭐⭐⭐⭐
**评分依据**：
- 抗压能力展现
- 面对挫折的态度
- 情绪管理能力
- 复原力和韧性

**具体表现**：
（列举面试中的具体案例）

### 3. 勤奋评分：_/100 ⭐⭐⭐⭐⭐
**评分依据**：
- 自驱力和主动性
- 持续投入的证据
- 超越常规的努力
- 成果导向的执行力

**具体表现**：
（列举面试中的具体案例）

## 二、总体评估

### 推荐意见：【推荐/考虑/不推荐】✅

**理由**：
（综合三维评分和岗位要求，给出明确建议）

### 主要优势（Top 3）
1. 
2. 
3. 

### 需要改进的方面（Top 3）
1. 
2. 
3. 

### 风险点识别
- 
- 

## 三、入职建议

### 岗位匹配度：_%
**分析**：

### 预期培养周期：_个月
**培养重点**：

### 薪资定位建议：
**理由**：

### 团队配置建议：
**建议**：

## 四、面试复盘

### 面试亮点
- 
- 

### 待验证事项
- 
- 

### 下次面试建议
如需进一步面试，建议重点考察：
1. 
2. 
3. 

---
**评估结论**：（用一句话总结这位候选人）
"""

        try:
            response = self.model.chat([user(content=final_prompt)])
            final_evaluation_text = self._extract_response_text(response)

            # 计算平均分
            avg_scores = self._calculate_average_scores()

            final_result = {
                "candidate_info": candidate_info,
                "round_count": len(self.round_evaluations),
                "round_evaluations": self.round_evaluations,
                "final_evaluation": final_evaluation_text,
                "average_scores": avg_scores,
                "evaluation_time": datetime.now().isoformat(),
            }

            return final_result

        except Exception as e:
            return {
                "candidate_info": candidate_info,
                "round_count": len(self.round_evaluations),
                "final_evaluation": f"生成最终评估时出错: {str(e)}",
                "average_scores": {"聪明度": 0, "皮实": 0, "勤奋": 0},
                "evaluation_time": datetime.now().isoformat(),
                "error": str(e),
            }

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

    def _parse_scores(self, evaluation_text: str) -> dict:
        """从评估文本中解析分数"""
        scores = {"聪明度": 0, "皮实": 0, "勤奋": 0}

        # 尝试提取分数（简单的正则匹配）
        import re

        for dimension in ["聪明度", "皮实", "勤奋"]:
            # 匹配模式如："聪明度：70/100" 或 "聪明度: 70/100"
            pattern = rf"{dimension}[:：]\s*(\d+(?:\.\d+)?)\s*/\s*100"
            match = re.search(pattern, evaluation_text)
            if match:
                try:
                    scores[dimension] = float(match.group(1))
                except (ValueError, AttributeError):
                    pass

        return scores

    def _calculate_average_scores(self) -> dict:
        """计算所有轮次的平均分"""
        if not self.round_evaluations:
            return {"聪明度": 0, "皮实": 0, "勤奋": 0}

        total_scores = {"聪明度": 0, "皮实": 0, "勤奋": 0}
        valid_counts = {"聪明度": 0, "皮实": 0, "勤奋": 0}

        for eval_data in self.round_evaluations:
            scores = eval_data.get("scores", {})
            for dimension in ["聪明度", "皮实", "勤奋"]:
                score = scores.get(dimension, 0)
                if score > 0:  # 只计算有效分数
                    total_scores[dimension] += score
                    valid_counts[dimension] += 1

        avg_scores = {}
        for dimension in ["聪明度", "皮实", "勤奋"]:
            if valid_counts[dimension] > 0:
                avg_scores[dimension] = round(
                    total_scores[dimension] / valid_counts[dimension], 1
                )
            else:
                avg_scores[dimension] = 0

        return avg_scores

    def reset_evaluations(self):
        """重置评估记录，用于新的面试"""
        self.round_evaluations = []

    def get_evaluation_summary(self) -> str:
        """获取评估摘要（用于显示）"""
        if not self.round_evaluations:
            return "暂无评估记录"

        summary = f"已完成 {len(self.round_evaluations)} 轮评估\n"
        avg_scores = self._calculate_average_scores()
        summary += f"平均分数 - 聪明度: {avg_scores['聪明度']}/100, "
        summary += f"皮实: {avg_scores['皮实']}/100, "
        summary += f"勤奋: {avg_scores['勤奋']}/100"

        return summary
