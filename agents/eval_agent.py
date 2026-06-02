"""
评估Agent - 负责根据面试者回复，评估回复与指标的切合度

功能：
- 评估候选人回答与HR指标的匹配度
- 基于聪明度、皮实、勤奋三维标准打分
- 提供每轮评估和最终综合评估
- 识别亮点和风险点
"""

from menglong import Model
from menglong.ml_model.schema.ml_request import UserMessage as user
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import re


class EvalAgent:
    """面试评估Agent - 基于专业评估标准进行候选人评估"""

    def __init__(self, evaluation_criteria_path: str = "test/eval.md"):
        """
        初始化评估Agent

        Args:
            evaluation_criteria_path: 评估标准文档路径
        """
        self.model = Model()
        self.evaluation_criteria = self._load_evaluation_criteria(
            evaluation_criteria_path
        )
        self.round_evaluations = []  # 存储每轮评估结果

    def _load_evaluation_criteria(self, criteria_path: str) -> str:
        """加载评估标准"""
        criteria_file = Path(criteria_path)
        if criteria_file.exists():
            with open(criteria_file, "r", encoding="utf-8") as f:
                return f.read()
        else:
            # 默认评估标准
            return """
## 评估维度

| 维度   | 核心验证标准                                                                              | 关键验证方法                                                                                                                                        | 补充说明                                                 |
| ---- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| 聪明度  | 1.解题逻辑性：开放性问题解法是否成立<br>2.底层理解力：能否快速抓住问题本质<br>3.迁移能力：能否基于底层逻辑举一反三                            | 1.必问开放性问题（如创新方案设计）<br>2.追问逻辑链：为什么这个解法能解决问题？ -> 请解释背后的底层逻辑<br>3. 验证举一反三能力（如延伸提问类似场景）                                                                   | 覆盖四步验证法（问开放题→看解法→追问原因→验逻辑）- 明确对应聪明定义（快速理解→深挖底层→迁移应用） |
| 勤奋度  | 1.时间投入意愿：是否愿为关键任务付出额外时间<br>2.持续行动证据：历史案例证明其长期投入与优化行动                                    | 1.口头确认：工作强度/临时任务承受力<br>2.事实验证： -> 举例说明某任务每日投入时长 -> 说明具体超出常规的行动（如流程优化）<br>3.交叉验证：成果与投入的因果关系                                                            | 包含"口头承诺+历史事实"双重验证- 强调"事实能否验证"闭环（如要求具体数据/成果佐证）        |
| 目标感  | 1．目标坚持性：是否为目标制定计划并持续行动<br>2.抗阻行动：遇困难时是否调整策略而非放弃                                         | 1.追问案例：请举例为达成目标付出持续努力的事实 -> 遇到阻力时的具体调整动作<br>2.验证事实：目标推进的关键证据（如阶段性成果）                                                                              | 聚焦对目标"不懈努力的事实验证"- 强调"持续努力"而非单次行动                     |
| 皮实度  | 1.抗压恢复力：受挫后能否快速调整并行动<br>2.批评转化力：能否从否定中提取改进价值                                            | 1.事实验证：举例说明过去在高压/受挫/被否定场景中的具体案例 -> 被批评/失败后的行动改进证据<br>2.压力测试：模拟高冲突场景观察反应                                                                           | 包含"口头意愿+历史事实"验证- 明确要求"事实能否验证"                        |
| 迎难而上 | 1.挑战接纳度：面对难题/高挑战工作时，心态上能积极正向接住任务而非推诿或回避；<br>2.解决导向行动：遇到难题时主动拆解问题、找方法/找资源，而不是找借口"先交差了事"； | 1.事实验证：举例说明曾主动承担或接住高挑战任务的具体案例 -> 推进过程中遇到的关键难点 -> 为解决难点采取的具体行动与资源调动 -> 最终结果及个人能力提升收获；<br>2.情境追问：设计"高难度任务突然压过来"的情境，追问其当下想法、优先级取舍和应对路径，以验证真实态度与行动倾向。 | 包含面对难题时的"心态与行动选择"，不推诿、不躲避，主动找方法而非糊弄交差，并把挑战作为自我成长路径   |
| 客户第一 | 1.决策优先级：在冲突场景中是否优先客户长期价值<br>2.决策逻辑合理性：选择是否立足客户真实需求                                      | 1.场景化问题测试（如客户利益vs公司KPI冲突）<br>2.双重追问 ->  你的选择是什么？ -> 为什么此选择对客户最有利？<br>3. 验证逻辑是否自洽                                                                      | 严格遵循"选择→原因"验证结构- 强调决策的客户价值立足点                        |
"""

    def evaluate_single_response(
        self,
        question: str,
        answer: str,
        candidate_info: Dict,
        question_intent: str = "",
    ) -> Dict:
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

1. **问题考察维度识别**：
    - 这个问题主要想考察候选人的哪个能力维度（聪明度/勤奋度/目标感/皮实度/迎难而上/客户第一）？
    - 输出内容至少包含一个表格，总结各维度的相关性得分和观测点：
        （如话题不涉及的维度，得分和观测点均填写“未涉及”）
        | 维度 | 话题相关性得分 | 观测点 |
        |---|---|---|
        | 聪明度 | 相关性分数 | 问题考察目的 |
        | 勤奋度 | 相关性分数 | 问题考察目的 |
        | 目标感 | 相关性分数 | 问题考察目的 |
        | 皮实度 | 相关性分数 | 问题考察目的 |
        | 迎难而上 | 相关性分数 | 问题考察目的 |
        | 客户第一 | 相关性分数 | 问题考察目的 |

2. **回答质量分析**：
   - 候选人的回答是否充分展现了该维度的能力？
   - 回答的具体性和真实性如何？
   - 是否有值得关注的亮点或风险点？
   - 如有评分，以百分制表示。

3. **六维评分**（按评估标准打分，只给出问题考察维度相关的分数，无关维度不评分）：
   - 聪明度：_/100（需说明评分依据）
   - 勤奋度：_/100（需说明评分依据）
   - 目标感：_/100（需说明评分依据）
   - 皮实度：_/100（需说明评分依据）
   - 迎难而上：_/100（需说明评分依据）
   - 客户第一：_/100（需说明评分依据）

4. **追问建议**：基于这个回答，建议下一轮如何深入追问？

5. **关键观察点**：有哪些需要在后续面试中重点验证的地方？

请以Mardwon格式结构化的方式输出评估结果。

"""

        try:
            response = self.model.chat([user(content=evaluation_prompt)])
            evaluation_text = self._extract_response_text(response)

            # 解析评分
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
                "scores": {
                    "聪明度": 0,
                    "勤奋度": 0,
                    "目标感": 0,
                    "皮实度": 0,
                    "迎难而上": 0,
                    "客户第一": 0,
                },
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }

    def evaluate_conversation(
        self,
        dialogue: List[Dict[str, str]],
        candidate_info: Dict,
        jd: str = "",
    ) -> Dict:
        """
        评估整场面试对话

        Args:
            dialogue: 对话列表，格式 [{"role": "interviewer/candidate", "content": "..."}]
            candidate_info: 候选人信息
            jd: 岗位描述

        Returns:
            dict: 包含每轮评估和总体评估的结果
        """
        # 重置评估记录
        self.reset_evaluations()

        # 提取问答对
        qa_pairs = self._extract_qa_pairs(dialogue)

        if not qa_pairs:
            return {
                "error": "未能提取到有效的问答对",
                "dialogue_rounds": len(dialogue),
                "evaluation_time": datetime.now().isoformat(),
            }

        # 准备候选人信息（添加JD）
        eval_candidate_info = {
            **candidate_info,
            "jd": jd[:500] if jd else "N/A",
        }

        # 逐个评估每个问答对
        print(f"\n📊 开始逐轮评估，共 {len(qa_pairs)} 个问答对...")

        for i, qa in enumerate(qa_pairs, 1):
            print(f"  评估第 {i}/{len(qa_pairs)} 轮...")

            self.evaluate_single_response(
                question=qa["question"],
                answer=qa["answer"],
                candidate_info=eval_candidate_info,
                question_intent="",  # 这里可以后续增强，自动分析问题意图
            )

        # 生成最终综合评估
        print("\n📝 生成最终综合评估...")
        final_result = self.generate_final_evaluation(
            candidate_info=eval_candidate_info,
            conversation_history=dialogue,
        )

        # 添加额外信息
        final_result.update(
            {
                "dialogue_rounds": len(dialogue),
                "qa_pairs_count": len(qa_pairs),
            }
        )

        return final_result

    def _extract_qa_pairs(self, dialogue: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        从对话中提取问答对

        Args:
            dialogue: 对话列表

        Returns:
            问答对列表，格式 [{"question": "...", "answer": "..."}]
        """
        qa_pairs = []
        current_question = None

        for turn in dialogue:
            role = turn.get("role", "")
            content = turn.get("content", "")

            if not content.strip():
                continue

            if role == "interviewer":
                # 如果有上一个问题还没配对答案，先保存
                if current_question:
                    qa_pairs.append(
                        {
                            "question": current_question,
                            "answer": "（候选人未回答）",
                        }
                    )
                current_question = content
            elif role == "candidate":
                # 候选人回答
                if current_question:
                    qa_pairs.append(
                        {
                            "question": current_question,
                            "answer": content,
                        }
                    )
                    current_question = None
                else:
                    # 候选人主动发言（如自我介绍）
                    qa_pairs.append(
                        {
                            "question": "（候选人主动发言）",
                            "answer": content,
                        }
                    )

        # 处理最后一个问题
        if current_question:
            qa_pairs.append(
                {
                    "question": current_question,
                    "answer": "（候选人未回答）",
                }
            )

        return qa_pairs

    def generate_final_evaluation(
        self, candidate_info: Dict, conversation_history: List[Dict] = None
    ) -> Dict:
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
- 评分：聪明度{eval_data["scores"].get("聪明度", "N/A")}/100, 勤奋度{eval_data["scores"].get("勤奋度", "N/A")}/100, 目标感{eval_data["scores"].get("目标感", "N/A")}/100, 皮实度{eval_data["scores"].get("皮实度", "N/A")}/100, 迎难而上{eval_data["scores"].get("迎难而上", "N/A")}/100, 客户第一{eval_data["scores"].get("客户第一", "N/A")}/100
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

## 一、六维能力综合评分

### 1. 聪明度评分：_/100 ⭐⭐⭐⭐⭐
**评分依据**：
- 逻辑思维能力表现
- 问题理解和抓重点能力
- 创新思维和独到见解
- 学习迁移能力

### 2. 勤奋度评分：_/100 ⭐⭐⭐⭐⭐
**评分依据**：
- 自驱力和主动性
- 持续投入的证据
- 超越常规的努力
- 成果导向的执行力

### 3. 目标感评分：_/100 ⭐⭐⭐⭐⭐
**评分依据**：
- 目标坚持性
- 抗阻行动
- 目标推进的关键证据

### 4. 皮实度评分：_/100 ⭐⭐⭐⭐⭐
**评分依据**：
- 抗压能力展现
- 面对挫折的态度
- 情绪管理能力
- 复原力和韧性

### 5. 迎难而上评分：_/100 ⭐⭐⭐⭐⭐
**评分依据**：
- 挑战接纳度
- 解决导向行动
- 挑战作为成长路径

### 6. 客户第一评分：_/100 ⭐⭐⭐⭐⭐
**评分依据**：
- 决策优先级
- 决策逻辑合理性
- 客户价值立足点

**具体表现**：
（列举面试中的具体案例，涵盖以上六个维度）

## 二、总体评估

### 推荐意见：【推荐/考虑/不推荐】✅
**理由**：

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
                "average_scores": {
                    "聪明度": 0,
                    "勤奋度": 0,
                    "目标感": 0,
                    "皮实度": 0,
                    "迎难而上": 0,
                    "客户第一": 0,
                },
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

    def _parse_scores(self, evaluation_text: str) -> Dict[str, float]:
        """从评估文本中解析分数"""
        scores = {
            "聪明度": 0,
            "勤奋度": 0,
            "目标感": 0,
            "皮实度": 0,
            "迎难而上": 0,
            "客户第一": 0,
        }

        # 尝试提取分数（正则匹配）
        for dimension in [
            "聪明度",
            "勤奋度",
            "目标感",
            "皮实度",
            "迎难而上",
            "客户第一",
        ]:
            # 匹配多种模式：
            # "聪明度评分：82/100"
            # "聪明度：70/100"
            # "聪明度: 70/100"
            patterns = [
                rf"{dimension}评分[:：]\s*(\d+(?:\.\d+)?)\s*/\s*100",
                rf"{dimension}[:：]\s*(\d+(?:\.\d+)?)\s*/\s*100",
                rf"### {dimension}评分[:：]\s*(\d+(?:\.\d+)?)\s*/\s*100",
            ]

            for pattern in patterns:
                match = re.search(pattern, evaluation_text)
                if match:
                    try:
                        scores[dimension] = float(match.group(1))
                        break  # 找到分数后跳出内循环
                    except (ValueError, AttributeError):
                        continue

        return scores

    def _parse_relevance_scores(self, evaluation_text: str) -> Dict[str, str]:
        """从评估文本中解析相关性得分"""
        relevance_scores = {
            "聪明度相关性": "未涉及",
            "勤奋度相关性": "未涉及",
            "目标感相关性": "未涉及",
            "皮实度相关性": "未涉及",
            "迎难而上相关性": "未涉及",
            "客户第一相关性": "未涉及",
        }

        # 尝试提取相关性得分（正则匹配）
        for dimension in [
            "聪明度相关性",
            "勤奋度相关性",
            "目标感相关性",
            "皮实度相关性",
            "迎难而上相关性",
            "客户第一相关性",
        ]:
            # pattern = rf"\|\s*{dimension}\s*\|\s*(\d+\/100|未涉及)\s*\|"
            pattern = rf"(?:\|\s*{dimension}\s*\||{dimension}[:：])\s*(\d+\/(?:100|未涉及)|未涉及)\s*"
            match = re.search(pattern, evaluation_text)
            if match:
                relevance_scores[dimension] = match.group(1).strip()

        return relevance_scores

    # ===== 加权平均分计算功能（已注释） =====
    # 以下代码实现了基于relevance_scores的加权平均分计算
    # 由于需求变更，暂时注释掉，保留原有的简单平均分计算

    # def _parse_relevance_weight(self, relevance_str: str) -> float:
    #     """
    #     将相关性得分字符串转换为权重值
    #
    #     Args:
    #         relevance_str: 相关性得分字符串，如 "80/100", "未涉及", "0/未涉及"
    #
    #     Returns:
    #         float: 权重值 (0.0-1.0)，对于无效或未涉及的返回0.0
    #     """
    #     if not relevance_str or relevance_str in ["未涉及", "0/未涉及"]:
    #         return 0.0
    #
    #     # 匹配 "数字/100" 格式
    #     match = re.match(r"(\d+(?:\.\d+)?)/100", relevance_str)
    #     if match:
    #         try:
    #             score = float(match.group(1))
    #             return score / 100.0  # 转换为0-1之间的权重
    #         except ValueError:
    #             return 0.0
    #
    #     return 0.0
    #
    # def _calculate_weighted_average(
    #     self, dimension: str, topic_results: List[Dict], scores: List[float]
    # ) -> float:
    #     """
    #     计算指定维度的加权平均分
    #
    #     Args:
    #         dimension: 维度名称 ("聪明度", "皮实", "勤奋")
    #         topic_results: 主题评估结果列表
    #         scores: 该维度的有效分数列表
    #
    #     Returns:
    #         float: 加权平均分，保留一位小数
    #     """
    #     if not scores or not topic_results:
    #         return 0.0
    #
    #     # 收集权重和分数
    #     weighted_sum = 0.0
    #     total_weight = 0.0
    #
    #     # 维度名称映射到相关性字段名
    #     relevance_key = f"{dimension}相关性"
    #
    #     score_index = 0
    #     for topic_result in topic_results:
    #         # 获取该主题的分数
    #         topic_scores = topic_result.get("scores", {})
    #         topic_score = topic_scores.get(dimension, 0)
    #
    #         # 只处理有效分数（大于0）
    #         if topic_score > 0 and score_index < len(scores):
    #             # 获取相关性得分并转换为权重
    #             relevance_scores = topic_result.get("relevance_scores", {})
    #             relevance_str = relevance_scores.get(relevance_key, "未涉及")
    #             weight = self._parse_relevance_weight(relevance_str)
    #
    #             # 如果权重为0（未涉及），使用默认权重0.1，避免完全忽略
    #             if weight == 0.0:
    #                 weight = 0.1
    #
    #             weighted_sum += scores[score_index] * weight
    #             total_weight += weight
    #             score_index += 1
    #
    #     # 计算加权平均分
    #     if total_weight > 0:
    #         weighted_avg = weighted_sum / total_weight
    #         return round(weighted_avg, 1)
    #     else:
    #         # 如果没有有效权重，返回简单平均分
    #         return round(sum(scores) / len(scores), 1) if scores else 0.0

    def _calculate_average_scores(self) -> Dict[str, float]:
        """计算所有轮次的平均分"""
        if not self.round_evaluations:
            return {"聪明度": 0, "皮实": 0, "勤奋": 0}

        total_scores = {
            "聪明度": 0,
            "勤奋度": 0,
            "目标感": 0,
            "皮实度": 0,
            "迎难而上": 0,
            "客户第一": 0,
        }
        valid_counts = {
            "聪明度": 0,
            "勤奋度": 0,
            "目标感": 0,
            "皮实度": 0,
            "迎难而上": 0,
            "客户第一": 0,
        }

        for eval_data in self.round_evaluations:
            scores = eval_data.get("scores", {})
            for dimension in [
                "聪明度",
                "勤奋度",
                "目标感",
                "皮实度",
                "迎难而上",
                "客户第一",
            ]:
                score = scores.get(dimension, 0)
                if score > 0:  # 只计算有效分数
                    total_scores[dimension] += score
                    valid_counts[dimension] += 1

        avg_scores = {}
        for dimension in [
            "聪明度",
            "勤奋度",
            "目标感",
            "皮实度",
            "迎难而上",
            "客户第一",
        ]:
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
        summary += f"勤奋度: {avg_scores['勤奋度']}/100, "
        summary += f"目标感: {avg_scores['目标感']}/100, "
        summary += f"皮实度: {avg_scores['皮实度']}/100, "
        summary += f"迎难而上: {avg_scores['迎难而上']}/100, "
        summary += f"客户第一: {avg_scores['客户第一']}/100"

        return summary

    def __repr__(self):
        return f"EvalAgent(evaluations={len(self.round_evaluations)})"

    def evaluate_single_topic(
        self,
        topic_name: str,
        dialogue: List[Dict[str, str]],
        candidate_info: Dict,
        jd: str = "",
    ) -> Dict:
        """
        评估单个主题的对话内容，作为整体进行评估

        Args:
            topic_name: 主题名称
            dialogue: 主题下的对话列表，格式 [{"interviewer": "...", "candidate": "..."}, ...]
            candidate_info: 候选人背景信息
            jd: 岗位描述

        Returns:
            dict: 包含主题评估结果的字典
        """
        if not dialogue:
            return {
                "topic": topic_name,
                "error": "主题对话内容为空",
                "evaluation": "",
                "relevance_scores": {
                    "聪明度相关性": "未涉及",
                    "勤奋度相关性": "未涉及",
                    "目标感相关性": "未涉及",
                    "皮实度相关性": "未涉及",
                    "迎难而上相关性": "未涉及",
                    "客户第一相关性": "未涉及",
                },
                "scores": {
                    "聪明度": 0,
                    "勤奋度": 0,
                    "目标感": 0,
                    "皮实度": 0,
                    "迎难而上": 0,
                    "客户第一": 0,
                },
                "timestamp": datetime.now().isoformat(),
            }

        # 构建主题对话内容
        dialogue_content = self._format_topic_dialogue_for_evaluation(dialogue)

        # 提取面试官问题和候选人回答
        interviewer_questions = []
        candidate_responses = []

        for turn in dialogue:
            if "interviewer" in turn and turn["interviewer"].strip():
                interviewer_questions.append(turn["interviewer"])
            if "candidate" in turn and turn["candidate"].strip():
                candidate_responses.append(turn["candidate"])

        evaluation_prompt = f"""
作为资深HR面试官，请对候选人在【{topic_name}】主题下的整体表现进行专业评估：

【评估标准】
{self.evaluation_criteria}

【候选人背景】
- 岗位: {candidate_info.get("position", "未知")}
- 聪明度要求: {candidate_info.get("intelligence_requirement", "N/A")}/100
{f"- 岗位描述: {jd}" if jd else "暂无岗位描述"}

【主题名称】
{topic_name}

【完整对话内容】
{dialogue_content}

请从以下几个方面对该主题进行整体评估：

## 1. 面试官问题意图分析
请分析面试官在这个主题下提出的问题主要想考察候选人的哪些方面：
- 主要考察维度（聪明度/皮实/勤奋）
- 分析给出各维度的问题统计
    1. 列举具体考察点（如：逻辑思维、抗压能力、学习能力等）
    2. 统计各维度相关问题的数量和所占主题内容量比例
    3. 分析各维度相关问题内容的多样性（广度）和覆盖度（深度）


## 2. 候选人回答契合度分析
评估候选人的回答是否契合面试官的考察意图：
- 是否理解了问题的核心考察点
- 回答内容是否有效展现了相关能力
- 是否提供了具体的案例和证据
- 回答的逻辑性和完整性如何

## 3. 主题相关性评估
通过与维度相关的问答的数量、内容占比、多样性和覆盖度，评估对话在各维度上的的相关性。
- 输出内容至少包含一个表格，总结各维度的相关性得分和观测点：
        （如主题不涉及的维度，得分和观测点均填写“未涉及”）
        | 维度 | 主题相关性得分 | 观测点 |
        |---|---|---|
        | 聪明度相关性 | 相关性分数/100 | 问题具体考察点 |
        | 勤奋度相关性 | 相关性分数/100 | 问题具体考察点 |
        | 目标感相关性 | 相关性分数/100 | 问题具体考察点 |
        | 皮实度相关性 | 相关性分数/100 | 问题具体考察点 |
        | 迎难而上相关性 | 相关性分数/100 | 问题具体考察点 |
        | 客户第一相关性 | 相关性分数/100 | 问题具体考察点 |

## 3. 主题表现评估
基于候选人在该主题下的整体表现，进行三维评分：

 ### 聪明度评分:_/100
**评分依据**
- 对问题的理解和把握程度
- 回答的逻辑性和条理性
- 是否展现了独到的思考和见解
- 学习迁移能力的体现

**具体表现**:
（结合对话内容具体分析）

### 勤奋度评分:_/100
**评分依据**
- 时间投入意愿
- 持续行动证据
- 成果与投入的因果关系

**具体表现**
（结合对话内容具体分析）

### 目标感评分:_/100
**评分依据**
- 目标坚持性
- 抗阻行动
- 目标推进的关键证据

**具体表现**
（结合对话内容具体分析）

### 皮实度评分:_/100
**评分依据**
- 抗压恢复力
- 批评转化力
- 压力测试反应

**具体表现**
（结合对话内容具体分析）

### 迎难而上评分:_/100
**评分依据**
- 挑战接纳度
- 解决导向行动
- 应对路径选择

**具体表现**
（结合对话内容具体分析）

### 客户第一评分:_/100
**评分依据**
- 决策优先级
- 决策逻辑合理性
- 客户价值立足点

**具体表现**
（结合对话内容具体分析）

## 4. 主题总结
- **主要优势**：在该主题下表现出的突出能力
- **改进空间**：需要加强的方面
- **关键洞察**：从该主题对话中获得的重要发现
- **风险提示**：需要注意的潜在问题

请以Mardwon格式结构化的方式输出评估结果,确保评分有具体依据。
"""
        # （如 "聪明度相关性: 80/100" , 如果不涉及 则为 "聪明度相关性: 0/未涉及")
        try:
            response = self.model.chat([user(content=evaluation_prompt)])
            evaluation_text = self._extract_response_text(response)

            # 解析相关性得分
            relevance_scores = self._parse_relevance_scores(evaluation_text)
            # 解析评分
            scores = self._parse_scores(evaluation_text)

            topic_result = {
                "topic": topic_name,
                "dialogue_count": len(dialogue),
                "interviewer_questions_count": len(interviewer_questions),
                "candidate_responses_count": len(candidate_responses),
                "evaluation": evaluation_text,
                "relevance_scores": relevance_scores,
                "scores": scores,
                "timestamp": datetime.now().isoformat(),
            }

            return topic_result

        except Exception as e:
            return {
                "topic": topic_name,
                "dialogue_count": len(dialogue),
                "evaluation": f"主题评估出错: {str(e)}",
                "relevance_scores": {
                    "聪明度相关性": "未涉及",
                    "勤奋度相关性": "未涉及",
                    "目标感相关性": "未涉及",
                    "皮实度相关性": "未涉及",
                    "迎难而上相关性": "未涉及",
                    "客户第一相关性": "未涉及",
                },
                "scores": {
                    "聪明度": 0,
                    "勤奋度": 0,
                    "目标感": 0,
                    "皮实度": 0,
                    "迎难而上": 0,
                    "客户第一": 0,
                },
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }

    def _format_topic_dialogue_for_evaluation(
        self, dialogue: List[Dict[str, str]]
    ) -> str:
        """
        格式化主题对话内容用于评估

        Args:
            dialogue: 对话列表，格式 [{"interviewer": "...", "candidate": "..."}, ...]

        Returns:
            str: 格式化的对话内容
        """
        formatted_lines = []

        for i, turn in enumerate(dialogue, 1):
            if "interviewer" in turn and turn["interviewer"].strip():
                formatted_lines.append(f"面试官-{i}: {turn['interviewer']}")
            if "candidate" in turn and turn["candidate"].strip():
                formatted_lines.append(f"候选人-{i}: {turn['candidate']}")

        return "\n\n".join(formatted_lines)

    def evaluate_topics(
        self,
        topics: List[Dict],
        candidate_info: Dict,
        jd: str = "",
    ) -> Dict:
        """
        评估按主题划分的对话数据

        Args:
            topics: 主题列表，格式：
                [
                    {
                        "topic": "主题名称",
                        "dialogue": [
                            {"interviewer": "问题", "candidate": "回答"},
                            {"interviewer": "问题", "candidate": "回答"},
                            ...
                        ]
                    },
                    ...
                ]
            candidate_info: 候选人信息
            jd: 岗位描述

        Returns:
            dict: 包含每个主题评估结果和总体评估的字典
        """
        print(f"\n📊 开始评估 {len(topics)} 个主题...")

        topic_results = []
        all_scores = {
            "聪明度": [],
            "勤奋度": [],
            "目标感": [],
            "皮实度": [],
            "迎难而上": [],
            "客户第一": [],
        }
        all_relevance_scores = {
            "聪明度相关性": [],
            "勤奋度相关性": [],
            "目标感相关性": [],
            "皮实度相关性": [],
            "迎难而上相关性": [],
            "客户第一相关性": [],
        }

        for i, topic_data in enumerate(topics, 1):
            topic_name = topic_data.get("topic", "未命名主题")
            dialogue = topic_data.get("dialogue", [])

            print(f"  评估主题 {i}/{len(topics)}: {topic_name}")

            # 使用新的主题评估方法
            topic_result = self.evaluate_single_topic(
                topic_name=topic_name,
                dialogue=dialogue,
                candidate_info=candidate_info,
                jd=jd,
            )

            topic_results.append(topic_result)
            # 收集相关性用于计算加权平均分
            relevance_scores_dict = topic_result.get("relevance_scores", {})
            for dimension in [
                "聪明度相关性",
                "勤奋度相关性",
                "目标感相关性",
                "皮实度相关性",
                "迎难而上相关性",
                "客户第一相关性",
            ]:
                relevance_scores = relevance_scores_dict.get(dimension, "未涉及")

                def normalized_relevance(relevance_scores):
                    if not relevance_scores or relevance_scores in [
                        "未涉及",
                        "0/未涉及",
                    ]:
                        return 0.0

                    # 匹配 "数字/100" 格式
                    match = re.match(r"(\d+(?:\.\d+)?)/100", relevance_scores)
                    if match:
                        try:
                            score = float(match.group(1))
                            return score / 100.0  # 转换为0-1之间的权重
                        except ValueError:
                            return 0.0

                    return 0.0

                relevance_scores = normalized_relevance(relevance_scores)
                # if score > 0:  # 只收集有效分数
                all_relevance_scores[dimension].append(relevance_scores)
            # 收集分数用于计算总体平均分
            scores = topic_result.get("scores", {})
            for dimension in [
                "聪明度",
                "勤奋度",
                "目标感",
                "皮实度",
                "迎难而上",
                "客户第一",
            ]:
                score = scores.get(dimension, 0)
                # if score > 0:  # 只收集有效分数
                all_scores[dimension].append(score)

        overall_relevance = {}
        for dimension in [
            "聪明度相关性",
            "勤奋度相关性",
            "目标感相关性",
            "皮实度相关性",
            "迎难而上相关性",
            "客户第一相关性",
        ]:
            if all_relevance_scores[dimension]:
                overall_relevance[dimension] = (
                    round(
                        sum(all_relevance_scores[dimension])
                        / len(all_relevance_scores[dimension]),
                        2,
                    )
                    * 100
                )
            else:
                overall_relevance[dimension] = 0.0

        def calculate_weighted_average(scores, weights):
            if not scores or len(scores) != len(weights):
                return 0.0

            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            total_weight = sum(weights)

            return round(weighted_sum / total_weight, 1) if total_weight > 0 else 0.0

        # 计算所有主题的加权平均分
        overall_scores = {}
        for dimension in [
            "聪明度",
            "勤奋度",
            "目标感",
            "皮实度",
            "迎难而上",
            "客户第一",
        ]:
            if all_scores[dimension]:
                # 使用简单平均分计算
                # overall_scores[dimension] = round(
                #     sum(all_scores[dimension]) / len(all_scores[dimension]), 1
                # )
                overall_scores[dimension] = calculate_weighted_average(
                    all_scores[dimension], all_relevance_scores[f"{dimension}相关性"]
                )
            else:
                overall_scores[dimension] = 0

        final_result = {
            "topic_count": len(topics),
            "topic_results": topic_results,
            "overall_relevance": overall_relevance,
            "overall_scores": overall_scores,
            "evaluation_time": datetime.now().isoformat(),
        }

        print("\n📝 主题评估完成")
        return final_result
