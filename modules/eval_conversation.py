"""
对话评估模块

功能：
- 清洗面试对话数据（格式化为标准对话）
- 评估面试对话质量
- 生成对话总结和评价
"""

from typing import Dict, List, Optional
import re
from datetime import datetime

from menglong import Model
from menglong.utils.log import print_message

from agents import EvalAgent
from manager.interview_data_manager import InterviewDataManager
from manager.models import ManagerConfig

import json
import hashlib
import os
from pathlib import Path

class ConversationEvaluator:
    """对话评估器 - 清洗和评估已有面试记录"""

    def __init__(self, csv_path: str = None,conversation: str = None):
        """
        初始化对话评估器

        Args:
            csv_path: CSV 数据文件路径
        """
        config = ManagerConfig()
        
        if csv_path:
            self.csv_path = csv_path
            config.data_source = csv_path
            
        if conversation:
            self.conversation = conversation
        
        # 创建 ManagerConfig 对象
        self.manager = InterviewDataManager(config)
        

        # 加载数据
        if csv_path:
            self.manager.load_data(csv_path)


        self.eval_agent = EvalAgent()

        # 用于对话清洗的模型
        self.model = Model()

    def clean_conversation(
        self, raw_dialogue: str, mode: str = "topic"
    ) -> List[Dict[str, str]]:
        """
        清洗对话数据，转换为标准格式

        Args:
            raw_dialogue: 原始对话文本
            mode: 清洗模式
                - "qa_pair": 提取问答对（面试官问题 + 候选人回答）
                - "topic": 按主题划分对话段落

        Returns:
            标准化的对话列表

            QA Pair 模式格式：
            [
                {"role": "interviewer", "content": "问题内容"},
                {"role": "candidate", "content": "回答内容"},
                ...
            ]

            Topic 模式格式：
            [
                {
                    "topic": "主题名称",
                    "dialogue": [
                        {"role": "interviewer", "content": "..."},
                        {"role": "candidate", "content": "..."},
                        ...
                    ]
                },
                ...
            ]
        """
        print_message(f"\n🔧 开始清洗对话数据（模式：{mode}）...")

        if not raw_dialogue or raw_dialogue.strip() == "":
            print_message("⚠️ 对话为空")
            return []

        # 根据模式选择不同的清洗策略
        if mode == "qa_pair":
            return self._clean_as_qa_pairs(raw_dialogue)
        elif mode == "topic":
            return self._clean_by_topics(raw_dialogue)
        else:
            print_message(f"⚠️ 未知模式 '{mode}'，使用默认 qa_pair 模式")
            return self._clean_as_qa_pairs(raw_dialogue)

    def _clean_as_qa_pairs(self, raw_dialogue: str) -> List[Dict[str, str]]:
        """
        清洗成问答对格式

        Args:
            raw_dialogue: 原始对话文本

        Returns:
            问答对列表
        """
        from menglong.ml_model.schema.ml_request import UserMessage as user

        # 构建清洗提示词
        cleaning_prompt = f"""你是一个专业的面试对话数据清洗助手。

请将以下面试对话数据清洗成问答对格式：

原始对话：
{raw_dialogue}

清洗要求：
1. 识别面试官和候选人
2. 去除时间戳（如 00:02:19）
3. 合并同一个人连续说的多句话，形成完整的问题或回答
4. 去除无意义的寒暄、重复内容、语气词
5. 每个问题必须配对一个回答（如果候选人未回答，标记为"未回答"）
6. 保留核心有效信息

输出格式（JSON）：
```json
[
  {{"role": "interviewer", "content": "面试官的完整问题"}},
  {{"role": "candidate", "content": "候选人的完整回答"}},
  {{"role": "interviewer", "content": "下一个问题"}},
  {{"role": "candidate", "content": "对应回答"}},
  ...
]
```

注意：
- 只输出 JSON，不要其他解释
- 问题和回答必须成对出现
- 内容要完整、清晰、有逻辑"""

        try:
            # 调用模型清洗数据（使用正确的消息格式）
            response = self.model.chat([user(content=cleaning_prompt)])

            # 提取 JSON
            import json

            # 尝试从 response 中提取 JSON
            response_text = response if isinstance(response, str) else str(response)

            # 查找 JSON 代码块
            json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接解析整个响应
                json_str = response_text

            # 解析 JSON
            cleaned_dialogue = json.loads(json_str)

            if not isinstance(cleaned_dialogue, list):
                print_message("⚠️ AI清洗结果格式不正确，使用规则解析...")
                return self._fallback_clean(raw_dialogue)

            print_message(f"✓ QA Pair 清洗完成，共 {len(cleaned_dialogue)} 轮对话")
            return cleaned_dialogue

        except Exception as e:
            print_message(f"⚠️ AI清洗失败: {str(e)}，使用规则解析...")
            return self._fallback_clean(raw_dialogue)

    def _clean_by_topics(self, raw_dialogue: str) -> List[Dict]:
        """
        按主题划分对话段落

        Args:
            raw_dialogue: 原始对话文本

        Returns:
            主题分段列表
        """
        from menglong.ml_model.schema.ml_request import UserMessage as user
        from menglong.utils.log import MessageType

        # # 先用规则解析获取基础对话
        # basic_dialogue = self._fallback_clean(raw_dialogue)

        # if not basic_dialogue:
        #     print_message("⚠️ 无法解析对话")
        #     return []

        # # 构建主题划分提示词
        # dialogue_text = "\n".join(
        #     [
        #         f"{'[面试官]' if turn['role'] == 'interviewer' else '[候选人]'}: {turn['content']}"
        #         for turn in basic_dialogue
        #     ]
        # )

        topic_prompt = f"""你是一个面试对话分析专家。

请将以下面试对话按主题划分成多个完整的对话段落：

对话内容：
{raw_dialogue}

任务要求：
1. 识别对话中的不同主题（如：自我介绍、项目经历、技术能力、职业规划等）
2. 将相关的问答划分到同一主题下
3. 每个主题包含完整的问答对话，主题数量控制在10个以内
4. 主题名称要简洁明确（3-10个字）

注意：
- 只输出 JSON，不要其他解释。JSON要完整。[] 和 {{}} 成对出现。```json 开头，``` 结尾。
- 每个主题至少包含一个完整的问答
- 主题划分要合理、清晰

输出格式（JSON）：
```json
[
  {{
    "topic": "主题1名称",
    "dialogue": [
      {{"interviewer": "问题"}},
      {{"candidate": "回答"}},
      ...
    ]
  }},
  {{
    "topic": "主题2名称",
    "dialogue": [...]
  }},
  ...
]
```

"""
        print_message(topic_prompt,msg_type=MessageType.USER)
        try:
            # 调用模型进行主题划分
            response = self.model.chat(
                [user(content=topic_prompt)]
            )
            print_message(response.text)

            import json

            # 提取 JSON
            response_text = response.text

            json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text

            # 解析 JSON
            topics = json.loads(json_str)

            if not isinstance(topics, list):
                print_message("⚠️ 主题划分格式不正确，使用基础对话")
                return [{"topic": "完整对话", "dialogue": raw_dialogue}]

            print_message(f"✓ Topic 清洗完成，共 {len(topics)} 个主题")
            return topics

        except Exception as e:
            print_message(f"⚠️ 主题划分失败: {str(e)}，返回基础对话")
            return [{"topic": "完整对话", "dialogue": raw_dialogue}]

    def _fallback_clean(self, raw_dialogue: str) -> List[Dict[str, str]]:
        """
        备用清洗方法 - 使用简单规则解析

        Args:
            raw_dialogue: 原始对话

        Returns:
            清洗后的对话列表
        """
        print_message("\n使用规则解析方法...")

        cleaned = []

        # 按行分割
        lines = raw_dialogue.split("\n")

        current_speaker = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 匹配说话人模式：Name(timestamp): content
            match = re.match(r"^([^(]+)\([^)]+\):\s*(.+)$", line)

            if match:
                speaker_name = match.group(1).strip()
                content = match.group(2).strip()

                # 判断是面试官还是候选人（简单规则：英文名为面试官）
                if re.match(r"^[A-Za-z\s]+$", speaker_name):
                    role = "interviewer"
                else:
                    role = "candidate"

                # 如果是新的说话人，保存之前的内容
                if current_speaker and current_speaker != role:
                    if current_content:
                        cleaned.append(
                            {
                                "role": current_speaker,
                                "content": " ".join(current_content),
                            }
                        )
                    current_content = []

                current_speaker = role
                current_content.append(content)
            else:
                # 延续上一个说话人的内容
                if current_content:
                    current_content.append(line)

        # 保存最后一个说话人的内容
        if current_speaker and current_content:
            cleaned.append(
                {"role": current_speaker, "content": " ".join(current_content)}
            )

        print_message(f"✓ 规则解析完成，共 {len(cleaned)} 轮对话")
        return cleaned

    def evaluate_conversation(
        self,
        cleaned_dialogue: List[Dict[str, str]],
        candidate_info: Optional[Dict] = None,
        jd: Optional[str] = None,
    ) -> Dict:
        """
        评估清洗后的对话

        Args:
            cleaned_dialogue: 清洗后的对话列表
            candidate_info: 候选人信息
            jd: 岗位描述

        Returns:
            评估结果
        """
        print_message("\n📊 开始评估对话...")

        if not cleaned_dialogue:
            return {
                "error": "对话为空，无法评估",
                "evaluated_at": datetime.now().isoformat(),
            }

        # 使用 EvalAgent 的 evaluate_conversation 方法评估
        try:
            # 准备候选人信息字典
            eval_candidate_info = {
                "name": candidate_info.get("name", "N/A") if candidate_info else "N/A",
                "position": candidate_info.get("position", "N/A")
                if candidate_info
                else "N/A",
                "intelligence_requirement": 75,  # 默认值
                "resume": candidate_info.get("resume", "N/A")[:500]
                if candidate_info
                else "N/A",
            }

            # 调用 EvalAgent 的整场对话评估方法
            evaluation_result = self.eval_agent.evaluate_conversation(
                dialogue=cleaned_dialogue,
                candidate_info=eval_candidate_info,
                jd=jd or "",
            )

            if "error" in evaluation_result:
                print_message(f"❌ 评估失败: {evaluation_result['error']}")
                return evaluation_result

            # 提取关键信息用于兼容性
            avg_scores = evaluation_result.get("average_scores", {})
            final_eval_text = evaluation_result.get("final_evaluation", "")

            # 生成简化的摘要（用于向后兼容）
            summary = self._generate_summary_from_eval(
                cleaned_dialogue, avg_scores, final_eval_text
            )

            result = {
                "dialogue_rounds": len(cleaned_dialogue),
                "qa_pairs_count": evaluation_result.get("qa_pairs_count", 0),
                "round_evaluations": evaluation_result.get("round_evaluations", []),
                "scores": avg_scores,  # 平均分数
                "final_evaluation": final_eval_text,  # 最终评估
                "summary": summary,  # 简化摘要
                "analysis": final_eval_text[:500],  # 评估意见摘录
                "evaluated_at": evaluation_result.get(
                    "evaluation_time", datetime.now().isoformat()
                ),
            }

            print_message("✓ 对话评估完成")
            return result

        except Exception as e:
            print_message(f"❌ 评估失败: {str(e)}")
            import traceback

            traceback.print_exc()
            return {"error": str(e), "evaluated_at": datetime.now().isoformat()}

    def _generate_summary(
        self, dialogue: List[Dict[str, str]], evaluation: Dict
    ) -> str:
        """
        生成对话摘要（旧版本兼容）

        Args:
            dialogue: 清洗后的对话
            evaluation: 评估结果

        Returns:
            摘要文本
        """
        # 统计信息
        interviewer_turns = sum(1 for turn in dialogue if turn["role"] == "interviewer")
        candidate_turns = sum(1 for turn in dialogue if turn["role"] == "candidate")

        # 提取评分
        scores = evaluation.get("scores", {})
        avg_score = sum(scores.values()) / len(scores) if scores else 0

        summary = f"""
对话摘要：
- 总轮次: {len(dialogue)}
- 面试官提问: {interviewer_turns} 次
- 候选人回答: {candidate_turns} 次
- 平均评分: {avg_score:.1f}/100

评估维度：
"""
        for dimension, score in scores.items():
            summary += f"- {dimension}: {score}/100\n"

        summary += f"\n评估意见:\n{evaluation.get('analysis', 'N/A')}"

        return summary

    def _generate_summary_from_eval(
        self,
        dialogue: List[Dict[str, str]],
        scores: Dict[str, float],
        final_evaluation: str,
    ) -> str:
        """
        从评估结果生成对话摘要

        Args:
            dialogue: 清洗后的对话
            scores: 平均分数
            final_evaluation: 最终评估文本

        Returns:
            摘要文本
        """
        # 统计信息
        interviewer_turns = sum(1 for turn in dialogue if turn["role"] == "interviewer")
        candidate_turns = sum(1 for turn in dialogue if turn["role"] == "candidate")

        # 计算平均分
        avg_score = sum(scores.values()) / len(scores) if scores else 0

        summary = f"""
对话摘要：
- 总轮次: {len(dialogue)}
- 面试官提问: {interviewer_turns} 次
- 候选人回答: {candidate_turns} 次
- 平均评分: {avg_score:.1f}/100

评估维度：
"""
        for dimension, score in scores.items():
            summary += f"- {dimension}: {score:.1f}/100\n"

        # 提取评估意见（前500字符）
        summary += f"\n评估意见:\n{final_evaluation[:500]}..."

        return summary

    def evaluate_record_by_id(
        self, record_id: int, round_name: str = "First Round"
    ) -> Dict:
        """
        根据记录 ID 评估面试对话

        Args:
            record_id: 记录 ID
            round_name: 轮次名称（First Round/Second Round/Final Round）

        Returns:
            评估结果
        """
        print_message(f"\n📋 加载记录 ID: {record_id}")

        # 加载记录 - 使用query()方法获取所有记录
        query_result = self.manager.query(limit=1000000)  # 设置足够大的limit

        # 根据 record.id 查找记录
        record = None
        for r in query_result.records:
            if r.id == record_id:
                record = r
                break

        if record is None:
            print_message(f"❌ 记录 ID {record_id} 不存在")
            return {"error": f"记录 ID {record_id} 不存在"}

        # 根据轮次获取对话数据
        # First Round对应conversation字段，其他轮次在metadata中
        if round_name == "First Round":
            raw_dialogue = record.conversation
            print_message(f"Raw Dialogue: {raw_dialogue[:100]}...")  # 打印前100字符
        else:
            # 尝试从metadata中获取其他轮次的对话
            dialogue_field = f"{round_name} Interview Dialogue"
            raw_dialogue = record.metadata.get(dialogue_field, "")

        if not raw_dialogue or raw_dialogue.strip() == "":
            print_message(f"⚠️ 记录中没有 {round_name} 的对话数据")
            return {"error": f"记录中没有 {round_name} 的对话数据"}

        # 清洗对话
        cleaned_dialogue = self.clean_conversation(raw_dialogue, mode="topic")

        # 准备候选人信息
        candidate_info = {
            "name": f"候选人_{record_id}",
            "position": record.position or "N/A",
            "resume": record.resume,
        }

        jd = record.jd

        # 评估对话
        result = self.evaluate_conversation_with_mode(
            cleaned_dialogue, candidate_info, jd, mode="topic"
        )

        print(result)

        # 添加原始信息
        result.update(
            {
                "record_id": record_id,
                "round": round_name,
                "job_title": record.position or "N/A",
                "cleaned_dialogue": cleaned_dialogue,
                "original_dialogue": raw_dialogue,
            }
        )

        return result

    def batch_evaluate(
        self,
        record_ids: Optional[List[int]] = None,
        round_name: str = "First Round",
        max_records: int = 10,
    ) -> List[Dict]:
        """
        批量评估多条记录

        Args:
            record_ids: 记录 ID 列表，None 表示评估所有
            round_name: 轮次名称
            max_records: 最大评估记录数

        Returns:
            评估结果列表
        """
        print_message("\n🔄 批量评估开始...")

        # 获取所有记录
        query_result = self.manager.query(limit=1000000)
        total_records = query_result.total

        if record_ids is None:
            record_ids = list(range(min(max_records, total_records)))
        else:
            record_ids = record_ids[:max_records]

        results = []

        for i, record_id in enumerate(record_ids, 1):
            print_message(f"\n进度: {i}/{len(record_ids)}")

            try:
                result = self.evaluate_record_by_id(record_id, round_name)
                results.append(result)
            except Exception as e:
                print_message(f"❌ 记录 {record_id} 评估失败: {str(e)}")
                results.append({"record_id": record_id, "error": str(e)})

        print_message(
            f"\n✓ 批量评估完成，成功 {sum(1 for r in results if 'error' not in r)}/{len(results)}"
        )

        return results

    def evaluate_all_rounds(self, record_id: int) -> Dict:
        """
        评估一条记录的所有轮次

        Args:
            record_id: 记录 ID

        Returns:
            包含所有轮次评估结果的字典
        """
        print_message(f"\n📋 评估记录 {record_id} 的所有轮次...")

        # 加载记录
        query_result = self.manager.query(limit=1000000)

        # 根据 record.id 查找记录
        record = None
        for r in query_result.records:
            if r.id == record_id:
                record = r
                break

        if record is None:
            print_message(f"❌ 记录 ID {record_id} 不存在")
            return {"error": f"记录 ID {record_id} 不存在"}

        # 定义所有轮次
        round_names = ["First Round", "Second Round", "Final Round"]

        all_results = {
            "record_id": record_id,
            "job_title": record.position or "N/A",
            "candidate_resume": (record.resume[:200] + "...")
            if len(record.resume) > 200
            else record.resume,
            "rounds": {},
            "summary": {},
            "evaluated_at": datetime.now().isoformat(),
        }

        # 评估每个轮次
        valid_rounds = 0
        total_scores = {"聪明度": [], "皮实": [], "勤奋": []}

        for round_name in round_names:
            print_message(f"\n  评估 {round_name}...")

            # 根据轮次获取对话数据
            if round_name == "First Round":
                raw_dialogue = record.conversation
            else:
                dialogue_field = f"{round_name} Interview Dialogue"
                raw_dialogue = record.metadata.get(dialogue_field, "")

            if not raw_dialogue or raw_dialogue.strip() == "":
                print_message(f"    ⚠️ {round_name} 无对话数据，跳过")
                all_results["rounds"][round_name] = {
                    "status": "no_data",
                    "message": "该轮次无对话数据",
                }
                continue

            try:
                # 清洗对话
                cleaned_dialogue = self.clean_conversation(raw_dialogue)

                # 准备候选人信息
                candidate_info = {
                    "name": f"候选人_{record_id}",
                    "position": record.position or "N/A",
                    "resume": record.resume,
                }

                jd = record.jd

                # 评估对话
                result = self.evaluate_conversation(
                    cleaned_dialogue, candidate_info, jd
                )

                if "error" in result:
                    all_results["rounds"][round_name] = {
                        "status": "error",
                        "error": result["error"],
                    }
                else:
                    all_results["rounds"][round_name] = {
                        "status": "success",
                        "dialogue_rounds": result.get("dialogue_rounds", 0),
                        "evaluation": result.get("evaluation", {}),
                        "summary": result.get("summary", ""),
                    }

                    # 收集分数用于总体统计
                    scores = result.get("evaluation", {}).get("scores", {})
                    for dimension, score in scores.items():
                        if dimension in total_scores:
                            total_scores[dimension].append(score)

                    valid_rounds += 1
                    print_message(f"    ✓ {round_name} 评估完成")

            except Exception as e:
                print_message(f"    ❌ {round_name} 评估失败: {str(e)}")
                all_results["rounds"][round_name] = {"status": "error", "error": str(e)}

        # 生成总体摘要
        if valid_rounds > 0:
            avg_scores = {
                dimension: sum(scores) / len(scores) if scores else 0
                for dimension, scores in total_scores.items()
            }
            overall_avg = (
                sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0
            )

            all_results["summary"] = {
                "valid_rounds": valid_rounds,
                "total_rounds": len(round_names),
                "average_scores": avg_scores,
                "overall_average": overall_avg,
                "performance_trend": self._analyze_trend(
                    all_results["rounds"], round_names
                ),
            }

            print_message(
                f"\n✓ 所有轮次评估完成！有效轮次: {valid_rounds}/{len(round_names)}"
            )
            print_message(f"  总体平均分: {overall_avg:.1f}/100")
        else:
            all_results["summary"] = {
                "valid_rounds": 0,
                "message": "所有轮次都无有效数据",
            }
            print_message("\n⚠️ 所有轮次都无有效对话数据")

        return all_results

    def _analyze_trend(self, rounds_results: Dict, round_names: List[str]) -> str:
        """
        分析表现趋势

        Args:
            rounds_results: 各轮次结果
            round_names: 轮次名称列表

        Returns:
            趋势分析文本
        """
        scores_by_round = []

        for round_name in round_names:
            round_result = rounds_results.get(round_name, {})
            if round_result.get("status") == "success":
                evaluation = round_result.get("evaluation", {})
                scores = evaluation.get("scores", {})
                if scores:
                    avg_score = sum(scores.values()) / len(scores)
                    scores_by_round.append(avg_score)

        if len(scores_by_round) < 2:
            return "数据不足，无法分析趋势"

        # 简单趋势分析
        if scores_by_round[-1] > scores_by_round[0] + 5:
            trend = "📈 表现呈上升趋势，候选人逐渐进入状态"
        elif scores_by_round[-1] < scores_by_round[0] - 5:
            trend = "📉 表现呈下降趋势，可能存在体力或压力问题"
        else:
            trend = "➡️ 表现稳定，整体水平保持一致"

        score_details = " → ".join([f"{s:.1f}" for s in scores_by_round])
        return f"{trend}\n   分数变化: {score_details}"

    def export_evaluation_report(
        self, evaluation_result: Dict, output_path: Optional[str] = None
    ) -> str:
        """
        导出评估报告

        Args:
            evaluation_result: 评估结果
            output_path: 输出路径

        Returns:
            输出文件路径
        """
        import json

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            record_id = evaluation_result.get("record_id", "unknown")
            output_path = f"reports/eval_conversation_{record_id}_{timestamp}.json"

        # 确保目录存在
        import os

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=2)

        print_message(f"\n✓ 评估报告已保存: {output_path}")
        return output_path

    def evaluate_conversation_with_mode(
        self,
        raw_dialogue: str,
        candidate_info: Optional[Dict] = None,
        jd: Optional[str] = None,
        mode: str = "topic",
        use_cache: bool = True,
    ) -> Dict:
        """
        根据清洗模式评估对话

        Args:
            raw_dialogue: 原始对话文本
            mode: 清洗模式 ("qa_pair" | "topic")
            candidate_info: 候选人信息
            jd: 岗位描述

        Returns:
            评估结果
        """
        print_message(f"\n📊 开始评估对话（模式：{mode}）...")

        if not raw_dialogue or raw_dialogue.strip() == "":
            return {
                "error": "对话为空，无法评估",
                "evaluated_at": datetime.now().isoformat(),
            }

        try:
            # 清洗对话
            if use_cache and self.exist_cache(raw_dialogue, mode):
                print("缓存存在，直接加载")
                cleaned_data = self.load_cleaned_dialogue(raw_dialogue, mode)
            else:
                print("缓存不在，重新清洗")
                cleaned_data = self.clean_conversation(raw_dialogue, mode=mode)
                self.save_cleaned_dialogue(cleaned_data,raw_dialogue, mode)

            if mode == "qa_pair":
                # QA Pair 模式直接评估
                result = self.evaluate_conversation(cleaned_data, candidate_info, jd)

            elif mode == "topic":
                # Topic 模式调用 EvalAgent 的 evaluate_topics 方法
                result = self.eval_agent.evaluate_topics(
                    topics=cleaned_data,
                    candidate_info=candidate_info,
                    jd=jd or "",
                )
            else:
                return {
                    "error": f"未知模式 '{mode}'",
                    "evaluated_at": datetime.now().isoformat(),
                }

            print_message("✓ 对话评估完成")
            return result

        except Exception as e:
            print_message(f"❌ 评估失败: {str(e)}")
            import traceback

            traceback.print_exc()
            return {"error": str(e), "evaluated_at": datetime.now().isoformat()}

    def exist_cache(self, raw_dialogue: str, mode: str):
        """
        退出缓存

        Args:
            raw_dialogue: 原始对话文本
            mode: 清洗模式
        """

        md5 = hashlib.md5(raw_dialogue.encode("utf-8")).hexdigest()
        cache_key = f"{mode}_{md5}".replace("\n", "").replace("\r", "")
        cache_path = self._get_cache_path(cache_key,mode)

        if cache_path.exists():
            return True
        return False




    def load_cleaned_dialogue(self, raw_dialogue: str, mode: str) -> List[Dict]:
        """
        从缓存中加载清洗后的对话

        Args:
            raw_dialogue: 原始对话文本
            mode: 清洗模式

        Returns:
            清洗后的对话数据
        """

        md5 = hashlib.md5(raw_dialogue.encode("utf-8")).hexdigest()
        cache_key = f"{mode}_{md5}".replace("\n", "").replace("\r", "")
        cache_path = self._get_cache_path(cache_key,mode)

        if not cache_path.exists():
            return []

        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_cache_path(self, cache_key: str,mode:str) -> Path:


        CLEANED_DIR = Path(f"data/cleaned/{mode}")
        os.makedirs(CLEANED_DIR,exist_ok=True)
        return Path(os.path.join(CLEANED_DIR, f"{cache_key}.json"))

    def save_cleaned_dialogue(self, cleaned_data: List[Dict], raw_dialogue: str, mode: str):
        """
        保存清洗后的对话到缓存

        Args:
            cleaned_data: 清洗后的对话数据
            raw_dialogue: 原始对话文本
            mode: 清洗模式
        """

        md5 = hashlib.md5(raw_dialogue.encode("utf-8")).hexdigest()
        cache_key = f"{mode}_{md5}".replace("\n", "").replace("\r", "")
        cache_path = self._get_cache_path(cache_key,mode)

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    def __repr__(self):
        return f"ConversationEvaluator(csv_path='{self.csv_path}')"
