from menglong import Model
from menglong.agents.role_play import RoleAgent
from menglong.ml_model.schema.ml_request import (
    SystemMessage as system,
    UserMessage as user,
    AssistantMessage as assistant,
    ToolMessage as tool,
)
import yaml
import json
import os
import glob
from datetime import datetime
from pathlib import Path
import traceback

from menglong.utils.log import configure, get_logger
from menglong.utils.log import print_json, print_message

from interview_assistant import InterviewAssistant
from eval_assistant import EvalAssistant
import pandas as pd


# 时间
configure(log_file=f"agent_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")


def test():
    print_message("Hello from interview-sim!")
    model = Model(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")
    res = model.chat([user(content="你好")])
    print_message(res)


class InterviewContextManager:
    def __init__(self):
        """初始化面试上下文管理器"""
        # ---
        self.interviewer_context = []  # model chat context for interviewer
        self.candidate_context = []  # model chat context for candidate
        self.hr_context = []  # model chat context for hr
        # ---
        # 对话历史
        self.conversation_history = []  # 对话历史记录
        self.hr_evaluations = []  # HR评估记录,评估每一次对话
        self.result_summary = []  # 记录关键有效满足hr需求的问答
        self.current_round = 0  # 当前对话轮次
        self.interview_topic = None  # 面试主题
        self.cache_context = None

    def add_message(self, role_name, content):
        """添加对话消息"""
        message = {
            "role": role_name,
            "content": content,
            "round": self.current_round,
        }
        self.conversation_history.append(message)

    def add_hr_evaluation(self, evaluation):
        """添加HR评估"""
        self.hr_evaluations.append(evaluation)

    def get_conversation_context(self):
        """获取当前对话上下文"""
        return self.conversation_history

    def get_hr_evaluations(self):
        """获取HR评估记录"""
        return self.hr_evaluations

    def get_result_summary(self):
        """获取关键有效问答总结"""
        return self.result_summary


class InterviewerAgent(RoleAgent):
    def __init__(self, role_info):
        super().__init__(role_config=role_info)


class CandidateAgent(RoleAgent):
    def __init__(self, role_info):
        super().__init__(role_config=role_info)


class RealDataCandidateAgent:
    """基于真实简历数据的候选人Agent"""

    def __init__(self, candidate_data):
        """
        初始化真实数据候选人Agent

        Args:
            candidate_data: 包含简历、JD、岗位信息等的字典
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

岗位对聪明度的要求：{self.intelligence_requirement}/10

请根据你的简历背景，以第一人称回答面试官的问题。要求：
1. 回答要符合简历中的经历和背景
2. 体现出适合该岗位的能力和特质
3. 回答要真实可信，不要夸大
4. 语气要自然、诚恳，体现求职者的谨慎和积极
5. 如果问题涉及简历中没有的经历，要诚实说明并展示学习意愿
6. 回答长度适中，既要详细又不要过于冗长

现在请准备回答面试官的问题。
"""

    def answer_question(self, question):
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
            answer = response.text  # self._extract_response_text(response)

            # 清理格式，确保回答自然
            answer = self._clean_answer(answer)

            return answer

        except Exception as e:
            return f"抱歉，我需要一点时间思考这个问题。能否请您再详细说明一下？"

    def _extract_response_text(self, response):
        """提取响应文本"""
        if hasattr(response, "content"):
            return response.content
        elif isinstance(response, dict) and "content" in response:
            return response["content"]
        elif isinstance(response, str):
            return response
        else:
            return str(response)

    def _clean_answer(self, answer):
        """清理回答格式"""
        # 移除可能的角色标识
        if answer.startswith(("候选人：", "求职者：", "我：")):
            answer = answer.split("：", 1)[1].strip()
        elif answer.startswith(("候选人:", "求职者:", "我:")):
            answer = answer.split(":", 1)[1].strip()

        # 移除多余的引号
        answer = answer.strip("\"'")

        return answer.strip()

    def get_candidate_info_summary(self):
        """获取候选人信息摘要"""
        return {
            "position": self.position,
            "intelligence_requirement": self.intelligence_requirement,
            "resume_preview": (
                self.resume[:200] + "..." if len(self.resume) > 200 else self.resume
            ),
            "jd_preview": self.jd[:200] + "..." if len(self.jd) > 200 else self.jd,
        }


class HRAgent(RoleAgent):
    def __init__(self, role_info):
        super().__init__(role_config=role_info)

    def opening_statement(self, topic=None):
        """HR开场白"""
        if topic:
            return f"[HR]:欢迎参加我们公司的面试！今天我们会通过提问交流的方式来了解您的背景和能力，尤其对{topic}岗位的适配情况。接下来交由我们的岗位面试官与您交流。请放轻松，展示您最好的一面。"
        return "[HR]:欢迎参加我们公司的面试！今天我们会通过一些问题来了解您的背景和能力。接下来交由我们的岗位面试官与您交流。请放轻松，展示您最好的一面。"

    def evaluate_response(self, question, answer):
        """评估候选人回答"""

        prompt = f"面试官的问题 : {question}\n\n候选人的回答 : {answer}\n\n请根据问题和回答对候选人进行评估是否符合冰山模型中的要求,如果符合要求，记录问题和关键回答和评估结果，并输出[OK]"

        res = self.chat(prompt)

        return res

    def should_end_interview(self, conversation_history):
        """判断是否应该结束面试"""
        # 简单的结束条件：超过3轮对话
        dialogue_count = len(
            [
                msg
                for msg in conversation_history
                if msg["role"] in ["岗位面试官", "候选人"]
            ]
        )
        return dialogue_count >= 6  # 3轮问答

    def evaluate_candidate(self, candidate):
        """最终评估候选人"""
        evaluations = self.collected_info.get("evaluations", [])
        if not evaluations:
            return "需要更多信息进行评估"

        avg_score = sum(eval_item["score"] for eval_item in evaluations) / len(
            evaluations
        )
        if avg_score >= 4:
            return "推荐录用"
        elif avg_score >= 3:
            return "可以考虑"
        else:
            return "不推荐"


class InterviewConversation:
    """Interview conversation"""

    def __init__(
        self, interviewer: InterviewerAgent, candidate: CandidateAgent, hr: HRAgent
    ):
        self.interviewer = interviewer
        self.candidate = candidate
        self.hr = hr
        self.context_manager = InterviewContextManager()
        self.max_rounds = 5  # 最大轮次

    def chat(self, topic=None):
        """三人面试对话流程"""
        if topic:
            self.context_manager.interview_topic = topic

        print_message("=== 面试开始 ===")

        # HR开场白
        hr_opening = self.hr.opening_statement(topic)
        print_message(f"{hr_opening}", title=self.hr.id, use_panel=True)
        self.context_manager.add_message(self.hr.id, hr_opening)

        for round_num in range(1, self.max_rounds + 1):
            self.context_manager.current_round = round_num
            print_message(f"\n--- 第 {round_num} 轮对话 ---")

            # 1. 面试官提问
            if round_num == 1:
                # 第一轮，面试官主动提问
                # self.context_manager.interviewer_context.append(
                #     user(content=hr_opening)
                # )
                interviewer_question = self.interviewer.chat(hr_opening)
                # self.context_manager.interviewer_context.append(
                #     assistant(content=interviewer_question)
                # )
                print_message(
                    f"{interviewer_question}", title=self.interviewer.id, use_panel=True
                )
            else:
                # 后续轮次，基于之前的对话继续提问
                if self.context_manager.cache_context is not None:
                    interviewer_question = self.interviewer.chat(
                        self.context_manager.cache_context
                    )
                    print_message(
                        f"{interviewer_question}",
                        title=self.interviewer.id,
                        use_panel=True,
                    )
                    self.context_manager.cache_context = None
                else:
                    raise ValueError("候选人回答必须是字符串类型, 且长度大于1")

            self.context_manager.add_message(self.interviewer.id, interviewer_question)

            # 2. 候选人回答
            # self.context_manager.candidate_context.append(
            #     user(content=interviewer_question)
            # )
            candidate_answer = self.candidate.chat(interviewer_question)
            # self.context_manager.candidate_context.append(
            #     assistant(content=candidate_answer)
            # )
            print_message(
                f"{candidate_answer}",
                title=self.candidate.id,
                use_panel=True,
            )
            self.context_manager.cache_context = candidate_answer
            self.context_manager.add_message(self.candidate.id, candidate_answer)

            # 3. HR评估
            hr_evaluation = self.hr.evaluate_response(
                interviewer_question, candidate_answer
            )
            if "[OK]" in hr_evaluation:
                self.context_manager.result_summary.append(hr_evaluation)
            print_message(f"{hr_evaluation}", title=self.hr.id, use_panel=True)
            self.context_manager.add_hr_evaluation(hr_evaluation)

            # 检查是否结束面试
            if self.hr.should_end_interview(
                self.context_manager.get_conversation_context()
            ):
                print_message(f"\n{self.hr.id}: 面试结束，感谢您的参与！")
                break

        print_message("\n=== 面试结束 ===")
        return self.context_manager.get_conversation_context()


def interview_scene():
    """模拟面试流程"""
    print_message("📋 面试模拟功能")
    print_message("请选择面试模式:")

    try:
        mode_choice = input(
            """
1. 标准模拟面试 (三方角色扮演)
2. 真实数据面试 (候选人使用真实简历与HR助手对话)
3. 返回主菜单

请输入选择 (1-3): """
        )

        if mode_choice == "1":
            standard_interview_simulation()
        elif mode_choice == "2":
            real_data_interview_simulation()
        elif mode_choice == "3":
            return
        else:
            print_message("无效选择")

    except Exception as e:
        print_message(f"❌ 面试模拟失败: {e}")
        print_message(traceback.format_exc())


def standard_interview_simulation():
    """标准三方角色扮演面试模拟"""
    # 读取 prompts 目录 下的 yaml配置文件
    dir_path = Path("prompts")
    role_infos = {}

    # 检查prompts目录是否存在
    if dir_path.exists():
        files = [f.name for f in dir_path.iterdir() if f.is_file()]
        print(f"文件列表: {files}")
        for file in files:
            if file.endswith(".yaml"):
                file_path = dir_path / file
                with open(file_path, "r") as f:
                    role_infos[file[:-5]] = yaml.safe_load(f)

        print_json(role_infos, title="角色信息")
    else:
        print_message("prompts目录不存在，使用默认配置")
        role_infos = {}

    # 创建三个角色
    interviewer = InterviewerAgent(role_info=role_infos.get("interviewer"))
    candidate = CandidateAgent(role_info=role_infos.get("candidate"))
    hr = HRAgent(role_info=role_infos.get("hr"))

    # 打印角色信息
    # res = interviewer.chat("请介绍一下自己")
    # rich_print(res)
    # res = candidate.chat("请介绍一下自己")
    # rich_print(res)
    # res = hr.chat("请介绍一下自己")
    # rich_print(res)

    # 创建面试对话
    interview = InterviewConversation(interviewer, candidate, hr)

    # # 开始面试
    conversation_result = interview.chat()
    # 打印对话结果
    print_message(conversation_result, title="对话结果", use_panel=True)

    # 打印最终结果
    print_message("\n=== 面试总结 ===")
    hr_evaluations = interview.context_manager.get_hr_evaluations()
    for i, eval_data in enumerate(hr_evaluations, 1):
        print_message(f"hr的评估[{i}]: {eval_data}")

    result_summary = interview.context_manager.get_result_summary()
    # 保存结果到文件
    with open(f"interview_result-{interview.candidate.id}.txt", "w") as f:
        f.write("\n=== 关键有效问答总结 ===\n")
        for i, summary in enumerate(result_summary, 1):
            f.write(f"总结[{i}]: {summary}\n")

    print("\n=== 面试总结结束 ===")


def real_data_interview_simulation():
    """真实数据面试模拟 - 候选人使用真实简历与HR助手对话"""
    try:
        print_message("🤖 真实数据面试模拟")
        print_message("=" * 50)

        # 初始化面试助手和评估助手
        assistant = InterviewAssistant()
        eval_assistant = EvalAssistant()  # 新增评估助手

        # 加载并显示候选人数据
        df = pd.read_csv("interview_data.csv")
        print_message("\n📋 可用候选人列表：")

        valid_candidates = []
        for i, row in df.iterrows():
            # 检查必要字段是否有效
            if not pd.isna(row.get("候选人脱敏简历")) and not pd.isna(
                row.get("岗位脱敏jd")
            ):
                position = row.get("岗位名称", "未知岗位")
                intelligence = row.get("该岗位要求的聪明度（满分十分）", "N/A")

                if pd.isna(position):
                    position = "未知岗位"
                if pd.isna(intelligence):
                    intelligence = "N/A"

                valid_candidates.append(i)
                print_message(f"{i}: {position} (要求聪明度: {intelligence})")

                # 只显示前15个
                if len(valid_candidates) >= 15:
                    break

        if not valid_candidates:
            print_message("❌ 没有找到有效的候选人数据")
            return

        # 用户选择候选人
        try:
            record_id = int(input(f"\n请选择候选人序号: "))
            if record_id not in valid_candidates:
                print_message("❌ 无效的序号")
                return
        except ValueError:
            print_message("❌ 请输入有效的数字")
            return

        # 加载选定的候选人数据
        candidate_row = df.iloc[record_id]
        candidate_data = {
            "resume": str(candidate_row.get("候选人脱敏简历", "")),
            "jd": str(candidate_row.get("岗位脱敏jd", "")),
            "position": str(candidate_row.get("岗位名称", "未知岗位")),
            "intelligence_requirement": candidate_row.get(
                "该岗位要求的聪明度（满分十分）", 0
            ),
        }

        print_message(f"\n✅ 已选择候选人:")
        print_message(f"岗位: {candidate_data['position']}")
        print_message(f"聪明度要求: {candidate_data['intelligence_requirement']}")
        print_message(f"简历预览: {candidate_data['resume'][:100]}...")
        print_message(f"JD预览: {candidate_data['jd'][:100]}...")

        # 生成面试问题
        print_message("\n🤖 正在为此候选人生成针对性面试问题...")
        questions_result = assistant.generate_interview_questions(
            resume=candidate_data["resume"],
            jd=candidate_data["jd"],
            focus_areas=["聪明度", "皮实", "勤奋"],
        )

        if "error" in questions_result:
            print_message(f"❌ 生成面试问题失败: {questions_result['error']}")
            return

        # 显示生成的面试方案
        print_message("\n📋 生成的面试方案:")
        print_message("=" * 60)
        questions_text = questions_result["generated_questions"]

        # 提取开场问题和核心问题（简化显示）
        preview = (
            questions_text[:800] + "..."
            if len(questions_text) > 800
            else questions_text
        )
        print_message(preview)

        # 询问是否开始模拟面试
        start_interview = input("\n🎯 是否开始模拟面试对话? (y/n): ").lower()
        if start_interview != "y":
            print_message("面试准备完成，可随时开始")
            return

        # 选择面试模式
        print_message("\n🎭 请选择面试模式:")
        print_message("1. 手动模式 - 您亲自回答面试问题")
        print_message("2. Agent模式 - AI候选人根据简历数据自动回答")

        mode_choice = input("请选择模式 (1-2): ").strip()

        if mode_choice == "2":
            # Agent模式：创建基于真实数据的候选人Agent
            print_message("\n🤖 正在创建AI候选人...")
            candidate_agent = RealDataCandidateAgent(candidate_data)
            agent_mode = True

            # 显示AI候选人信息摘要
            info_summary = candidate_agent.get_candidate_info_summary()
            print_message("✅ AI候选人已就绪，基于以下简历背景:")
            print_message(f"   岗位: {info_summary['position']}")
            print_message(
                f"   聪明度要求: {info_summary['intelligence_requirement']}/100"
            )
            print_message(f"   简历预览: {info_summary['resume_preview']}")
            print_message("\n🎭 AI候选人将根据简历背景智能回答面试问题")
        else:
            # 手动模式（默认）
            agent_mode = False
            print_message("✅ 手动模式已启用，您将亲自回答面试问题")

        # 开始交互式面试
        print_message("\n" + "=" * 60)
        print_message("🎤 面试开始!")
        print_message("=" * 60)
        print_message("提示: 输入 'quit' 结束面试")

        conversation_history = []
        round_count = 0
        max_rounds = 8  # 最多8轮对话

        # HR开场白
        hr_opening = f"欢迎来到面试！我是HR，今天我们将针对{candidate_data['position']}岗位进行面试。请放松心情，展示您最好的一面。"
        print_message(f"\n[HR助手]: {hr_opening}")
        conversation_history.append({"role": "HR", "content": hr_opening})

        while round_count < max_rounds:
            round_count += 1
            print_message(f"\n--- 第 {round_count} 轮对话 ---")

            # HR提问 (基于生成的问题和对话历史)
            if round_count == 1:
                # 第一轮，使用开场问题
                hr_question = (
                    "首先，请您简单介绍一下自己，包括您的教育背景和主要工作经历。"
                )
            else:
                # 后续轮次，基于面试方案和对话历史生成问题
                context_for_hr = f"""
基于以下面试方案和对话历史，请作为HR面试官提出下一个问题：

面试方案摘要:
{questions_text[:1000]}

当前对话历史:
{format_conversation_history(conversation_history[-4:])}  # 只使用最近4轮对话

请提出一个针对性的问题，重点评估候选人的聪明度、皮实或勤奋程度。问题要具体、有深度。
"""
                try:
                    model = Model()
                    response = model.chat([user(content=context_for_hr)])
                    hr_question = assistant._extract_response_text(response)

                    # 清理HR问题格式
                    if hr_question.startswith("[HR"):
                        hr_question = hr_question.split(":", 1)[1].strip()
                    elif hr_question.startswith("HR:"):
                        hr_question = hr_question.split(":", 1)[1].strip()

                except Exception as e:
                    print_message(f"⚠️ 生成问题时出错: {e}")
                    # 使用预设问题
                    default_questions = [
                        "能否详细说说您在处理困难项目时的具体经历？",
                        "假如遇到工作压力很大的情况，您通常如何应对？",
                        "您认为自己最大的优势是什么？请举个具体例子。",
                        "描述一次您主动学习新技能的经历。",
                    ]
                    hr_question = default_questions[
                        min(round_count - 2, len(default_questions) - 1)
                    ]

            print_message(f"[HR助手]: {hr_question}")
            conversation_history.append({"role": "HR", "content": hr_question})

            # 候选人回答 - 根据模式选择
            if agent_mode:
                # Agent模式：AI候选人自动回答
                print_message("\n[AI候选人思考中...]")
                candidate_answer = candidate_agent.answer_question(hr_question)
                print_message(f"[AI候选人]: {candidate_answer}")
            else:
                # 手动模式：用户输入回答
                print_message("\n[您的回答]:")
                candidate_answer = input(">> ").strip()

                if candidate_answer.lower() == "quit":
                    print_message("面试结束，感谢您的参与！")
                    break

                if not candidate_answer:
                    print_message("⚠️ 请输入您的回答")
                    continue

            conversation_history.append({"role": "候选人", "content": candidate_answer})

            # HR评估 - 使用专业评估助手
            print_message("\n[HR评估中...]")

            # 使用EvalAssistant进行评估
            eval_result = eval_assistant.evaluate_single_response(
                question=hr_question,
                answer=candidate_answer,
                candidate_info=candidate_data,
                question_intent="",  # 可以根据面试方案提取问题意图
            )

            evaluation = eval_result.get("evaluation", "评估失败")
            print_message(f"\n[HR评估]: {evaluation}")

            # 保存评估到对话历史
            conversation_history.append({"role": "HR评估", "content": evaluation})

        # 面试总结
        print_message("\n" + "=" * 60)
        print_message("📋 面试总结")
        print_message("=" * 60)

        # 保存面试记录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        interview_record = {
            "candidate_info": candidate_data,
            "interview_questions_plan": questions_result,
            "conversation_history": conversation_history,
            "interview_time": timestamp,
            "total_rounds": round_count,
            "interview_mode": "agent" if agent_mode else "manual",
            "mode_description": "AI候选人自动回答" if agent_mode else "用户手动回答",
        }

        record_file = f"real_data_interview_{timestamp}.json"
        try:
            import json

            with open(record_file, "w", encoding="utf-8") as f:
                json.dump(interview_record, f, ensure_ascii=False, indent=2)
            print_message(f"📋 面试记录已保存到: {record_file}")
        except Exception as e:
            print_message(f"⚠️ 保存面试记录失败: {e}")

        # 生成最终评估
        print_message("\n🎯 正在生成最终综合评估...")

        try:
            # 使用EvalAssistant生成最终评估
            final_result = eval_assistant.generate_final_evaluation(
                candidate_info=candidate_data, conversation_history=conversation_history
            )

            final_evaluation = final_result.get("final_evaluation", "评估失败")
            avg_scores = final_result.get("average_scores", {})

            print_message("\n" + "=" * 80)
            print_message("🎯 最终综合评估与复盘")
            print_message("=" * 80)
            print_message(final_evaluation)

            print_message("\n📊 各轮评估平均分:")
            print_message(f"聪明度: {avg_scores.get('聪明度', 0)}/100")
            print_message(f"皮实: {avg_scores.get('皮实', 0)}/100")
            print_message(f"勤奋: {avg_scores.get('勤奋', 0)}/100")

            # 更新面试记录，包含评估详情
            interview_record["evaluation_details"] = {
                "round_evaluations": eval_assistant.round_evaluations,
                "final_evaluation": final_result,
                "average_scores": avg_scores,
            }

        except Exception as e:
            print_message(f"⚠️ 生成最终评估时出错: {e}")
            print_message("感谢参与本次面试！")

    except ImportError as e:
        print_message(f"❌ 导入模块失败: {e}")
        print_message("请确认 interview_assistant.py 文件存在")
    except FileNotFoundError:
        print_message("❌ 未找到 interview_data.csv 文件")
    except Exception as e:
        print_message(f"❌ 真实数据面试模拟失败: {e}")
        print_message(traceback.format_exc())


def format_conversation_history(history):
    """格式化对话历史"""
    formatted = ""
    for msg in history:
        formatted += f"{msg['role']}: {msg['content']}\n\n"
    return formatted
    # # 读取 prompts 目录 下的 yaml配置文件
    # dir_path = Path("prompts")
    # role_infos = {}

    # # 检查prompts目录是否存在
    # if dir_path.exists():
    #     files = [f.name for f in dir_path.iterdir() if f.is_file()]
    #     print(f"文件列表: {files}")
    #     for file in files:
    #         if file.endswith(".yaml"):
    #             file_path = dir_path / file
    #             with open(file_path, "r") as f:
    #                 role_infos[file[:-5]] = yaml.safe_load(f)

    #     print_json(role_infos, title="角色信息")
    # else:
    #     print_message("prompts目录不存在，使用默认配置")
    #     role_infos = {}

    # # 创建三个角色
    # interviewer = InterviewerAgent(role_info=role_infos.get("interviewer"))
    # candidate = CandidateAgent(role_info=role_infos.get("candidate"))
    # hr = HRAgent(role_info=role_infos.get("hr"))

    # # 打印角色信息
    # # res = interviewer.chat("请介绍一下自己")
    # # rich_print(res)
    # # res = candidate.chat("请介绍一下自己")
    # # rich_print(res)
    # # res = hr.chat("请介绍一下自己")
    # # rich_print(res)

    # # 创建面试对话
    # interview = InterviewConversation(interviewer, candidate, hr)

    # # # 开始面试
    # conversation_result = interview.chat()
    # # 打印对话结果
    # print_message(conversation_result, title="对话结果", use_panel=True)

    # # 打印最终结果
    # print_message("\n=== 面试总结 ===")
    # hr_evaluations = interview.context_manager.get_hr_evaluations()
    # for i, eval_data in enumerate(hr_evaluations, 1):
    #     print_message(f"hr的评估[{i}]: {eval_data}")

    # result_summary = interview.context_manager.get_result_summary()
    # # 保存结果到文件
    # with open(f"interview_result-{interview.candidate.id}.txt", "w") as f:
    #     f.write("\n=== 关键有效问答总结 ===\n")
    #     for i, summary in enumerate(result_summary, 1):
    #         f.write(f"总结[{i}]: {summary}\n")

    # print("\n=== 面试总结结束 ===")


def extract_interview_experience():
    """提取面试经验"""
    print_message("🧠 开始提取面试经验...")

    try:
        from interview_processor import InterviewDataProcessor

        # 初始化数据处理器
        processor = InterviewDataProcessor("interview_data.csv")

        # 显示checkpoint状态
        print_message("\n📁 Checkpoint状态:")
        experience_files = processor.list_experience_files()
        print_message(f"📁 现有经验文件: {len(experience_files)} 个")
        for exp_file in experience_files[-3:]:  # 显示最新3个
            print_message(
                f"  - {exp_file['filename']}: record_{exp_file['record_id']}, {exp_file['timestamp'][:10]}"
            )

        # 获取记录选择和总结模式（仅在交互式环境中）
        import sys

        selected_indices = None
        summary_mode = "incremental"  # 默认增量模式

        if sys.stdin.isatty():
            try:
                # 显示数据预览
                total_valid = processor.display_data_preview()

                if total_valid == 0:
                    print_message("❌ 没有有效数据可供选择")
                    return None

                # 获取用户选择
                selection_input = input(
                    f"\n📝 请输入要分析的记录序号 (默认随机选择5条): "
                ).strip()

                if selection_input:
                    selected_indices = processor._parse_indices_input(
                        selection_input, total_valid
                    )
                    if not selected_indices:
                        print_message("❌ 无效的选择，使用默认设置")
                        selected_indices = None

                # 选择总结模式
                print_message("\n📋 选择总结模式:")
                print_message("1. 增量更新 (incremental) - 默认，新经验与已有经验合并")
                print_message("2. 温故知新 (review) - 包含历史经验的重新总结")
                print_message("3. 全量重新总结 (full_refresh) - 完全重新分析所有经验")

                mode_input = input("请选择模式 (1-3, 默认1): ").strip()
                mode_map = {"1": "incremental", "2": "review", "3": "full_refresh"}
                summary_mode = mode_map.get(mode_input, "incremental")

                print_message(f"✅ 已选择模式: {summary_mode}")

            except (EOFError, ValueError) as e:
                print_message(f"使用默认设置: 随机选择5条记录, 模式: {summary_mode}")

        # 提取经验
        experience_data = processor.extract_interview_experience(
            selected_indices=selected_indices,
            resume_from_checkpoint=False,  # 新设计不需要checkpoint恢复
            summary_mode=summary_mode,
        )

        if "error" in experience_data:
            print_message(f"❌ 提取失败: {experience_data['error']}")
            return None

        # 显示结果摘要
        print_message("\n📊 经验提取结果摘要:")
        print_message(f"总记录数: {experience_data['total_records']}")
        print_message(f"有效记录数: {experience_data['valid_records']}")
        print_message(f"采样记录数: {experience_data['sampled_records']}")
        print_message(f"成功提取: {experience_data['successful_extractions']}")
        print_message(f"总结模式: {experience_data['summary_mode']}")

        # 显示token统计
        if "token_stats" in experience_data:
            token_stats = experience_data["token_stats"]
            print_message("\n💰 Token使用统计:")
            print_message(f"输入tokens: {token_stats['total_input_tokens']:,}")
            print_message(f"输出tokens: {token_stats['total_output_tokens']:,}")
            print_message(
                f"总tokens: {token_stats['total_input_tokens'] + token_stats['total_output_tokens']:,}"
            )
            print_message(f"API调用次数: {token_stats['api_calls']}")
            print_message(f"预估总成本: ${token_stats['total_cost']:.4f}")

        # 显示整合后的经验（截取前500字符）
        if "integrated_experience" in experience_data:
            integrated_exp = experience_data["integrated_experience"]
            preview = (
                integrated_exp[:500] + "..."
                if len(integrated_exp) > 500
                else integrated_exp
            )
            print_message(f"\n🎯 整合经验预览:\n{preview}")

        # 保存报告
        report_path = processor.save_experience_report(experience_data)

        if sys.stdin.isatty():
            try:
                view_detail = input("\n👀 是否查看详细经验内容? (y/n): ").lower()
                if view_detail == "y" and "integrated_experience" in experience_data:
                    print_message("\n📖 完整经验内容:")
                    print_message(experience_data["integrated_experience"])
            except EOFError:
                print_message("\n💡 在非交互式环境中运行，跳过详细查看")

        return experience_data

    except ImportError as e:
        print_message(f"❌ 导入模块失败: {e}")
        print_message("详细错误信息:")
        print_message(traceback.format_exc())
    except Exception as e:
        print_message(f"❌ 提取经验失败: {e}")
        print_message("详细错误信息:")
        print_message(traceback.format_exc())
        return None


def analyze_interview_data():
    """分析面试数据"""
    print_message("🔍 开始分析面试数据...")

    try:
        from interview_processor import InterviewDataProcessor

        # 初始化数据处理器
        processor = InterviewDataProcessor("interview_data.csv")

        # 获取基本信息
        basic_info = processor.get_basic_info()
        print("\n📊 数据基本信息:")
        print_json(basic_info)

        # 分析岗位分布
        position_analysis = processor.analyze_positions()
        print_message("\n📈 岗位分析结果:")
        print_message(position_analysis)

        # 分析面试结果
        interview_results = processor.analyze_interview_results()
        print_message("\n📈 面试结果分析:")
        print_json(interview_results)

        # 搜索特定候选人（仅在交互式环境中）
        import sys

        if sys.stdin.isatty():  # 检查是否在交互式终端中
            try:
                search_keyword = input(
                    "\n🔍 请输入搜索关键词 (例如：市场、销售、技术): "
                )
                if search_keyword:
                    candidates = processor.search_candidates(search_keyword)
                    print_message(f"\n找到 {len(candidates)} 个相关候选人")
                    if len(candidates) > 0:
                        print_message("前3个候选人信息:")
                        for i in range(min(3, len(candidates))):
                            candidate_info = processor.extract_candidate_info(
                                candidates.index[i]
                            )
                            print_message(f"\n候选人 {i + 1}:")
                            print_message(
                                f"岗位: {candidate_info['raw_data'].get('岗位名称', '未知')}"
                            )
                            if candidate_info["resume_info"]:
                                print_message(
                                    f"简历信息: {candidate_info['resume_info']}"
                                )
            except EOFError:
                print_message("\n⚠️ 检测到非交互式环境，跳过搜索功能")
        else:
            print_message("\n💡 在非交互式环境中运行，跳过搜索功能")

        # 生成可视化图表（仅在交互式环境中）
        if sys.stdin.isatty():
            try:
                generate_viz = input("\n📊 是否生成可视化图表? (y/n): ").lower()
                if generate_viz == "y":
                    processor.plot_position_distribution()

                # 导出报告
                export_report = input("\n📋 是否导出分析报告? (y/n): ").lower()
                if export_report == "y":
                    report_path = processor.export_analysis_report()
                    print(f"✅ 报告已导出到: {report_path}")
            except EOFError:
                print_message("\n⚠️ 检测到非交互式环境，跳过可视化和导出功能")
        else:
            print_message("\n💡 在非交互式环境中运行，跳过可视化和导出功能")

        return processor

    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("详细错误信息:")
        print(traceback.format_exc())
        print("请先运行: uv add pandas matplotlib seaborn")
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print("详细错误信息:")
        print(traceback.format_exc())
        print("请确认 interview_data.csv 文件存在")
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        print("详细错误信息:")
        print(traceback.format_exc())
        return None


def interview_assistant_mode():
    """面试辅助追问模式"""
    try:
        from interview_assistant import InterviewAssistant

        print_message("🤖 面试辅助追问系统")
        print_message("=" * 50)

        assistant = InterviewAssistant()

        while True:
            choice = input(
                """
请选择功能：
1. 基于CSV数据生成面试问题
2. 手动输入简历和JD生成问题
3. 候选人匹配度分析
4. 返回主菜单

请输入选择 (1-4): """
            )

            if choice == "1":
                generate_questions_from_csv(assistant)
            elif choice == "2":
                generate_questions_manual(assistant)
            elif choice == "3":
                analyze_candidate_fit_manual(assistant)
            elif choice == "4":
                break
            else:
                print_message("无效选择，请重新输入")

    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("请确认 interview_assistant.py 文件存在")
    except Exception as e:
        print(f"❌ 面试辅助系统启动失败: {e}")
        print(traceback.format_exc())


def generate_questions_from_csv(assistant: "InterviewAssistant"):
    """从CSV数据生成面试问题"""
    try:
        # 加载并显示数据
        candidate_data = assistant.load_candidate_data()
        if not candidate_data:
            print("❌ 无法加载候选人数据")
            return

        # 显示可用的候选人列表
        import pandas as pd

        df = pd.read_csv("interview_data.csv")
        print("\n📋 可用候选人列表：")
        for i, row in df.iterrows():
            position = row.get("岗位名称", "未知岗位")
            intelligence = row.get("该岗位要求的聪明度（满分十分）", "N/A")

            # 处理NaN值
            if pd.isna(position):
                position = "未知岗位"
            if pd.isna(intelligence):
                intelligence = "N/A"

            # 只显示前20个有效记录
            if i < 20 and not pd.isna(row.get("候选人脱敏简历")):
                print(f"{i}: {position} (要求聪明度: {intelligence})")
            elif i >= 20:
                break

        # 用户选择候选人
        try:
            record_id = int(input(f"\n请选择候选人序号 (0-19): "))
            if record_id < 0 or record_id >= 20:
                print("❌ 无效的序号")
                return
        except ValueError:
            print("❌ 请输入有效的数字")
            return

        # 加载选定的候选人数据
        candidate_data = assistant.load_candidate_data(record_id=record_id)
        if not candidate_data:
            print("❌ 加载候选人数据失败")
            return

        print(f"\n✅ 已选择候选人: {candidate_data.get('position', '未知岗位')}")
        print(f"📝 简历预览: {candidate_data['resume'][:100]}...")
        print(f"💼 JD预览: {candidate_data['jd'][:100]}...")

        # 选择评估重点
        focus_areas = select_focus_areas()

        # 生成面试问题
        result = assistant.generate_interview_questions(
            resume=candidate_data["resume"],
            jd=candidate_data["jd"],
            focus_areas=focus_areas,
        )

        if "error" not in result:
            # 显示生成的问题
            print("\n" + "=" * 60)
            print("🎯 生成的面试问题方案:")
            print("=" * 60)
            print(result["generated_questions"])

            # 保存结果，传递候选人ID
            output_path = assistant.save_interview_plan(
                result, candidate_id=f"record_{record_id}"
            )

            # 显示token使用情况
            token_usage = result.get("token_usage", {})
            print(f"\n💰 Token使用统计:")
            print(f"输入tokens: {token_usage.get('input_tokens', 0):,}")
            print(f"输出tokens: {token_usage.get('output_tokens', 0):,}")
            print(f"总tokens: {token_usage.get('total_tokens', 0):,}")
        else:
            print(f"❌ 生成失败: {result['error']}")

    except Exception as e:
        print(f"❌ 生成面试问题失败: {e}")
        print(traceback.format_exc())


def generate_questions_manual(assistant: "InterviewAssistant"):
    """手动输入简历和JD生成问题"""
    try:
        print("\n📝 请输入候选人信息:")

        print("\n1. 请输入候选人简历 (输入完成后按Enter，然后输入'END'结束):")
        resume_lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            resume_lines.append(line)
        resume = "\n".join(resume_lines)

        if not resume.strip():
            print("❌ 简历不能为空")
            return

        print("\n2. 请输入岗位描述 (输入完成后按Enter，然后输入'END'结束):")
        jd_lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            jd_lines.append(line)
        jd = "\n".join(jd_lines)

        if not jd.strip():
            print("❌ 岗位描述不能为空")
            return

        # 选择评估重点
        focus_areas = select_focus_areas()

        # 生成面试问题
        result = assistant.generate_interview_questions(
            resume=resume, jd=jd, focus_areas=focus_areas
        )

        if "error" not in result:
            # 显示生成的问题
            print("\n" + "=" * 60)
            print("🎯 生成的面试问题方案:")
            print("=" * 60)
            print(result["generated_questions"])

            # 保存结果，手动输入模式使用manual标识
            output_path = assistant.save_interview_plan(result, candidate_id="manual")

            # 显示token使用情况
            token_usage = result.get("token_usage", {})
            print(f"\n💰 Token使用统计:")
            print(f"输入tokens: {token_usage.get('input_tokens', 0):,}")
            print(f"输出tokens: {token_usage.get('output_tokens', 0):,}")
            print(f"总tokens: {token_usage.get('total_tokens', 0):,}")
        else:
            print(f"❌ 生成失败: {result['error']}")

    except Exception as e:
        print(f"❌ 生成面试问题失败: {e}")
        print(traceback.format_exc())


def analyze_candidate_fit_manual(assistant: "InterviewAssistant"):
    """手动分析候选人匹配度"""
    try:
        print("\n🔍 候选人匹配度分析")

        print("\n1. 请输入候选人简历 (输入完成后按Enter，然后输入'END'结束):")
        resume_lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            resume_lines.append(line)
        resume = "\n".join(resume_lines)

        if not resume.strip():
            print("❌ 简历不能为空")
            return

        print("\n2. 请输入岗位描述 (输入完成后按Enter，然后输入'END'结束):")
        jd_lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            jd_lines.append(line)
        jd = "\n".join(jd_lines)

        if not jd.strip():
            print("❌ 岗位描述不能为空")
            return

        # 分析匹配度
        result = assistant.analyze_candidate_fit(resume=resume, jd=jd)

        if "error" not in result:
            # 显示分析结果
            print("\n" + "=" * 60)
            print("📊 候选人匹配度分析结果:")
            print("=" * 60)
            print(result["fit_analysis"])

            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"candidate_fit_analysis_{timestamp}.json"
            try:
                import json

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"\n📋 分析结果已保存到: {output_path}")
            except Exception as e:
                print(f"⚠️ 保存失败: {e}")

            # 显示token使用情况
            token_usage = result.get("token_usage", {})
            print(f"\n💰 Token使用统计:")
            print(f"输入tokens: {token_usage.get('input_tokens', 0):,}")
            print(f"输出tokens: {token_usage.get('output_tokens', 0):,}")
            print(f"总tokens: {token_usage.get('total_tokens', 0):,}")
        else:
            print(f"❌ 分析失败: {result['error']}")

    except Exception as e:
        print(f"❌ 匹配度分析失败: {e}")
        print(traceback.format_exc())


def view_json_reports():
    """查看JSON报告工具"""
    print_message("📊 JSON报告查看器")
    print_message("=" * 50)

    try:
        # 获取所有JSON文件
        json_files = []

        # 扫描根目录的JSON文件
        root_json_files = glob.glob("*.json")
        for file in root_json_files:
            json_files.append(("根目录", file))

        # 扫描checkpoints目录的JSON文件
        if os.path.exists("checkpoints"):
            checkpoint_files = glob.glob("checkpoints/*.json")
            for file in checkpoint_files:
                json_files.append(("checkpoints", file))

        if not json_files:
            print("❌ 未找到任何JSON报告文件")
            return

        # 按类型分组显示
        print("📁 可用的JSON报告文件:")
        print()

        # 分类显示
        general_guidelines = []
        individual_experiences = []
        interview_plans = []
        other_files = []

        for category, file_path in json_files:
            if "general_interview_guidelines" in file_path:
                general_guidelines.append((category, file_path))
            elif "individual_interview_experience" in file_path:
                individual_experiences.append((category, file_path))
            elif "interview_plan" in file_path:
                interview_plans.append((category, file_path))
            else:
                other_files.append((category, file_path))

        file_index = 1
        all_files = []

        # 显示通用面试提问经验
        if general_guidelines:
            print("📚 通用面试提问经验:")
            for category, file_path in general_guidelines:
                print(f"  {file_index}. {file_path}")
                all_files.append((category, file_path))
                file_index += 1
            print()

        # 显示个人面试经验总结
        if individual_experiences:
            print("� 个人面试经验总结:")
            for category, file_path in individual_experiences:
                filename = os.path.basename(file_path)
                print(f"  {file_index}. {filename}")
                all_files.append((category, file_path))
                file_index += 1
            print()

        # 显示面试方案
        if interview_plans:
            print("� 面试方案:")
            for category, file_path in interview_plans:
                print(f"  {file_index}. {file_path}")
                all_files.append((category, file_path))
                file_index += 1
            print()

        # 显示其他文件
        if other_files:
            print("📄 其他JSON文件:")
            for category, file_path in other_files:
                print(f"  {file_index}. {file_path}")
                all_files.append((category, file_path))
                file_index += 1
            print()

        while True:
            choice = input(
                f"请选择要查看的文件 (1-{len(all_files)}), 输入 'l' 重新列出, 或输入 'q' 退出: "
            ).strip()

            if choice.lower() == "q":
                break
            elif choice.lower() == "l":
                # 重新显示文件列表
                print("\n📁 可用的JSON报告文件:")
                print()

                # 重新显示通用面试提问经验
                if general_guidelines:
                    print("📚 通用面试提问经验:")
                    for i, (category, file_path) in enumerate(general_guidelines, 1):
                        print(f"  {i}. {file_path}")
                    print()

                # 重新显示个人面试经验总结
                if individual_experiences:
                    start_idx = len(general_guidelines) + 1
                    print("👤 个人面试经验总结:")
                    for i, (category, file_path) in enumerate(
                        individual_experiences, start_idx
                    ):
                        filename = os.path.basename(file_path)
                        print(f"  {i}. {filename}")
                    print()

                # 重新显示面试方案
                if interview_plans:
                    start_idx = (
                        len(general_guidelines) + len(individual_experiences) + 1
                    )
                    print("📋 面试方案:")
                    for i, (category, file_path) in enumerate(
                        interview_plans, start_idx
                    ):
                        print(f"  {i}. {file_path}")
                    print()

                # 重新显示其他文件
                if other_files:
                    start_idx = (
                        len(general_guidelines)
                        + len(individual_experiences)
                        + len(interview_plans)
                        + 1
                    )
                    print("� 其他JSON文件:")
                    for i, (category, file_path) in enumerate(other_files, start_idx):
                        print(f"  {i}. {file_path}")
                    print()
                continue

            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(all_files):
                    category, file_path = all_files[choice_idx]
                    view_json_file(file_path, show_full_content=True)
                else:
                    print("❌ 无效的选择，请重新输入")
            except ValueError:
                print("❌ 请输入有效的数字、'l' 或 'q'")

    except Exception as e:
        print(f"❌ 查看JSON报告失败: {e}")
        print(traceback.format_exc())


def view_json_file(file_path, show_full_content=False):
    """查看单个JSON文件"""
    try:
        print_message(f"\n📄 正在查看: {file_path}")
        print_message("=" * 60)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 根据文件类型显示不同的信息
        if "general_interview_guidelines" in file_path:
            display_general_guidelines(data, show_full_content)
        elif "individual_interview_experience" in file_path:
            display_individual_experience(data, show_full_content)
        elif "interview_plan" in file_path:
            display_plan_report(data, show_full_content)
        else:
            display_generic_json(data, show_full_content)

        print_message("=" * 60)
        input("按回车键继续...")

    except Exception as e:
        print(f"❌ 读取文件失败: {e}")


def display_general_guidelines(data, show_full_content=False):
    """显示通用面试提问经验"""
    print(f"📊 总记录数: {data.get('total_records', 'N/A')}")
    print(f"✅ 有效记录数: {data.get('valid_records', 'N/A')}")
    print(f"🎯 采样记录数: {data.get('sampled_records', 'N/A')}")
    print(f"🧠 成功提取数: {data.get('successful_extractions', 'N/A')}")
    print(f"📅 总结模式: {data.get('summary_mode', 'N/A')}")

    # Token统计
    if "token_stats" in data:
        tokens = data["token_stats"]
        print(f"\n💰 Token使用统计:")
        print(f"  输入Tokens: {tokens.get('total_input_tokens', 0):,}")
        print(f"  输出Tokens: {tokens.get('total_output_tokens', 0):,}")
        print(f"  总成本: ${tokens.get('total_cost', 0):.6f}")

    # 新提取的经验
    if "new_experiences" in data and data["new_experiences"]:
        print(f"\n 新提取的经验 ({len(data['new_experiences'])}条):")
        max_experiences = (
            len(data["new_experiences"])
            if show_full_content
            else min(3, len(data["new_experiences"]))
        )

        for i, exp in enumerate(data["new_experiences"][:max_experiences], 1):
            print(f"\n  [{i}] 记录ID: {exp.get('record_id', 'N/A')}")
            resume = exp.get("resume_summary", "")
            if resume:
                if show_full_content:
                    print(f"      简历摘要: {resume}")
                else:
                    print(f"      简历摘要: {resume[:100]}...")
            print(f"      分析时间: {exp.get('analysis_time', 'N/A')}")

            # 如果显示完整内容，也显示提取的经验
            if show_full_content and "extracted_experience" in exp:
                print_message(f"      提取的经验:\n{exp['extracted_experience']}")

        if not show_full_content and len(data["new_experiences"]) > 3:
            print(f"      ... 还有 {len(data['new_experiences']) - 3} 条经验")

    # 集成经验
    if "integrated_experience" in data:
        exp_text = data["integrated_experience"]
        print(f"\n📚 集成经验{'完整内容' if show_full_content else '预览'}:")

        if show_full_content:
            print_message(exp_text)
        else:
            lines = exp_text.split("\n")
            for line in lines[:10]:  # 显示前10行
                if line.strip():
                    print(f"    {line}")
            if len(lines) > 10:
                print(f"    ... (还有 {len(lines) - 10} 行)")


def display_individual_experience(data, show_full_content=False):
    """显示个人面试经验总结"""
    print(f"🔄 记录ID: {data.get('record_id', 'N/A')}")
    print(f"📁 样本名称: {data.get('sample_name', 'N/A')}")
    print(f"⏰ 时间戳: {data.get('timestamp', 'N/A')}")

    # 经验内容
    if "experience" in data:
        exp = data["experience"]
        print(f"\n📋 经验内容:")
        print(f"  记录ID: {exp.get('record_id', 'N/A')}")
        print(f"  分析时间: {exp.get('analysis_time', 'N/A')}")

        # 简历摘要
        if "resume_summary" in exp:
            resume = exp["resume_summary"]
            print_message(
                f"\n  👤 简历摘要{'完整内容' if show_full_content else '预览'}:"
            )

            if show_full_content:
                print(f"      {resume}")
            else:
                lines = resume.split("\n")
                for line in lines[:3]:
                    if line.strip():
                        print(f"      {line}")
                if len(lines) > 3:
                    print(f"      ... (还有 {len(lines) - 3} 行)")

        # 评估内容
        if "evaluation" in exp:
            evaluation = exp["evaluation"]
            print(f"\n  📊 评估{'完整内容' if show_full_content else '预览'}:")
            if show_full_content:
                print_message(f"      {evaluation}")
            else:
                print(f"      {evaluation[:200]}...")

        # 提取的经验
        if "extracted_experience" in exp:
            extracted = exp["extracted_experience"]
            print(f"\n  🧠 提取经验{'完整内容' if show_full_content else '预览'}:")

            if show_full_content:
                print_message(extracted)
            else:
                lines = extracted.split("\n")
                for line in lines[:5]:
                    if line.strip():
                        print(f"      {line}")
                if len(lines) > 5:
                    print(f"      ... (还有 {len(lines) - 5} 行)")


def display_experience_report(data, show_full_content=False):
    """显示经验提取报告"""
    print(f"📊 总记录数: {data.get('total_records', 'N/A')}")
    print(f"✅ 有效记录数: {data.get('valid_records', 'N/A')}")
    print(f"🎯 采样记录数: {data.get('sampled_records', 'N/A')}")
    print(f"🧠 成功提取数: {data.get('successful_extractions', 'N/A')}")
    print(f"📅 总结模式: {data.get('summary_mode', 'N/A')}")

    # Token统计
    if "token_stats" in data:
        tokens = data["token_stats"]
        print(f"\n💰 Token使用统计:")
        print(f"  输入Tokens: {tokens.get('total_input_tokens', 0):,}")
        print(f"  输出Tokens: {tokens.get('total_output_tokens', 0):,}")
        print(f"  总成本: ${tokens.get('total_cost', 0):.6f}")

    # 新提取的经验
    if "new_experiences" in data and data["new_experiences"]:
        print(f"\n 新提取的经验 ({len(data['new_experiences'])}条):")
        max_experiences = (
            len(data["new_experiences"])
            if show_full_content
            else min(3, len(data["new_experiences"]))
        )

        for i, exp in enumerate(data["new_experiences"][:max_experiences], 1):
            print(f"\n  [{i}] 记录ID: {exp.get('record_id', 'N/A')}")
            resume = exp.get("resume_summary", "")
            if resume:
                if show_full_content:
                    print_message(f"      简历摘要: {resume}")
                else:
                    print(f"      简历摘要: {resume[:100]}...")
            print(f"      分析时间: {exp.get('analysis_time', 'N/A')}")

            # 如果显示完整内容，也显示提取的经验
            if show_full_content and "extracted_experience" in exp:
                print_message(f"      提取的经验:\n{exp['extracted_experience']}")

        if not show_full_content and len(data["new_experiences"]) > 3:
            print(f"      ... 还有 {len(data['new_experiences']) - 3} 条经验")

    # 集成经验
    if "integrated_experience" in data:
        exp_text = data["integrated_experience"]
        print(f"\n📚 集成经验{'完整内容' if show_full_content else '预览'}:")

        if show_full_content:
            print_message(exp_text)
        else:
            lines = exp_text.split("\n")
            for line in lines[:10]:  # 显示前10行
                if line.strip():
                    print(f"    {line}")
            if len(lines) > 10:
                print(f"    ... (还有 {len(lines) - 10} 行)")


def display_plan_report(data, show_full_content=False):
    """显示面试计划报告"""
    print(f"📅 生成时间: {data.get('generation_time', 'N/A')}")

    # 候选人简历
    if "candidate_resume" in data:
        resume = data["candidate_resume"]
        print(f"\n👤 候选人简历{'完整内容' if show_full_content else '预览'}:")

        if show_full_content:
            print_message(resume)
        else:
            lines = resume.split("\n")
            for line in lines[:5]:
                if line.strip():
                    print(f"    {line}")
            if len(lines) > 5:
                print(f"    ... (还有 {len(lines) - 5} 行)")

    # 职位描述
    if "job_description" in data:
        jd = data["job_description"]
        print(f"\n💼 职位描述{'完整内容' if show_full_content else '预览'}:")

        if show_full_content:
            print_message(jd)
        else:
            lines = jd.split("\n")
            for line in lines[:5]:
                if line.strip():
                    print(f"    {line}")
            if len(lines) > 5:
                print(f"    ... (还有 {len(lines) - 5} 行)")

    # 关注领域
    if "focus_areas" in data:
        print(f"\n🎯 关注领域: {', '.join(data['focus_areas'])}")

    # Token使用情况
    if "token_usage" in data:
        tokens = data["token_usage"]
        print(f"\n💰 Token使用统计:")
        print(f"  输入Tokens: {tokens.get('input_tokens', 0):,}")
        print(f"  输出Tokens: {tokens.get('output_tokens', 0):,}")
        print(f"  总Tokens: {tokens.get('total_tokens', 0):,}")

    # 生成的问题预览
    if "generated_questions" in data:
        questions = data["generated_questions"]
        print(f"\n❓ 生成的问题{'完整内容' if show_full_content else '预览'}:")

        if show_full_content:
            print(questions)
        else:
            lines = questions.split("\n")
            for line in lines[:15]:
                if line.strip():
                    print(f"    {line}")
            if len(lines) > 15:
                print(f"    ... (还有 {len(lines) - 15} 行)")


def display_checkpoint_file(data, show_full_content=False):
    """显示checkpoint文件"""
    print(f"🔄 记录ID: {data.get('record_id', 'N/A')}")
    print(f"📁 样本名称: {data.get('sample_name', 'N/A')}")
    print(f"⏰ 时间戳: {data.get('timestamp', 'N/A')}")

    # 经验内容
    if "experience" in data:
        exp = data["experience"]
        print(f"\n📋 经验内容:")
        print(f"  记录ID: {exp.get('record_id', 'N/A')}")
        print(f"  分析时间: {exp.get('analysis_time', 'N/A')}")

        # 简历摘要
        if "resume_summary" in exp:
            resume = exp["resume_summary"]
            print(f"\n  👤 简历摘要{'完整内容' if show_full_content else '预览'}:")

            if show_full_content:
                print(f"      {resume}")
            else:
                lines = resume.split("\n")
                for line in lines[:3]:
                    if line.strip():
                        print(f"      {line}")
                if len(lines) > 3:
                    print(f"      ... (还有 {len(lines) - 3} 行)")

        # 评估内容
        if "evaluation" in exp:
            evaluation = exp["evaluation"]
            print(f"\n  📊 评估{'完整内容' if show_full_content else '预览'}:")
            if show_full_content:
                print(f"      {evaluation}")
            else:
                print(f"      {evaluation[:200]}...")

        # 提取的经验
        if "extracted_experience" in exp:
            extracted = exp["extracted_experience"]
            print(f"\n  🧠 提取经验{'完整内容' if show_full_content else '预览'}:")

            if show_full_content:
                print(extracted)
            else:
                lines = extracted.split("\n")
                for line in lines[:5]:
                    if line.strip():
                        print(f"      {line}")
                if len(lines) > 5:
                    print(f"      ... (还有 {len(lines) - 5} 行)")


def display_generic_json(data, show_full_content=False):
    """显示通用JSON数据"""

    def print_value(key, value, indent=0, full_content=False):
        prefix = "  " * indent
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            items_to_show = (
                list(value.items()) if full_content else list(value.items())[:5]
            )

            for k, v in items_to_show:
                print_value(k, v, indent + 1, full_content)

            if not full_content and len(value) > 5:
                print(f"{prefix}  ... (还有 {len(value) - 5} 个键)")

        elif isinstance(value, list):
            print(f"{prefix}{key}: [列表，{len(value)} 个项目]")
            items_to_show = value if full_content else value[:3]

            for i, item in enumerate(items_to_show):
                if isinstance(item, (dict, list)):
                    print(f"{prefix}  [{i}]: {type(item).__name__}")
                    if full_content and isinstance(item, dict):
                        for k, v in item.items():
                            print_value(k, v, indent + 2, full_content)
                else:
                    item_str = str(item)
                    if not full_content and len(item_str) > 100:
                        item_str = item_str[:100] + "..."
                    print(f"{prefix}  [{i}]: {item_str}")

            if not full_content and len(value) > 3:
                print(f"{prefix}  ... (还有 {len(value) - 3} 个项目)")

        elif isinstance(value, str):
            if not full_content and len(value) > 200:
                print(f"{prefix}{key}: {value[:200]}...")
            else:
                print(f"{prefix}{key}: {value}")
        else:
            print(f"{prefix}{key}: {value}")

    print(f"📄 JSON数据{'完整内容' if show_full_content else '预览'}:")
    items_to_show = list(data.items()) if show_full_content else list(data.items())[:10]

    for key, value in items_to_show:
        print_value(key, value, full_content=show_full_content)

    if not show_full_content and len(data) > 10:
        print(f"  ... (还有 {len(data) - 10} 个键)")


def select_focus_areas():
    """选择评估重点领域"""
    print("\n📋 请选择评估重点 (可多选，用逗号分隔，如: 1,2,3):")
    print("1. 聪明度")
    print("2. 皮实")
    print("3. 勤奋")
    print("4. 全部 (默认)")

    choice = input("请输入选择: ").strip()

    if not choice or choice == "4":
        return ["聪明度", "皮实", "勤奋"]

    focus_map = {"1": "聪明度", "2": "皮实", "3": "勤奋"}

    try:
        selected = [
            focus_map[c.strip()] for c in choice.split(",") if c.strip() in focus_map
        ]
        if not selected:
            print("⚠️ 无效选择，使用默认 (全部)")
            return ["聪明度", "皮实", "勤奋"]
        return selected
    except:
        print("⚠️ 输入格式错误，使用默认 (全部)")
        return ["聪明度", "皮实", "勤奋"]


def main():
    print_message("🎯 面试模拟系统")
    print_message("=" * 50)

    try:
        choice = input(
            """
请选择功能：
1. 运行面试模拟
2. 分析面试数据
3. 提取面试经验
4. 面试辅助追问
5. 查看JSON报告
6. 退出

请输入选择 (1-6): """
        )

        if choice == "1":
            interview_scene()
        elif choice == "2":
            analyze_interview_data()
        elif choice == "3":
            extract_interview_experience()
        elif choice == "4":
            interview_assistant_mode()
        elif choice == "5":
            view_json_reports()
        elif choice == "6":
            print_message("再见！")
        else:
            print_message("无效选择，退出程序")

    except KeyboardInterrupt:
        print_message("\n用户中断程序")
    except Exception as e:
        print_message(f"❌ 程序执行失败: {e}")
        print_message("详细错误信息:")
        print_message(traceback.format_exc())


if __name__ == "__main__":
    main()
