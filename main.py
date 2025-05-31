from menglong import Model
from menglong.agents.role_play import RoleAgent
from menglong.agents.conversation import ATAConversation
from menglong.ml_model.schema.ml_request import (
    SystemMessage as system,
    UserMessage as user,
    AssistantMessage as assistant,
    ToolMessage as tool,
)
import yaml
from datetime import datetime
from pathlib import Path

from menglong.utils.log import rich_print, rich_print_json
from menglong.utils.log import configure_logger, print

# 时间
configure_logger(log_file=f"agent_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")


def test():
    rich_print("Hello from interview-sim!")
    model = Model(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")
    res = model.chat([user(content="你好")])
    rich_print(res)


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

        print("=== 面试开始 ===")

        # HR开场白
        hr_opening = self.hr.opening_statement(topic)
        print(f"{hr_opening}", title=self.hr.id, use_panel=True)
        self.context_manager.add_message(self.hr.id, hr_opening)

        for round_num in range(1, self.max_rounds + 1):
            self.context_manager.current_round = round_num
            print(f"\n--- 第 {round_num} 轮对话 ---")

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
                print(
                    f"{interviewer_question}", title=self.interviewer.id, use_panel=True
                )
            else:
                # 后续轮次，基于之前的对话继续提问
                if self.context_manager.cache_context is not None:
                    interviewer_question = self.interviewer.chat(
                        self.context_manager.cache_context
                    )
                    print(
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
            print(
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
            print(f"{hr_evaluation}", title=self.hr.id, use_panel=True)
            self.context_manager.add_hr_evaluation(hr_evaluation)

            # 检查是否结束面试
            if self.hr.should_end_interview(
                self.context_manager.get_conversation_context()
            ):
                print(f"\n{self.hr.id}: 面试结束，感谢您的参与！")
                break

        print("\n=== 面试结束 ===")
        return self.context_manager.get_conversation_context()


def interview_scene():
    """模拟面试流程"""
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

        rich_print_json(role_infos, title="角色信息")
    else:
        print("prompts目录不存在，使用默认配置")
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
    rich_print(conversation_result, title="对话结果", use_panel=True)

    # 打印最终结果
    print("\n=== 面试总结 ===")
    hr_evaluations = interview.context_manager.get_hr_evaluations()
    for i, eval_data in enumerate(hr_evaluations, 1):
        print(f"hr的评估[{i}]: {eval_data}")

    result_summary = interview.context_manager.get_result_summary()
    # 保存结果到文件
    with open(f"interview_result-{interview.candidate.id}.txt", "w") as f:
        f.write("\n=== 关键有效问答总结 ===\n")
        for i, summary in enumerate(result_summary, 1):
            f.write(f"总结[{i}]: {summary}\n")

    print("\n=== 面试总结结束 ===")


def main():
    interview_scene()


if __name__ == "__main__":
    main()
