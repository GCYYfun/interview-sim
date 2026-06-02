"""
Agents模块 - 面试系统的智能Agent集合

包含以下Agent：
- CandidateAgent: 候选人Agent，基于简历自动对话，用于数据增强/蒸馏
- HRAgent: HR Agent，根据简历和回答问HR指标相关问题
- EvalAgent: 评估Agent，评估回复与指标的切合度
- ExperienceAgent: 经验Agent，抽取通用面试经验形成文档
- InterviewAgent: 面试Agent，生成面试问题和分析匹配度
- ProfessionAgent: 专业Agent（TODO），动态添加专业领域支持
"""

from .candidate_agent import CandidateAgent
from .hr_agent import HRAgent
from .eval_agent import EvalAgent
from .experience_agent import ExperienceAgent
from .interview_agent import InterviewAgent
from .profession_agent import ProfessionAgent

__all__ = [
    "CandidateAgent",
    "HRAgent",
    "EvalAgent",
    "ExperienceAgent",
    "InterviewAgent",
    "ProfessionAgent",
]

__version__ = "2.0.0"
