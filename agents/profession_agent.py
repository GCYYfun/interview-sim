"""
专业Agent - 根据需要动态添加的专业领域Agent

功能（TODO）：
- 根据不同专业领域（技术、产品、运营等）提供专业化面试支持
- 可以动态配置和扩展
- 提供领域特定的问题库和评估标准

设计思路：
1. 基类：ProfessionAgent
2. 子类：TechAgent, ProductAgent, OperationsAgent 等
3. 配置驱动：通过配置文件定义不同专业的评估标准

示例用法（未来）：
```python
# 技术岗位Agent
tech_agent = TechAgent(
    domain="后端开发",
    tech_stack=["Python", "Django", "PostgreSQL"],
    level="高级工程师"
)

# 产品岗位Agent
product_agent = ProductAgent(
    domain="B端产品",
    focus_areas=["需求分析", "项目管理", "数据分析"]
)
```

当前状态：TODO - 占位文件，等待后续实现
"""

from typing import Dict, List


class ProfessionAgent:
    """
    专业Agent基类（占位）

    后续实现时需要考虑：
    1. 如何定义专业领域的评估标准
    2. 如何与HR Agent协同工作
    3. 如何动态加载专业知识库
    4. 如何支持多专业复合岗位
    """

    def __init__(self, profession_config: Dict = None):
        """
        初始化专业Agent

        Args:
            profession_config: 专业配置
                {
                    "domain": str,           # 专业领域
                    "sub_domains": List[str], # 子领域
                    "tech_stack": List[str],  # 技术栈（技术岗位）
                    "level": str,            # 级别要求
                    "evaluation_focus": List[str]  # 评估重点
                }
        """
        self.config = profession_config or {}
        self.domain = self.config.get("domain", "通用")

        # TODO: 加载专业知识库
        # TODO: 初始化专业评估标准
        # TODO: 配置专业问题模板

    def generate_professional_question(self, context: Dict) -> str:
        """
        生成专业领域的面试问题

        TODO: 实现
        """
        raise NotImplementedError("待实现：专业问题生成")

    def evaluate_professional_answer(
        self, question: str, answer: str, context: Dict
    ) -> Dict:
        """
        评估专业领域的回答

        TODO: 实现
        """
        raise NotImplementedError("待实现：专业回答评估")

    def get_professional_insights(self, conversation_history: List[Dict]) -> Dict:
        """
        提供专业洞察

        TODO: 实现
        """
        raise NotImplementedError("待实现：专业洞察提取")


# TODO: 实现具体的专业Agent


class TechAgent(ProfessionAgent):
    """技术岗位专业Agent（TODO）"""

    pass


class ProductAgent(ProfessionAgent):
    """产品岗位专业Agent（TODO）"""

    pass


class OperationsAgent(ProfessionAgent):
    """运营岗位专业Agent（TODO）"""

    pass


class DesignAgent(ProfessionAgent):
    """设计岗位专业Agent（TODO）"""

    pass


# 后续扩展方向：
# 1. 支持配置文件驱动的Agent创建
# 2. 提供Agent组合机制（一个岗位多个专业维度）
# 3. 与经验库集成，积累专业领域知识
# 4. 支持自定义专业评估模型
