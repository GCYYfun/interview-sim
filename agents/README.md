# Agents 模块

面试系统的智能Agent集合，负责面试过程中的各个环节。

## 📦 模块结构

```
agents/
├── __init__.py           # 模块导出
├── candidate_agent.py    # 候选人Agent
├── hr_agent.py          # HR Agent
├── eval_agent.py        # 评估Agent
├── experience_agent.py  # 经验Agent
├── interview_agent.py   # 面试Agent
└── profession_agent.py  # 专业Agent（TODO）
```

## 🤖 Agent 说明

### 1. CandidateAgent - 候选人Agent

**职责**：
- 根据候选人简历信息，自动与HR Agent对话
- 模拟真实候选人的回答
- 用于数据增强/蒸馏

**用法**：
```python
from agents import CandidateAgent

candidate_data = {
    "resume": "张三，5年Python开发经验...",
    "jd": "招聘高级Python工程师...",
    "position": "高级Python工程师",
    "intelligence_requirement": 80
}

candidate = CandidateAgent(candidate_data)
answer = candidate.answer_question("请介绍你的项目经验")
```

**关键方法**：
- `answer_question(question)`: 回答面试问题
- `get_candidate_info_summary()`: 获取候选人信息摘要

---

### 2. HRAgent - HR Agent

**职责**：
- 负责根据简历和回答问HR指标相关问题
- 考察候选人的聪明度、皮实、勤奋等指标
- 评估候选人回答是否符合岗位要求
- 判断面试进度

**用法**：
```python
from agents import HRAgent

hr = HRAgent()

# 开场白
opening = hr.opening_statement("Python工程师")

# 生成问题
question = hr.generate_question(
    candidate_resume="...",
    jd="...",
    dimension="聪明度"  # 可选：聪明度/皮实/勤奋
)

# 评估回答
eval_result = hr.evaluate_response(
    question="你的问题",
    answer="候选人回答",
    candidate_info={...}
)

# 判断是否结束
should_end = hr.should_end_interview(conversation_history)
```

**关键方法**：
- `opening_statement(topic)`: 开场白
- `generate_question(...)`: 生成面试问题
- `evaluate_response(...)`: 评估回答
- `should_end_interview(...)`: 判断是否结束
- `generate_final_summary(...)`: 生成最终总结

---

### 3. EvalAgent - 评估Agent

**职责**：
- 评估候选人回复与指标的切合度
- 基于聪明度、皮实、勤奋三维标准打分
- 提供每轮评估和最终综合评估
- 识别亮点和风险点

**用法**：
```python
from agents import EvalAgent

evaluator = EvalAgent()

# 评估单个回答
eval_result = evaluator.evaluate_single_response(
    question="面试问题",
    answer="候选人回答",
    candidate_info={
        "position": "岗位名称",
        "intelligence_requirement": 80
    },
    question_intent="考察聪明度"  # 可选
)

# 查看评分
scores = eval_result['scores']
print(f"聪明度: {scores['聪明度']}/100")

# 生成最终评估
final_eval = evaluator.generate_final_evaluation(
    candidate_info={...},
    conversation_history=[...]
)

# 重置评估（新面试）
evaluator.reset_evaluations()
```

**关键方法**：
- `evaluate_single_response(...)`: 评估单个回答
- `generate_final_evaluation(...)`: 生成最终综合评估
- `reset_evaluations()`: 重置评估记录
- `get_evaluation_summary()`: 获取评估摘要

**评估标准**：
- 聪明度：逻辑思维、问题理解、学习能力、创新思维
- 皮实：抗压能力、挫折恢复力、情绪管理、韧性
- 勤奋：自驱力、主动性、持续投入、结果导向

---

### 4. ExperienceAgent - 经验Agent

**职责**：
- 根据面试者的简历、岗位描述、面试对话、评价结果
- 分析抽取通用的面试经验
- 形成文档，可供HR Agent使用
- **注意**：后续会结合到data manager中

**用法**：
```python
from agents import ExperienceAgent

exp_agent = ExperienceAgent()

# 从单次面试提取经验
experience = exp_agent.extract_experience(
    resume="候选人简历",
    jd="岗位描述",
    conversation_history=[...],
    evaluation_result={...}
)

# 保存经验
exp_agent.save_experience(experience)

# 整合多个经验
guide = exp_agent.consolidate_experiences()

# 获取所有经验
all_exp = exp_agent.get_all_experiences()
```

**关键方法**：
- `extract_experience(...)`: 从单次面试提取经验
- `consolidate_experiences(...)`: 整合多个经验成指南
- `save_experience(...)`: 保存经验到文件
- `get_all_experiences()`: 获取所有已提取经验

**输出内容**：
- 问题设计经验
- 回答评估要点
- 三维能力识别技巧
- 岗位匹配洞察
- 面试技巧总结

---

### 5. InterviewAgent - 面试Agent

**职责**：
- 基于简历和JD生成针对性的面试问题
- 分析候选人与岗位的匹配度
- 提供深度追问策略
- 基于经验库优化问题设计

**用法**：
```python
from agents import InterviewAgent

interview_agent = InterviewAgent()

# 生成面试问题
questions_result = interview_agent.generate_interview_questions(
    resume="候选人简历",
    jd="岗位描述",
    focus_areas=["聪明度", "皮实", "勤奋"],
    position="Python工程师"
)

# 分析候选人匹配度
match_result = interview_agent.analyze_candidate_match(
    resume="候选人简历",
    jd="岗位描述",
    position="Python工程师"
)

# 保存面试方案
filepath = interview_agent.save_interview_plan(
    questions_result,
    candidate_id="candidate_001"
)
```

**关键方法**：
- `generate_interview_questions(...)`: 生成面试问题和策略
- `analyze_candidate_match(...)`: 分析候选人匹配度
- `save_interview_plan(...)`: 保存面试方案到文件
- `load_latest_experience()`: 加载最新的面试经验库

**生成的问题包含**：
- 候选人分析（亮点、风险、匹配度）
- 开场破冰问题
- 核心能力测试问题（聪明度/皮实/勤奋）
- 深度追问策略
- 风险预警
- 结尾问题

**经验库支持**：
- 自动加载 `general_interview_guidelines_*.json`
- 如无经验库，使用默认面试技巧
- 基于经验库优化问题设计

---

### 6. ProfessionAgent - 专业Agent（TODO）

**职责**（待实现）：
- 根据需要动态添加专业领域支持
- 提供领域特定的问题库和评估标准
- 支持技术、产品、运营、设计等不同岗位

**计划功能**：
```python
# 未来用法
from agents import TechAgent, ProductAgent

# 技术岗位
tech = TechAgent(
    domain="后端开发",
    tech_stack=["Python", "Django"],
    level="高级工程师"
)

# 产品岗位
product = ProductAgent(
    domain="B端产品",
    focus_areas=["需求分析", "项目管理"]
)
```

---

## 🎯 完整使用示例

参见 `test/test_agents.py`：

```python
from agents import CandidateAgent, HRAgent, EvalAgent, ExperienceAgent, InterviewAgent

# 1. 准备数据
candidate_data = {...}

# 2. 创建Agents
candidate = CandidateAgent(candidate_data)
hr = HRAgent()
evaluator = EvalAgent()
interview_agent = InterviewAgent()

# 3. 生成面试问题（面试前准备）
questions_result = interview_agent.generate_interview_questions(
    resume=candidate_data['resume'],
    jd=candidate_data['jd'],
    position=candidate_data['position']
)

# 4. 面试流程
opening = hr.opening_statement(candidate_data['position'])
for question in questions:
    answer = candidate.answer_question(question)
    eval_result = evaluator.evaluate_single_response(...)

# 5. 最终评估
final_eval = evaluator.generate_final_evaluation(...)

# 6. 提取经验
exp_agent = ExperienceAgent()
experience = exp_agent.extract_experience(...)
```

运行演示：
```bash
python test/test_agents.py
```

---

## 🔄 与其他模块的关系

```
agents/                    # Agent模块（智能对话）
  ├─ CandidateAgent       # 使用 menglong.Model
  ├─ HRAgent              # 使用 RoleAgent
  ├─ EvalAgent            # 使用 menglong.Model
  └─ ExperienceAgent      # 使用 menglong.Model
      │
      └─> 后续集成到 manager/experience_extractor.py

manager/                   # 数据管理模块
  ├─ data_loader.py
  ├─ data_processor.py
  └─ experience_extractor.py  # 将整合ExperienceAgent

modules/                   # 功能模块（业务逻辑）
  └─ interview_sim.py      # 使用agents进行面试模拟

cli/                       # Rich控制台
  └─ app.py               # 调用modules，modules使用agents
```

---

## 📝 设计原则

1. **职责单一**：每个Agent专注一个职责
2. **松耦合**：Agent之间通过标准接口交互
3. **可扩展**：支持新增专业Agent
4. **数据驱动**：基于配置和数据灵活调整
5. **可测试**：每个Agent可独立测试

---

## 🚀 后续计划

- [ ] 完善ProfessionAgent实现
- [ ] ExperienceAgent集成到manager
- [ ] 支持Agent组合（多Agent协作）
- [ ] 添加Agent性能监控
- [ ] 支持自定义评估标准

---

## 📚 相关文档

- [Manager模块文档](../manager/README.md)
- [重构方案](../REFACTOR.md)
- [测试用例](../test/test_agents.py)
