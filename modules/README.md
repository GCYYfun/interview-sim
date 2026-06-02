# 功能模块 (Modules)

本目录包含所有核心功能模块，每个模块封装了特定的业务功能。

## 📦 模块列表

### 1. InterviewSimulator (面试模拟)
**文件**: `interview_sim.py`

**功能**: 执行完整的面试流程

**主要方法**:
- `run_interview()` - 运行面试（支持auto/manual模式）
- `_run_auto_interview()` - AI自动对话模式
- `_run_manual_interview()` - 人工手动输入模式

**使用示例**:
```python
from modules import InterviewSimulator

simulator = InterviewSimulator()

# 模拟面试
result = simulator.run_interview(
    resume="path/to/resume.txt",
    jd="path/to/jd.txt",
    mode="auto",  # 或 "manual"
    max_rounds=5
)
```

**依赖**: InterviewAgent, CandidateAgent, EvalAgent

---

### 2. DataAnalyzer (数据分析)
**文件**: `data_analysis.py`

**功能**: 分析和可视化面试数据

**主要方法**:
- `show_statistics()` - 显示统计信息
- `search_candidates()` - 搜索候选人
- `generate_visualizations()` - 生成可视化图表
- `export_summary()` - 导出摘要报告

**使用示例**:
```python
from modules import DataAnalyzer

analyzer = DataAnalyzer(csv_path="interview_data.csv")

# 查看统计
analyzer.show_statistics()

# 搜索候选人
results = analyzer.search_candidates(position="Python开发")

# 生成图表
analyzer.generate_visualizations(output_dir="charts/")
```

**依赖**: manager.InterviewDataManager

---

### 3. ExperienceExtractor (经验提取)
**文件**: `experience_extract.py`

**功能**: 从面试记录提取和整合经验

**主要方法**:
- `extract_from_records()` - 从面试记录提取经验
- `consolidate_all_experiences()` - 整合所有经验
- `view_experiences()` - 查看经验列表
- `export_experiences()` - 导出经验

**使用示例**:
```python
from modules import ExperienceExtractor

extractor = ExperienceExtractor(
    csv_path="interview_data.csv",
    exp_dir="experiences/"
)

# 从记录提取
experience = extractor.extract_from_records(
    record_ids=[1, 2, 3],
    position="Python开发"
)

# 整合所有经验
consolidated = extractor.consolidate_all_experiences()
```

**依赖**: ExperienceAgent, manager

---

### 4. InterviewAssistant (面试助手)
**文件**: `interview_assist.py`

**功能**: 面试准备和辅助

**主要方法**:
- `generate_questions()` - 生成面试问题
- `analyze_match()` - 分析简历和JD匹配度
- `prepare_interview()` - 完整面试准备
- `load_from_history()` - 从历史数据加载

**使用示例**:
```python
from modules import InterviewAssistant

assistant = InterviewAssistant()

# 准备面试
plan = assistant.prepare_interview(
    resume="path/to/resume.txt",
    jd="path/to/jd.txt"
)

# 仅生成问题
questions = assistant.generate_questions(resume, jd)

# 仅分析匹配度
match_result = assistant.analyze_match(resume, jd)
```

**依赖**: InterviewAgent, manager

---

### 5. ReportViewer (报告查看)
**文件**: `report_viewer.py`

**功能**: 查看和管理报告文件

**主要方法**:
- `list_all_reports()` - 列出所有报告
- `view_report()` - 查看报告详情
- `search_reports()` - 搜索报告
- `export_report()` - 导出报告

**使用示例**:
```python
from modules import ReportViewer

viewer = ReportViewer()

# 列出报告
reports = viewer.list_all_reports(report_type="interview")

# 查看报告
data = viewer.view_report(
    filepath="temp/interview_result.json",
    show_full=True
)

# 搜索报告
matched = viewer.search_reports(keyword="Python")

# 导出报告
viewer.export_report(
    filepath="temp/report.json",
    output_format="md",
    output_path="reports/report.md"
)
```

**依赖**: 无（独立模块）

---

### 6. ConversationEvaluator (对话评估) ⭐ NEW
**文件**: `eval_conversation.py`

**功能**: 清洗和评估已有面试对话记录

**主要方法**:
- `clean_conversation()` - 清洗对话数据为标准格式
- `evaluate_conversation()` - 评估清洗后的对话
- `evaluate_record_by_id()` - 根据记录ID评估对话
- `batch_evaluate()` - 批量评估多条记录
- `export_evaluation_report()` - 导出评估报告

**使用示例**:
```python
from modules import ConversationEvaluator

evaluator = ConversationEvaluator(csv_path="interview_data.csv")

# 评估单条记录
result = evaluator.evaluate_record_by_id(
    record_id=0,
    round_name="First Round"
)

# 批量评估
results = evaluator.batch_evaluate(
    record_ids=[0, 1, 2],
    round_name="First Round",
    max_records=10
)

# 导出报告
evaluator.export_evaluation_report(
    result,
    output_path="reports/eval_result.json"
)
```

**特殊功能**:
- **智能清洗**: 使用 AI 模型自动清洗对话格式
  - 识别面试官和候选人
  - 去除时间戳和无效内容
  - 合并连续对话
  - 转换为标准 JSON 格式

- **备用解析**: 如果 AI 清洗失败，使用规则解析
  - 正则匹配说话人和内容
  - 自动判断角色（英文名=面试官，中文名=候选人）

- **多维评估**: 
  - 使用 EvalAgent 进行三维评分
  - 生成对话摘要和统计
  - 提供详细评估意见

**依赖**: EvalAgent, manager.InterviewDataManager, Model

---

## 🔧 统一导入

所有模块都可以通过 `modules` 包统一导入：

```python
from modules import (
    InterviewSimulator,
    DataAnalyzer,
    ExperienceExtractor,
    InterviewAssistant,
    ReportViewer,
    ConversationEvaluator,  # ⭐ NEW
)
```

## 📝 设计原则

1. **单一职责**: 每个模块专注于一个特定功能
2. **依赖注入**: 通过构造函数传递依赖，便于测试
3. **统一输出**: 使用 `menglong.utils.log.print_message` 统一输出格式
4. **错误处理**: 完善的异常处理和用户提示
5. **可扩展性**: 预留扩展接口，便于功能增强
6. **智能清洗**: 对话评估模块使用 AI 模型进行数据清洗 ⭐

## 🔄 模块关系

```
modules/
  ├─ interview_sim.py      [使用: InterviewAgent, CandidateAgent, EvalAgent]
  ├─ data_analysis.py      [使用: manager.InterviewDataManager]
  ├─ experience_extract.py [使用: ExperienceAgent, manager]
  ├─ interview_assist.py   [使用: InterviewAgent, manager]
  ├─ report_viewer.py      [独立模块]
  └─ eval_conversation.py  [使用: EvalAgent, manager, Model] ⭐ NEW
```

## 📊 数据流

```
CSV数据 → DataAnalyzer → 统计分析
         ↓
    InterviewDataManager
         ↓
    InterviewSimulator → 面试结果 → ReportViewer
         ↓
    ExperienceExtractor → 经验文件
         ↓
    InterviewAssistant → 面试方案
         ↓
    ConversationEvaluator → 对话评估报告 ⭐ NEW
         ↓ (清洗)
    AI Model 清洗对话 → 标准格式对话
         ↓ (评估)
    EvalAgent → 评估结果
```

## � 对话评估模块详解

### 数据清洗流程

```
原始对话（含时间戳、混乱格式）
    ↓
AI Model 智能清洗
    ├─ 识别说话人（面试官/候选人）
    ├─ 去除时间戳
    ├─ 合并连续对话
    └─ 标准化格式
    ↓
[
  {"role": "interviewer", "content": "..."},
  {"role": "candidate", "content": "..."},
  ...
]
    ↓
EvalAgent 评估
    ↓
评估报告（评分 + 分析 + 摘要）
```

### 对话格式示例

**原始格式**:
```
Emma Wang(00:02:19): 老同学能听得见吗？
马智慧(00:02:24): 可以可以听得见...
Emma Wang(00:02:27): 我们都开一下摄像头吧...
```

**清洗后格式**:
```json
[
  {
    "role": "interviewer",
    "content": "老同学能听得见吗？我们都开一下摄像头吧..."
  },
  {
    "role": "candidate", 
    "content": "可以可以听得见..."
  }
]
```

## �🎯 下一步

这些模块将由 `cli/` 目录下的Rich控制台应用统一调用，提供友好的命令行交互界面。

**对话评估** 功能已集成到 CLI 主菜单的第 6 项！ ⭐
````
