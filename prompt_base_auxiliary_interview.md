# Auxiliary interview

##  Data

Candidate
    - Background Information

Job Description
    - Overview
    - Requirements
    - Responsibilities

Interviewer
    - HR interviewer
    - Professional interviewer

Indicator
    - Smartness
    - Resilience
    - Diligence

## Prompt Processor and Get Interview Insights

```
请基于以下面试数据，提取HR在面试中识别候选人"聪明度"、"皮实"和"勤奋"这三个指标的经验技巧。主要体现为提问技巧、追问策略和评估要点。

## 背景信息

### 候选人简历
{resume}

### 岗位JD
{jd}

### 面试对话
{conversation}

### HR评价
{evaluation}

## 任务要求

请分析这次面试中HR是如何通过提问和互动来评估候选人的：
1. **聪明度** - 逻辑思维、学习能力、问题分析能力
2. **皮实** - 抗压能力、韧性、面对困难的态度
3. **勤奋** - 工作热情、主动性、持续学习意愿

请从以下角度提取经验：

### 1. 有效提问技巧
- 针对聪明度的提问方式和角度
- 针对皮实的提问方式和角度  
- 针对勤奋的提问方式和角度

### 2. 关键追问策略
- 当候选人回答不够深入时的追问技巧
- 如何通过追问挖掘真实能力
- 什么样的回答需要进一步验证

### 3. 评估要点
- 每个指标的关键观察点
- 优秀回答的特征
- 需要警惕的回答模式

### 4. 适用场景
- 这些技巧适合什么类型的岗位
- 什么背景的候选人
- 什么阶段使用

请用结构化的方式输出，包含具体的问题示例和判断标准。
"""
```

Input ： data

process each conversation in data to extract interview insights

summarize each case insights into a general experience

continue update, ( may be in graph structure)

### Use Interview Insights to Generate Targeted Interview Questions and Evaluation Criteria

message = {
    Q: Experience,
    A: Interview Questions
}

Q = LLM( messages )