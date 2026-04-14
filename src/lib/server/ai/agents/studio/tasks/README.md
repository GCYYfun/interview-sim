# Agent 任务数据组织

Agent数据现已按任务类型组织，每个任务包含三个核心文件。

## 目录结构

```
src/lib/server/ai/agents/studio/
├── tasks/
│   ├── benchmark/              # 基准测试任务
│   │   ├── system.md          # 系统提示
│   │   ├── knowledge.md       # 知识库
│   │   └── task.md            # 任务指南
│   │
│   └── interview-assessment/  # 面试评估任务
│       ├── system.md          # 系统提示
│       ├── knowledge.md       # 知识库
│       └── task.md            # 任务指南
│
├── index.ts                    # Agent执行逻辑
└── types.ts                    # 类型定义
```

## 文件说明

### system.md
- 定义Agent的角色和职责
- 设定评估维度和标准
- 明确输出格式要求

### knowledge.md
- 领域知识和最佳实践
- 评估原则和方法
- 行业标准和规范

### task.md
- 具体任务执行指南
- 数据获取流程
- 分析方法和步骤

## 添加新任务

要添加新的Agent任务，只需：

1. 在 `tasks/` 下创建新目录，例如 `tasks/my-task/`
2. 创建三个文件：`system.md`, `knowledge.md`, `task.md`
3. 在 `types.ts` 的 `StudioAgentKey` 中添加新的Agent类型
4. 在 `index.ts` 的 `taskTypeMap` 中映射新类型到目录名称

这样的结构使得Agent数据：
- ✅ 按任务类型清晰组织
- ✅ 易于维护和扩展
- ✅ 便于版本控制
- ✅ 支持快速添加新Agent
