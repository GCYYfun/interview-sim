# Interview Simulation & Evaluation AI

本项目是一个基于大模型（LLM）的闭环面试系统，包含 **面试模拟器** 和 **面试评估器** 两大核心组件。旨在通过 AI 代理模拟真实的面试场景，并提供深度的面试表现分析。

## 核心功能

1.  **面试模拟 (`simulator.py`)**: 
    - 由面试官（Interviewer）和候选人（Candidate）两个 AI 代理进行自主对话。
    - 支持基于职位描述（JD）和候选人简历（Resume）的个性化面试。
    - 支持流式输出和候选人/面试官的“思考（Thinking）”逻辑展示。
    - 自动保存 MD/JSON 格式的完整对话记录。
2.  **面试评估 (`evaluator.py`)**: 
    - 对生成的模拟记录或真实面试文本进行深度分析。
    - 包含**主题拆解**（Topic Analysis）和**多维度能力评估**（Evaluation Report）。
    - 支持交互式选择、批量过滤和临时调试模式。

## 环境配置

本项目使用 `uv` 进行高效的包管理。

### 安装依赖

```bash
# 1. 克隆项目后，安装依赖
uv sync

# 2. 激活虚拟环境 (可选)
source .venv/bin/activate
```

### 更新 menglong sdk
```bash
uv add --refresh --upgrade-package menglong https://github.com/gcyyfun/menglong.git
```

### SDK 配置

项目已迁移至 `menglong` SDK。请在项目根目录创建 `.configs.toml` 并配置 Provider 密钥：

```toml
[default]
model_id = "menglong/global.anthropic.claude-sonnet-4-5-20250929-v1:0"

[providers.menglong]
base_url = "YOUR_BASE_URL"
api_key = "YOUR_API_KEY"
# 其他 Provider 配置...
```

---

## 快速开始

### 1. 运行模拟面试 (Simulator)

#### 基础示例：自主面试
```bash
uv run simulator.py --jd rm_jd --resume zhangsan_resume
```

#### 进阶示例：参考模式 (Guided)
面试官会参考已有录音转文字记录来引导提问。
```bash
uv run simulator.py --jd rm_jd --resume zhangsan_resume --transcript zhangsan_rm_transcript_1
```

#### 调试示例：模型竞技与长文面试
```bash
uv run simulator.py --jd rm_jd --resume zhangsan_resume \
  --interviewer-model "menglong/global.anthropic.claude-sonnet-4-5-20250929-v1:0" \
  --max-turns 40
```

---

### 2. 运行面试评估 (Evaluator)

#### 基础示例：交互式选择
```bash
uv run evaluator.py
```

#### 进阶示例：特定候选人与条件过滤
```bash
# 针对特定记录批量运行
uv run evaluator.py --name zhangsan_rm_transcript_1

# 仅评估特定岗位记录
uv run evaluator.py --jd rm --candidate zhangsan
```

#### 调试示例：仅主题分析与调试模式
```bash
# 仅生成主题大纲，不评分
uv run evaluator.py --name zhangsan_rm_transcript_1 --step topic

# 调试提示词逻辑，不覆盖正式报告
uv run evaluator.py --name zhangsan_rm_transcript_1 --temp --force
```

---

## 项目目录结构

- `agents/`: AI 代理实现（面试官、候选人、评估专家）
- `simulation/`: 面试模拟环境协调逻辑
- `components/`: 文件解析、数据管理等核心组件
- `data/resources/`: 输入源 (JD, Resume, Conversations)
- `data/generated/simulations/`: `simulator.py` 生成的 MD/JSON 记录
- `data/generated/reports/`: `evaluator.py` 生成的评估报告
- `prompts/`: 提示词管理逻辑

## 进阶技巧
- **Quiet 模式**: 使用 `--quiet` 在后台静默运行模拟。
- **Temp 模式**: 使用 `--temp` 将结果保存到 `data/generated/temp/`，带时间戳，适合实验。

