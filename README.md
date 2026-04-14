# HuShi (智能面试评估工作台)

HuShi 是一款基于 LLM 处理面试相关材料（简历、面试对话、职位描述）的智能工作台。它能够通过多阶段流水线对候选人进行深度分析、匹配度评估并生成结构化的评估报告。

## 核心特性

- **多阶段 AI 流水线**：从初步 Job Fit 分析到最终综合评估报告自动生成。
- **面试评估大脑**：利用 DeepSeek/Gemini 等大模型深入理解复杂的面试对话。
- **中断恢复机制**：支持 Pipeline 的断点续跑，确保复杂任务的稳定性。
- **实时控制台**：支持流式输出 AI 思考过程和工具调用日志。
- **调试模式**：内置 `HUSHI_LLM_DEBUG` 模式，可实时追踪后台 Agent 执行细节。

---

## 快速启动

### 1. 安装依赖

推荐使用 `pnpm` 安装环境：

```bash
pnpm install
```

### 2. 配置环境变量

复制 `.env.example` 并重命名为 `.env`，填入必要的 API Key 和模型配置：

```bash
cp .env.example .env
```

核心配置项说明：
- `DATABASE_URL`: 本地 SQLite 路径（默认为 `file:local.db`）。
- `BETTER_AUTH_SECRET`: 用户认证密钥（可以使用 `openssl rand -base64 32` 生成）。
- `MENGLONG_API_KEY`: 访问底层 LLM 服务的密钥。
- `HUSHI_LLM_DEBUG`: 设置为 `true` 可在前端展示 DEBUG 按钮。

### 3. 初始化数据库

本项目使用 Drizzle ORM，首次启动前需要将 Schema 同步到本地数据库文件：

```bash
npm run db:push
```

### 4. 启动开发服务器

```bash
npm run dev
```

现在访问 [http://localhost:5173](http://localhost:5173) 即可开启。

---

## 用户账户

项目目前预设了以下测试账号（密码同用户名），在进入登录页后系统会自动完成初始化：

- `test_user_4141`
- `test_user_4142`
- `test_user_4143`

## 技术栈

- **Frontend**: SvelteKit 5 (Runes), TailwindCSS
- **Backend**: SvelteKit Server Actions, Better-Auth
- **ORM**: Drizzle ORM (LibSQL/SQLite)
- **AI Engine**: Pipeline Runner + MengLong SDK
