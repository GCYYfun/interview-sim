# @buddy/menglong

MengLong (朦胧) TypeScript SDK — 一个统一的 LLM 适配层，旨在提供与 Python 版 MengLong 高度对齐的开发体验。

## 特性

- 🚀 **统一 API**：在 MengLong, OpenAI, 和 DeepSeek 之间无缝切换。
- 🛠️ **工具调用优先**：原生支持 Action/Outcome 结构化工具调用逻辑。
- 🖼️ **多模态支持**：提供简单易用的工厂函数，支持 文本、图片、文档 等多种内容。
- 📊 **结构化日志**：内置 `ConsoleLogger`，可一键开启 LLM 原始请求/响应调试。
- 🌊 **稳定流式响应**：基于 SSE 的流式处理，支持增量工具调用合并。

## 安装

```bash
pnpm add @buddy/menglong
```

## 快速上手

### 1. 简单聊天

```typescript
import { Model } from '@buddy/menglong';

const model = new Model('menglong/deepseek-chat');

const resp = await model.chat(['你好，请问你是？']);
console.log(resp.text);
```

### 2. 管理对话上下文 (Context)

```typescript
import { Model, Context } from '@buddy/menglong';

const model = new Model('openai/gpt-4o');
const ctx = new Context();

ctx.user('谁是第一个登上月球的人？')
   .assistant('是尼尔·阿姆斯特朗。')
   .user('那是什么时候？');

const resp = await model.chat(ctx);
console.log(resp.text);
```

### 3. 流式响应

```typescript
import { Model, User } from '@buddy/menglong';

const model = new Model('menglong/menglong-chat');

for await (const chunk of model.streamChat([User('写一首关于人工智能的诗')])) {
  process.stdout.write(chunk.output?.delta?.text ?? '');
}
```

### 4. 工具调用 (Action/Outcome)

```typescript
import { Model, User, Assistant, Tool } from '@buddy/menglong';

const model = new Model('openai/gpt-4o', {
  tools: [{
    type: 'function',
    function: {
      name: 'get_weather',
      description: '获取当前天气',
      parameters: { /* JSON Schema */ }
    }
  }]
});

// 模型决定调用工具 (Action)
const resp = await model.chat(['东京的天气怎么样？']);
if (resp.tool_calls) {
  const call = resp.tool_calls[0];
  console.log(`调用函数 ${call.name}，参数: ${call.arguments}`);
  
  // 提供执行结果 (Outcome)
  const result = await model.chat([
      User('东京的天气怎么样？'),
      Assistant(undefined, [call]),
      Tool(call.id, '晴天，25°C', 'get_weather')
  ]);
  console.log(result.text);
}
```

## 调试与日志

启用内置日志可以清晰地查看所有原始请求 Payload 和响应内容：

```typescript
import { Model, ConsoleLogger } from '@buddy/menglong';

const model = new Model('menglong/menglong-chat', {
  logger: new ConsoleLogger()
});
```

## 配置文件支持

SDK 会按以下顺序搜索 `.configs.toml` 配置文件：
1. 环境变量：`MENGLONG_CONFIG`
2. 当前目录（向上级目录递增搜索）
3. Home 目录：`~/.configs.toml`

## 开源协议

MIT
