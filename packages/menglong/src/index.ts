/**
 * MengLong TypeScript SDK
 * 
 * 统一 LLM 接入层，支持多 Provider 适配。
 * 
 * @example
 * ```typescript
 * import { Model, User, Context } from '@buddy/menglong';
 * 
 * const model = new Model('menglong/gemini-2.0-flash');
 * 
 * // 简单字符串请求
 * const resp = await model.chat(['你好！']);
 * console.log(resp.text);
 * 
 * // 使用 Context 管理历史
 * const ctx = new Context();
 * ctx.user('给我讲个笑话').assistant('...').user('再讲一个');
 * const resp2 = await model.chat(ctx);
 * 
 * // 流式响应
 * for await (const chunk of model.streamChat([User('写首诗')])) {
 *   process.stdout.write(chunk.output?.delta?.text ?? '');
 * }
 * ```
 */

// Logger
export { type MengLongLogger, ConsoleLogger, silentLogger } from './utils/logger.js';

// Schemas
export * from './schemas/chat.js';
export * from './schemas/tool.js';

// Config
export type * from './utils/config/config_type.js';
export { loadConfig, parseConfig, loadConfigFromFile } from './utils/config/config_loader.js';

// Model
export { Model, type ModelOptions } from './models/model.js';

// Providers (advanced usage)
export { BaseProvider } from './models/providers/base.js';
export { ProviderRegistry } from './models/providers/registry.js';
export { MengLongProvider } from './models/providers/menglong.js';
export { OpenAIProvider, type OpenAIResponse, type OpenAIChoice } from './models/providers/openai.js';
export { DeepSeekProvider } from './models/providers/deepseek.js';

