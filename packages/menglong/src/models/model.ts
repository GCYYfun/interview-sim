/**
 * MengLong TypeScript SDK
 * Model - 统一模型门面 (Facade)
 * 
 * 对应 Python 版 menglong/models/model.py
 * 
 * @example
 * ```typescript
 * import { Model, User, ConsoleLogger } from '@buddy/menglong';
 * 
 * const model = new Model('menglong/gemini-2.0-flash', {
 *   logger: new ConsoleLogger()
 * });
 * const response = await model.chat([User('你好！')]);
 * console.log(response.text);
 * ```
 */

import type { Message, Response, StreamResponse } from '../schemas/chat.js';
import { Context } from '../schemas/chat.js';
import type { ToolInfo } from '../schemas/tool.js';
import type { Config, ProviderConfig } from '../utils/config/config_type.js';
import { loadConfig } from '../utils/config/config_loader.js';
import { ProviderRegistry } from './providers/registry.js';
import { BaseProvider } from './providers/base.js';
import type { MengLongLogger } from '../utils/logger.js';

import { silentLogger } from '../utils/logger.js';

// 确保内置 providers 被注册
import './providers/menglong.js';
import './providers/openai.js';
import './providers/deepseek.js';

/** 模型初始化选项 */
export interface ModelOptions {
  /** 配置文件路径 (可选，默认自动搜索) */
  configPath?: string;
  /** 直接传入配置对象 (优先级最高) */
  config?: Config;
  /** 日志记录器 (可选) */
  logger?: MengLongLogger;
}

export class Model {
  private readonly config: Config;
  private readonly logger: MengLongLogger;
  private readonly defaultModelId: string;
  private readonly providerCache = new Map<string, BaseProvider>();

  constructor(defaultModelId?: string, options: ModelOptions = {}) {
    this.logger = options.logger ?? silentLogger;
    this.config = options.config ?? loadConfig(options.configPath);
    this.defaultModelId = defaultModelId ?? this.config.default?.model_id ?? '';
    
    this.logger.debug(`Model initialized with defaultModelId: ${this.defaultModelId}`);
  }

  // ==================== 内部工具方法 ====================

  private _parseModelId(modelId: string): [string, string] {
    if (!modelId || !modelId.includes('/')) {
      throw new Error(`Invalid model_id format: '${modelId}'. Expected 'provider/model-name'.`);
    }
    const idx = modelId.indexOf('/');
    return [modelId.slice(0, idx), modelId.slice(idx + 1)];
  }

  private _getProviderAndModel(modelOverride?: string): [BaseProvider, string] {
    const targetId = modelOverride ?? this.defaultModelId;
    if (!targetId) {
      throw new Error('No model specified and no default_model_id set in config.');
    }

    const [providerName, modelName] = this._parseModelId(targetId);

      if (!this.providerCache.has(providerName)) {
        this.logger.debug(`Initializing provider: ${providerName}`);
        const provider = ProviderRegistry.getInstance(providerName, this.config);
        // 为 Provider 注入 logger
        if (provider instanceof BaseProvider) {
          provider.setLogger(this.logger);
        }
        this.providerCache.set(providerName, provider);
      }


    return [this.providerCache.get(providerName)!, modelName];
  }

  private _ensureMessages(messages: (Message | string)[] | Context): Message[] {
    if (messages instanceof Context) {
      return messages.toArray();
    }
    return messages.map((m) => {
      if (typeof m === 'string') return { role: 'user' as const, content: m };
      return m;
    });
  }

  private _ensureTools(tools?: unknown[]): ToolInfo[] | undefined {
    if (!tools || tools.length === 0) return undefined;
    
    return tools.map((t) => {
      if (t && typeof t === 'object') {
        const toolObj = t as Record<string, unknown>;
        
        // 1. 如果带有 handler 且是 ToolDefinition 格式，调用 schema() (适配 agent-core)
        if (typeof toolObj.handler === 'function' && typeof toolObj.schema === 'function') {
          return (toolObj as unknown as { schema(): ToolInfo }).schema();
        }
        
        // 2. 如果已经符合标准的 ToolInfo 格式 (嵌套结构: {type:'function', function:{...}})
        if (toolObj.type === 'function' && toolObj.function) {
          return toolObj as unknown as ToolInfo;
        }

        
        // 3. 如果是扁平结构 (agent-core 极简格式)，进行转换
        if (toolObj.name && toolObj.parameters) {
          return {
            type: 'function' as const,
            function: {
              name: String(toolObj.name),
              description: String(toolObj.description || ''),
              parameters: toolObj.parameters,
            }
          } as ToolInfo;
        }
      }
      
      return t as unknown as ToolInfo;
    });

  }

  // ==================== 核心接口 ====================

  /** 异步聊天请求 */
  async chat(
    messages: (Message | string)[] | Context,
    model?: string,
    kwargs: Record<string, unknown> = {},
  ): Promise<Response> {
    const [provider, modelName] = this._getProviderAndModel(model);
    const msgs = this._ensureMessages(messages);

    if (kwargs.tools) {
      kwargs = { ...kwargs, tools: this._ensureTools(kwargs.tools as unknown[]) };
    }

    this.logger.debug(`[Model.chat()] provider: ${provider.constructor.name}, model: ${modelName}`);
    return provider.chat(msgs, modelName, kwargs);
  }

  /** 异步流式聊天请求 */
  async *streamChat(
    messages: (Message | string)[] | Context,
    model?: string,
    kwargs: Record<string, unknown> = {},
  ): AsyncGenerator<StreamResponse> {
    const [provider, modelName] = this._getProviderAndModel(model);
    const msgs = this._ensureMessages(messages);

    if (kwargs.tools) {
      kwargs = { ...kwargs, tools: this._ensureTools(kwargs.tools as unknown[]) };
    }

    this.logger.debug(`[Model.streamChat()] provider: ${provider.constructor.name}, model: ${modelName}`);
    yield* provider.streamChat(msgs, modelName, kwargs);
  }

  /** 向量嵌入请求 */
  async embed(texts: string[], model?: string, kwargs: Record<string, unknown> = {}): Promise<unknown> {
    const [provider, modelName] = this._getProviderAndModel(model);
    return provider.embed(texts, modelName, kwargs);
  }

  /** 注册额外 Provider */
  static registerProvider(name: string, providerClass: new (config: ProviderConfig) => BaseProvider): void {
    ProviderRegistry.register(name, providerClass);
  }

  static listAvailableProviders(): string[] {
    return ProviderRegistry.listProviders();
  }
}

