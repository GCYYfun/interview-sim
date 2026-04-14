/**
 * MengLong TypeScript SDK
 * BaseProvider 抽象基类
 * 
 * 对应 Python 版 menglong/models/providers/base.py
 */

import type { Message, Response, StreamResponse } from '../../schemas/chat.js';
import type { ProviderConfig, ModelConfig } from '../../utils/config/config_type.js';
import type { ToolInfo } from '../../schemas/tool.js';
import type { MengLongLogger } from '../../utils/logger.js';
import { silentLogger } from '../../utils/logger.js';

export abstract class BaseProvider {
  protected readonly config: ProviderConfig;
  protected readonly providerName: string;
  protected logger: MengLongLogger = silentLogger;

  constructor(config: ProviderConfig, providerName: string) {
    this.config = config;
    this.providerName = providerName;
  }

  /** 为 Provider 注入 Logger */
  public setLogger(logger: MengLongLogger): void {
    this.logger = logger;
  }

  // ==================== 生命周期钩子（子类实现） ====================

  /** [由内向外] 将 SDK 内部 Message 转换为供应商特定格式 */
  protected abstract _convertMessages(messages: Message[]): unknown;

  /** [由外向内] 将供应商原始响应归一化为 SDK 内部 Response */
  protected abstract _normalizeResponse(raw: unknown, model: string): Response;

  /** [由外向内] 将供应商流式碎片归一化为 SDK 内部 StreamResponse */
  protected abstract _normalizeStreamChunk(raw: unknown, model: string): StreamResponse;

  /** [由内向外] 将标准化工具列表转换为供应商特定格式 */
  protected abstract _convertTools(tools: ToolInfo[]): unknown;

  /**
   * [由内向外] 统一转换控制参数
   * 1. 从配置加载模型默认参数
   * 2. 合并运行时传入的覆盖参数
   * 3. 过滤 undefined 值
   */
  protected _convertParams(model: string, kwargs: Record<string, unknown> = {}): Record<string, unknown> {
    const params: Record<string, unknown> = {};

    // 从 config.models[model] 加载静态默认参数
    const modelConfig = this.config.models?.[model] as ModelConfig | undefined;
    if (modelConfig) {
      for (const [k, v] of Object.entries(modelConfig)) {
        if (v !== undefined && v !== null) params[k] = v;
      }
    }

    // 合并运行时参数
    for (const [k, v] of Object.entries(kwargs)) {
      if (v !== undefined && v !== null) params[k] = v;
    }

    return params;
  }

  // ==================== 核心能力接口（子类实现） ====================

  abstract chat(messages: Message[], model: string, kwargs?: Record<string, unknown>): Promise<Response>;

  abstract streamChat(messages: Message[], model: string, kwargs?: Record<string, unknown>): AsyncGenerator<StreamResponse>;

  /** 向量嵌入（可选，不支持时抛出） */
  embed(texts: string[], model: string, kwargs: Record<string, unknown> = {}): Promise<unknown> {
    void texts; void model; void kwargs;
    throw new Error(`Provider '${this.providerName}' does not support embeddings.`);
  }

}

