/**
 * MengLong TypeScript SDK
 * MengLong Provider 实现
 * 
 * 对应 Python 版 menglong/models/providers/menglong.py
 * 使用原生 fetch API，支持 SSE 格式的流式响应。
 */

import { BaseProvider } from './base.js';
import { ProviderRegistry } from './registry.js';
import type { Message, Response, StreamResponse, AnyContentPart, Outcome } from '../../schemas/chat.js';
import { createResponse } from '../../schemas/chat.js';
import type { ProviderConfig } from '../../utils/config/config_type.js';
import type { ToolInfo } from '../../schemas/tool.js';

export class MengLongProvider extends BaseProvider {

  private readonly baseUrl: string;
  private readonly apiKey: string | undefined;
  private readonly timeout: number;

  constructor(config: ProviderConfig) {
    super(config, 'menglong');

    const env = typeof process !== 'undefined' ? process.env : {};
    this.baseUrl = (config.base_url as string | undefined ?? env.MENGLONG_BASE_URL ?? 'http://localhost:8000/menglong').replace(/\/$/, '');
    this.apiKey = (config.api_key as string | undefined) || env.MENGLONG_API_KEY;
    this.timeout = (config.timeout as number | undefined) ?? 60;
  }


  // ==================== 请求头 ====================

  private _getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  // ==================== 生命周期钩子 ====================

  protected _convertMessages(messages: Message[]): unknown[] {
    return messages.map((msg) => {
      let content = msg.content;

      if (Array.isArray(content)) {
        // 过滤掉 content part 中的 undefined 字段，确保序列化干净
        const cleaned = content.map((part: unknown) => {
          return Object.fromEntries(
            Object.entries(part as Record<string, unknown>).filter(([, v]) => v !== undefined)
          ) as unknown;
        });

        // 对于 assistant 消息，如果全部是纯文本 part，合并为字符串
        // （OpenAI-compat 网关通常期望 assistant content 为字符串）
        if (msg.role === 'assistant') {
          const allText = cleaned.every((p: unknown) => (p as Record<string, unknown>).type === 'text');
          if (allText) {
            content = cleaned.map((p: unknown) => (p as Record<string, unknown>).text as string).join('\n') as unknown as typeof content;
          } else {
            content = cleaned as unknown as typeof content;
          }
        } else {
          content = cleaned as unknown as typeof content;
        }
      }

      const m: Record<string, unknown> = {
        role: msg.role,
        content: content,
      };

      if (msg.role === 'tool' && Array.isArray(content)) {
        const outcome = (content as AnyContentPart[]).find((p: AnyContentPart): p is Outcome => p.type === 'outcome');
        if (outcome) {
          m['tool_id'] = outcome.id;
        }
      }
      return m;
    });
  }

  protected _normalizeResponse(raw: unknown, modelName: string): Response {
    const data = raw as Record<string, unknown>;

    // 如果响应已包含 SDK 格式的 output 字段，直接使用
    if (data.output !== undefined) {
      return createResponse(raw as Response);
    }

    // 否则当作 OpenAI-compat 格式处理（MengLong 网关可能透传 OpenAI 格式）
    type OAIChoice = {
      message?: {
        content?: string;
        tool_calls?: Array<{ id: string; type: string; function: { name: string; arguments: string } }>;
      };
      finish_reason?: string;
    };
    type OAIUsage = { prompt_tokens: number; completion_tokens: number; total_tokens: number };
    const choices = data.choices as OAIChoice[] | undefined;
    const choice = choices?.[0];
    const message = choice?.message;
    const usage = data.usage as OAIUsage | undefined;

    return createResponse({
      model: (data.model as string | undefined) || modelName,
      usage: usage ? {
        input_tokens: usage.prompt_tokens,
        output_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
      } : undefined,
      output: {
        content: message?.content ? { text: message.content } : undefined,
        actions: message?.tool_calls?.map((tc) => ({
          type: 'action' as const,
          id: tc.id,
          name: tc.function?.name,
          arguments: tc.function?.arguments ? (() => {
            try { return JSON.parse(tc.function.arguments) as Record<string, unknown>; }
            catch { return { _raw: tc.function.arguments } as Record<string, unknown>; }
          })() : {}
        })),
        status: choice?.finish_reason
      }
    });
  }

  protected _normalizeStreamChunk(raw: unknown, _model: string): StreamResponse {
    void _model;
    const chunk = raw as Record<string, unknown>;

    // 简化：直接返回，像 Python 版本一样
    // MengLong 网关应该返回 SDK 原生格式
    return chunk as StreamResponse;
  }



  protected _convertTools(tools: ToolInfo[]): unknown[] {
    // MengLong 使用标准格式，无需转换
    return tools;
  }

  // ==================== 核心接口 ====================

  async chat(messages: Message[], model: string, kwargs: Record<string, unknown> = {}): Promise<Response> {
    const params = this._convertParams(model, kwargs);

    if (params.tools) {
      params.tools = this._convertTools(params.tools as ToolInfo[]);
    }

    const payload = {
      model,
      messages: this._convertMessages(messages),
      ...params,
    };

    this.logger.debug(`[MengLongProvider.chat()] Payload:`, payload);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout * 1000);

    try {
      const res = await fetch(`${this.baseUrl}/chat`, {
        method: 'POST',
        headers: this._getHeaders(),
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      clearTimeout(timer);

      if (!res.ok) {
        const text = await res.text().catch(() => '');
        this.logger.error(`[MengLongProvider.chat()] HTTP ${res.status}: ${text}`);
        throw new Error(`MengLong HTTP ${res.status}: ${text}`);
      }

      const data = await res.json();
      return this._normalizeResponse(data, model);
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        throw new Error(`MengLong request timed out after ${this.timeout}s`);
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }

  async *streamChat(messages: Message[], model: string, kwargs: Record<string, unknown> = {}): AsyncGenerator<StreamResponse> {
    const params = this._convertParams(model, kwargs);

    if (params.tools) {
      params.tools = this._convertTools(params.tools as ToolInfo[]);
    }

    const payload = {
      model,
      messages: this._convertMessages(messages),
      stream: true,
      ...params,
    };

    this.logger.debug(`[MengLongProvider.streamChat()] Payload:`, payload);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout * 1000);

    try {
      const res = await fetch(`${this.baseUrl}/chat`, {
        method: 'POST',
        headers: this._getHeaders(),
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      clearTimeout(timer);

      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`MengLong HTTP ${res.status}: ${text}`);
      }

      if (!res.body) throw new Error('No response body for streaming');

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          if (trimmed.startsWith('data: ')) {
            const dataStr = trimmed.slice(6);
            if (dataStr === '[DONE]') return;

            try {
              const chunk = JSON.parse(dataStr) as Record<string, unknown>;

              // 简化：直接调用 normalize，像 Python 版本一样
              // MengLong 网关应该返回 SDK 原生格式
              const normalized = this._normalizeStreamChunk(chunk, model);

              // 调试日志
              // this.logger.debug(`[MengLongProvider.streamChat] normalized chunk:`, {
              //   hasOutput: !!normalized.output,
              //   deltaKeys: Object.keys(normalized.output?.delta || {}),
              //   outputEnd: normalized.output?.end,
              //   actionsCount: normalized.output?.actions?.length
              // });

              yield normalized;
            } catch (error) {
              // 跳过可能的 JSON 片段解析错误
              this.logger.debug(`[MengLongProvider.streamChat] JSON parse error:`, error);
            }
          }
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        throw new Error(`MengLong stream timed out after ${this.timeout}s`);
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }
}

// 自动注册
ProviderRegistry.register('menglong', MengLongProvider);
