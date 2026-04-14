/**
 * MengLong TypeScript SDK
 * OpenAI Provider 实现
 * 
 * 对应 Python 版 menglong/models/providers/openai.py
 */

import { BaseProvider } from './base.js';
import { ProviderRegistry } from './registry.js';
import type { Message, Response, StreamResponse, AnyContentPart, Action, Outcome } from '../../schemas/chat.js';
import { createResponse } from '../../schemas/chat.js';
import type { ProviderConfig } from '../../utils/config/config_type.js';
import type { ToolInfo } from '../../schemas/tool.js';

export interface OpenAIChoice {
  message?: {
    content?: string;
    reasoning_content?: string;
    tool_calls?: Array<{
      id: string;
      type: 'function';
      function: {
        name: string;
        arguments: string;
      };
    }>;
  };
  delta?: {
    content?: string;
    reasoning_content?: string;
    tool_calls?: Array<{
      index: number;
      id?: string;
      function?: {
        name?: string;
        arguments?: string;
      };
    }>;
  };
  finish_reason?: string;
}

export interface OpenAIResponse {
  model: string;
  choices: OpenAIChoice[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export class OpenAIProvider extends BaseProvider {
  protected readonly baseUrl: string;
  protected readonly apiKey: string | undefined;
  protected readonly timeout: number;

  constructor(config: ProviderConfig, name = 'openai') {
    super(config, name);
    const env = typeof process !== 'undefined' ? process.env : {};
    this.baseUrl = (config.base_url as string | undefined ?? env.OPENAI_BASE_URL ?? 'https://api.openai.com/v1').replace(/\/$/, '');
    this.apiKey = (config.api_key as string | undefined) || env.OPENAI_API_KEY;
    this.timeout = (config.timeout as number | undefined) ?? 60;
  }


  // ==================== 请求头 ====================

  protected _getHeaders(): Record<string, string> {
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
      const content = msg.content;
      let serializedContent: unknown;
      let toolCalls: unknown[] | undefined;

      if (Array.isArray(content)) {
        if (msg.role === 'assistant') {
          // OpenAI 要求 assistant 的工具调用放在外部 tool_calls 字段
          const textParts: string[] = [];
          const calls: unknown[] = [];
          
          for (const part of content) {
            if (part.type === 'text') {
              textParts.push(part.text);
            } else if (part.type === 'action') {
              const action = part as Action;
              calls.push({
                id: action.id || `call_${Math.random().toString(36).slice(2, 10)}`,
                type: 'function',
                function: {
                  name: action.name,
                  arguments: typeof action.arguments === 'string' 
                    ? action.arguments 
                    : JSON.stringify(action.arguments || {})
                }
              });
            }
          }
          
          if (calls.length > 0) {
            toolCalls = calls;
            // 如果有工具调用且没文本，content 设为 null
            serializedContent = textParts.length > 0 ? textParts.join('\n') : null;
          } else {
            serializedContent = textParts.join('\n');
          }
        } else if (msg.role === 'tool') {
          // OpenAI 要求 tool 角色的内容必须是字符串（结果）
          const results: string[] = [];
          for (const part of content) {
            if (part.type === 'outcome') {
              results.push(part.result);
            } else if (part.type === 'text') {
              results.push(part.text);
            }
          }
          serializedContent = results.join('\n');
        } else {
          // 常规多模态消息 (User/System)
          serializedContent = content.map((part: AnyContentPart) => {
            return Object.fromEntries(
              Object.entries(part).filter(([, v]) => v !== undefined),
            );
          });
        }
      } else {
        serializedContent = content;
      }

      const m: Record<string, unknown> = {
        role: msg.role,
        content: serializedContent,
      };

      if (toolCalls) {
        m['tool_calls'] = toolCalls;
      }

      if (msg.role === 'tool' && Array.isArray(content)) {
        const outcome = content.find((p: AnyContentPart): p is Outcome => p.type === 'outcome');
        if (outcome) {
          m['tool_call_id'] = outcome.id;
        }
      }
      return m;
    });
  }


  protected _normalizeResponse(raw: unknown, modelName: string): Response {
    const data = raw as OpenAIResponse;
    const choice = data.choices?.[0];
    const message = choice?.message;
    
    return createResponse({
      model: data.model || modelName,
      usage: data.usage ? {
        input_tokens: data.usage.prompt_tokens,
        output_tokens: data.usage.completion_tokens,
        total_tokens: data.usage.total_tokens,
      } : undefined,
      output: {
        content: message?.content ? { text: message.content } : undefined,
        actions: message?.tool_calls?.map((tc) => ({
          type: 'action' as const,
          id: tc.id,
          name: tc.function?.name,
          arguments: tc.function?.arguments ? (() => { 
            try { return JSON.parse(tc.function.arguments); } 
            catch { return { _raw: tc.function.arguments }; } 
          })() : {}
        })),
        status: choice?.finish_reason
      }
    });
  }

  protected _normalizeStreamChunk(raw: unknown, modelName: string): StreamResponse {
    const data = raw as OpenAIResponse;
    const choice = data.choices?.[0];
    const delta = choice?.delta;
    
    return {
      model: data.model || modelName,
      output: {
        delta: {
          text: delta?.content || undefined,
          reasoning: delta?.reasoning_content || undefined,
        },
        end: choice?.finish_reason || undefined
      },
      usage: data.usage ? {
        input_tokens: data.usage.prompt_tokens,
        output_tokens: data.usage.completion_tokens,
        total_tokens: data.usage.total_tokens,
      } : undefined,
    };
  }

  protected _convertTools(tools: ToolInfo[]): unknown[] {
    return tools.map(t => {
      // 如果已经是嵌套格式，直接返回
      if (t.type === 'function' && t.function) return t;
      // 否则异常，Model._ensureTools 应该已经处理好
      return t;
    });
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

    this.logger.debug(`[OpenAIProvider.chat()] Payload:`, payload);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout * 1000);

    try {
      const res = await fetch(`${this.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: this._getHeaders(),
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      clearTimeout(timer);

      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`OpenAI HTTP ${res.status}: ${text}`);
      }

      const data = await res.json();
      return this._normalizeResponse(data, model);
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

    this.logger.debug(`[OpenAIProvider.streamChat()] Payload:`, payload);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout * 1000);

    try {
      const res = await fetch(`${this.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: this._getHeaders(),
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      clearTimeout(timer);

      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`OpenAI HTTP ${res.status}: ${text}`);
      }

      if (!res.body) throw new Error('No response body for streaming');

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      // OpenAI 流式 tool_calls 是增量形式
      const accToolCalls: Record<number, {
        id: string;
        name: string;
        arguments: string;
      }> = {};

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
              const chunk = JSON.parse(dataStr) as OpenAIResponse;
              const choice = chunk.choices?.[0];
              const delta = choice?.delta;
              const finishReason = choice?.finish_reason;

              if (delta?.tool_calls) {
                for (const tc of delta.tool_calls) {
                  const idx = tc.index ?? 0;
                  if (!accToolCalls[idx]) {
                    accToolCalls[idx] = { id: tc.id || '', name: tc.function?.name || '', arguments: '' };
                  }
                  if (tc.id) accToolCalls[idx].id = tc.id;
                  if (tc.function?.name) accToolCalls[idx].name = tc.function.name;
                  if (tc.function?.arguments) accToolCalls[idx].arguments += tc.function.arguments;
                }
                continue;
              }

              if (finishReason === 'tool_calls') {
                const actions = Object.values(accToolCalls).map(tc => ({
                  type: 'action' as const,
                  id: tc.id,
                  name: tc.name,
                  arguments: tc.arguments ? (() => { 
                    try { return JSON.parse(tc.arguments); } 
                    catch { return { _raw: tc.arguments }; } 
                  })() : {}
                })) as Action[];
                yield {
                  model: chunk.model || model,
                  output: { end: 'tool_calls', actions },
                };
                return;
              }

              yield this._normalizeStreamChunk(chunk, model);
            } catch {
              // Ignore partial JSON
            }
          }
        }
      }
    } finally {
      clearTimeout(timer);
    }
  }
}

// 自动注册
ProviderRegistry.register('openai', OpenAIProvider);
