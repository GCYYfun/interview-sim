/**
 * MengLong TypeScript SDK
 * DeepSeek Provider 实现
 * 
 * 继承自 OpenAIProvider
 */

import { ProviderRegistry } from './registry.js';
import { OpenAIProvider, type OpenAIResponse } from './openai.js';
import type { ProviderConfig } from '../../utils/config/config_type.js';
import type { Response, StreamResponse } from '../../schemas/chat.js';

export class DeepSeekProvider extends OpenAIProvider {
  constructor(config: ProviderConfig) {
    const env = typeof process !== 'undefined' ? process.env : {};
    super({
      ...config,
      // 赋予默认的 Base_url 如果 config 里没写
      base_url: config.base_url || env.DEEPSEEK_BASE_URL || 'https://api.deepseek.com',
      api_key: config.api_key || env.DEEPSEEK_API_KEY
    }, 'deepseek');
  }


  protected _normalizeResponse(raw: unknown, modelName: string): Response {
    const data = raw as OpenAIResponse;
    const res = super._normalizeResponse(data, modelName);
    
    // 提取推理内容 (DeepSeek 特有)
    const choice = data.choices?.[0];
    const reasoning = choice?.message?.reasoning_content;
    
    if (reasoning && res.output?.content) {
      res.output.content.reasoning = reasoning;
    }
    
    return res;
  }

  protected _normalizeStreamChunk(raw: unknown, modelName: string): StreamResponse {
    const data = raw as OpenAIResponse;
    const res = super._normalizeStreamChunk(data, modelName);
    
    const choice = data.choices?.[0];
    const reasoning = choice?.delta?.reasoning_content;
    
    if (reasoning && res.output?.delta) {
      res.output.delta.reasoning = reasoning;
    }
    
    return res;
  }
}

// 自动注册
ProviderRegistry.register('deepseek', DeepSeekProvider);


