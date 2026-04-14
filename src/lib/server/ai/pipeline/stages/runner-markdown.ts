import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { Model, Context, MengLongProvider, ProviderRegistry, ConsoleLogger } from '$menglong';
import { env } from '$env/dynamic/private';
import type { Config } from '$menglong';
import type { PipelineEvent } from '../types';
import { debugLog, log } from '../debug-log';

// pipeline 额外支持 infinigence 前缀，均走 menglong 网关
if (!ProviderRegistry.getProviderClass('infinigence')) {
	log('[runner-markdown] 注册 infinigence provider');
	ProviderRegistry.register('infinigence', MengLongProvider);
} else {
	log('[runner-markdown] infinigence provider 已注册');
}

export function createPipelineModel() {
	const modelId = env.PIPELINE_LLM_MODEL?.trim() || env.HUSHI_LLM_MODEL?.trim() || 'infinigence/deepseek-v3.2-thinking';
	const apiKey = env.MENGLONG_API_KEY?.trim() || undefined;
	const baseUrl = env.MENGLONG_BASE_URL?.trim() || 'http://localhost:8000/menglong';
	const config: Config = {
		default: { model_id: modelId },
		providers: {
			menglong:    { base_url: baseUrl, api_key: apiKey, timeout: 600 },
			anthropic:   { base_url: baseUrl, api_key: apiKey, timeout: 600 },
			infinigence: { base_url: baseUrl, api_key: apiKey, timeout: 600 }
		}
	};
	debugLog('createPipelineModel', { modelId, baseUrl });
	// 只有 HUSHI_LLM_DEBUG=true 时才启用 SDK 日志
	const isDebug = process.env.HUSHI_LLM_DEBUG === 'true';
	const logger = isDebug ? new ConsoleLogger() : undefined;
	return new Model(modelId, {
		config,
		logger
	});
}

const PROMPT_DIR = resolve('prompt');
const promptCache = new Map<string, string>();

export async function loadPrompt(filename: string): Promise<string> {
	if (!promptCache.has(filename)) {
		const content = await readFile(resolve(PROMPT_DIR, filename), 'utf-8');
		promptCache.set(filename, content);
	}
	return promptCache.get(filename)!;
}

/** 替换 prompt 中的 {{PIPELINE_XXX}} 占位符 */
export function fillPlaceholders(template: string, vars: Record<string, string>): string {
	return Object.entries(vars).reduce(
		(text, [key, value]) => text.replaceAll(`{{${key}}}`, value),
		template
	);
}

/**
 * Markdown输出模式 stage runner：单轮流式调用，直接输出Markdown文本，不要求工具调用。
 *
 * 模型在单轮流式调用中完成分析，可输出 thinking，最后直接输出Markdown文本。
 */
export async function* runStageMarkdown(
	promptFile: string,
	vars: Record<string, string>,
	promptDir = 'pipeline/basic',
	stageId: import('../types').PipelineStageId = 'stage2'
): AsyncGenerator<PipelineEvent, string> {
	yield { type: 'status', message: '准备数据...' };

	const template = await loadPrompt(`${promptDir}/${promptFile}`);
	// 直接使用填充后的模板，不添加工具调用指令
	const userMessage = fillPlaceholders(template, vars);

	const systemMessage = '你是专业的结构化面试评估专家。严格按照用户消息中的规则和数据进行分析，直接输出完整的Markdown评估报告。';

	const model = createPipelineModel();
	const context = new Context();
	context.system(systemMessage);
	context.user(userMessage);

	yield { type: 'status', message: '分析中...' };

	// 添加控制台调试日志
	log(`[${stageId}] 开始流式调用，直接输出Markdown`);
	log(`[${stageId}] 系统消息:`, systemMessage);
	log(`[${stageId}] 用户消息长度:`, userMessage.length);

	let accumulatedThinking = '';
	let accumulatedText = '';
	let finalUsage: { input_tokens: number; output_tokens: number; total_tokens: number } | undefined;

	debugLog(`${stageId}/stream_start`, { promptFile, stageId, varsCount: Object.keys(vars).length });

	log(`[${stageId}] 开始调用 model.streamChat()...`);

	try {
		for await (const chunk of model.streamChat(context, undefined, {
			max_tokens: 60000  // 设置足够的 token 空间
		})) {
			debugLog(`${stageId}/chunk`, {
				outputEnd: chunk.output?.end,
				deltaText: chunk.output?.delta?.text?.slice(0, 100),
				deltaReasoning: chunk.output?.delta?.reasoning?.slice(0, 100)
			});

			// 处理思考内容
			if (chunk.output?.delta?.reasoning) {
				accumulatedThinking += chunk.output.delta.reasoning;
				yield {
					type: 'llm_chunk',
					stage: stageId,
					step: 1,
					reasoning: chunk.output.delta.reasoning
				};
			}

			// 处理文本内容
			if (chunk.output?.delta?.text) {
				accumulatedText += chunk.output.delta.text;
				yield {
					type: 'llm_chunk',
					stage: stageId,
					step: 1,
					text: chunk.output.delta.text
				};
				// 如果是 stage2，额外产出一个 chunk 事件给前端主显区（评估结果区域）
				if (stageId === 'stage2') {
					yield { type: 'chunk', text: chunk.output.delta.text };
				}
			}

			// 记录使用情况
			if (chunk.usage) {
				finalUsage = chunk.usage;
			}
		}
	} catch (error) {
		console.error(`[${stageId}] streamChat 错误:`, error);
		throw error;
	}

	log(`[${stageId}] 流式调用结束，统计:`, {
		thinkingLen: accumulatedThinking.length,
		textLen: accumulatedText.length,
		usage: finalUsage
	});

	debugLog(`${stageId}/stream_end`, {
		thinkingLen: accumulatedThinking.length,
		textLen: accumulatedText.length,
		usage: finalUsage
	});

	// 发送最终的 llm_call 事件汇总信息
	const llmCallEvent = {
		type: 'llm_call' as const,
		stage: stageId,
		step: 1,
		prompt: userMessage,
		...(accumulatedThinking ? { thinking: accumulatedThinking } : {}),
		...(accumulatedText ? { response: accumulatedText } : {}),
		...(finalUsage ? { usage: finalUsage } : {})
	};

	log(`[${stageId}] 发送 llm_call 事件:`, {
		hasThinking: !!accumulatedThinking,
		thinkingLength: accumulatedThinking.length,
		hasText: !!accumulatedText,
		textLength: accumulatedText.length,
		hasUsage: !!finalUsage
	});

	yield llmCallEvent;

	// 检查是否有文本响应
	if (!accumulatedText) {
		yield {
			type: 'tool' as const,
			name: 'error',
			message: `LLM 没有返回任何文本内容。可能是 prompt 格式问题。`,
			stage: stageId
		};
		throw new Error('Markdown模式 LLM 未返回任何文本内容');
	}

	return accumulatedText;
}
