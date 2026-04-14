import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { Model, Context, MengLongProvider, ProviderRegistry, ConsoleLogger } from '$menglong';
import { env } from '$env/dynamic/private';
import type { Config } from '$menglong';
import type { PipelineEvent, DimensionKey } from '../types';
import { createSubmitEvaluationResultTool, parseSubmittedResult } from '../tools/submit-evaluation-result';
import { debugLog, log } from '../debug-log';
import { debugStageLog, debugToolCall, debugStreamChunk, debugPipelineEvent } from '../debug-log-enhanced';

// pipeline 额外支持 infinigence 前缀，均走 menglong 网关
if (!ProviderRegistry.getProviderClass('infinigence')) {
	log('[runner-basic] 注册 infinigence provider');
	ProviderRegistry.register('infinigence', MengLongProvider);
} else {
	log('[runner-basic] infinigence provider 已注册');
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

/** 从 dimension-rubric.md 中按 ## 标题提取指定维度章节 */
export function extractDimensionSection(rubric: string, dimension: DimensionKey): string {
	const lines = rubric.split('\n');
	const startIdx = lines.findIndex((l) => l.startsWith(`## ${dimension}`));
	if (startIdx === -1) return `[未找到维度章节: ${dimension}]`;
	let endIdx = lines.length;
	for (let i = startIdx + 1; i < lines.length; i++) {
		if (lines[i].startsWith('## ')) { endIdx = i; break; }
	}
	return lines.slice(startIdx, endIdx).join('\n').trim();
}

/** 替换 prompt 中的 {{PIPELINE_XXX}} 占位符 */
export function fillPlaceholders(template: string, vars: Record<string, string>): string {
	return Object.entries(vars).reduce(
		(text, [key, value]) => text.replaceAll(`{{${key}}}`, value),
		template
	);
}

/**
 * 基础模式 stage runner：单轮流式调用，支持 thinking + tool call。
 *
 * 模型在单轮流式调用中完成分析，可输出 thinking，最后调用 submit_evaluation_result 工具。
 */
export async function* runStageBasic(
	promptFile: string,
	vars: Record<string, string>,
	promptDir = 'pipeline/basic',
	stageId: import('../types').PipelineStageId = 'stage1a'
): AsyncGenerator<PipelineEvent, unknown> {
	yield { type: 'status', message: '准备数据...' };

	const template = await loadPrompt(`${promptDir}/${promptFile}`);
	// 在用户消息末尾添加工具调用指令
	const userMessage = fillPlaceholders(template, vars) + `

重要：请调用 submit_evaluation_result 工具提交你的评估结果，不要直接输出 JSON 文本。

工具说明：
- 工具名称：submit_evaluation_result
- 功能：提交本阶段的结构化评估结果
- 参数：{"result": "JSON字符串"}
- 注意：result 字段必须是合法 JSON 字符串，按照上面的输出 Schema 格式化`;

	const systemMessage = '你是专业的结构化面试评估专家。严格按照用户消息中的规则和数据进行分析，完成分析后调用 submit_evaluation_result 工具提交结构化评估结果。';

	const model = createPipelineModel();
	const context = new Context();
	context.system(systemMessage);
	context.user(userMessage);

	const submitTool = createSubmitEvaluationResultTool();

	yield { type: 'status', message: '分析中...' };

	// 添加控制台调试日志
	log(`[${stageId}] 开始流式调用，工具定义:`, JSON.stringify(submitTool.schema(), null, 2));
	log(`[${stageId}] 系统消息:`, systemMessage);
	log(`[${stageId}] 用户消息长度:`, userMessage.length);

	let accumulatedThinking = '';
	let accumulatedText = '';
	let finalActions: import('$menglong').Action[] | undefined;
	let finalUsage: { input_tokens: number; output_tokens: number; total_tokens: number } | undefined;

	// 用于累积流式工具调用
	const accumulatedTools: Record<number, { id: string; name: string; arguments: string }> = {};

	debugLog(`${stageId}/stream_start`, { promptFile, stageId, varsCount: Object.keys(vars).length });

	// 使用流式调用，强制要求调用工具
	log(`[${stageId}] 开始调用 model.streamChat()...`);
	log(`[${stageId}] 工具定义:`, JSON.stringify(submitTool.schema(), null, 2));
	log(`[${stageId}] 消息数量:`, context.messages.length);

	try {
		for await (const chunk of model.streamChat(context, undefined, {
			tools: [submitTool.schema()],
			tool_choice: 'auto',  // 使用 auto，让 LLM 决定是否调用工具
			max_tokens: 60000  // 设置足够的 token 空间，确保能完成工具调用
		})) {
			// 调试：打印每个 chunk 的完整结构
			log(`[${stageId}] 收到 chunk:`, JSON.stringify(chunk, null, 2));

		debugLog(`${stageId}/chunk`, {
			outputEnd: chunk.output?.end,
			actionsCount: chunk.output?.actions?.length,
			deltaText: chunk.output?.delta?.text?.slice(0, 100),
			deltaReasoning: chunk.output?.delta?.reasoning?.slice(0, 100)
		});

		// 处理思考内容
		if (chunk.output?.delta?.reasoning) {
			debugStreamChunk(stageId, 'thinking', chunk.output.delta.reasoning);
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
			debugStreamChunk(stageId, 'text', chunk.output.delta.text);
			accumulatedText += chunk.output.delta.text;
			yield {
				type: 'llm_chunk',
				stage: stageId,
				step: 1,
				text: chunk.output.delta.text
			};
		}

		// 处理流式工具调用增量（根据 Python SDK 测试）
		if (chunk.output?.delta?.actions) {
			debugStageLog(stageId, 'tool_chunks', chunk.output.delta.actions);
			log(`[${stageId}] 收到流式工具调用增量:`, chunk.output.delta.actions);
			for (const action of chunk.output.delta.actions) {
				const idx = action.index ?? 0;
				if (!accumulatedTools[idx]) {
					accumulatedTools[idx] = { id: '', name: '', arguments: '' };
					// 当开始累积一个新工具时，发送事件到 UI
					if (action.name) {
						yield {
							type: 'tool' as const,
							name: action.name,
							message: `正在调用工具 ${action.name}...`,
							stage: stageId,
							args: action.arguments
						};
					}
				}
				if (action.id) accumulatedTools[idx].id = action.id;
				if (action.name) accumulatedTools[idx].name = action.name;
				if (action.arguments) {
					if (typeof action.arguments === 'string') {
						accumulatedTools[idx].arguments += action.arguments;
						// 发送工具调用参数增量到 UI
						yield {
							type: 'tool' as const,
							name: action.name || 'unknown',
							message: `接收参数: ${action.arguments.slice(0, 50)}${action.arguments.length > 50 ? '...' : ''}`,
							stage: stageId,
							args: action.arguments
						};
					} else {
						// 如果 arguments 已经是对象，转换为字符串
						accumulatedTools[idx].arguments = JSON.stringify(action.arguments);
						yield {
							type: 'tool' as const,
							name: action.name || 'unknown',
							message: `接收参数对象`,
							stage: stageId,
							args: action.arguments
						};
					}
				}
			}
			log(`[${stageId}] 累积后的工具:`, accumulatedTools);
		}

		// 处理 tool call 结束
		if (chunk.output?.end === 'tool_calls' && chunk.output?.actions) {
			finalActions = chunk.output.actions;
			debugLog(`${stageId}/tool_calls_end`, {
				actions: finalActions.map(a => ({ name: a.name, argsLen: JSON.stringify(a.arguments).length }))
			});
		}

		// 如果流式结束且有累积的工具，组装成最终 actions
		if (chunk.output?.end === 'tool_calls' && Object.keys(accumulatedTools).length > 0 && !finalActions) {
			finalActions = Object.values(accumulatedTools).map(tc => ({
				type: 'action' as const,
				id: tc.id,
				name: tc.name,
				arguments: tc.arguments ? (() => {
					try { return JSON.parse(tc.arguments) as Record<string, unknown>; }
					catch { return { _raw: tc.arguments } as Record<string, unknown>; }
				})() : {}
			}));
			log(`[${stageId}] 从流式增量组装的 actions:`, finalActions);
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
		actionsCount: finalActions?.length,
		finalActions: finalActions,
		usage: finalUsage
	});

	debugLog(`${stageId}/stream_end`, {
		thinkingLen: accumulatedThinking.length,
		textLen: accumulatedText.length,
		actionsCount: finalActions?.length,
		usage: finalUsage
	});

	// 发送最终的 llm_call 事件汇总信息
	// 即使没有 thinking 或 text，也确保发送事件以便 UI 显示
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

	// 检查是否有 tool call
	const submitAction = finalActions?.find(a => a.name === 'submit_evaluation_result');
	if (submitAction) {
		// 发送工具调用完成的消息
		yield {
			type: 'tool' as const,
			name: 'submit_evaluation_result',
			message: '工具调用完成，正在解析评估结果...',
			stage: stageId,
			args: submitAction.arguments
		};

		log(`[${stageId}] 提交的评估结果参数:`, submitAction.arguments);
		const args = normalizeArgs(submitAction.arguments);

		// 发送结果解析完成的消息
		yield {
			type: 'tool' as const,
			name: 'submit_evaluation_result',
			message: '评估结果解析完成',
			stage: stageId,
			result: parseSubmittedResult(args)
		};

		return parseSubmittedResult(args);
	}

	// 如果没有 tool call，检查是否有文本响应
	if (accumulatedText) {
		yield {
			type: 'tool' as const,
			name: 'error',
			message: `LLM 没有调用工具，而是返回了文本。可能是 prompt 格式问题。`,
			stage: stageId
		};
		throw new Error(`基础模式 LLM 未调用 submit_evaluation_result，而是返回了文本: ${accumulatedText.slice(0, 200)}`);
	}

	yield {
		type: 'tool' as const,
		name: 'error',
		message: `LLM 既没有调用工具也没有返回文本。可能是工具调用失败或 prompt 格式问题。`,
		stage: stageId
	};
	throw new Error('基础模式 LLM 未调用 submit_evaluation_result 且无文本响应');
}

function normalizeArgs(input: import('$menglong').Action['arguments']): Record<string, unknown> {
	if (!input) return {};
	if (typeof input === 'string') {
		try { return JSON.parse(input) as Record<string, unknown>; }
		catch { return { raw: input }; }
	}
	return input as Record<string, unknown>;
}
