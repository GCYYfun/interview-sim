import { Context } from '$menglong';
import type { PipelineEvent } from '../types';
import { debugLog, log } from '../debug-log';
import { createPipelineModel, loadPrompt, fillPlaceholders } from './runner-shared';

export { loadPrompt, fillPlaceholders };


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
		const msg = error instanceof Error ? error.message : JSON.stringify(error);
		console.error(`[${stageId}] streamChat 错误:`, error);
		throw new Error(`[${stageId}] streamChat 失败: ${msg}`);
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
