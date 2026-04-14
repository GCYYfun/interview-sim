import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import type { ExecutableTool } from '$lib/server/ai/generator/types';
import { runGenerator } from '$lib/server/ai/generator';
import { Context } from '$menglong';
import type { PipelineEvent } from '../types';
import { parseSubmittedResult } from '../tools/submit-evaluation-result';
import { createPipelineModel } from './runner-basic';

const PROMPT_DIR = resolve('prompt');

// 缓存 prompt 文件，避免多次 IO
const promptCache = new Map<string, string>();

export async function loadPrompt(filename: string): Promise<string> {
	if (!promptCache.has(filename)) {
		const content = await readFile(resolve(PROMPT_DIR, filename), 'utf-8');
		promptCache.set(filename, content);
	}
	return promptCache.get(filename)!;
}

/**
 * 运行一个 pipeline stage 的 Generator，直到 LLM 调用 submit_evaluation_result 为止。
 *
 * 与通用 runGenerator 的区别：
 * - 终止条件是 tool call 为 submit_evaluation_result（而非无工具调用）
 * - 不做流式文本输出，只转发 tool / status / warning 事件
 * - 返回 submit 的 result 数据（已解析的对象）
 */
export async function* runStageGenerator(
	systemPrompt: string,
	userPrompt: string,
	tools: ExecutableTool[],
	maxSteps = 20
): AsyncGenerator<PipelineEvent, unknown> {
	const model = createPipelineModel();
	const context = new Context();
	context.system(systemPrompt);
	context.user(userPrompt);

	const toolMap = new Map(tools.map((t) => [t.name, t]));

	for (let step = 0; step < maxSteps; step++) {
		yield { type: 'status', message: `推理第 ${step + 1} 轮...` };

		const response = await model.chat(context, undefined, {
			tools: tools.map((t) => t.schema()),
			tool_choice: 'auto',  // 使用 auto，让 LLM 决定是否调用工具
			max_tokens: 60000
		});

		const actions = response.tool_calls ?? [];

		if (actions.length === 0) {
			// LLM 没有调用任何工具就停止了——说明它没有提交结果
			// 尝试从 response.text 解析 JSON 作为兜底
			const text = response.text?.trim() ?? '';
			if (text) {
				try {
					return JSON.parse(text);
				} catch {
					throw new Error(`Stage LLM 未调用 submit_evaluation_result，且回复不是合法 JSON。回复片段: ${text.slice(0, 200)}`);
				}
			}
			throw new Error('Stage LLM 未调用 submit_evaluation_result，且没有文本回复。');
		}

		// 构建 assistant 消息（含 tool_use blocks）
		const content: import('$menglong').AnyContentPart[] = [];
		if (response.text) content.push({ type: 'text', text: response.text });
		for (const action of actions) content.push(action);
		context.add({ role: 'assistant', content });

		for (const action of actions) {
			const args = normalizeArgs(action.arguments);

			// 如果 LLM 调用了 submit_evaluation_result，提取结果并结束
			if (action.name === 'submit_evaluation_result') {
				const result = parseSubmittedResult(args);
				yield { type: 'tool', name: action.name, message: '提交评估结果...', result };
				return result;
			}

			const handler = toolMap.get(action.name);
			if (!handler) {
				const msg = `工具 "${action.name}" 未注册`;
				context.tool(action.id ?? crypto.randomUUID(), msg, action.name);
				yield { type: 'warning', message: msg };
				continue;
			}

			yield { type: 'tool', name: action.name, message: `调用 ${action.name}...`, args };
			const result = await handler.handler(args);
			context.tool(
				action.id ?? crypto.randomUUID(),
				typeof result === 'string' ? result : JSON.stringify(result, null, 2),
				action.name
			);
			yield { type: 'tool', name: action.name, message: `${action.name} 完成`, result };
		}
	}

	throw new Error(`Stage 超出最大步数 ${maxSteps}，未收到 submit_evaluation_result`);
}

function normalizeArgs(input: import('$menglong').Action['arguments']): Record<string, unknown> {
	if (!input) return {};
	if (typeof input === 'string') {
		try { return JSON.parse(input) as Record<string, unknown>; }
		catch { return { raw: input }; }
	}
	return input as Record<string, unknown>;
}
