import {
	Context,
	type Action,
	type AnyContentPart,
	type Message,
	type Response
} from '$menglong';
import { createAiModel } from '$lib/server/ai/config';
import type { ExecutableTool, GeneratorEvent, GeneratorRunInput, GeneratorStreamEvent } from './types';

function normalizeToolArgs(input: Action['arguments']) {
	if (!input) return {};
	if (typeof input === 'string') {
		try {
			return JSON.parse(input) as Record<string, unknown>;
		} catch {
			return { raw: input };
		}
	}
	return input as Record<string, unknown>;
}

function stringifyToolResult(result: unknown) {
	if (typeof result === 'string') return result;
	return JSON.stringify(result, null, 2);
}

function buildAssistantToolMessage(response: Response, actions: Action[]): Message {
	const content: AnyContentPart[] = [];

	if (response.text) {
		content.push({
			type: 'text',
			text: response.text
		});
	}

	for (const action of actions) {
		content.push(action);
	}

	return {
		role: 'assistant',
		content
	};
}

export async function* runGenerator(input: GeneratorRunInput): AsyncGenerator<GeneratorStreamEvent> {
	const model = createAiModel();
	const context = new Context();
	const toolMap = new Map<string, ExecutableTool>((input.tools ?? []).map((tool) => [tool.name, tool]));
	const maxSteps = input.maxSteps ?? 50;

	context.system(input.systemPrompt);
	context.user(input.userPrompt);
	yield { type: 'status', message: '正在初始化...' };

	let lastResponse: Response | undefined;

	for (let step = 0; step < maxSteps; step += 1) {
		yield { type: 'status', message: `第 ${step + 1} 轮推理中...` };

		// 最后一步或判断不会有工具调用时用 streamChat，其余用 chat
		// 先用 chat，工具调用完毕后的最终轮改用 streamChat
		const response = await model.chat(context, undefined, {
			tools: (input.tools ?? []).map((tool) => tool.schema())
		});
		lastResponse = response;

		const actions = response.tool_calls ?? [];

		if (actions.length === 0) {
			// 这一轮没有工具调用，说明是最终输出轮
			// 用 streamChat 重新生成，让 token 实时流出
			yield { type: 'status', message: '正在生成报告...' };

			let finalText = '';
			for await (const chunk of model.streamChat(context, undefined, {
				tools: []
			})) {
				const delta = chunk.output?.delta?.text ?? '';
				if (delta) {
					finalText += delta;
					yield { type: 'chunk', text: delta };
				}
			}

			yield { type: 'done', finalText, usage: response.usage };
			return;
		}

		context.add(buildAssistantToolMessage(response, actions));

		for (const action of actions) {
			const handler = toolMap.get(action.name);
			if (!handler) {
				const missingMessage = `工具 "${action.name}" 未注册。`;
				context.tool(action.id ?? crypto.randomUUID(), missingMessage, action.name);
				yield { type: 'warning', message: missingMessage };
				continue;
			}

			yield { type: 'tool', name: action.name, message: `调用 ${action.name}...` };
			const params = normalizeToolArgs(action.arguments);
			const result = await handler.handler(params);
			context.tool(action.id ?? crypto.randomUUID(), stringifyToolResult(result), action.name);
			yield { type: 'tool', name: action.name, message: `${action.name} 完成` };
		}
	}

	// 超出最大步数，强制用最后的 context 生成一次
	yield { type: 'status', message: '已达最大轮数，正在生成结果...' };
	let finalText = '';
	for await (const chunk of model.streamChat(context, undefined, { tools: [] })) {
		const delta = chunk.output?.delta?.text ?? '';
		if (delta) {
			finalText += delta;
			yield { type: 'chunk', text: delta };
		}
	}
	yield { type: 'done', finalText, usage: lastResponse?.usage };
}
