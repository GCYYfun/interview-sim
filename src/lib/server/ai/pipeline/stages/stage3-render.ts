import { Context } from '$menglong';
import type { ExecutableTool } from '$lib/server/ai/generator/types';
import { createGetPipelineCheckpointTool } from '../tools/get-pipeline-checkpoint';
import { loadPrompt, createPipelineModel } from './runner-basic';
import type { PipelineCheckpoint, PipelineEvent } from '../types';

export async function* runStage3(
	getCheckpoint: () => PipelineCheckpoint,
	mode: 'basic' | 'advanced' = 'basic'
): AsyncGenerator<PipelineEvent, string> {
	const systemPrompt = await loadPrompt('report-renderer.md');
	const model = createPipelineModel();

	if (mode === 'basic') {
		// 基础模式：直接读 checkpoint 数据拼入 prompt，一次 streamChat
		const cp = getCheckpoint();
		const stage1aData = JSON.stringify(cp.stages.stage1a.data ?? {}, null, 2);
		const stage2Data  = JSON.stringify(cp.stages.stage2.data  ?? {}, null, 2);

		const context = new Context();
		context.system(systemPrompt);
		context.user(`请根据以下评估数据生成完整的面试评估报告（Markdown 格式）。

## 岗位匹配评估（stage1a）
\`\`\`json
${stage1aData}
\`\`\`

## 综合评估结果（stage2）
\`\`\`json
${stage2Data}
\`\`\`

请严格按照报告格式规范输出完整 Markdown 报告。`);

		yield { type: 'status', message: '正在生成报告...' };

		let finalText = '';
		for await (const chunk of model.streamChat(context, undefined, { max_tokens: 60000 })) {
			const delta = chunk.output?.delta?.text ?? '';
			if (delta) {
				finalText += delta;
				yield { type: 'chunk', text: delta };
			}
		}

		if (!finalText) throw new Error('报告生成返回空内容');
		return finalText;
	}

	// 进阶模式：agent 工具循环读取 checkpoint 数据，再流式生成报告
	const context = new Context();
	context.system(systemPrompt);

	const checkpointTool = createGetPipelineCheckpointTool(getCheckpoint);
	const tools: ExecutableTool[] = [checkpointTool];

	context.user(`请根据评估结果生成完整的面试评估报告（Markdown 格式）。

步骤：
1. 调用 get_pipeline_checkpoint(stage2) 获取综合评估结果
2. 调用 get_pipeline_checkpoint(stage1a) 获取岗位匹配结果
3. 按照报告格式规范生成完整 Markdown 报告`);

	for (let step = 0; step < 10; step++) {
		yield { type: 'status', message: `报告准备中（第 ${step + 1} 轮）...` };

		const response = await model.chat(context, undefined, {
			tools: tools.map((t) => t.schema()),
			max_tokens: 60000
		});

		const actions = response.tool_calls ?? [];
		if (actions.length === 0) break;

		const content: import('$menglong').AnyContentPart[] = [];
		if (response.text) content.push({ type: 'text', text: response.text });
		for (const action of actions) content.push(action);
		context.add({ role: 'assistant', content });

		for (const action of actions) {
			const args = normalizeArgs(action.arguments);
			const handler = tools.find((t) => t.name === action.name);
			if (!handler) {
				const msg = `工具 "${action.name}" 未注册`;
				context.tool(action.id ?? crypto.randomUUID(), msg, action.name);
				yield { type: 'warning', message: msg };
				continue;
			}
			yield { type: 'tool', name: action.name, message: `调用 ${action.name}...` };
			const result = await handler.handler(args);
			context.tool(
				action.id ?? crypto.randomUUID(),
				typeof result === 'string' ? result : JSON.stringify(result, null, 2),
				action.name
			);
			yield { type: 'tool', name: action.name, message: `${action.name} 完成` };
		}
	}

	yield { type: 'status', message: '正在生成报告...' };
	let finalText = '';
	for await (const chunk of model.streamChat(context, undefined, { tools: [], max_tokens: 60000 })) {
		const delta = chunk.output?.delta?.text ?? '';
		if (delta) {
			finalText += delta;
			yield { type: 'chunk', text: delta };
		}
	}

	if (!finalText) throw new Error('报告生成返回空内容');
	return finalText;
}

function normalizeArgs(input: import('$menglong').Action['arguments']): Record<string, unknown> {
	if (!input) return {};
	if (typeof input === 'string') {
		try { return JSON.parse(input) as Record<string, unknown>; }
		catch { return { raw: input }; }
	}
	return input as Record<string, unknown>;
}
