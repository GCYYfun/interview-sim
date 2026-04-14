import { defineTool } from '$menglong';

/**
 * submit_evaluation_result —— 结构化提交工具。
 *
 * 每个评分 stage 的 Generator 只注册这一个"输出工具"。
 * LLM 调用此工具即代表完成本 stage 的推理，result 字段即为最终 JSON 数据。
 * schema 采用宽松 object，实际结构由各 stage 的 prompt 规则约束。
 *
 * 设计原则：
 * - 不在 TypeScript 层做 schema 校验（保留 prompt 灵活性）
 * - handler 不做任何处理，直接返回 result，由调用方从 tool_call.arguments 提取
 */
export function createSubmitEvaluationResultTool() {
	return defineTool({
		name: 'submit_evaluation_result',
		description:
			'提交本阶段的结构化评估结果。当你完成所有分析推理后，调用此工具提交最终 JSON。提交后本阶段结束。',
		parameters: {
			type: 'object',
			properties: {
				result: {
					type: 'string',
					description:
						'评估结果的 JSON 字符串。必须是合法 JSON，字段结构按照当前 stage 的 prompt 要求输出。'
				}
			},
			required: ['result']
		},
		handler: async (params) => {
			// handler 不做任何事——调用方直接从 tool_call.arguments.result 提取数据
			// 这里返回 ack 仅用于让 LLM 知道工具已被接收
			return { status: 'submitted', received: true };
		}
	});
}

/** 从 tool call arguments 中解析 result JSON，解析失败则抛出 */
export function parseSubmittedResult(args: Record<string, unknown>): unknown {
	const raw = String(args.result ?? '').trim();
	if (!raw) throw new Error('submit_evaluation_result: result 字段为空');
	return JSON.parse(raw);
}
