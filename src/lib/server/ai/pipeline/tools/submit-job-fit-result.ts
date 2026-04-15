import { defineTool } from '$menglong';

/**
 * submit_job_fit_result —— stage1a 专属提交工具。
 *
 * 参数结构对应 job-fit-rater.md 的输出 JSON Schema，
 * 把字段约束从 prompt 下沉到 tool schema 层，让 LLM 按结构填充。
 */
export function createSubmitJobFitResultTool() {
	return defineTool({
		name: 'submit_job_fit_result',
		description:
			'提交岗位匹配评估结果。完成所有分析推理后，调用此工具提交结构化结果。提交后本阶段结束。',
		parameters: {
			type: 'object',
			properties: {
				fit_verdict: {
					type: 'string',
					enum: ['高匹配', '中匹配', '低匹配'],
					description: '匹配结论(Fit Verdict)'
				},
				fit_summary: {
					type: 'string',
					description: '简述(Fit Summary)：对匹配情况的整体说明'
				},
				strengths: {
					type: 'array',
					items: { type: 'string' },
					description: '匹配亮点(Strengths)：候选人与岗位契合的关键点列表'
				},
				risks: {
					type: 'array',
					items: { type: 'string' },
					description: '风险点(Risks)：候选人与岗位不匹配或存疑的点列表'
				},
				evidence: {
					type: 'array',
					description: '证据(Evidence)：支撑结论的原始引用列表',
					items: {
						type: 'object',
						properties: {
							source: {
								type: 'string',
								enum: ['CV', 'Transcript', 'JD'],
								description: '证据来源'
							},
							quote_or_fact: {
								type: 'string',
								description: '引用原文或关键事实'
							},
							context_optional: {
								type: 'string',
								description: '可选的补充上下文'
							}
						},
						required: ['source', 'quote_or_fact']
					}
				},
				confidence: {
					type: 'number',
					minimum: 0,
					maximum: 1,
					description: '置信度(Confidence)：0–1 之间的小数，反映证据充分程度'
				}
			},
			required: ['fit_verdict', 'fit_summary', 'strengths', 'risks', 'evidence', 'confidence']
		},
		handler: async () => {
			return { status: 'submitted', received: true };
		}
	});
}

/** 将工具参数重组为 job-fit-rater.md 输出 Schema 格式 */
export function parseJobFitResult(args: Record<string, unknown>): unknown {
	return {
		'岗位匹配简述(Job Fit Summary)': {
			'匹配结论(Fit Verdict)': args.fit_verdict,
			'简述(Fit Summary)': args.fit_summary,
			'匹配亮点(Strengths)': args.strengths,
			'风险点(Risks)': args.risks,
			'证据(Evidence)': args.evidence,
			'置信度(Confidence)': args.confidence
		}
	};
}
