import type { ExecutableTool } from '$lib/server/ai/generator/types';
import { createNotebookGeneratorTools } from '$lib/server/ai/generator/tools';
import { createSubmitEvaluationResultTool } from '../tools/submit-evaluation-result';
import { createSubmitJobFitResultTool, parseJobFitResult } from '../tools/submit-job-fit-result';
import { runStageGenerator, loadPrompt } from './runner';
import { runStageBasic } from './runner-basic';
import type { PipelineEvent } from '../types';

type NotebookDetail = NonNullable<
	Awaited<ReturnType<typeof import('$lib/server/workbench').getNotebookDetailForUser>>
>;

export async function* runStage1a(
	detail: NotebookDetail,
	mode: 'basic' | 'advanced' = 'basic'
): AsyncGenerator<PipelineEvent, unknown> {
	if (mode === 'basic') {
		// 基础模式：直接读取 source 内容，填入占位符，一次 LLM 调用
		const jd = detail.sources.find((s) => s.sourceType === 'job-description')?.content ?? '[JD 未提供]';
		const cv = detail.sources.find((s) => s.sourceType === 'resume')?.content ?? '[简历未提供]';
		const transcript = detail.sources.find((s) => s.sourceType === 'conversation')?.content ?? '[面试记录未提供]';

		return yield* runStageBasic(
			'job-fit-rater.md',
			{ PIPELINE_JD: jd, PIPELINE_CV: cv, PIPELINE_TRANSCRIPT: transcript },
			'pipeline/basic',
			'stage1a',
			createSubmitJobFitResultTool(),
			parseJobFitResult
		);
	}

	// 进阶模式：Agent 自取数据
	const systemPrompt = await loadPrompt('job-fit-rater.md');
	const userPrompt = `请对此候选人进行岗位匹配评估。
使用工具读取简历（resume）、面试记录（conversation）和职位描述（job-description），然后按照评分规则进行分析，最后调用 submit_evaluation_result 提交结果 JSON。`;

	const notebookTools = createNotebookGeneratorTools(detail);
	const tools: ExecutableTool[] = [
		notebookTools.find((t) => t.name === 'get_source_by_type')!,
		createSubmitEvaluationResultTool()
	];
	return yield* runStageGenerator(systemPrompt, userPrompt, tools);
}
