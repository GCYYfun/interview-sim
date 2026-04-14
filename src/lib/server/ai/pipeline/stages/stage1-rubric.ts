import type { ExecutableTool } from '$lib/server/ai/generator/types';
import { createNotebookGeneratorTools } from '$lib/server/ai/generator/tools';
import { createGetDimensionRubricTool } from '../tools/get-dimension-rubric';
import { createSubmitEvaluationResultTool } from '../tools/submit-evaluation-result';
import { runStageGenerator, loadPrompt } from './runner';
import { runStageBasic, loadPrompt as loadBasicPrompt, extractDimensionSection } from './runner-basic';
import type { DimensionKey, PipelineEvent } from '../types';
import { DIMENSION_STAGE_MAP } from '../types';

type NotebookDetail = NonNullable<
	Awaited<ReturnType<typeof import('$lib/server/workbench').getNotebookDetailForUser>>
>;

export async function* runStage1Rubric(
	detail: NotebookDetail,
	dimension: DimensionKey,
	mode: 'basic' | 'advanced' = 'basic'
): AsyncGenerator<PipelineEvent, unknown> {
	if (mode === 'basic') {
		const dimensionRubric = await loadBasicPrompt('dimension-rubric.md');
		const section = extractDimensionSection(dimensionRubric, dimension);
		const transcript = detail.sources.find((s) => s.sourceType === 'conversation')?.content ?? '[面试记录未提供]';
		const cv = detail.sources.find((s) => s.sourceType === 'resume')?.content ?? '[简历未提供]';
		const jd = detail.sources.find((s) => s.sourceType === 'job-description')?.content ?? '[JD 未提供]';

		return yield* runStageBasic('rubric-rater.md', {
			PIPELINE_DIMENSION_NAME: dimension,
			PIPELINE_DIMENSION_RUBRIC: section,
			PIPELINE_TRANSCRIPT: transcript,
			PIPELINE_CV: cv,
			PIPELINE_JD: jd
		}, 'pipeline/basic', DIMENSION_STAGE_MAP[dimension]);
	}

	// 进阶模式：Agent 自取数据
	const systemPrompt = await loadPrompt('rubric-rater.md');
	const userPrompt = `请评估候选人的【${dimension}】维度。

步骤：
1. 调用 get_dimension_rubric 获取【${dimension}】的评分标准（Rubric）
2. 调用 get_source_by_type 分别读取：conversation（面试记录）、resume（简历）、job-description（职位描述）
3. 按照 Rubric 规则，对候选人在【${dimension}】维度的表现进行结构化评分
4. 调用 submit_evaluation_result 提交评分结果 JSON（严格按照 rubric-rater 的输出 Schema）`;

	const notebookTools = createNotebookGeneratorTools(detail);
	const tools: ExecutableTool[] = [
		createGetDimensionRubricTool(),
		notebookTools.find((t) => t.name === 'get_source_by_type')!,
		createSubmitEvaluationResultTool()
	];
	return yield* runStageGenerator(systemPrompt, userPrompt, tools);
}
