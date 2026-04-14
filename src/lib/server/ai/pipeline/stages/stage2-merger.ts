import type { ExecutableTool } from '$lib/server/ai/generator/types';
import { createGetPipelineCheckpointTool } from '../tools/get-pipeline-checkpoint';
import { createSubmitEvaluationResultTool } from '../tools/submit-evaluation-result';
import { runStageGenerator, loadPrompt } from './runner';
import { runStageBasic } from './runner-basic';
import { runStageMarkdown } from './runner-markdown';
import type { PipelineCheckpoint, PipelineEvent } from '../types';
import { log } from '../debug-log';

export async function* runStage2(
	getCheckpoint: () => PipelineCheckpoint,
	mode: 'basic' | 'advanced' = 'basic'
): AsyncGenerator<PipelineEvent, unknown> {
	log(`[stage2] 开始运行，模式: ${mode}`);

	if (mode === 'basic') {
		const cp = getCheckpoint();
		log(`[stage2] checkpoint 数据:`, {
			stage1a: cp.stages.stage1a.data ? '有数据' : '无数据',
			stage1b: cp.stages.stage1b.data ? '有数据' : '无数据',
			stage1c: cp.stages.stage1c.data ? '有数据' : '无数据',
			stage1d: cp.stages.stage1d.data ? '有数据' : '无数据',
			stage1e: cp.stages.stage1e.data ? '有数据' : '无数据',
			stage1f: cp.stages.stage1f.data ? '有数据' : '无数据'
		});

		// 使用新的Markdown输出模式，直接生成Markdown报告
		return yield* runStageMarkdown('final-merger-markdown.md', {
			PIPELINE_STAGE1A_RESULT: JSON.stringify(cp.stages.stage1a.data ?? {}, null, 2),
			PIPELINE_STAGE1B_RESULT: JSON.stringify(cp.stages.stage1b.data ?? {}, null, 2),
			PIPELINE_STAGE1C_RESULT: JSON.stringify(cp.stages.stage1c.data ?? {}, null, 2),
			PIPELINE_STAGE1D_RESULT: JSON.stringify(cp.stages.stage1d.data ?? {}, null, 2),
			PIPELINE_STAGE1E_RESULT: JSON.stringify(cp.stages.stage1e.data ?? {}, null, 2),
			PIPELINE_STAGE1F_RESULT: JSON.stringify(cp.stages.stage1f.data ?? {}, null, 2)
		}, 'pipeline/basic', 'stage2');
	}

	// 进阶模式：Agent 自取 checkpoint 数据
	const systemPrompt = await loadPrompt('final-merger.md');
	const userPrompt = `请对候选人进行综合汇总评估。

步骤：
1. 调用 get_pipeline_checkpoint 分别读取：stage1a、stage1b、stage1c、stage1d、stage1e、stage1f 的数据
2. 按照 Final Merger 规则进行加权计算，生成完整顶层 JSON
3. 调用 submit_evaluation_result 提交结果（严格按照 final-merger 的输出 Schema）`;

	const tools: ExecutableTool[] = [
		createGetPipelineCheckpointTool(getCheckpoint),
		createSubmitEvaluationResultTool()
	];
	return yield* runStageGenerator(systemPrompt, userPrompt, tools);
}
