import { defineTool } from '$menglong';
import type { PipelineCheckpoint, PipelineStageId } from '../types';

/**
 * 工厂函数，注入当前 pipeline 的 checkpoint 引用。
 * Stage 2/3 用此工具读取上游 stage 的输出数据。
 */
export function createGetPipelineCheckpointTool(getCheckpoint: () => PipelineCheckpoint) {
	return defineTool({
		name: 'get_pipeline_checkpoint',
		description:
			'读取指定 pipeline stage 的已完成输出数据（JSON）。用于 Stage 2/3 获取上游评分结果。',
		parameters: {
			type: 'object',
			properties: {
				stage: {
					type: 'string',
					description: 'stage ID，例如 stage1a / stage1b / stage2',
					enum: ['stage1a', 'stage1b', 'stage1c', 'stage1d', 'stage1e', 'stage1f', 'stage2']
				}
			},
			required: ['stage']
		},
		handler: async (params) => {
			const stageId = String(params.stage ?? '') as PipelineStageId;
			const checkpoint = getCheckpoint();
			const state = checkpoint.stages[stageId];
			if (!state) {
				return { status: 'not_found', message: `Stage "${stageId}" 不存在` };
			}
			if (state.status !== 'done' || state.data === null) {
				return { status: 'not_ready', message: `Stage "${stageId}" 尚未完成（状态: ${state.status}）` };
			}
			return { status: 'ok', stage: stageId, data: state.data };
		}
	});
}
