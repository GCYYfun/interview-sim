// ─── Checkpoint ──────────────────────────────────────────────────────────────

export type PipelineStageId =
	| 'stage1a'
	| 'stage1b'
	| 'stage1c'
	| 'stage1d'
	| 'stage1e'
	| 'stage1f'
	| 'stage2'
	| 'stage3';

export type PipelineStageStatus = 'pending' | 'running' | 'done' | 'failed';

export type PipelineStageState = {
	status: PipelineStageStatus;
	attempt: number;
	data: unknown | null;
	error?: string;
};

export type PipelineCheckpoint = {
	pipelineRunId: string;
	notebookId: string;
	createdAt: number;
	updatedAt: number;
	stages: Record<PipelineStageId, PipelineStageState>;
};

// ─── Stream Events ────────────────────────────────────────────────────────────

/** 进度播报事件，由 pipeline 编排器推送到前端 */
export type PipelineEvent =
	| { type: 'stage_start'; stage: PipelineStageId; message: string }
	| { type: 'stage_complete'; stage: PipelineStageId; message: string; durationMs?: number }
	| { type: 'stage_retry'; stage: PipelineStageId; message: string; attempt: number }
	| { type: 'stage_error'; stage: PipelineStageId; message: string }
	| { type: 'llm_chunk'; stage: PipelineStageId; step: number; text?: string; reasoning?: string }
	| { type: 'llm_call'; stage: PipelineStageId; step: number; prompt?: string; thinking?: string; response?: string; usage?: { input_tokens: number; output_tokens: number; total_tokens: number } }
	| { type: 'tool'; name: string; message: string; stage?: PipelineStageId; args?: any; result?: any }
	| { type: 'status'; message: string }
	| { type: 'warning'; message: string }
	| { type: 'chunk'; text: string }
	| { type: 'done'; finalText: string }
	| { type: 'error'; message: string }
	| { type: 'save'; title: string; summary: string; content: string; rawJson?: unknown };

// ─── Dimension keys ───────────────────────────────────────────────────────────

export type DimensionKey = '聪明度' | '勤奋度' | '目标感' | '皮实度' | '客户第一';

export const DIMENSION_STAGE_MAP: Record<DimensionKey, PipelineStageId> = {
	聪明度: 'stage1b',
	勤奋度: 'stage1c',
	目标感: 'stage1d',
	皮实度: 'stage1e',
	客户第一: 'stage1f'
};

export const STAGE_DIMENSION_MAP: Partial<Record<PipelineStageId, DimensionKey>> = {
	stage1b: '聪明度',
	stage1c: '勤奋度',
	stage1d: '目标感',
	stage1e: '皮实度',
	stage1f: '客户第一'
};

export const MAX_RETRIES = 3;
