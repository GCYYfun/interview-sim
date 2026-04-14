import { eq, desc } from 'drizzle-orm';
import { db } from '$lib/server/db';
import { pipelineCheckpoint } from '$lib/server/db/schema';
import type { PipelineCheckpoint, PipelineStageId, PipelineStageState } from './types';
import { MAX_RETRIES } from './types';

// ─── Init ─────────────────────────────────────────────────────────────────────

function emptyCheckpoint(notebookId: string): PipelineCheckpoint {
	const now = Date.now();
	return {
		pipelineRunId: crypto.randomUUID(),
		notebookId,
		createdAt: now,
		updatedAt: now,
		stages: {
			stage1a: { status: 'pending', attempt: 0, data: null },
			stage1b: { status: 'pending', attempt: 0, data: null },
			stage1c: { status: 'pending', attempt: 0, data: null },
			stage1d: { status: 'pending', attempt: 0, data: null },
			stage1e: { status: 'pending', attempt: 0, data: null },
			stage1f: { status: 'pending', attempt: 0, data: null },
			stage2:  { status: 'pending', attempt: 0, data: null },
			stage3:  { status: 'pending', attempt: 0, data: null }
		}
	};
}

// ─── DB helpers ───────────────────────────────────────────────────────────────

type CheckpointRow = typeof pipelineCheckpoint.$inferSelect;

function deserialize(row: CheckpointRow): PipelineCheckpoint {
	return JSON.parse(row.state) as PipelineCheckpoint;
}

/**
 * 对 resume 的 checkpoint 做状态修正：
 * - `running` → `pending`：上次崩溃时卡住的 stage，重新跑
 * - `done` 但 data 为 null → `pending`：标记成功但没有数据，重新跑
 */
function sanitizeForResume(cp: PipelineCheckpoint): PipelineCheckpoint {
	const stages = { ...cp.stages } as typeof cp.stages;
	for (const key of Object.keys(stages) as PipelineStageId[]) {
		const s = stages[key];
		if (s.status === 'running') {
			stages[key] = { ...s, status: 'pending' };
		} else if (s.status === 'done' && (s.data === null || s.data === undefined)) {
			stages[key] = { ...s, status: 'pending' };
		}
	}
	return { ...cp, stages };
}

/** 获取或新建 checkpoint。
 * - resume=true：查找最近一条未完成记录，sanitize 后续跑
 * - resume=false（默认）：总是新建，忽略旧 checkpoint
 */
export async function getOrCreateCheckpoint(notebookId: string, resume = false): Promise<{
	rowId: string;
	checkpoint: PipelineCheckpoint;
	isResume: boolean;
}> {
	if (resume) {
		const existing = await db
			.select()
			.from(pipelineCheckpoint)
			.where(eq(pipelineCheckpoint.notebookId, notebookId))
			.orderBy(desc(pipelineCheckpoint.createdAt))
			.limit(1);

		const row = existing[0];
		if (row && row.status === 'running') {
			const sanitized = sanitizeForResume(deserialize(row));
			await db
				.update(pipelineCheckpoint)
				.set({ state: JSON.stringify(sanitized), updatedAt: Date.now() })
				.where(eq(pipelineCheckpoint.id, row.id));
			return { rowId: row.id, checkpoint: sanitized, isResume: true };
		}
	}

	// 新建
	const checkpoint = emptyCheckpoint(notebookId);
	const now = Date.now();
	const id = crypto.randomUUID();
	await db.insert(pipelineCheckpoint).values({
		id,
		notebookId,
		state: JSON.stringify(checkpoint),
		status: 'running',
		createdAt: now,
		updatedAt: now
	});
	return { rowId: id, checkpoint, isResume: false };
}

/**
 * 更新单个 stage 状态并持久化。
 * - 写 done 时校验 data 不为空，否则降级为 failed
 * - 每次从 DB 读最新快照再 patch，防止并行 stage 互相覆盖
 */
export async function updateStageCheckpoint(
	rowId: string,
	_staleCheckpoint: PipelineCheckpoint,
	stageId: PipelineStageId,
	patch: Partial<PipelineStageState>
): Promise<PipelineCheckpoint> {
	// done 但 data 为空 → 降级为 failed，不写入无效的成功状态
	if (patch.status === 'done' && (patch.data === null || patch.data === undefined)) {
		patch = { ...patch, status: 'failed', error: 'stage 完成但 data 为空' };
	}

	// 从 DB 读最新快照再 patch，防止并行 stage 互相覆盖
	const rows = await db.select().from(pipelineCheckpoint).where(eq(pipelineCheckpoint.id, rowId));
	const current: PipelineCheckpoint = rows[0] ? deserialize(rows[0]) : _staleCheckpoint;

	const updated: PipelineCheckpoint = {
		...current,
		updatedAt: Date.now(),
		stages: {
			...current.stages,
			[stageId]: { ...current.stages[stageId], ...patch }
		}
	};
	await db
		.update(pipelineCheckpoint)
		.set({ state: JSON.stringify(updated), updatedAt: updated.updatedAt })
		.where(eq(pipelineCheckpoint.id, rowId));
	return updated;
}

/** 从 DB 重新加载 checkpoint */
export async function reloadCheckpoint(rowId: string): Promise<PipelineCheckpoint> {
	const rows = await db.select().from(pipelineCheckpoint).where(eq(pipelineCheckpoint.id, rowId));
	if (!rows[0]) throw new Error(`Checkpoint ${rowId} not found`);
	return deserialize(rows[0]);
}

/** 标记整个 pipeline 完成或失败 */
export async function finalizeCheckpoint(
	rowId: string,
	checkpoint: PipelineCheckpoint,
	status: 'done' | 'failed'
): Promise<void> {
	const now = Date.now();
	const updated: PipelineCheckpoint = { ...checkpoint, updatedAt: now };
	await db
		.update(pipelineCheckpoint)
		.set({ state: JSON.stringify(updated), status, updatedAt: now })
		.where(eq(pipelineCheckpoint.id, rowId));
}

/** 判断某个 stage 是否需要运行 */
export function shouldRunStage(state: PipelineStageState): boolean {
	if (state.status === 'done') return false;
	if (state.status === 'failed' && state.attempt >= MAX_RETRIES) return false;
	return true;
}
