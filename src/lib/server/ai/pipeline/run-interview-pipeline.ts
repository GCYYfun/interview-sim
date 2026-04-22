import type { PipelineCheckpoint, PipelineEvent, PipelineStageId, DimensionKey } from './types';
import { MAX_RETRIES } from './types';
import {
	getOrCreateCheckpoint,
	updateStageCheckpoint,
	finalizeCheckpoint,
	reloadCheckpoint,
	shouldRunStage
} from './checkpoint';
import { runStage1a } from './stages/stage1-job-fit';
import { runStage1Rubric } from './stages/stage1-rubric';
import { runStage2 } from './stages/stage2-merger';
import { logger } from '$lib/server/logger';

type NotebookDetail = NonNullable<
	Awaited<ReturnType<typeof import('$lib/server/workbench').getNotebookDetailForUser>>
>;

type User = Parameters<typeof import('$lib/server/workbench').getNotebookDetailForUser>[0];

/**
 * 运行单个 stage，含重试逻辑。
 * 达到最大重试次数后 throw，由外层 catch 终止整个 pipeline。
 */
async function runStageWithRetry(
	stageId: PipelineStageId,
	rowId: string,
	checkpoint: PipelineCheckpoint,
	run: () => AsyncGenerator<PipelineEvent, unknown>,
	push: (event: PipelineEvent) => void,
	onCheckpointUpdate: (updated: PipelineCheckpoint) => void
): Promise<{ checkpoint: PipelineCheckpoint; data: unknown }> {
	let current = checkpoint;

	while (true) {
		const attempt = (current.stages[stageId].attempt ?? 0) + 1;

		if (attempt > MAX_RETRIES) {
			throw new Error(`Stage ${stageId} 超过最大重试次数 ${MAX_RETRIES}`);
		}

		current = await updateStageCheckpoint(rowId, current, stageId, { status: 'running', attempt });

		try {
			const gen = run();
			let data: unknown = undefined;
			while (true) {
				const { value, done } = await gen.next();
				if (done) { data = value; break; }
				push(value as PipelineEvent);
			}

			current = await updateStageCheckpoint(rowId, current, stageId, { status: 'done', data });
			onCheckpointUpdate(current);
			return { checkpoint: current, data };
		} catch (err) {
			const error = err instanceof Error ? err.message : String(err);
			logger.error('pipeline', `${stageId} attempt ${attempt} failed`, { stageId, attempt, error }, 'pipeline');
			push({ type: 'stage_retry', stage: stageId, message: `${stageId} 失败：${error.slice(0, 120)}`, attempt });
			current = await updateStageCheckpoint(rowId, current, stageId, { status: 'failed', error });
			onCheckpointUpdate(current);
			if (attempt >= MAX_RETRIES) throw new Error(`Stage ${stageId} 达到最大重试次数：${error}`);
		}
	}
}

export type PipelineMode = 'basic' | 'advanced';

/**
 * 主 Pipeline 编排器（async generator，流式输出 PipelineEvent）
 *
 * 容错策略：所有 stage 必须全部成功，任意一个达到最大重试次数则整体失败。
 * 断点续跑：resume 时已成功的 stage 跳过，未成功的重新跑。
 */
export async function* runInterviewPipeline(
	_user: User,
	detail: NotebookDetail,
	mode: 'basic' | 'advanced' = 'basic',
	resume = false
): AsyncGenerator<PipelineEvent> {
	const notebookId = detail.record.id;

	// ─── Checkpoint 初始化 ────────────────────────────────────────────────────
	const { rowId, checkpoint: initCheckpoint, isResume } = await getOrCreateCheckpoint(notebookId, resume);
	let checkpoint = initCheckpoint;

	logger.info('pipeline', `pipeline ${isResume ? 'resumed' : 'started'}`, {
		notebookId,
		rowId,
		mode,
		candidate: detail.record.subject
	}, 'pipeline');

	let latestCheckpoint = checkpoint;
	const getLatestCheckpoint = () => latestCheckpoint;
	const updateLatest = (cp: PipelineCheckpoint) => { latestCheckpoint = cp; };

	const queue: PipelineEvent[] = [];
	const push = (e: PipelineEvent) => queue.push(e);

	yield { type: 'status', message: isResume ? '恢复上次评估进度...' : '开始面试评估流水线' };

	try {
		// ─── Stage 1a + 1b-1f 并行（全部必须成功）───────────────────────────
		const DIMENSIONS: DimensionKey[] = ['聪明度', '勤奋度', '目标感', '皮实度', '客户第一'];
		const RUBRIC_STAGES: PipelineStageId[] = ['stage1b', 'stage1c', 'stage1d', 'stage1e', 'stage1f'];

		const parallelTasks: Promise<void>[] = [];

		// Stage 1a
		if (shouldRunStage(checkpoint.stages.stage1a)) {
			push({ type: 'stage_start', stage: 'stage1a', message: '评估岗位匹配度...' });
			const t1a = Date.now();
			parallelTasks.push(
				runStageWithRetry('stage1a', rowId, checkpoint, () => runStage1a(detail, mode), push, updateLatest)
					.then(({ checkpoint: cp }) => {
						const verdict =
							(cp.stages.stage1a.data as Record<string, Record<string, Record<string, string>>> | null)
								?.['岗位匹配简述(Job Fit Summary)']?.['匹配结论(Fit Verdict)'] ?? '完成';
						push({ type: 'stage_complete', stage: 'stage1a', message: `岗位匹配：${verdict}`, durationMs: Date.now() - t1a });
					})
			);
		}

		// Stage 1b-1f
		for (let i = 0; i < DIMENSIONS.length; i++) {
			const dimension = DIMENSIONS[i];
			const stageId = RUBRIC_STAGES[i];
			if (shouldRunStage(checkpoint.stages[stageId])) {
				push({ type: 'stage_start', stage: stageId, message: `评估维度：${dimension}...` });
				const tStart = Date.now();
				parallelTasks.push(
					runStageWithRetry(stageId, rowId, checkpoint, () => runStage1Rubric(detail, dimension, mode), push, updateLatest)
						.then(({ checkpoint: cp }) => {
							const overall = (cp.stages[stageId].data as Record<string, Record<string, unknown>> | null)
								?.['维度总体评价(Dimension Overall)'];
							const level = (overall as Record<string, string> | undefined)?.['档位(Level)'] ?? '完成';
							const conf = (overall as Record<string, number> | undefined)?.['置信度(Confidence)'];
							const confStr = conf !== undefined ? ` · 置信度 ${Math.round(Number(conf) * 100)}%` : '';
							push({ type: 'stage_complete', stage: stageId, message: `${dimension}：${level}${confStr}`, durationMs: Date.now() - tStart });
						})
				);
			}
		}

		// 等待全部并行任务，通过 catch 同步捕获并避免造成 UnhandledPromiseRejection 导致 Node 服务崩溃
		let allSettled = false;
		let allDoneError: Error | null = null;
		Promise.all(parallelTasks).then(() => { allSettled = true; }).catch((e) => {
			allDoneError = e instanceof Error ? e : new Error(String(e));
			allSettled = true;
		});

		while (!allSettled) {
			while (queue.length > 0) yield queue.shift()!;
			await new Promise<void>((r) => setTimeout(r, 50));
		}
		if (allDoneError) throw allDoneError; // 重新抛出并行任务中的异常
		while (queue.length > 0) yield queue.shift()!;

		// 并行完成后从 DB 重新加载，拿到所有 stage 的最终状态
		checkpoint = await reloadCheckpoint(rowId);
		latestCheckpoint = checkpoint;

		// ─── Stage 2 ──────────────────────────────────────────────────────────
		if (shouldRunStage(checkpoint.stages.stage2)) {
			yield { type: 'stage_start', stage: 'stage2', message: '综合汇总，计算总体评分...' };
			const t2 = Date.now();

			// 为了实现 Stage 2 的流式输出，我们需要像 Stage 1 那样在一个循环中消费 queue
			let stage2Done = false;
			let stage2Error: Error | null = null;
			let stage2Res: any = null;
			runStageWithRetry(
				'stage2', rowId, checkpoint,
				() => runStage2(getLatestCheckpoint, mode),
				push, updateLatest
			).then((res) => {
				stage2Res = res;
				stage2Done = true;
			}).catch(err => {
				stage2Error = err instanceof Error ? err : new Error(String(err));
				stage2Done = true;
			});

			// 在 Stage 2 运行期间，持续 yield 队列中的事件
			while (!stage2Done) {
				while (queue.length > 0) yield queue.shift()!;
				await new Promise<void>((r) => setTimeout(r, 50));
			}

			if (stage2Error) throw stage2Error;
			checkpoint = stage2Res.checkpoint;
			while (queue.length > 0) yield queue.shift()!;

			const evalData = (checkpoint.stages.stage2.data as Record<string, Record<string, unknown>> | null)
				?.['候选人总体评价(Overall Evaluation)'];
			const level = (evalData as Record<string, string> | undefined)?.['总体档位(Overall Level)'] ?? '完成';
			const score = (evalData as Record<string, unknown> | undefined)?.['总体分数(Overall Score)'] ?? '';
			const conf = (evalData as Record<string, number> | undefined)?.['总体置信度(Overall Confidence)'];
			const confStr = conf !== undefined ? ` · 置信度 ${Math.round(Number(conf) * 100)}%` : '';
			yield { type: 'stage_complete', stage: 'stage2', message: `总体档位：${level}（分数 ${score}${confStr}）`, durationMs: Date.now() - t2 };
		}


		// 两阶段 pipeline：stage2 完成后直接保存结果
		const stage2Data = checkpoint.stages.stage2.data;
		if (stage2Data && typeof stage2Data === "string") {
			// stage2 返回的是 Markdown 文本
			const finalText = stage2Data;
			const summary = finalText.replace(/\s+/g, " ").trim().slice(0, 280);
			yield {
				type: "save",
				title: "面试评估报告",
				summary,
				content: finalText,
				rawJson: null
			};
		} else if (stage2Data && typeof stage2Data === "object") {
			// stage2 返回的是 JSON 对象（可能来自旧版本）
			const jsonStr = JSON.stringify(stage2Data, null, 2);
			const summary = jsonStr.replace(/\s+/g, " ").trim().slice(0, 280);
			yield {
				type: "save",
				title: "面试评估报告",
				summary,
				content: jsonStr,
				rawJson: stage2Data
			};
		}

		// 记录报告生成统计信息
		logger.info('pipeline', 'report generated', {
			userId: _user.id,
			userName: _user.name || _user.email,
			candidateName: detail.record.subject,
			notebookId,
			notebookTitle: detail.record.title,
			rowId
		}, 'pipeline');

		await finalizeCheckpoint(rowId, checkpoint, "done");
	} catch (err) {
		const message = err instanceof Error ? err.message : String(err);
		logger.error('pipeline', 'pipeline failed', { notebookId, rowId, error: message }, 'pipeline');
		// 失败时保持 DB 中 status=running，下次触发时 sanitizeForResume 会清理卡住的 stage，实现断点续跑
		// 只有全部成功才 finalize 为 done
		yield { type: 'error', message: `发生严重异常，请立即联系负责人，防止服务崩溃！详细信息：${message}` };
	}
}
