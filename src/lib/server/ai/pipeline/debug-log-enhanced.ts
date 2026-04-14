/**
 * Pipeline debug logger — 增强版，每个stage单独文件，记录全部过程
 */
import { appendFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { PipelineStageId } from './types';

const LOG_DIR = resolve('.svelte-kit', 'debug', 'pipeline');

function ensureStageLogDir(stageId: PipelineStageId) {
	const stageDir = resolve(LOG_DIR, stageId);
	try {
		mkdirSync(stageDir, { recursive: true });
	} catch { /* ignore */ }
	return stageDir;
}

function getStageLogFile(stageId: PipelineStageId): string {
	const stageDir = ensureStageLogDir(stageId);
	return resolve(stageDir, 'pipeline.log');
}

/**
 * 记录stage级别的详细日志
 */
export function debugStageLog(stageId: PipelineStageId, tag: string, data: unknown) {
	const isDebug = process.env.HUSHI_LLM_DEBUG === 'true';
	if (!isDebug) return;
	try {
		const logFile = getStageLogFile(stageId);
		const timestamp = new Date().toISOString();
		let logLine: string;

		if (typeof data === 'string') {
			logLine = `${timestamp} [${tag}] ${data}`;
		} else if (data === null || data === undefined) {
			logLine = `${timestamp} [${tag}]`;
		} else {
			// 对于复杂对象，格式化JSON
			const jsonStr = JSON.stringify(data, null, 2);
			// 如果需要缩进，确保每行都有标签
			const lines = jsonStr.split('\n');
			logLine = lines.map((line, i) => {
				if (i === 0) return `${timestamp} [${tag}] ${line}`;
				return `${' '.repeat(31 + tag.length + 2)}${line}`;
			}).join('\n');
		}

		appendFileSync(logFile, logLine + '\n');
		console.log(`[DEBUG ${stageId}] ${tag}:`, data);
	} catch (error) {
		console.error(`写入stage调试日志失败: ${error}`);
	}
}

/**
 * 记录工具调用
 */
export function debugToolCall(stageId: PipelineStageId, toolName: string, action: 'start' | 'args' | 'complete' | 'error', data?: unknown) {
	debugStageLog(stageId, `tool.${toolName}.${action}`, data || '');
}

/**
 * 记录流式输出
 */
export function debugStreamChunk(stageId: PipelineStageId, type: 'thinking' | 'text' | 'tool_chunk', chunk: string) {
	const truncated = chunk.length > 200 ? chunk.slice(0, 200) + '...' : chunk;
	debugStageLog(stageId, `stream.${type}`, truncated);
}

/**
 * 记录pipeline事件
 */
export function debugPipelineEvent(stageId: PipelineStageId, eventType: string, data: unknown) {
	debugStageLog(stageId, `event.${eventType}`, data);
}

/**
 * 初始化debug目录
 */
export function initDebugLog() {
	const isDebug = process.env.HUSHI_LLM_DEBUG === 'true';
	if (!isDebug) return;
	try {
		mkdirSync(LOG_DIR, { recursive: true });
		writeFileSync(resolve(LOG_DIR, 'README.txt'), `Pipeline Debug Logs - 每个stage单独文件夹

每个stage文件夹包含：
- pipeline.log文件：该stage的详细日志（每次运行会追加内容）
- 日志内容包括：thinking流式输出、tool调用、事件等
- 格式：时间戳 [标签] 内容

调试时查看控制台输出或直接查看这些日志文件。
`);
	} catch { /* ignore */ }
}

// 自动初始化
initDebugLog();