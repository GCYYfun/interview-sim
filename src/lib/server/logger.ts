/**
 * 服务端统一日志模块
 *
 * - ERROR / WARN：始终写入文件，方便生产环境排查问题
 * - INFO / DEBUG：仅在 HUSHI_LLM_DEBUG=true 时写入
 * - 日志目录：.svelte-kit/logs/
 *   - web.log      — HTTP 请求 / 业务错误
 *   - pipeline.log — AI Pipeline 详细过程（与 debug-log.ts 共存，格式统一）
 *
 * 使用示例：
 *   import { logger } from '$lib/server/logger';
 *   logger.info('analyze', 'request received', { userId, slug });
 *   logger.error('analyze', 'stream failed', err);
 */
import { appendFileSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

// ─── 常量 ────────────────────────────────────────────────────────────────────

const LOG_DIR = resolve('.svelte-kit', 'logs');
const WEB_LOG = resolve(LOG_DIR, 'web.log');
const PIPELINE_LOG = resolve(LOG_DIR, 'pipeline.log');

let _dirReady = false;

function ensureDir() {
	if (_dirReady) return;
	try {
		mkdirSync(LOG_DIR, { recursive: true });
	} catch { /* ignore */ }
	_dirReady = true;
}

// ─── 日志级别 ─────────────────────────────────────────────────────────────────

type Level = 'DEBUG' | 'INFO' | 'WARN' | 'ERROR';

const LEVEL_RANK: Record<Level, number> = {
	DEBUG: 0,
	INFO:  1,
	WARN:  2,
	ERROR: 3,
};

function isDebugMode() {
	return process.env.HUSHI_LLM_DEBUG === 'true';
}

/** 当前允许写入的最低级别：INFO 始终记录，DEBUG 需要开启 HUSHI_LLM_DEBUG=true */
function minLevel(): Level {
	return isDebugMode() ? 'DEBUG' : 'INFO';
}

// ─── 格式化 ───────────────────────────────────────────────────────────────────

function formatData(data: unknown): string {
	if (data === undefined || data === null) return '';
	if (data instanceof Error) {
		return ` | ${data.message}${data.stack ? '\n' + data.stack : ''}`;
	}
	if (typeof data === 'string') return ` | ${data}`;
	try {
		return ` | ${JSON.stringify(data)}`;
	} catch {
		return ` | [unserializable]`;
	}
}

function buildLine(level: Level, tag: string, message: string, data?: unknown): string {
	const ts = new Date().toISOString();
	return `${ts} [${level.padEnd(5)}] [${tag}] ${message}${formatData(data)}\n`;
}

// ─── 写入 ─────────────────────────────────────────────────────────────────────

function write(target: 'web' | 'pipeline', line: string) {
	ensureDir();
	const file = target === 'pipeline' ? PIPELINE_LOG : WEB_LOG;
	try {
		appendFileSync(file, line);
	} catch { /* ignore */ }
}

// ─── 核心 log 函数 ────────────────────────────────────────────────────────────

function log(
	level: Level,
	tag: string,
	message: string,
	data?: unknown,
	target: 'web' | 'pipeline' = 'web'
) {
	if (LEVEL_RANK[level] < LEVEL_RANK[minLevel()]) return;

	const line = buildLine(level, tag, message, data);
	write(target, line);

	// 同步到控制台
	if (level === 'ERROR') {
		console.error(line.trimEnd());
	} else if (level === 'WARN') {
		console.warn(line.trimEnd());
	} else if (isDebugMode()) {
		console.log(line.trimEnd());
	}
}

// ─── 公开 API ─────────────────────────────────────────────────────────────────

export const logger = {
	debug: (tag: string, message: string, data?: unknown, target?: 'web' | 'pipeline') =>
		log('DEBUG', tag, message, data, target),
	info:  (tag: string, message: string, data?: unknown, target?: 'web' | 'pipeline') =>
		log('INFO',  tag, message, data, target),
	warn:  (tag: string, message: string, data?: unknown, target?: 'web' | 'pipeline') =>
		log('WARN',  tag, message, data, target),
	error: (tag: string, message: string, data?: unknown, target?: 'web' | 'pipeline') =>
		log('ERROR', tag, message, data, target),
};

/**
 * 记录 HTTP 请求（供 hooks.server.ts 调用）
 */
export function logRequest(method: string, url: string, userId?: string) {
	log('INFO', 'http', `${method} ${url}`, userId ? { userId } : undefined);
}

/**
 * 记录 HTTP 响应
 */
export function logResponse(method: string, url: string, status: number, durationMs: number) {
	const level: Level = status >= 500 ? 'ERROR' : status >= 400 ? 'WARN' : 'INFO';
	log(level, 'http', `${method} ${url} → ${status} (${durationMs}ms)`);
}
