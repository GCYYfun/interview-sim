/**
 * Pipeline debug logger — 写到 .svelte-kit/debug/pipeline.log
 * 使用 process.env 而非 $env/dynamic/private（该模块在请求上下文外使用）
 */
import { appendFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const LOG_DIR  = resolve('.svelte-kit', 'debug');
const LOG_FILE = resolve(LOG_DIR, 'pipeline.log');
let initialized = false;

function ensureInit() {
	if (initialized) return;
	const isDebug = process.env.HUSHI_LLM_DEBUG === 'true';
	if (!isDebug) {
		initialized = true;
		return;
	}
	try {
		mkdirSync(LOG_DIR, { recursive: true });
		// 每次进程启动清空旧日志
		writeFileSync(LOG_FILE, `=== pipeline debug log started ${new Date().toISOString()} ===\n`);
	} catch { /* ignore */ }
	initialized = true;
}

export function debugLog(tag: string, data: unknown) {
	const isDebug = process.env.HUSHI_LLM_DEBUG === 'true';
	if (!isDebug) return;

	// 总是写入调试日志，便于排查问题
	ensureInit();
	const line = `[${new Date().toISOString()}] [${tag}] ${
		typeof data === 'string' ? data : JSON.stringify(data)
	}\n`;
	try {
		appendFileSync(LOG_FILE, line);
		// 同时输出到控制台，便于即时查看
		console.log(`[DEBUG] ${tag}:`, data);
	} catch (error) {
		console.error(`写入调试日志失败: ${error}`);
	}
}

/**
 * 受 HUSHI_LLM_DEBUG 控制的 console.log 封装
 */
export function log(...args: any[]) {
	if (process.env.HUSHI_LLM_DEBUG === 'true') {
		console.log(...args);
	}
}
