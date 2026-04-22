/**
 * runner-shared.ts — Pipeline runner 共享工具
 *
 * 被 runner-basic.ts 和 runner-markdown.ts 共同引用，
 * 避免 createPipelineModel / loadPrompt / fillPlaceholders 重复定义。
 */
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { Model, MengLongProvider, ProviderRegistry, ConsoleLogger } from '$menglong';
import { env } from '$env/dynamic/private';
import type { Config } from '$menglong';
import { debugLog, log } from '../debug-log';

// ─── Provider 注册（模块级，只执行一次）────────────────────────────────────────
// runner-basic 和 runner-markdown 都会 import 本模块，
// 由于 JS 模块只初始化一次，provider 注册也只会执行一次，不会重复注册。
if (!ProviderRegistry.getProviderClass('infinigence')) {
	log('[runner-shared] 注册 infinigence provider');
	ProviderRegistry.register('infinigence', MengLongProvider);
} else {
	log('[runner-shared] infinigence provider 已注册');
}

// ─── Model 工厂 ───────────────────────────────────────────────────────────────

export function createPipelineModel() {
	const modelId =
		env.PIPELINE_LLM_MODEL?.trim() ||
		env.HUSHI_LLM_MODEL?.trim() ||
		'infinigence/deepseek-v3.2-thinking';
	const apiKey = env.MENGLONG_API_KEY?.trim() || undefined;
	const baseUrl = env.MENGLONG_BASE_URL?.trim() || 'http://localhost:8000/menglong';
	const config: Config = {
		default: { model_id: modelId },
		providers: {
			menglong:    { base_url: baseUrl, api_key: apiKey, timeout: 600 },
			anthropic:   { base_url: baseUrl, api_key: apiKey, timeout: 600 },
			infinigence: { base_url: baseUrl, api_key: apiKey, timeout: 600 }
		}
	};
	debugLog('createPipelineModel', { modelId, baseUrl });
	const sdkLogger = process.env.HUSHI_LLM_DEBUG === 'true' ? new ConsoleLogger() : undefined;
	return new Model(modelId, { config, logger: sdkLogger });
}

// ─── Prompt 加载（带缓存）────────────────────────────────────────────────────

const PROMPT_DIR = resolve('prompt');

/** 共享 prompt 缓存：两个 runner 共用同一个 Map，同一文件只读一次磁盘 */
const promptCache = new Map<string, string>();

export async function loadPrompt(filename: string): Promise<string> {
	if (!promptCache.has(filename)) {
		const content = await readFile(resolve(PROMPT_DIR, filename), 'utf-8');
		promptCache.set(filename, content);
	}
	return promptCache.get(filename)!;
}

// ─── Prompt 填充 ──────────────────────────────────────────────────────────────

/** 替换 prompt 中的 {{PIPELINE_XXX}} 占位符 */
export function fillPlaceholders(template: string, vars: Record<string, string>): string {
	return Object.entries(vars).reduce(
		(text, [key, value]) => text.replaceAll(`{{${key}}}`, value),
		template
	);
}
