import { env } from '$env/dynamic/private';
import { ConsoleLogger, Model, MengLongProvider, ProviderRegistry, type Config } from '$menglong';

// 将 anthropic 和 infinigence 前缀路由到 menglong provider（网关统一处理，返回格式一致）
if (!ProviderRegistry.getProviderClass('anthropic')) {
	console.log('[config] 注册 anthropic provider');
	ProviderRegistry.register('anthropic', MengLongProvider);
}
if (!ProviderRegistry.getProviderClass('infinigence')) {
	console.log('[config] 注册 infinigence provider');
	ProviderRegistry.register('infinigence', MengLongProvider);
}
console.log('[config] 已注册的 providers:', ProviderRegistry.listProviders());

export class AiConfigurationError extends Error {
	constructor(message: string) {
		super(message);
		this.name = 'AiConfigurationError';
	}
}

export function getDefaultModelId() {
	return env.HUSHI_LLM_MODEL?.trim() || 'menglong/deepseek-chat';
}

export function createAiModel() {
	const modelId = getDefaultModelId();
	const config: Config = {
		default: {
			model_id: modelId
		},
		providers: {
			menglong: {
				base_url: env.MENGLONG_BASE_URL?.trim() || 'http://localhost:8000/menglong',
				api_key: env.MENGLONG_API_KEY?.trim() || undefined,
				timeout: 300
			},
			anthropic: {
				base_url: env.MENGLONG_BASE_URL?.trim() || 'http://localhost:8000/menglong',
				api_key: env.MENGLONG_API_KEY?.trim() || undefined,
				timeout: 300
			},
			infinigence: {
				base_url: env.MENGLONG_BASE_URL?.trim() || 'http://localhost:8000/menglong',
				api_key: env.MENGLONG_API_KEY?.trim() || undefined,
				timeout: 300
			}
		}
	};

	return new Model(modelId, {
		config,
		logger: env.HUSHI_LLM_DEBUG === 'true' ? new ConsoleLogger() : undefined
	});
}
