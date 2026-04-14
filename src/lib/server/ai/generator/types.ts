import type { Action, Response, Usage } from '$menglong';

export type ExecutableTool = {
	name: string;
	description: string;
	parameters: {
		type: 'object';
		properties: Record<string, { type: string; description: string; enum?: string[] }>;
		required?: string[];
	};
	handler: (params: Record<string, unknown>) => Promise<unknown> | unknown;
	schema: () => {
		type: 'function';
		function: {
			name: string;
			description: string;
			parameters: {
				type: 'object';
				properties: Record<string, { type: string; description: string; enum?: string[] }>;
				required?: string[];
			};
		};
	};
};

export type GeneratorRunInput = {
	systemPrompt: string;
	userPrompt: string;
	tools?: ExecutableTool[];
	maxSteps?: number;
};

export type GeneratorEvent =
	| { type: 'status'; message: string }
	| { type: 'tool'; name: string; message: string }
	| { type: 'warning'; message: string };

export type GeneratorStreamEvent =
	| { type: 'status'; message: string }
	| { type: 'tool'; name: string; message: string }
	| { type: 'warning'; message: string }
	| { type: 'chunk'; text: string }
	| { type: 'done'; finalText: string; usage?: Usage }
	| { type: 'error'; message: string };

// 保留旧类型兼容
export type GeneratorRunResult = {
	events: GeneratorEvent[];
	finalText: string;
	response?: Response;
	usage?: Usage;
	lastActions?: Action[];
};
