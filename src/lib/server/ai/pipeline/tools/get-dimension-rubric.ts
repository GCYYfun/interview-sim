import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { defineTool } from '$menglong';
import type { DimensionKey } from '../types';

const PROMPT_DIR = resolve('prompt');

let cachedRubric: string | null = null;

async function loadDimensionRubric(): Promise<string> {
	if (!cachedRubric) {
		cachedRubric = await readFile(resolve(PROMPT_DIR, 'dimension-rubric.md'), 'utf-8');
	}
	return cachedRubric;
}

/** 从 dimension-rubric.md 中按 ## 标题切割出指定维度章节 */
function extractSection(rubric: string, dimension: DimensionKey): string {
	const lines = rubric.split('\n');
	const startIdx = lines.findIndex((line) => line.startsWith(`## ${dimension}`));
	if (startIdx === -1) return `[未找到维度章节: ${dimension}]`;

	// 找到下一个 ## 标题（同级），作为结束
	let endIdx = lines.length;
	for (let i = startIdx + 1; i < lines.length; i++) {
		if (lines[i].startsWith('## ')) {
			endIdx = i;
			break;
		}
	}
	return lines.slice(startIdx, endIdx).join('\n').trim();
}

export function createGetDimensionRubricTool() {
	return defineTool({
		name: 'get_dimension_rubric',
		description:
			'获取指定评估维度的评分标准（Rubric）章节，包含子维度定义、Trigger/Gate、证据点锚点和档位分数表。',
		parameters: {
			type: 'object',
			properties: {
				dimension: {
					type: 'string',
					description: '要获取的维度名称',
					enum: ['聪明度', '勤奋度', '目标感', '皮实度', '客户第一']
				}
			},
			required: ['dimension']
		},
		handler: async (params) => {
			const dimension = String(params.dimension ?? '') as DimensionKey;
			const rubric = await loadDimensionRubric();
			const section = extractSection(rubric, dimension);
			return { dimension, rubric_section: section };
		}
	});
}
