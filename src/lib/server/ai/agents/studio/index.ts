import { readFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createNotebookGeneratorTools, runGenerator } from '$lib/server/ai/generator';
import { getNotebookDetailForUser } from '$lib/server/workbench';
import { runInterviewPipeline } from '$lib/server/ai/pipeline';
import type { StudioAgentInput, StudioAgentKey } from './types';

type StudioAgentUser = Parameters<typeof getNotebookDetailForUser>[0];

// Task type mappings
const taskTypeMap: Record<StudioAgentKey, string> = {
	'audio-overview': 'benchmark',
	'interview-assessment': 'interview-assessment'
};

const studioTitles: Record<StudioAgentKey, string> = {
	'audio-overview': '基准测试',
	'interview-assessment': '面试评估'
};

// Get the directory of the current file (works in both dev and build)
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function readAgentFile(...segments: string[]) {
	const filePath = resolve(__dirname, 'tasks', ...segments);
	return readFile(filePath, 'utf8');
}

function buildUserPrompt(
	detail: NonNullable<Awaited<ReturnType<typeof getNotebookDetailForUser>>>,
	input: StudioAgentInput
) {
	const sourceInfo = [];
	if (input.resumeSourceId) {
		const resume = detail.sources.find(s => s.id === input.resumeSourceId);
		if (resume) sourceInfo.push(`Resume: ${resume.title}`);
	}
	if (input.conversationSourceId) {
		const conv = detail.sources.find(s => s.id === input.conversationSourceId);
		if (conv) sourceInfo.push(`Interview Conversation: ${conv.title}`);
	}
	if (input.jdSourceId) {
		const jd = detail.sources.find(s => s.id === input.jdSourceId);
		if (jd) sourceInfo.push(`Job Description: ${jd.title}`);
	}

	return [
		`Run the Studio tool: ${studioTitles[input.tool]}.`,
		`Notebook title: ${detail.record.title}`,
		`Subject: ${detail.record.subject}`,
		`Tag: ${detail.record.tag}`,
		`Focus: ${input.focus?.trim() || 'overall notebook objective'}`,
		`Tone: ${input.tone?.trim() || 'concise'}`,
		`Length: ${input.length?.trim() || 'standard'}`,
		`Include citations: ${input.includeCitations ? 'yes' : 'no'}`,
		sourceInfo.length > 0 ? `Selected sources: ${sourceInfo.join(', ')}` : '',
		'Use tools before finalizing if you need notebook evidence or source content.',
		'Return only the finished markdown deliverable.'
	].filter(Boolean).join('\n');
}

function summarizeOutput(text: string) {
	const normalized = text.replace(/\s+/g, ' ').trim();
	return normalized.slice(0, 280);
}

export async function* streamStudioAgent(
	user: StudioAgentUser,
	slug: string,
	input: StudioAgentInput
): AsyncGenerator<string> {
	const detail = await getNotebookDetailForUser(user, slug);
	if (!detail) {
		yield JSON.stringify({ type: 'error', message: 'Notebook not found.' }) + '\n';
		return;
	}

	// ─── interview-assessment 走专用 Pipeline ────────────────────────────────
	if (input.tool === 'interview-assessment') {
		for await (const event of runInterviewPipeline(user, detail, input.evaluationMode ?? 'basic', input.resumeCheckpoint ?? false)) {
			yield JSON.stringify(event) + '\n';
		}
		return;
	}

	// ─── 其他 agent 走通用 Generator ────────────────────────────────────────
	const taskType = taskTypeMap[input.tool];
	const [systemPrompt, knowledgePrompt, taskPrompt] = await Promise.all([
		readAgentFile(taskType, 'system.md'),
		readAgentFile(taskType, 'knowledge.md'),
		readAgentFile(taskType, 'task.md')
	]);

	let finalText = '';

	for await (const event of runGenerator({
		systemPrompt: [systemPrompt.trim(), knowledgePrompt.trim(), taskPrompt.trim()].join('\n\n'),
		userPrompt: buildUserPrompt(detail, input),
		tools: createNotebookGeneratorTools(detail)
	})) {
		if (event.type === 'done') {
			finalText = event.finalText;
		}
		yield JSON.stringify(event) + '\n';
	}

	// 返回 save 所需数据
	yield JSON.stringify({
		type: 'save',
		title: studioTitles[input.tool],
		summary: summarizeOutput(finalText),
		content: finalText
	}) + '\n';
}

export type { StudioAgentInput, StudioAgentKey } from './types';
