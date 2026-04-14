import { and, asc, desc, eq } from 'drizzle-orm';
import { client, db } from '$lib/server/db';
import {
	notebook,
	notebookMessage,
	notebookOutput,
	notebookSource,
	user
} from '$lib/server/db/schema';
import type { SourceItem, SourceTemplateItem } from '$lib/data/workbench';
import { sourceTemplates } from '$lib/data/workbench';

type UserRecord = typeof user.$inferSelect;
type NotebookRecord = typeof notebook.$inferSelect;
type SourceRecord = typeof notebookSource.$inferSelect;
type OutputRecord = typeof notebookOutput.$inferSelect;
type MessageRecord = typeof notebookMessage.$inferSelect;

const defaultAccent = 'linear-gradient(135deg, rgba(255, 255, 255, 0.03), rgba(23, 25, 31, 0.05))';

function slugify(text: string): string {
	return text
		.toLowerCase()
		.trim()
		.replace(/[\s_]+/g, '-')
		.replace(/[^\w-]+/g, '')
		.replace(/--+/g, '-')
		.replace(/^-+|-+$/g, '');
}

function titleCaseTool(tool: string): string {
	return tool
		.split('-')
		.map((word) => word.charAt(0).toUpperCase() + word.slice(1))
		.join(' ');
}

function timeAgoLabel(date: string | number | Date): string {
	const now = new Date();
	const past = new Date(date);
	const diffMs = now.getTime() - past.getTime();
	const diffSecs = Math.floor(diffMs / 1000);
	const diffMins = Math.floor(diffSecs / 60);
	const diffHours = Math.floor(diffMins / 60);
	const diffDays = Math.floor(diffHours / 24);

	if (diffSecs < 60) return 'Just now';
	if (diffMins < 60) return `${diffMins}m ago`;
	if (diffHours < 24) return `${diffHours}h ago`;
	if (diffDays < 7) return `${diffDays}d ago`;
	
	return past.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function summariseSourceType(sourceType: string) {
	switch (sourceType) {
		case 'resume':
			return 'Resume';
		case 'conversation':
			return 'Conversation';
		case 'job-description':
			return 'Job Description';
		case 'website':
			return 'Website';
		case 'portfolio':
			return 'Portfolio';
		default:
			return 'Source';
	}
}

function buildAssistantReply(
	prompt: string,
	selectedSource: SourceRecord | undefined,
	record: NotebookRecord
) {
	const normalized = prompt.toLowerCase();
	if (normalized.includes('strength')) {
		return `${record.subject} reads strongest in ${record.tag.toLowerCase()}, with the clearest signal coming from ${selectedSource?.title ?? 'the current source set'}. The evidence points to repeatable ownership, clear delivery outcomes, and a stronger-than-average narrative for scoped leadership.`;
	}
	if (normalized.includes('gap') || normalized.includes('risk')) {
		return `The biggest open question is depth around ${selectedSource ? selectedSource.meta.toLowerCase() : 'execution details'}. I would validate how ${record.subject} measures impact, collaborates cross-functionally, and handles ambiguous requirements before making a final recommendation.`;
	}
	if (normalized.includes('interview')) {
		return `Suggested interview angle: ask ${record.subject} to walk through one project from problem framing to shipped outcome, then probe tradeoffs, stakeholder management, and how success was measured.`;
	}
	return `Using ${selectedSource?.title ?? 'the available sources'}, I would summarize ${record.subject} as a strong fit for workflows centered on ${record.tag.toLowerCase()}. If you want, I can next break this into strengths, risks, or interview questions.`;
}

function buildStudioOutput(tool: string, record: NotebookRecord, sourceCount: number) {
	const toolName = titleCaseTool(tool);
	const summaryMap: Record<string, string> = {
		'audio-overview': `${record.subject} has ${sourceCount} active sources. This brief is structured as a spoken overview that highlights headline strengths, key experience signals, and the most useful follow-up probes.`,
		'resume-analysis': `${record.subject} shows the clearest signal in ${record.tag.toLowerCase()}, with evidence concentrated around repeatable delivery, source quality, and role-aligned impact statements.`,
		'skill-matching': `${record.subject} aligns best where role requirements emphasize strategic ownership, documentation quality, and cross-functional execution. The most defensible match comes from the primary source set.`,
		'interview-qs': `Prepared a focused interview pack for ${record.subject}, covering craft depth, decision making, stakeholder alignment, and evidence gaps that still need validation.`
	};

	return {
		title: toolName,
		summary:
			summaryMap[tool] ??
			`${toolName} generated for ${record.subject} using ${sourceCount} ingested sources and the current notebook summary.`,
		content: `${toolName}\n\nPrepared for ${record.subject} with ${sourceCount} active sources. This saved output is a lightweight non-streaming draft intended to capture the current notebook state before deeper analysis is run.`
	};
}

function chunkTextForStream(text: string) {
	return text
		.split('\n')
		.flatMap((line) => (line.trim() ? [`${line}\n`] : ['\n']))
		.filter(Boolean);
}

function isEditableSourceRecord(source: SourceRecord) {
	const lowerTitle = source.title.toLowerCase();
	const editableExtensions = ['.md', '.txt', '.csv', '.json'];
	if (editableExtensions.some((extension) => lowerTitle.endsWith(extension))) {
		return true;
	}

	return ['conversation', 'website', 'job-description', 'dataset'].includes(source.sourceType);
}

function buildMockStudioAnalysis(
	tool: string,
	record: NotebookRecord,
	selectedSource: SourceRecord | undefined,
	config: {
		focus?: string;
		tone?: string;
		length?: string;
		includeCitations?: boolean;
	}
) {
	const toolName = titleCaseTool(tool);
	const sourceLabel = selectedSource?.title ?? 'the current source set';
	const focus = config.focus?.trim() || 'overall candidate fit';
	const tone = config.tone?.trim() || 'concise';
	const length = config.length?.trim() || 'standard';
	const citationLine = config.includeCitations
		? `\nEvidence anchors: ${sourceLabel}; ${selectedSource?.meta ?? 'Notebook summary'}`
		: '';

	const text = `${toolName}

Objective
Produce a ${tone} ${length} analysis for ${record.subject}, centered on ${focus}.

Context
The selected source is ${sourceLabel}. This notebook is tagged ${record.tag} and currently positions ${record.subject} around ${record.summary}

Assessment
${record.subject} shows the strongest signal in areas supported directly by ${sourceLabel}. The source material indicates repeatable ownership, structured execution, and enough context to form a working recommendation. The biggest advantage in this pass is clarity: the notebook already frames the candidate around a concrete narrative instead of disconnected bullet points.

Key Findings
1. The strongest evidence is tied to delivery and decision-making, not just surface-level title inflation.
2. The narrative is easier to defend when grounded in the selected source, which gives this analysis a cleaner spine.
3. The main risk is not lack of signal, but where supporting evidence still needs validation or triangulation.

Recommended Next Step
Use this output as the working summary in the right-hand Saved outputs area, then expand with another source if you want a broader or more comparative read.${citationLine}`;

	return {
		title: toolName,
		summary: `Generated ${tone} ${length} analysis for ${record.subject}, focused on ${focus}.`,
		text,
		chunks: chunkTextForStream(text)
	};
}

export async function getWorkspaceHome(user: UserRecord) {
	const records = await db
		.select()
		.from(notebook)
		.where(eq(notebook.userId, user.id))
		.orderBy(desc(notebook.updatedAt));

	return records.map((record, index) => ({
		id: record.id,
		slug: record.slug,
		title: record.title,
		tag: record.tag,
		subject: record.subject,
		summary: record.summary,
		location: record.location,
		contact: record.contact,
		link: record.link,
		role: record.role as 'OWNER' | 'EDITOR' | 'VIEWER',
		accent: record.accent,
		sourcesLabel: '0 sources',
		sourceCount: 0,
		createdDate: new Date(record.createdAt).toLocaleDateString('en-US', {
			month: 'short',
			day: '2-digit',
			year: 'numeric'
		}),
		updatedAgo: index === 0 ? 'Just now' : timeAgoLabel(record.updatedAt)
	}));
}

export async function enrichWorkspaceList(items: Awaited<ReturnType<typeof getWorkspaceHome>>) {
	const counts = await Promise.all(
		items.map(async (item) => {
			const sources = await db
				.select({ id: notebookSource.id })
				.from(notebookSource)
				.where(eq(notebookSource.notebookId, item.id));

			return {
				notebookId: item.id,
				sourceCount: sources.length
			};
		})
	);

	return items.map((item) => {
		const count = counts.find((entry) => entry.notebookId === item.id)?.sourceCount ?? 0;

		return {
			...item,
			sourceCount: count,
			sourcesLabel: `${count} ${count === 1 ? 'source' : 'sources'}`
		};
	});
}

export async function createNotebookForUser(
	user: UserRecord,
	input: { title: string; subject: string; tag?: string }
) {
	const title = input.title.trim();
	const subject = input.subject.trim();
	const tag = input.tag?.trim() || '';
	const slugBase = slugify(`${title}-${subject}`);
	const now = Date.now();

	const existing = await db
		.select({ slug: notebook.slug })
		.from(notebook)
		.where(and(eq(notebook.userId, user.id), eq(notebook.slug, slugBase)))
		.limit(1);

	const slug = existing.length > 0 ? `${slugBase}-${now.toString().slice(-4)}` : slugBase;
	const notebookId = crypto.randomUUID();

	await db.insert(notebook).values({
		id: notebookId,
		userId: user.id,
		slug,
		title,
		tag,
		summary: `${subject} notebook created.`,
		subject,
		location: '',
		contact: '',
		link: '',
		role: 'OWNER',
		accent: defaultAccent,
		createdAt: now,
		updatedAt: now
	});

	await db.insert(notebookMessage).values({
		id: crypto.randomUUID(),
		notebookId,
		role: 'assistant',
		content: `Notebook created for ${subject}. Add a resume, conversation, or job description to begin synthesis.`,
		createdAt: now
	});

	return slug;
}

export async function getNotebookDetailForUser(
	user: UserRecord,
	slug: string,
	selectedSourceId?: string | null
) {
	const records = await db
		.select()
		.from(notebook)
		.where(and(eq(notebook.userId, user.id), eq(notebook.slug, slug)))
		.limit(1);

	const record = records[0];
	if (!record) return null;

	const sources = await db
		.select()
		.from(notebookSource)
		.where(eq(notebookSource.notebookId, record.id))
		.orderBy(asc(notebookSource.createdAt));

	const outputs = await db
		.select()
		.from(notebookOutput)
		.where(eq(notebookOutput.notebookId, record.id))
		.orderBy(desc(notebookOutput.createdAt));

	const messages = await db
		.select()
		.from(notebookMessage)
		.where(eq(notebookMessage.notebookId, record.id))
		.orderBy(asc(notebookMessage.createdAt));

	const selectedSource =
		sources.find((source) => source.id === selectedSourceId) ??
		sources.find((source) => source.isSelected) ??
		sources[0];

	if (selectedSource && !selectedSource.isSelected) {
		await setSelectedSource(record.id, selectedSource.id);
	}

	return {
		record,
		selectedSource,
		sources,
		outputs,
		messages,
		sourceTemplates
	};
}

async function setSelectedSource(notebookId: string, sourceId: string) {
	const sources = await db
		.select({ id: notebookSource.id })
		.from(notebookSource)
		.where(eq(notebookSource.notebookId, notebookId));

	for (const source of sources) {
		await db
			.update(notebookSource)
			.set({ isSelected: source.id === sourceId })
			.where(eq(notebookSource.id, source.id));
	}
}

export async function addSourceToNotebook(
	user: UserRecord,
	slug: string,
	input: {
		title: string;
		sourceType: string;
		content: string;
		meta?: string;
		subtitle?: string;
	}
) {
	const detail = await getNotebookDetailForUser(user, slug);
	if (!detail) return null;

	const now = Date.now();
	const title = input.title.trim();
	const sourceType = input.sourceType.trim();
	const content = input.content.trim();
	const meta = input.meta?.trim() || `${summariseSourceType(sourceType)} source`;

	const iconMap: Record<string, SourceItem['icon']> = {
		resume: 'file-text',
		conversation: 'message-square',
		'job-description': 'briefcase',
		website: 'link',
		portfolio: 'file',
		dataset: 'file'
	};

	const existingSources = detail.sources.length;
	await db.insert(notebookSource).values({
		id: crypto.randomUUID(),
		notebookId: detail.record.id,
		title,
		subtitle: input.subtitle?.trim() || `${summariseSourceType(sourceType)} • Added just now`,
		meta,
		icon: iconMap[sourceType] ?? 'file-text',
		content,
		sourceType,
		isSelected: existingSources === 0,
		createdAt: now
	});

	await db
		.update(notebook)
		.set({
			summary: `Source library expanded with ${title}. The notebook is ready for a refreshed analysis pass.`,
			updatedAt: now
		})
		.where(eq(notebook.id, detail.record.id));

	return detail.record.slug;
}

export async function createStudioOutput(user: UserRecord, slug: string, tool: string) {
	const detail = await getNotebookDetailForUser(user, slug);
	if (!detail) return null;

	const now = Date.now();
	const output = buildStudioOutput(tool, detail.record, detail.sources.length);

	await db.insert(notebookOutput).values({
		id: crypto.randomUUID(),
		notebookId: detail.record.id,
		title: output.title,
		summary: output.summary,
		content: output.content,
		updatedAgoLabel: 'Just now',
		progress: tool === 'audio-overview' ? 76 : null,
		duration: tool === 'audio-overview' ? '09:42' : null,
		createdAt: now
	});

	await db
		.update(notebook)
		.set({
			updatedAt: now
		})
		.where(eq(notebook.id, detail.record.id));

	return detail.record.slug;
}

export async function createMockStudioOutput(
	user: UserRecord,
	slug: string,
	input: {
		tool: string;
		focus?: string;
		tone?: string;
		length?: string;
		includeCitations?: boolean;
	}
) {
	const detail = await getNotebookDetailForUser(user, slug);
	if (!detail) return null;

	const now = Date.now();
	const generated = buildMockStudioAnalysis(input.tool, detail.record, detail.selectedSource, {
		focus: input.focus,
		tone: input.tone,
		length: input.length,
		includeCitations: input.includeCitations
	});

	await db.insert(notebookOutput).values({
		id: crypto.randomUUID(),
		notebookId: detail.record.id,
		title: generated.title,
		summary: generated.text.slice(0, 280),
		content: generated.text,
		updatedAgoLabel: 'Just now',
		progress: null,
		duration: null,
		createdAt: now
	});

	await db
		.update(notebook)
		.set({
			summary: generated.summary,
			updatedAt: now
		})
		.where(eq(notebook.id, detail.record.id));

	return generated;
}

export async function saveNotebookOutput(
	user: UserRecord,
	slug: string,
	input: {
		title: string;
		summary: string;
		content: string;
	}
) {
	const detail = await getNotebookDetailForUser(user, slug);
	if (!detail) return null;

	const now = Date.now();

	await db.insert(notebookOutput).values({
		id: crypto.randomUUID(),
		notebookId: detail.record.id,
		title: input.title.trim(),
		summary: input.summary.trim(),
		content: input.content.trim(),
		updatedAgoLabel: 'Just now',
		progress: null,
		duration: null,
		createdAt: now
	});

	await db
		.update(notebook)
		.set({
			summary: input.summary.trim(),
			updatedAt: now
		})
		.where(eq(notebook.id, detail.record.id));

	return true;
}

export async function createNotebookMessage(user: UserRecord, slug: string, prompt: string) {
	const detail = await getNotebookDetailForUser(user, slug);
	if (!detail) return null;

	const content = prompt.trim();
	const now = Date.now();

	await db.insert(notebookMessage).values({
		id: crypto.randomUUID(),
		notebookId: detail.record.id,
		role: 'user',
		content,
		createdAt: now
	});

	await db.insert(notebookMessage).values({
		id: crypto.randomUUID(),
		notebookId: detail.record.id,
		role: 'assistant',
		content: buildAssistantReply(content, detail.selectedSource, detail.record),
		createdAt: now + 1
	});

	await db
		.update(notebook)
		.set({
			updatedAt: now
		})
		.where(eq(notebook.id, detail.record.id));

	return detail.record.slug;
}

export async function chooseNotebookSource(user: UserRecord, slug: string, sourceId: string) {
	const detail = await getNotebookDetailForUser(user, slug);
	if (!detail) return null;

	const selected = detail.sources.find((source) => source.id === sourceId);
	if (!selected) return null;

	await setSelectedSource(detail.record.id, sourceId);
	await db.update(notebook).set({ updatedAt: Date.now() }).where(eq(notebook.id, detail.record.id));

	return detail.record.slug;
}

export async function updateNotebookDetails(
	user: UserRecord,
	slug: string,
	input: {
		title: string;
		subject: string;
		tag: string;
		location: string;
		contact: string;
		link: string;
		summary: string;
	}
) {
	const detail = await getNotebookDetailForUser(user, slug);
	if (!detail) return null;

	const now = Date.now();
	const title = input.title.trim();
	const subject = input.subject.trim();
	const tag = input.tag.trim();
	const location = input.location.trim();
	const contact = input.contact.trim();
	const link = input.link.trim();
	const summary = input.summary.trim();
	const nextSlugBase = slugify(`${title}-${subject}`);

	const existing = await db
		.select({ id: notebook.id, slug: notebook.slug })
		.from(notebook)
		.where(and(eq(notebook.userId, user.id), eq(notebook.slug, nextSlugBase)));

	const slugTakenByOther = existing.some((entry) => entry.id !== detail.record.id);
	const nextSlug = slugTakenByOther ? `${nextSlugBase}-${now.toString().slice(-4)}` : nextSlugBase;

	await db
		.update(notebook)
		.set({
			title,
			subject,
			tag,
			location,
			contact,
			link,
			summary,
			slug: nextSlug,
			updatedAt: now
		})
		.where(eq(notebook.id, detail.record.id));

	return nextSlug;
}

export async function deleteNotebookForUser(user: UserRecord, slug: string) {
	const detail = await getNotebookDetailForUser(user, slug);
	if (!detail) return null;

	await db.delete(notebookSource).where(eq(notebookSource.notebookId, detail.record.id));
	await db.delete(notebookOutput).where(eq(notebookOutput.notebookId, detail.record.id));
	await db.delete(notebookMessage).where(eq(notebookMessage.notebookId, detail.record.id));
	await db.delete(notebook).where(eq(notebook.id, detail.record.id));

	return true;
}

export async function deleteNotebookSource(user: UserRecord, slug: string, sourceId: string) {
	const detail = await getNotebookDetailForUser(user, slug);
	if (!detail) return null;

	const source = detail.sources.find((entry) => entry.id === sourceId);
	if (!source) return null;

	await db.delete(notebookSource).where(eq(notebookSource.id, sourceId));

	const remainingSources = detail.sources.filter((entry) => entry.id !== sourceId);
	if (source.isSelected && remainingSources[0]) {
		await setSelectedSource(detail.record.id, remainingSources[0].id);
	}

	await db
		.update(notebook)
		.set({
			updatedAt: Date.now()
		})
		.where(eq(notebook.id, detail.record.id));

	return true;
}

export async function deleteNotebookOutput(user: UserRecord, slug: string, outputId: string) {
	const detail = await getNotebookDetailForUser(user, slug);
	if (!detail) return null;

	const output = detail.outputs.find((entry) => entry.id === outputId);
	if (!output) return null;

	await db.delete(notebookOutput).where(eq(notebookOutput.id, outputId));
	await db
		.update(notebook)
		.set({
			updatedAt: Date.now()
		})
		.where(eq(notebook.id, detail.record.id));

	return true;
}

export async function updateNotebookOutputTitle(
	user: UserRecord,
	slug: string,
	outputId: string,
	title: string
) {
	const detail = await getNotebookDetailForUser(user, slug);
	if (!detail) return null;

	const output = detail.outputs.find((entry) => entry.id === outputId);
	if (!output) return null;

	await db
		.update(notebookOutput)
		.set({
			title: title.trim()
		})
		.where(eq(notebookOutput.id, outputId));

	await db
		.update(notebook)
		.set({
			updatedAt: Date.now()
		})
		.where(eq(notebook.id, detail.record.id));

	return true;
}

export async function updateNotebookSource(
	user: UserRecord,
	slug: string,
	sourceId: string,
	input: {
		title: string;
		meta: string;
		content: string;
	}
) {
	const detail = await getNotebookDetailForUser(user, slug);
	if (!detail) return null;

	const source = detail.sources.find((entry) => entry.id === sourceId);
	if (!source || !isEditableSourceRecord(source)) return null;

	await db
		.update(notebookSource)
		.set({
			title: input.title.trim(),
			meta: input.meta.trim(),
			content: input.content.trim()
		})
		.where(eq(notebookSource.id, sourceId));

	await db
		.update(notebook)
		.set({
			updatedAt: Date.now()
		})
		.where(eq(notebook.id, detail.record.id));

	return true;
}

export function mapNotebookDetailView(
	detail: NonNullable<Awaited<ReturnType<typeof getNotebookDetailForUser>>>
) {
	return {
		id: detail.record.id,
		slug: detail.record.slug,
		title: detail.record.title,
		tag: detail.record.tag,
		role: detail.record.role as 'OWNER' | 'EDITOR' | 'VIEWER',
		subject: detail.record.subject,
		location: detail.record.location,
		contact: detail.record.contact,
		link: detail.record.link,
		summary: detail.record.summary,
		selectedSourceId: detail.selectedSource?.id ?? null,
		selectedSourceContent:
			detail.selectedSource?.content ??
			'No source selected yet. Add a source to start building a synthesis-ready notebook.',
		sources: detail.sources.map((source) => ({
			id: source.id,
			title: source.title,
			subtitle: source.subtitle,
			meta: source.meta,
			content: source.content,
			sourceType: source.sourceType,
			icon: source.icon as SourceItem['icon'],
			selected: source.id === detail.selectedSource?.id,
			editable: isEditableSourceRecord(source)
		})),
		studioTools: [
			{
				key: 'audio-overview',
				title: '基准测试',
				description: '验证markdown渲染和工具执行。',
				icon: 'mic'
			},
			{
				key: 'interview-assessment',
				title: '面试评估',
				description: '基于简历、面试对话和职位描述评估候选人。',
				icon: 'bar-chart'
			}
		] as const,
		savedOutputs: detail.outputs.map((output) => ({
			id: output.id,
			title: output.title,
			summary: output.summary,
			content: output.content,
			updatedAgo: timeAgoLabel(output.createdAt),
			progress: output.progress ? output.progress / 100 : undefined,
			duration: output.duration ?? undefined
		})),
		messages: detail.messages.map((message) => ({
			id: message.id,
			role: message.role as 'user' | 'assistant',
			content: message.content
		})),
		storageLabel: `${Math.max(4.5, detail.sources.length * 1.8).toFixed(1)} MB / 100 MB`,
		storageRatio: Math.min(92, Math.max(8, detail.sources.length * 12))
	};
}

export function getSourceTemplates() {
	return sourceTemplates as SourceTemplateItem[];
}
