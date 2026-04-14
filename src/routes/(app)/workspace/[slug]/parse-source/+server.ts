import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { createAiModel, getDefaultModelId } from '$lib/server/ai/config';
import { User, System } from '$menglong';
import { promises as fs } from 'fs';
import path from 'path';
import { PDF } from '@libpdf/core';
import mammoth from 'mammoth';

// ---------- helpers ----------

function getInterviewSourceDir() {
	return path.resolve('interview_source');
}

async function ensureDir(dir: string) {
	await fs.mkdir(dir, { recursive: true });
}

const SOURCE_TYPE_PREFIX: Record<string, string> = {
	resume: 'resume',
	conversation: 'conversation',
	'job-description': 'jd'
};

function buildFileName(sourceType: string, name: string, notebookSlug: string): string {
	const prefix = SOURCE_TYPE_PREFIX[sourceType] ?? sourceType;
	const safeName = name
		.replace(/[^a-zA-Z0-9\u4e00-\u9fa5._-]/g, '_')
		.replace(/_{2,}/g, '_')
		.replace(/^_|_$/g, '');
	const safeSlug = notebookSlug.replace(/[^a-zA-Z0-9\u4e00-\u9fa5._-]/g, '_');
	return `${prefix}_${safeName}_by_${safeSlug}`;
}

async function backupRawFile(
	file: File,
	sourceType: string,
	notebookSlug: string,
	title: string
): Promise<string> {
	const base = getInterviewSourceDir();
	const dir = path.join(base, 'raw', sourceType);
	await ensureDir(dir);

	const ext = file.name.includes('.') ? '.' + file.name.split('.').pop() : '';
	const safeName = buildFileName(sourceType, title, notebookSlug) + ext;
	const filePath = path.join(dir, safeName);
	await fs.writeFile(filePath, Buffer.from(await file.arrayBuffer()));
	return filePath;
}

async function backupMdFile(
	content: string,
	sourceType: string,
	notebookSlug: string,
	title: string
): Promise<string> {
	const base = getInterviewSourceDir();
	const dir = path.join(base, 'md', sourceType);
	await ensureDir(dir);

	const safeName = buildFileName(sourceType, title, notebookSlug) + '.md';
	const filePath = path.join(dir, safeName);
	await fs.writeFile(filePath, content, 'utf-8');
	return filePath;
}

// ---------- 文件类型判断 ----------

function isBinaryFile(file: File): boolean {
	if (file.type === 'application/pdf') return true;
	if (
		file.type === 'application/msword' ||
		file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
	)
		return true;
	const ext = file.name.split('.').pop()?.toLowerCase();
	return ext === 'pdf' || ext === 'doc' || ext === 'docx';
}

function isDocx(file: File): boolean {
	if (file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') return true;
	return file.name.split('.').pop()?.toLowerCase() === 'docx';
}

/**
 * 用 mammoth 提取 docx 纯文本，失败返回 null。
 */
async function extractDocxText(buf: ArrayBuffer): Promise<string | null> {
	try {
		const result = await mammoth.extractRawText({ buffer: Buffer.from(buf) });
		return result.value.trim() || null;
	} catch {
		return null;
	}
}

/**
 * 用 @libpdf/core 提取 PDF 原始文字，按页拼接为纯文本。
 * 提取失败或内容为空时返回 null，调用方降级为 LLM 视觉解析。
 */
async function extractPdfText(buf: ArrayBuffer): Promise<string | null> {
	try {
		const pdf = await PDF.load(new Uint8Array(buf));
		const pages = await pdf.extractText();
		let text = '';
		for (const page of Object.values(pages) as { lines: { text: string }[] }[]) {
			for (const line of page.lines) {
				if (line.text?.trim()) text += line.text + '\n';
			}
			text += '\n';
		}
		return text.trim() || null;
	} catch {
		return null;
	}
}

// 匹配对话行起始格式：姓名(HH:MM:SS):
const DIALOGUE_LINE_RE = /^(.+?)\((\d{2}:\d{2}:\d{2})\):\s*/;

/**
 * 将 libpdf 提取的对话原文整理为 Markdown。
 * 规则：
 *  - 匹配 `姓名(HH:MM:SS): 内容` 的行视为新对话段
 *  - 不带前缀的行是上一段的续行，拼接到前一段
 *  - 输出格式：**姓名** `HH:MM:SS`\n\n> 内容
 */
function formatConversationMarkdown(rawText: string): string {
	const lines = rawText.split('\n');

	interface Segment { speaker: string; time: string; parts: string[] }
	const segments: Segment[] = [];

	for (const line of lines) {
		const trimmed = line.trim();
		if (!trimmed) continue;

		const match = trimmed.match(DIALOGUE_LINE_RE);
		if (match) {
			const content = trimmed.slice(match[0].length).trim();
			segments.push({ speaker: match[1].trim(), time: match[2], parts: content ? [content] : [] });
		} else if (segments.length > 0) {
			// 续行：拼到最后一段
			segments[segments.length - 1].parts.push(trimmed);
		}
	}

	if (segments.length === 0) return rawText; // 无法识别格式，返回原文

	return segments
		.map(({ speaker, time, parts }) => {
			const content = parts.join('');
			return `**${speaker}** \`${time}\`\n\n> ${content}`;
		})
		.join('\n\n');
}

// ---------- LLM system prompt ----------

function buildSystemPrompt(sourceType: string): string {
	if (sourceType === 'conversation') {
		return `你是一位专业的面试记录整理助手。

你的任务是将原始对话内容整理为结构清晰的 Markdown，**必须保留每一句话的完整原文，严禁概括、删减或改写任何对话内容**。

要求：
1. 识别对话双方（面试官 / 候选人），保留发言者标注（如有时间戳也保留）
2. 每句对话单独成段，完整保留原文措辞
3. 去除明显乱码、水印、页眉页脚等噪声字符
4. 若原文有章节结构，用 Markdown 标题（##）标注
5. 直接输出 Markdown，不加任何说明或前缀
6. 输出语言与原文一致`;
	}

	const roleMap: Record<string, string> = {
		resume: '你是一位专业的招聘信息整理助手。',
		'job-description': '你是一位专业的职位信息整理助手。'
	};
	const role = roleMap[sourceType] ?? '你是一位专业的文档整理助手。';
	return `${role}

你的任务是将用户提供的原始文档内容，整理、清洗并转换为结构清晰的 Markdown 格式。

要求：
1. 保留所有关键信息，不得遗漏重要内容
2. 使用 Markdown 标题（#/##/###）组织层级
3. 列表信息使用 Markdown 列表（- 或数字）
4. 去除乱码、无关格式符号、页眉页脚等噪声
5. 直接输出 Markdown 内容，不要加任何解释或前缀
6. 输出语言与原文一致`;
}

// ---------- POST handler ----------

export const POST: RequestHandler = async ({ request, params, locals }) => {
	console.log('[parse-source] 开始处理请求');
	const user = locals.user;
	console.log('[parse-source] user:', user ? `已登录 (${user.email})` : '未登录');
	console.log('[parse-source] session:', locals.session ? '有 session' : '无 session');

	if (!user) {
		console.error('[parse-source] 401 错误: 用户未认证');
		error(401, 'Unauthorized');
	}

	let formData: FormData;
	try {
		formData = await request.formData();
	} catch {
		return json({ error: 'Invalid form data.' }, { status: 400 });
	}

	const file = formData.get('file');
	const sourceType = formData.get('sourceType')?.toString().trim() ?? '';
	const title = formData.get('title')?.toString().trim() || '';

	if (!(file instanceof File) || file.size === 0) {
		return json({ error: '请先上传文件。' }, { status: 400 });
	}
	if (!sourceType) {
		return json({ error: '请选择来源类型。' }, { status: 400 });
	}

	const notebookSlug = params.slug;
	const resolvedTitle = title || file.name.replace(/\.[^.]+$/, '');

	// ---- 备份原始文件 ----
	let rawPath = '';
	try {
		rawPath = await backupRawFile(file, sourceType, notebookSlug, resolvedTitle);
	} catch (e) {
		console.error('[parse-source] backup raw failed:', e);
	}

	// ---- 读取文件内容 ----
	const isBinary = isBinaryFile(file);
	let rawText = '';
	let pdfBase64 = '';
	let extractedText: string | null = null;

	if (isBinary) {
		const buf = await file.arrayBuffer().catch(() => null);
		if (!buf) {
			return json({ error: '文件读取失败，请检查文件格式。' }, { status: 422 });
		}

		if (sourceType === 'conversation') {
			// conversation：优先 libpdf 提取原文，保留完整对话细节
			extractedText = await extractPdfText(buf);
		} else if (isDocx(file)) {
			// docx：用 mammoth 提取纯文本，再交给 LLM 整理
			rawText = (await extractDocxText(buf)) ?? '';
			if (!rawText) {
				return json({ error: 'DOCX 文件解析失败，请检查文件是否损坏。' }, { status: 422 });
			}
		}

		// PDF（非 conversation）或 docx 提取失败：降级为 base64 传 LLM
		if (!extractedText && !rawText) {
			pdfBase64 = Buffer.from(buf).toString('base64');
		}
	} else {
		rawText = await file.text().catch(() => '');
		if (!rawText) {
			return json({ error: '文件读取失败，请检查文件格式。' }, { status: 422 });
		}
	}

	// ---- conversation + libpdf：直接格式化，不走 LLM ----
	if (extractedText) {
		const mdContent = formatConversationMarkdown(extractedText);

		const encoder = new TextEncoder();
		return new Response(
			new ReadableStream({
				async start(controller) {
					const chunkSize = 256;
					for (let i = 0; i < mdContent.length; i += chunkSize) {
						controller.enqueue(
							encoder.encode(
								JSON.stringify({ type: 'chunk', text: mdContent.slice(i, i + chunkSize) }) + '\n'
							)
						);
						// 小延迟让前端逐步渲染内容
						await new Promise((r) => setTimeout(r, 8));
					}

					try {
						const mdPath = await backupMdFile(mdContent, sourceType, notebookSlug, resolvedTitle);
						controller.enqueue(
							encoder.encode(JSON.stringify({ type: 'done', rawPath, mdPath }) + '\n')
						);
					} catch (e) {
						console.error('[parse-source] backup md failed:', e);
						controller.enqueue(
							encoder.encode(JSON.stringify({ type: 'done', rawPath }) + '\n')
						);
					}
					controller.close();
				}
			}),
			{ headers: { 'content-type': 'text/plain; charset=utf-8', 'cache-control': 'no-store' } }
		);
	}

	// ---- LLM 流式整理（非 conversation，或 libpdf 提取失败降级） ----
	const parseModel = pdfBase64
		? 'anthropic/us.anthropic.claude-haiku-4-5-20251001-v1:0'
		: getDefaultModelId();
	const model = createAiModel();
	const systemPrompt = buildSystemPrompt(sourceType);

	const userContent = pdfBase64
		? '请将文档内容整理为 Markdown 格式，保留所有关键信息，直接输出 Markdown。'
		: `请将以下内容整理为结构清晰的 Markdown 格式，保留所有关键信息，直接输出 Markdown：\n\n---\n${rawText.slice(0, 20000)}\n---`;

	const messages = pdfBase64
		? [System(systemPrompt), User(userContent, { pdf: pdfBase64 })]
		: [System(systemPrompt), User(userContent)];

	const encoder = new TextEncoder();
	let fullContent = '';

	return new Response(
		new ReadableStream({
			async start(controller) {
				try {
					for await (const chunk of model.streamChat(messages, parseModel)) {
						const delta = chunk.output?.delta?.text ?? '';
						if (delta) {
							fullContent += delta;
							controller.enqueue(
								encoder.encode(JSON.stringify({ type: 'chunk', text: delta }) + '\n')
							);
						}
					}
				} catch (e) {
					const msg = e instanceof Error ? e.message : 'LLM 解析失败';
					controller.enqueue(
						encoder.encode(JSON.stringify({ type: 'error', message: msg }) + '\n')
					);
					controller.close();
					return;
				}

				try {
					const mdPath = await backupMdFile(fullContent, sourceType, notebookSlug, resolvedTitle);
					controller.enqueue(
						encoder.encode(JSON.stringify({ type: 'done', rawPath, mdPath }) + '\n')
					);
				} catch (e) {
					console.error('[parse-source] backup md failed:', e);
					controller.enqueue(encoder.encode(JSON.stringify({ type: 'done', rawPath }) + '\n'));
				}

				controller.close();
			}
		}),
		{
			headers: {
				'content-type': 'text/plain; charset=utf-8',
				'cache-control': 'no-store'
			}
		}
	);
};
