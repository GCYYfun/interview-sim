import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { AiConfigurationError } from '$lib/server/ai/config';
import { streamStudioAgent } from '$lib/server/ai/agents/studio';
import { saveNotebookOutput } from '$lib/server/workbench';
import { logger } from '$lib/server/logger';

export const POST: RequestHandler = async ({ request, params, locals }) => {
	const user = locals.user;
	if (!user) {
		error(401, 'Unauthorized');
	}

	const body = (await request.json()) as {
		tool?: string;
		focus?: string;
		tone?: string;
		length?: string;
		includeCitations?: boolean;
		resumeSourceId?: string;
		conversationSourceId?: string;
		jdSourceId?: string;
		evaluationMode?: 'basic' | 'advanced';
		resumeCheckpoint?: boolean;
	};

	if (!body.tool) {
		return json({ error: 'Tool is required.' }, { status: 400 });
	}

	logger.info('analyze', 'stream start', {
		userId: user.id,
		slug: params.slug,
		tool: body.tool,
		mode: body.evaluationMode,
		resume: body.resumeCheckpoint
	});

	const encoder = new TextEncoder();

	return new Response(
		new ReadableStream({
			async start(controller) {
				try {
					for await (const line of streamStudioAgent(user, params.slug, {
						tool: body.tool as 'audio-overview' | 'interview-assessment',
						focus: body.focus,
						tone: body.tone,
						length: body.length,
						includeCitations: body.includeCitations,
						resumeSourceId: body.resumeSourceId,
						conversationSourceId: body.conversationSourceId,
						jdSourceId: body.jdSourceId,
						evaluationMode: body.evaluationMode,
					resumeCheckpoint: body.resumeCheckpoint
					})) {
						// 解析 save 事件，保存到数据库（不推给客户端）
						try {
							const event = JSON.parse(line.trim());
							if (event.type === 'save') {
								logger.info('analyze', 'saving output to db', { slug: params.slug, title: event.title });
								await saveNotebookOutput(user, params.slug, {
									title: event.title,
									summary: event.summary,
									content: event.content
								}).catch((saveErr) => {
									logger.error('analyze', 'saveNotebookOutput failed', saveErr);
									console.error(saveErr);
								});
								continue;
							}
						} catch { /* 不是 JSON 就直接推 */ }

						controller.enqueue(encoder.encode(line));
					}
					logger.info('analyze', 'stream complete', { userId: user.id, slug: params.slug });
				} catch (err) {
					if (err instanceof AiConfigurationError) {
						logger.warn('analyze', 'AI configuration error', { message: err.message, slug: params.slug });
						controller.enqueue(
							encoder.encode(JSON.stringify({ type: 'error', message: err.message }) + '\n')
						);
					} else {
						const message = err instanceof Error ? err.message : 'Studio analysis failed unexpectedly.';
						logger.error('analyze', 'unexpected stream error', err);
						controller.enqueue(
							encoder.encode(JSON.stringify({ type: 'error', message: `发生未知异常，请马上联系开发，防止服务崩溃！详情：${message}` }) + '\n')
						);
					}
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
