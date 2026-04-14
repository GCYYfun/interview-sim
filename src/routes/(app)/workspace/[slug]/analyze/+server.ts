import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { AiConfigurationError } from '$lib/server/ai/config';
import { streamStudioAgent } from '$lib/server/ai/agents/studio';
import { saveNotebookOutput } from '$lib/server/workbench';

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
								await saveNotebookOutput(user, params.slug, {
									title: event.title,
									summary: event.summary,
									content: event.content
								}).catch(console.error);
								continue;
							}
						} catch { /* 不是 JSON 就直接推 */ }

						controller.enqueue(encoder.encode(line));
					}
				} catch (err) {
					if (err instanceof AiConfigurationError) {
						controller.enqueue(
							encoder.encode(JSON.stringify({ type: 'error', message: err.message }) + '\n')
						);
					} else {
						const message = err instanceof Error ? err.message : 'Studio analysis failed unexpectedly.';
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
