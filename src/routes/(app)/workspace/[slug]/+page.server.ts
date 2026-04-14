import { error, fail, redirect } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import {
	addSourceToNotebook,
	chooseNotebookSource,
	createNotebookMessage,
	createStudioOutput,
	deleteNotebookForUser,
	deleteNotebookOutput,
	deleteNotebookSource,
	getNotebookDetailForUser,
	getSourceTemplates,
	mapNotebookDetailView,
	updateNotebookSource,
	updateNotebookOutputTitle,
	updateNotebookDetails
} from '$lib/server/workbench';

export const load: PageServerLoad = async ({ params, locals, url }) => {
	const user = locals.user;
	if (!user) {
		error(401, 'Unauthorized');
	}

	const detail = await getNotebookDetailForUser(user, params.slug, url.searchParams.get('source'));

	if (!detail) {
		error(404, 'Notebook not found');
	}

	return {
		notebook: mapNotebookDetailView(detail),
		sourceTemplates: getSourceTemplates(),
		debug: process.env.HUSHI_LLM_DEBUG === 'true'
	};
};

export const actions: Actions = {
	addSource: async ({ request, params, locals }) => {
		const user = locals.user;
		if (!user) {
			error(401, 'Unauthorized');
		}

		const formData = await request.formData();
		const title = formData.get('title')?.toString().trim() ?? '';
		const sourceType = formData.get('sourceType')?.toString().trim() ?? '';
		const meta = formData.get('meta')?.toString().trim() ?? '';
		const content = formData.get('content')?.toString().trim() ?? '';
		const file = formData.get('file');

		let resolvedTitle = title;
		let resolvedContent = content;
		let resolvedSubtitle = '';

		if (file instanceof File && file.name) {
			resolvedTitle = resolvedTitle || file.name;
			const fileSizeKB = Math.round(file.size / 1024);

			const isBinary =
				file.type === 'application/pdf' ||
				file.type === 'application/msword' ||
				file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
				['pdf', 'doc', 'docx'].includes(file.name.split('.').pop()?.toLowerCase() ?? '');

			if (resolvedContent) {
				// 前端已通过 parse-source API 解析完成，直接使用
				const ext = file.name.split('.').pop()?.toUpperCase() || 'File';
				resolvedSubtitle = `${ext} • ${fileSizeKB} KB • AI 解析完成`;
			} else if (isBinary) {
				// 二进制文件必须先通过 AI 解析
				return fail(400, {
					sourceError: '请先点击「AI 解析为 Markdown」按钮完成解析，再添加源。'
				});
			} else {
				// 文本文件未预解析（理论上前端应先解析，但此处做兜底）
				return fail(400, {
					sourceError: '请先点击「AI 解析为 Markdown」按钮完成解析，再添加源。'
				});
			}
		}

		if (!resolvedTitle || !sourceType || !resolvedContent) {
			return fail(400, {
				sourceError: '请填写标题，选择来源类型，并上传文件或直接粘贴内容。'
			});
		}

		await addSourceToNotebook(user, params.slug, {
			title: resolvedTitle,
			sourceType,
			meta,
			content: resolvedContent,
			subtitle: resolvedSubtitle
		});
		return { success: true };
	},
	selectSource: async ({ request, params, locals }) => {
		const user = locals.user;
		if (!user) {
			error(401, 'Unauthorized');
		}

		const formData = await request.formData();
		const sourceId = formData.get('sourceId')?.toString() ?? '';

		if (!sourceId) {
			return fail(400, { sourceError: 'Source selection is missing.' });
		}

		await chooseNotebookSource(user, params.slug, sourceId);
		return { success: true };
	},
	updateNotebook: async ({ request, params, locals }) => {
		const user = locals.user;
		if (!user) {
			error(401, 'Unauthorized');
		}

		const formData = await request.formData();
		const title = formData.get('title')?.toString().trim() ?? '';
		const subject = formData.get('subject')?.toString().trim() ?? '';
		const tag = formData.get('tag')?.toString().trim() ?? '';
		const location = formData.get('location')?.toString().trim() ?? '';
		const contact = formData.get('contact')?.toString().trim() ?? '';
		const link = formData.get('link')?.toString().trim() ?? '';
		const summary = formData.get('summary')?.toString().trim() ?? '';

		if (!title || !subject || !tag || !location || !contact || !link || !summary) {
			return fail(400, {
				notebookError: 'Please complete all notebook fields before saving.',
				notebookValues: { title, subject, tag, location, contact, link, summary }
			});
		}

		const nextSlug = await updateNotebookDetails(user, params.slug, {
			title,
			subject,
			tag,
			location,
			contact,
			link,
			summary
		});

		if (!nextSlug) {
			return fail(404, { notebookError: 'Notebook not found.' });
		}

		if (nextSlug !== params.slug) {
			redirect(303, `/workspace/${nextSlug}`);
		}

		return { success: true };
	},
	deleteSource: async ({ request, params, locals }) => {
		const user = locals.user;
		if (!user) {
			error(401, 'Unauthorized');
		}

		const formData = await request.formData();
		const sourceId = formData.get('sourceId')?.toString() ?? '';

		if (!sourceId) {
			return fail(400, { sourceError: 'Source deletion target is missing.' });
		}

		await deleteNotebookSource(user, params.slug, sourceId);
		return { success: true };
	},
	deleteOutput: async ({ request, params, locals }) => {
		const user = locals.user;
		if (!user) {
			error(401, 'Unauthorized');
		}

		const formData = await request.formData();
		const outputId = formData.get('outputId')?.toString() ?? '';

		if (!outputId) {
			return fail(400, { outputError: 'Output deletion target is missing.' });
		}

		await deleteNotebookOutput(user, params.slug, outputId);
		return { success: true };
	},
	updateOutput: async ({ request, params, locals }) => {
		const user = locals.user;
		if (!user) {
			error(401, 'Unauthorized');
		}

		const formData = await request.formData();
		const outputId = formData.get('outputId')?.toString() ?? '';
		const title = formData.get('title')?.toString().trim() ?? '';

		if (!outputId || !title) {
			return fail(400, {
				outputError: 'Output title is required.',
				outputValues: { outputId, title }
			});
		}

		const updated = await updateNotebookOutputTitle(user, params.slug, outputId, title);
		if (!updated) {
			return fail(404, { outputError: 'Output not found.' });
		}

		return { success: true };
	},
	updateSource: async ({ request, params, locals }) => {
		const user = locals.user;
		if (!user) {
			error(401, 'Unauthorized');
		}

		const formData = await request.formData();
		const sourceId = formData.get('sourceId')?.toString() ?? '';
		const title = formData.get('title')?.toString().trim() ?? '';
		const meta = formData.get('meta')?.toString().trim() ?? '';
		const content = formData.get('content')?.toString().trim() ?? '';

		if (!sourceId || !title || !meta || !content) {
			return fail(400, {
				sourceEditError: 'Please complete all editable source fields before saving.',
				sourceEditValues: { sourceId, title, meta, content }
			});
		}

		const updated = await updateNotebookSource(user, params.slug, sourceId, {
			title,
			meta,
			content
		});

		if (!updated) {
			return fail(400, {
				sourceEditError: 'This source is read-only and can only be viewed.'
			});
		}

		return { success: true };
	},
	deleteNotebook: async ({ params, locals }) => {
		const user = locals.user;
		if (!user) {
			error(401, 'Unauthorized');
		}

		await deleteNotebookForUser(user, params.slug);
		redirect(303, '/workspace');
	},
	runTool: async ({ request, params, locals }) => {
		const user = locals.user;
		if (!user) {
			error(401, 'Unauthorized');
		}

		const formData = await request.formData();
		const tool = formData.get('tool')?.toString() ?? '';
		if (!tool) {
			return fail(400, { toolError: 'Tool is required.' });
		}

		await createStudioOutput(user, params.slug, tool);
		return { success: true };
	},
	sendMessage: async ({ request, params, locals }) => {
		const user = locals.user;
		if (!user) {
			error(401, 'Unauthorized');
		}

		const formData = await request.formData();
		const prompt = formData.get('prompt')?.toString().trim() ?? '';

		if (!prompt) {
			return fail(400, { messageError: 'Please enter a question first.' });
		}

		await createNotebookMessage(user, params.slug, prompt);
		return { success: true };
	}
};
