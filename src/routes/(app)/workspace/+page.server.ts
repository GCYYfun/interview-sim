import { fail, redirect } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import {
	createNotebookForUser,
	deleteNotebookForUser,
	enrichWorkspaceList,
	getWorkspaceHome,
	updateNotebookDetails
} from '$lib/server/workbench';

export const load: PageServerLoad = async ({ locals }) => {
	const user = locals.user;
	if (!user) {
		redirect(302, '/login');
	}

	const notebooks = await enrichWorkspaceList(await getWorkspaceHome(user));

	return {
		notebooks
	};
};

export const actions: Actions = {
	createNotebook: async ({ request, locals }) => {
		const user = locals.user;
		if (!user) {
			redirect(302, '/login');
		}

		const formData = await request.formData();
		const title = formData.get('title')?.toString().trim() ?? '';
		const subject = formData.get('subject')?.toString().trim() ?? '';
		const tag = formData.get('tag')?.toString().trim() ?? '';

		if (!title || !subject) {
			return fail(400, {
				createError: 'Title and subject are required.',
				createValues: { title, subject, tag }
			});
		}

		const slug = await createNotebookForUser(user, { title, subject, tag });
		redirect(303, `/workspace/${slug}`);
	},
	updateNotebook: async ({ request, locals }) => {
		const user = locals.user;
		if (!user) {
			redirect(302, '/login');
		}

		const formData = await request.formData();
		const slug = formData.get('slug')?.toString().trim() ?? '';
		const title = formData.get('title')?.toString().trim() ?? '';
		const subject = formData.get('subject')?.toString().trim() ?? '';
		const tag = formData.get('tag')?.toString().trim() ?? '';

		if (!slug || !title || !subject || !tag) {
			return fail(400, {
				manageError: 'Slug, title, subject, and tag are required.',
				manageValues: { slug, title, subject, tag }
			});
		}

		const notebooks = await getWorkspaceHome(user);
		const current = notebooks.find((entry) => entry.slug === slug);
		if (!current) {
			return fail(404, { manageError: 'Notebook not found.' });
		}

		const nextSlug = await updateNotebookDetails(user, slug, {
			title,
			subject,
			tag,
			location: current.location,
			contact: current.contact,
			link: current.link,
			summary: current.summary
		});

		if (!nextSlug) {
			return fail(404, { manageError: 'Notebook not found.' });
		}

		return { success: true, managedSlug: nextSlug };
	},
	deleteNotebook: async ({ request, locals }) => {
		const user = locals.user;
		if (!user) {
			redirect(302, '/login');
		}

		const formData = await request.formData();
		const slug = formData.get('slug')?.toString().trim() ?? '';

		if (!slug) {
			return fail(400, { manageError: 'Notebook deletion target is missing.' });
		}

		await deleteNotebookForUser(user, slug);
		return { success: true };
	}
};
