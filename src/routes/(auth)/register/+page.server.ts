import { fail, redirect } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import { auth } from '$lib/server/auth';

export const load: PageServerLoad = async ({ locals }) => {
	if (locals.user) {
		redirect(302, '/workspace');
	}
};

export const actions: Actions = {
	register: async ({ request }) => {
		const data = await request.formData();
		const name = data.get('name') as string;
		const email = data.get('email') as string;
		const password = data.get('password') as string;

		if (!name || !email || !password) {
			return fail(400, { error: 'All fields are required.' });
		}
		if (password.length < 8) {
			return fail(400, { error: 'Password must be at least 8 characters.' });
		}

		try {
			await auth.api.signUpEmail({
				body: { name, email, password },
				asResponse: false
			});
		} catch {
			return fail(400, { error: 'Email already in use or registration failed.' });
		}

		redirect(302, '/workspace');
	}
};
