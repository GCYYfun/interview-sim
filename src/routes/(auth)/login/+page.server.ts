import { fail, redirect } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import { auth } from '$lib/server/auth';
import { db } from '$lib/server/db';
import { user } from '$lib/server/db/schema';
import { eq } from 'drizzle-orm';

let seedRan = false;

export const load: PageServerLoad = async ({ locals }) => {
	if (!seedRan) {
		seedRan = true;
		try {
			// 取消 admin 账户
			await db.delete(user).where(eq(user.email, 'admin@local.test')).catch(() => {});
			// 添加测试账户
			const usersToAdd = ['test_user_4141', 'test_user_4142', 'test_user_4143'];
			for (const u of usersToAdd) {
				await auth.api.signUpEmail({
					body: { email: `${u}@local.test`, password: u, name: u },
					asResponse: false
				}).catch((e) => console.log('user exists or error:', e));
			}
			console.log('[AUTH] Test users seeded, admin removed.');
		} catch (e) {
			console.error('[AUTH] Seeding error:', e);
		}
	}

	// Already logged in → go to app
	if (locals.user) {
		redirect(302, '/workspace');
	}
};

export const actions: Actions = {
	login: async ({ request }) => {
		const data = await request.formData();
		const rawEmail = data.get('email') as string;
		const password = data.get('password') as string;
		const email = rawEmail.includes('@') ? rawEmail : `${rawEmail}@local.test`;

		if (!email || !password) {
			return fail(400, { error: 'Email and password are required.' });
		}

		try {
			await auth.api.signInEmail({
				body: { email, password },
				asResponse: false
			});
		} catch {
			return fail(401, { error: 'Invalid email or password.' });
		}

		redirect(302, '/workspace');
	}
};
