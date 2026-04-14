import { redirect } from '@sveltejs/kit';
import type { Actions } from './$types';
import { auth } from '$lib/server/auth';

export const actions: Actions = {
	default: async ({ request, cookies }) => {
		try {
			// 1. 让 better-auth 处理服务端登出（使 session 失效）
			await auth.api.signOut({ headers: request.headers });
		} catch (e) {
			console.error('Logout error during API call:', e);
		}

		// 2. 强制清除浏览器 Cookie（双重保险）
		cookies.delete('better-auth.session_token', { path: '/' });

		// 3. 抛出重定向
		throw redirect(302, '/login');
	}
};
