import type { Handle } from '@sveltejs/kit';
import { building } from '$app/environment';
import { auth } from '$lib/server/auth';
import { svelteKitHandler } from 'better-auth/svelte-kit';
import { logRequest, logResponse } from '$lib/server/logger';

/** 不需要记录日志的路径前缀（静态资源 / HMR / favicon 等） */
const SKIP_LOG_PREFIXES = ['/_app/', '/__svelte', '/favicon', '/robots.txt'];

function shouldLog(pathname: string): boolean {
	return !SKIP_LOG_PREFIXES.some((p) => pathname.startsWith(p));
}

const handleBetterAuth: Handle = async ({ event, resolve }) => {
	const session = await auth.api.getSession({ headers: event.request.headers });

	if (session) {
		event.locals.session = session.session;
		event.locals.user = session.user;
	}

	const { method } = event.request;
	const { pathname } = event.url;
	const t0 = Date.now();

	if (shouldLog(pathname)) {
		logRequest(method, pathname, event.locals.user?.id);
	}

	const response = await svelteKitHandler({ event, resolve, auth, building });

	if (shouldLog(pathname)) {
		logResponse(method, pathname, response.status, Date.now() - t0);
	}

	return response;
};

export const handle: Handle = handleBetterAuth;
