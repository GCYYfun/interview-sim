<script lang="ts">
	import { enhance } from '$app/forms';
	import Icon from '$lib/components/ui/Icon.svelte';

	let { form } = $props<{ form?: { error?: string } }>();
	let loading = $state(false);
</script>

<svelte:head>
	<title>Login · HuShi</title>
</svelte:head>

<div class="auth-bg">
	<div class="auth-orb auth-orb-left"></div>
	<div class="auth-orb auth-orb-right"></div>

	<div class="auth-container">
		<div class="auth-logo">
			<div class="auth-logo-icon">
				<Icon name="brain" size={20} />
			</div>
			<span class="auth-logo-name">CuratorAI</span>
		</div>
		<p class="auth-logo-sub">THE DIGITAL CURATOR</p>

		<div class="auth-card">
			<h1 class="auth-title">Welcome Back</h1>
			<p class="auth-subtitle">Synthesize your professional trajectory.</p>

			{#if form?.error}
				<div class="auth-error">{form.error}</div>
			{/if}

			<form
				method="POST"
				action="?/login"
				use:enhance={() => {
					loading = true;
					return async ({ update }) => {
						loading = false;
						await update();
					};
				}}
			>
				<div class="form-group">
					<label class="label" for="email">Email Address</label>
					<input
						id="email"
						name="email"
						type="text"
						class="auth-input"
						placeholder="name@workspace.com or admin"
						required
						autocomplete="username"
					/>
				</div>

				<div class="form-group">
					<div class="form-label-row">
						<label class="label" for="password">Password</label>
						<a href="/register" class="auth-link">Forgot Password?</a>
					</div>
					<input
						id="password"
						name="password"
						type="password"
						class="auth-input"
						placeholder="••••••••"
						required
						autocomplete="current-password"
					/>
				</div>

				<button type="submit" class="auth-submit" disabled={loading}>
					{#if loading}
						<span class="spinner"></span> Signing in...
					{:else}
						LOGIN TO WORKSPACE
					{/if}
				</button>
			</form>

			<div class="auth-divider"><span>OR CONTINUE WITH</span></div>

			<div class="auth-social">
				<button class="auth-social-btn coming-soon" title="Coming soon">
					<Icon name="google" size={16} />
					Google
				</button>
				<button class="auth-social-btn coming-soon" title="Coming soon">
					<Icon name="github" size={16} />
					GitHub
				</button>
			</div>
		</div>

		<p class="auth-footer">
			Don't have an account?
			<a href="/register" class="auth-link">Request access</a>
		</p>

		<div class="auth-meta">
			<a href="/login" class="auth-meta-link">Security</a>
			<span>·</span>
			<a href="/login" class="auth-meta-link">Privacy</a>
			<span>·</span>
			<span class="auth-meta-link">Synthesis Engine v2.4</span>
		</div>
	</div>
</div>

<style>
	.auth-bg {
		min-height: 100vh;
		display: grid;
		place-items: center;
		padding: 40px 16px;
		position: relative;
		overflow: hidden;
		background:
			radial-gradient(circle at left 20%, rgba(67, 182, 255, 0.08), transparent 22%),
			radial-gradient(circle at right 12%, rgba(52, 161, 217, 0.12), transparent 24%),
			linear-gradient(90deg, #0d1317 0%, #070809 48%, #0a1115 100%);
	}

	.auth-orb {
		position: absolute;
		border-radius: 999px;
		filter: blur(12px);
		opacity: 0.45;
		pointer-events: none;
	}

	.auth-orb-left {
		inset: auto auto 12% 4%;
		width: 340px;
		height: 340px;
		background: radial-gradient(circle, rgba(54, 163, 222, 0.18), transparent 68%);
	}

	.auth-orb-right {
		inset: 8% 2% auto auto;
		width: 420px;
		height: 420px;
		background: radial-gradient(circle, rgba(54, 163, 222, 0.18), transparent 68%);
	}

	.auth-container {
		width: 100%;
		max-width: 390px;
		position: relative;
		z-index: 1;
		display: grid;
		justify-items: center;
	}

	.auth-logo {
		display: inline-flex;
		align-items: center;
		gap: 10px;
	}

	.auth-logo-icon {
		width: 32px;
		height: 32px;
		display: grid;
		place-items: center;
		border-radius: 999px;
		background: rgba(89, 182, 255, 0.18);
		color: #82d1ff;
	}

	.auth-logo-name {
		font-size: 2rem;
		font-weight: 700;
		letter-spacing: -0.05em;
	}

	.auth-logo-sub {
		margin: 8px 0 22px;
		color: var(--color-text-secondary);
		font-size: 0.9rem;
		letter-spacing: 0.12em;
		text-transform: uppercase;
	}

	.auth-card {
		width: 100%;
		background: rgba(30, 30, 30, 0.92);
		border: 1px solid rgba(255, 255, 255, 0.06);
		border-radius: 10px;
		padding: 24px 22px;
		box-shadow: 0 28px 70px rgba(0, 0, 0, 0.36);
	}

	.auth-title {
		margin: 0;
		font-size: 2rem;
		letter-spacing: -0.04em;
	}

	.auth-subtitle {
		margin: 8px 0 24px;
		color: var(--color-text-secondary);
	}

	.auth-error {
		margin-bottom: 16px;
		padding: 10px 12px;
		border-radius: 8px;
		background: rgba(248, 81, 73, 0.12);
		color: #ff8f88;
		border: 1px solid rgba(248, 81, 73, 0.3);
	}

	.form-group {
		margin-bottom: 16px;
	}

	.form-label-row {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		margin-bottom: 6px;
	}

	.label {
		margin-bottom: 6px;
		font-size: 0.8rem;
		color: #aeb4ba;
	}

	.auth-input {
		width: 100%;
		border: 1px solid transparent;
		border-radius: 6px;
		background: #2a2a2a;
		color: var(--color-text-primary);
		padding: 12px 14px;
		outline: none;
	}

	.auth-input:focus {
		border-color: rgba(105, 200, 255, 0.6);
	}

	.auth-link {
		color: #65c5ff;
		text-decoration: none;
		font-size: 0.82rem;
		font-weight: 600;
	}

	.auth-submit {
		width: 100%;
		display: inline-flex;
		justify-content: center;
		align-items: center;
		gap: 8px;
		margin-top: 8px;
		padding: 12px 16px;
		border: 0;
		border-radius: 6px;
		background: linear-gradient(135deg, #86d5ff, #55b7ff);
		color: #083251;
		font-weight: 800;
		letter-spacing: 0.06em;
	}

	.auth-divider {
		display: flex;
		align-items: center;
		gap: 12px;
		margin: 18px 0;
		color: var(--color-text-muted);
		font-size: 0.75rem;
		font-weight: 700;
		letter-spacing: 0.1em;
	}

	.auth-divider::before,
	.auth-divider::after {
		content: '';
		flex: 1;
		height: 1px;
		background: rgba(255, 255, 255, 0.06);
	}

	.auth-social {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 10px;
	}

	.auth-social-btn {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: 8px;
		border: 0;
		border-radius: 8px;
		padding: 12px;
		background: rgba(255, 255, 255, 0.03);
		color: var(--color-text-primary);
		font-weight: 600;
	}

	.auth-footer {
		margin: 18px 0 0;
		color: var(--color-text-secondary);
	}

	.auth-meta {
		display: flex;
		gap: 8px;
		align-items: center;
		margin-top: 88px;
		color: var(--color-text-muted);
		font-size: 0.75rem;
		letter-spacing: 0.1em;
		text-transform: uppercase;
	}

	.auth-meta-link {
		color: inherit;
		text-decoration: none;
	}

	.spinner {
		width: 12px;
		height: 12px;
		border: 2px solid rgba(8, 50, 81, 0.3);
		border-top-color: #083251;
		border-radius: 50%;
		animation: spin 0.6s linear infinite;
	}

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}
</style>
