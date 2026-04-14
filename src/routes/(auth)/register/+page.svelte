<script lang="ts">
	import { enhance } from '$app/forms';
	import Icon from '$lib/components/ui/Icon.svelte';

	let { form } = $props<{ form?: { error?: string } }>();
	let loading = $state(false);
</script>

<svelte:head>
	<title>Register · HuShi</title>
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
			<h1 class="auth-title">Request Access</h1>
			<p class="auth-subtitle">Create your account to get started.</p>

			{#if form?.error}
				<div class="auth-error">{form.error}</div>
			{/if}

			<form
				method="POST"
				action="?/register"
				use:enhance={() => {
					loading = true;
					return async ({ update }) => {
						loading = false;
						await update();
					};
				}}
			>
				<div class="form-group">
					<label class="label" for="name">Full Name</label>
					<input
						id="name"
						name="name"
						type="text"
						class="auth-input"
						placeholder="Your name"
						required
						autocomplete="name"
					/>
				</div>

				<div class="form-group">
					<label class="label" for="email">Email Address</label>
					<input
						id="email"
						name="email"
						type="email"
						class="auth-input"
						placeholder="name@workspace.com"
						required
						autocomplete="email"
					/>
				</div>

				<div class="form-group">
					<label class="label" for="password">Password</label>
					<input
						id="password"
						name="password"
						type="password"
						class="auth-input"
						placeholder="Minimum 8 characters"
						required
						minlength="8"
						autocomplete="new-password"
					/>
				</div>

				<button type="submit" class="auth-submit" disabled={loading}>
					{#if loading}
						<span class="spinner"></span> Creating account...
					{:else}
						CREATE ACCOUNT
					{/if}
				</button>
			</form>
		</div>

		<p class="auth-footer">
			Already have an account?
			<a href="/login" class="auth-link">Sign in</a>
		</p>
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

	.auth-footer {
		margin: 18px 0 0;
		color: var(--color-text-secondary);
	}

	.auth-link {
		color: #65c5ff;
		text-decoration: none;
		font-weight: 600;
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
