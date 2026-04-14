<script lang="ts">
	import { page } from '$app/stores';
	import { enhance } from '$app/forms';
	import Icon from '$lib/components/ui/Icon.svelte';

	let { children, data } = $props();

	let user = $derived(data?.user);
	let showUserMenu = $state(false);
</script>

<div class="app-shell">
	<!-- Top Navbar -->
	<header class="navbar">
		<div class="navbar-left">
			<a href="/workspace" class="navbar-brand">
				<div class="brand-icon">
					<Icon name="brain" size={16} />
				</div>
				<span class="brand-name">HuShi</span>
			</a>
		</div>

		<div class="navbar-center">
			<div class="search-wrap">
				<Icon name="search" size={14} class="search-icon" />
				<input
					type="text"
					class="search-input"
					placeholder="Search ..."
					aria-label="Search notebooks"
				/>
			</div>
		</div>

		<div class="navbar-right">
			<button
				class="nav-icon-btn coming-soon"
				title="Notifications (coming soon)"
				aria-label="Notifications"
			>
				<Icon name="bell" size={16} />
			</button>
			<button class="nav-icon-btn coming-soon" title="Settings (coming soon)" aria-label="Settings">
				<Icon name="settings" size={16} />
			</button>
			<div class="user-menu-wrap">
				<button
					class="user-avatar"
					onclick={() => (showUserMenu = !showUserMenu)}
					aria-label="User menu"
				>
					{#if user?.name}
						{user.name[0].toUpperCase()}
					{:else}
						U
					{/if}
				</button>
				{#if showUserMenu}
					<!-- svelte-ignore a11y_click_events_have_key_events -->
					<!-- svelte-ignore a11y_no_static_element_interactions -->
					<div class="user-dropdown">
						<div class="user-info">
							<span class="user-name">{user?.name ?? 'User'}</span>
							<span class="user-email">{user?.email ?? ''}</span>
						</div>
						<div class="divider"></div>
						<form method="POST" action="/logout" use:enhance>
							<button type="submit" class="dropdown-item">
								<Icon name="logout" size={14} />
								Sign out
							</button>
						</form>
					</div>
				{/if}
			</div>
		</div>
	</header>

	<!-- Page Content -->
	<main class="app-main">
		{@render children()}
	</main>
</div>

<style>
	.app-shell {
		display: flex;
		flex-direction: column;
		min-height: 100vh;
		background: var(--color-bg-base);
	}

	/* ── Navbar ─────────────────────────────────── */
	.navbar {
		height: 48px;
		border-bottom: 1px solid var(--color-border);
		background: var(--color-bg-base);
		display: flex;
		align-items: center;
		padding: 0 20px;
		gap: 16px;
		position: sticky;
		top: 0;
		z-index: 40;
	}

	.navbar-left {
		flex: 0 0 auto;
	}

	.navbar-brand {
		display: flex;
		align-items: center;
		gap: 8px;
		text-decoration: none;
		color: var(--color-text-primary);
		transition:
			transform 0.18s ease,
			opacity 0.18s ease;
	}

	.navbar-brand:hover {
		opacity: 0.92;
		transform: translateY(-1px);
	}

	.brand-icon {
		width: 26px;
		height: 26px;
		background: var(--color-accent-dim);
		border: 1px solid rgba(59, 158, 255, 0.3);
		border-radius: 6px;
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--color-accent);
		transition:
			transform 0.18s ease,
			box-shadow 0.18s ease;
	}

	.navbar-brand:hover .brand-icon {
		transform: scale(1.04);
		box-shadow: 0 8px 20px rgba(59, 158, 255, 0.16);
	}

	.brand-name {
		font-size: 14px;
		font-weight: 700;
		letter-spacing: -0.01em;
	}

	.navbar-center {
		flex: 1;
		max-width: 400px;
		margin: 0 auto;
	}

	.search-wrap {
		position: relative;
		display: flex;
		align-items: center;
	}

	:global(.search-icon) {
		position: absolute;
		left: 10px;
		color: var(--color-text-muted);
		pointer-events: none;
	}

	.search-input {
		width: 100%;
		background: var(--color-bg-elevated);
		border: 1px solid var(--color-border);
		border-radius: 6px;
		color: var(--color-text-primary);
		font-size: 13px;
		padding: 5px 12px 5px 32px;
		outline: none;
		transition: border-color 0.15s;
	}
	.search-input::placeholder {
		color: var(--color-text-muted);
	}
	.search-input:focus {
		border-color: var(--color-accent);
		box-shadow: 0 0 0 3px rgba(59, 158, 255, 0.14);
	}

	.navbar-right {
		flex: 0 0 auto;
		display: flex;
		align-items: center;
		gap: 4px;
		margin-left: auto;
	}

	.nav-icon-btn {
		width: 32px;
		height: 32px;
		border-radius: 6px;
		border: none;
		background: transparent;
		color: var(--color-text-secondary);
		display: flex;
		align-items: center;
		justify-content: center;
		cursor: pointer;
		transition: all 0.15s;
	}
	.nav-icon-btn:hover {
		background: var(--color-bg-elevated);
		color: var(--color-text-primary);
	}

	.user-menu-wrap {
		position: relative;
	}

	.user-avatar {
		width: 30px;
		height: 30px;
		border-radius: 50%;
		background: var(--color-accent);
		color: #fff;
		font-size: 12px;
		font-weight: 700;
		border: none;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
		margin-left: 4px;
		transition: opacity 0.15s;
	}
	.user-avatar:hover {
		opacity: 0.85;
	}

	.user-dropdown {
		position: absolute;
		top: calc(100% + 8px);
		right: 0;
		min-width: 200px;
		background: var(--color-bg-elevated);
		border: 1px solid var(--color-border);
		border-radius: 8px;
		padding: 8px 0;
		box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
		z-index: 100;
	}

	.user-info {
		padding: 8px 14px 10px;
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.user-name {
		font-size: 13px;
		font-weight: 600;
		color: var(--color-text-primary);
	}

	.user-email {
		font-size: 11px;
		color: var(--color-text-muted);
	}

	.dropdown-item {
		display: flex;
		align-items: center;
		gap: 8px;
		width: 100%;
		padding: 7px 14px;
		background: none;
		border: none;
		color: var(--color-text-secondary);
		font-size: 13px;
		cursor: pointer;
		text-align: left;
		transition: all 0.15s;
	}
	.dropdown-item:hover {
		background: var(--color-bg-overlay);
		color: var(--color-text-primary);
	}

	/* ── Main content ───────────────────────────── */
	.app-main {
		flex: 1;
		overflow: auto;
	}
</style>
