<script lang="ts">
	import { enhance } from '$app/forms';
	import { invalidateAll } from '$app/navigation';
	import Icon from '$lib/components/ui/Icon.svelte';
	import type { ActionData, PageData } from './$types';

	let { data, form } = $props<{ data: PageData; form?: ActionData }>();

	const roleClassMap = {
		OWNER: 'badge-owner',
		EDITOR: 'badge-editor',
		VIEWER: 'badge-viewer'
	} as const;

	const roleOptions = ['ALL', 'OWNER', 'EDITOR', 'VIEWER'] as const;

	let selectedRole = $state<(typeof roleOptions)[number]>('ALL');
	let showCreateModal = $state(false);
	let showManageModal = $state(false);
	let activeNotebook = $state<(typeof data.notebooks)[number] | null>(null);
	type NotebookItem = PageData['notebooks'][number];

	let filteredNotebooks = $derived(
		selectedRole === 'ALL'
			? data.notebooks
			: data.notebooks.filter((notebook: NotebookItem) => notebook.role === selectedRole)
	);

	let featuredNotebooks = $derived(filteredNotebooks.slice(0, 3));

	function openManageModal(notebook: (typeof data.notebooks)[number]) {
		activeNotebook = notebook;
		showManageModal = true;
	}

	function getRoleClass(role: keyof typeof roleClassMap) {
		return roleClassMap[role];
	}
</script>

<svelte:head>
	<title>Workspace · HuShi</title>
</svelte:head>

<section class="workspace-page">
	<div class="hero-row">
		<div>
			<h1>Workspace</h1>
			<p>Manage, enrich, and synthesize your recruiting materials in one place.</p>
		</div>
		<button type="button" class="create-button" onclick={() => (showCreateModal = true)}>
			<Icon name="plus" size={16} />
			Create new
		</button>
	</div>

	<section class="workspace-section">
		<div class="section-head">
			<span>Featured notebooks</span>
			<div class="section-meta">{filteredNotebooks.length} active</div>
		</div>

		<div class="featured-grid">
			{#if featuredNotebooks.length > 0}
				{#each featuredNotebooks as notebook}
					<div class="featured-wrap">
						<a
							class="featured-card"
							href={`/workspace/${notebook.slug}`}
							style={`background:${notebook.accent}`}
						>
							<div class="featured-tag">{notebook.tag}</div>
							<div>
								<h2>{notebook.title}</h2>
								<p>{notebook.summary}</p>
							</div>
							<div class="featured-meta">
								<span><Icon name="file-text" size={12} /> {notebook.sourcesLabel}</span>
								<span><Icon name="clock" size={12} /> {notebook.updatedAgo}</span>
							</div>
						</a>
						<button
							type="button"
							class="card-manage"
							aria-label="Manage notebook"
							onclick={() => openManageModal(notebook)}
						>
							<Icon name="settings" size={14} />
						</button>
					</div>
				{/each}
			{:else}
				<div class="empty-card">
					<strong>No notebooks match this filter.</strong>
					<span>Create a new notebook or switch role filters to see more.</span>
				</div>
			{/if}
		</div>
	</section>

	<section class="workspace-section">
		<div class="section-head">
			<span>Recent notebooks</span>
			<div class="filter-chip">
				<label for="role-filter">Filter by</label>
				<select id="role-filter" bind:value={selectedRole}>
					{#each roleOptions as option}
						<option value={option}>
							{option === 'ALL' ? 'All roles' : option}
						</option>
					{/each}
				</select>
			</div>
		</div>

		<div class="table-card">
			<div class="table-head">
				<span>Title</span>
				<span>Sources</span>
				<span>Created Date</span>
				<span>Role</span>
				<span></span>
			</div>

			{#if filteredNotebooks.length > 0}
				{#each filteredNotebooks as notebook}
					<div class="table-row">
						<a class="table-row-main" href={`/workspace/${notebook.slug}`}>
							<div class="title-cell">
								<Icon name="list" size={15} />
								<div>
									<strong>{notebook.title}</strong>
									<small>{notebook.subject}</small>
								</div>
							</div>
							<span>{notebook.sourcesLabel}</span>
							<span>{notebook.createdDate}</span>
							<span class={`badge ${getRoleClass(notebook.role)}`}>{notebook.role}</span>
							<span class="row-action">
								<Icon name="chevron-right" size={16} />
							</span>
						</a>
						<button
							type="button"
							class="row-manage"
							aria-label="Manage notebook"
							onclick={() => openManageModal(notebook)}
						>
							<Icon name="settings" size={14} />
						</button>
					</div>
				{/each}
			{:else}
				<div class="empty-state">
					<strong>No notebooks to show.</strong>
					<span>Try another role filter or create a new notebook.</span>
				</div>
			{/if}
		</div>
	</section>
</section>

{#if showCreateModal}
	<div
		class="modal-overlay"
		role="presentation"
		tabindex="-1"
		onclick={() => (showCreateModal = false)}
		onkeydown={(event) => {
			if (event.key === 'Escape') showCreateModal = false;
		}}
	>
		<div
			class="modal-box create-modal"
			role="dialog"
			aria-modal="true"
			aria-labelledby="create-notebook-title"
			tabindex="-1"
			onclick={(event) => event.stopPropagation()}
			onkeydown={(event) => event.stopPropagation()}
		>
			<button
				type="button"
				class="modal-close"
				aria-label="Close modal"
				onclick={() => (showCreateModal = false)}
			>
				<Icon name="x" size={16} />
			</button>

			<h2 id="create-notebook-title">Create notebook</h2>
			<p>Start a new recruiting workspace with a title, a subject, and a role-specific tag.</p>

			<form
				method="POST"
				action="?/createNotebook"
				use:enhance={() => {
					return async ({ update, result }) => {
						await update();
						if (result.type === 'failure') return;
						showCreateModal = false;
					};
				}}
				class="create-form"
			>
				<label>
					<span class="label">Notebook title</span>
					<input
						name="title"
						class="input-field"
						placeholder="Resume Analysis"
						value={form?.createValues?.title ?? ''}
						required
					/>
				</label>

				<label>
					<span class="label">Subject</span>
					<input
						name="subject"
						class="input-field"
						placeholder="Jordan Smith"
						value={form?.createValues?.subject ?? ''}
						required
					/>
				</label>

				<label>
					<span class="label">Tag</span>
					<input
						name="tag"
						class="input-field"
						placeholder="AI ANALYSIS"
						value={form?.createValues?.tag ?? ''}
					/>
				</label>

				{#if form?.createError}
					<p class="form-error">{form.createError}</p>
				{/if}

				<div class="form-actions">
					<button type="button" class="btn btn-secondary" onclick={() => (showCreateModal = false)}>
						Cancel
					</button>
					<button type="submit" class="btn btn-primary">Create notebook</button>
				</div>
			</form>
		</div>
	</div>
{/if}

{#if showManageModal && activeNotebook}
	<div
		class="modal-overlay"
		role="presentation"
		tabindex="-1"
		onclick={() => (showManageModal = false)}
		onkeydown={(event) => {
			if (event.key === 'Escape') showManageModal = false;
		}}
	>
		<div
			class="modal-box create-modal"
			role="dialog"
			aria-modal="true"
			aria-labelledby="manage-notebook-title"
			tabindex="-1"
			onclick={(event) => event.stopPropagation()}
			onkeydown={(event) => event.stopPropagation()}
		>
			<button
				type="button"
				class="modal-close"
				aria-label="Close modal"
				onclick={() => (showManageModal = false)}
			>
				<Icon name="x" size={16} />
			</button>

			<h2 id="manage-notebook-title">Manage notebook</h2>
			<p>Rename this Workspace item or remove it if it is no longer needed.</p>

			<form
				method="POST"
				action="?/updateNotebook"
				use:enhance={() => {
					return async ({ update, result }) => {
						await update();
						if (result.type === 'failure') return;
						await invalidateAll();
						showManageModal = false;
					};
				}}
				id="updateNotebookForm"
				class="create-form"
			>
				<input type="hidden" name="slug" value={activeNotebook.slug} />

				<label>
					<span class="label">Notebook title</span>
					<input
						name="title"
						class="input-field"
						value={form?.manageValues?.slug === activeNotebook.slug
							? form.manageValues?.title
							: activeNotebook.title}
						required
					/>
				</label>

				<label>
					<span class="label">Subject</span>
					<input
						name="subject"
						class="input-field"
						value={form?.manageValues?.slug === activeNotebook.slug
							? form.manageValues?.subject
							: activeNotebook.subject}
						required
					/>
				</label>

				<label>
					<span class="label">Tag</span>
					<input
						name="tag"
						class="input-field"
						value={form?.manageValues?.slug === activeNotebook.slug
							? form.manageValues?.tag
							: activeNotebook.tag}
						required
					/>
				</label>

				{#if form?.manageError}
					<p class="form-error">{form.manageError}</p>
				{/if}
			</form>

			<div class="manage-actions-row">
				<form method="POST" action="?/deleteNotebook" class="delete-notebook-form">
					<input type="hidden" name="slug" value={activeNotebook.slug} />
					<button type="submit" class="btn btn-danger" formnovalidate>
						Delete notebook
					</button>
				</form>
				<div class="manage-primary">
					<button type="button" class="btn btn-secondary" onclick={() => (showManageModal = false)}>
						Cancel
					</button>
					<button type="submit" class="btn btn-primary" form="updateNotebookForm">Save changes</button>
				</div>
			</div>
		</div>
	</div>
{/if}

<style>
	.workspace-page {
		max-width: 1360px;
		margin: 0 auto;
		padding: 28px 18px 56px;
	}

	.hero-row {
		display: flex;
		justify-content: space-between;
		gap: 16px;
		align-items: flex-start;
		margin-bottom: 34px;
	}

	.hero-row h1 {
		margin: 0;
		font-size: clamp(2.2rem, 4vw, 3rem);
		line-height: 1;
		letter-spacing: -0.04em;
	}

	.hero-row p {
		margin: 8px 0 0;
		color: var(--color-text-secondary);
		font-size: 1.02rem;
	}

	.create-button {
		display: inline-flex;
		align-items: center;
		gap: 10px;
		background: linear-gradient(135deg, #89d6ff, #59b6ff);
		color: #052741;
		font-weight: 700;
		border: 0;
		border-radius: 8px;
		padding: 15px 18px;
		box-shadow: 0 10px 24px rgba(89, 182, 255, 0.18);
		transition:
			transform 0.18s ease,
			box-shadow 0.18s ease,
			filter 0.18s ease;
	}

	.create-button:hover {
		transform: translateY(-2px);
		box-shadow: 0 16px 34px rgba(89, 182, 255, 0.24);
		filter: brightness(1.02);
	}

	.workspace-section + .workspace-section {
		margin-top: 42px;
	}

	.section-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 16px;
		margin-bottom: 18px;
	}

	.section-head > span {
		color: var(--color-text-secondary);
		text-transform: uppercase;
		letter-spacing: 0.12em;
		font-size: 0.85rem;
		font-weight: 700;
	}

	.section-meta {
		color: var(--color-text-muted);
		font-size: 0.9rem;
	}

	.featured-grid {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 16px;
	}

	.featured-wrap {
		position: relative;
	}

	.featured-card,
	.empty-card {
		min-height: 168px;
		display: flex;
		flex-direction: column;
		justify-content: space-between;
		border-radius: 10px;
		border: 1px solid rgba(255, 255, 255, 0.05);
		padding: 18px;
		color: inherit;
		transition:
			transform 0.2s ease,
			border-color 0.2s ease,
			box-shadow 0.2s ease;
	}

	.featured-card {
		text-decoration: none;
	}

	.featured-card:hover {
		transform: translateY(-4px);
		border-color: rgba(140, 217, 255, 0.16);
		box-shadow: 0 18px 38px rgba(0, 0, 0, 0.22);
	}

	.card-manage,
	.row-manage {
		border: 1px solid rgba(255, 255, 255, 0.08);
		background: rgba(7, 12, 18, 0.78);
		color: var(--color-text-secondary);
		transition:
			transform 0.18s ease,
			border-color 0.18s ease,
			color 0.18s ease,
			background 0.18s ease;
	}

	.card-manage:hover,
	.row-manage:hover {
		transform: translateY(-2px);
		border-color: rgba(140, 217, 255, 0.2);
		color: var(--color-text-primary);
		background: rgba(16, 24, 34, 0.92);
	}

	.card-manage {
		position: absolute;
		top: 12px;
		right: 12px;
		width: 34px;
		height: 34px;
		border-radius: 10px;
		display: inline-grid;
		place-items: center;
	}

	.empty-card {
		grid-column: 1 / -1;
		background: rgba(255, 255, 255, 0.03);
	}

	.featured-card h2 {
		margin: 14px 0 10px;
		font-size: 1.75rem;
		line-height: 1.15;
		letter-spacing: -0.04em;
	}

	.featured-card p,
	.empty-card span {
		margin: 0;
		color: var(--color-text-secondary);
	}

	.featured-tag {
		color: #55bfff;
		font-weight: 700;
		letter-spacing: 0.08em;
		font-size: 0.8rem;
	}

	.featured-meta {
		display: flex;
		gap: 16px;
		flex-wrap: wrap;
		color: var(--color-text-muted);
		font-size: 0.85rem;
	}

	.featured-meta span {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}

	.filter-chip {
		display: flex;
		align-items: center;
		gap: 10px;
		color: var(--color-text-muted);
		font-size: 0.86rem;
	}

	.filter-chip select {
		background: rgba(255, 255, 255, 0.03);
		color: var(--color-text-primary);
		border: 1px solid var(--color-border);
		border-radius: 8px;
		padding: 8px 12px;
		transition:
			border-color 0.18s ease,
			box-shadow 0.18s ease,
			background 0.18s ease;
	}

	.filter-chip select:hover,
	.filter-chip select:focus {
		border-color: rgba(140, 217, 255, 0.28);
		box-shadow: 0 0 0 3px rgba(59, 158, 255, 0.12);
	}

	.table-card {
		background: rgba(255, 255, 255, 0.02);
		border: 1px solid rgba(255, 255, 255, 0.05);
		border-radius: 10px;
		overflow: hidden;
	}

	.table-head,
	.table-row-main {
		display: grid;
		grid-template-columns: minmax(0, 2.3fr) 0.7fr 0.9fr 0.55fr 44px;
		gap: 16px;
		align-items: center;
	}

	.table-head {
		padding: 16px 18px;
		color: var(--color-text-muted);
		font-size: 0.8rem;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		border-bottom: 1px solid rgba(255, 255, 255, 0.05);
	}

	.table-row {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto;
		gap: 10px;
		align-items: center;
		padding: 0 18px;
		border-bottom: 1px solid rgba(255, 255, 255, 0.04);
	}

	.table-row-main {
		padding: 16px 0;
		text-decoration: none;
		color: inherit;
		transition:
			background 0.18s ease,
			transform 0.18s ease,
			border-color 0.18s ease;
	}

	.table-row-main:hover {
		background: rgba(255, 255, 255, 0.035);
		transform: translateX(4px);
		border-color: rgba(140, 217, 255, 0.12);
	}

	.table-row:last-child {
		border-bottom: 0;
	}

	.title-cell {
		display: flex;
		align-items: center;
		gap: 10px;
		min-width: 0;
	}

	.title-cell > div {
		display: grid;
		gap: 3px;
		min-width: 0;
	}

	.title-cell strong,
	.title-cell small {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.title-cell small {
		color: var(--color-text-muted);
	}

	.row-action {
		display: inline-flex;
		justify-content: flex-end;
		color: var(--color-text-muted);
	}

	.row-manage {
		width: 34px;
		height: 34px;
		border-radius: 10px;
		display: inline-grid;
		place-items: center;
	}

	.empty-state {
		padding: 24px 18px;
		display: grid;
		gap: 8px;
	}

	.empty-state span {
		color: var(--color-text-secondary);
	}

	.create-modal {
		max-width: 520px;
	}

	.create-modal > p {
		margin: 8px 0 20px;
		color: var(--color-text-secondary);
	}

	.create-form {
		display: grid;
		gap: 16px;
	}

	.create-form label {
		display: grid;
		gap: 8px;
	}

	.form-actions {
		display: flex;
		justify-content: flex-end;
		gap: 12px;
		margin-top: 8px;
	}

	.manage-actions {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		margin-top: 8px;
	}

	.manage-actions-row {
		display: grid;
		grid-template-columns: max-content minmax(0, auto);
		align-items: center;
		gap: 12px;
		margin-top: 8px;
	}

	.delete-notebook-form {
		display: inline-flex;
	}

	.manage-primary {
		display: flex;
		gap: 12px;
		justify-self: end;
	}

	.btn-danger {
		background: rgba(255, 108, 122, 0.12);
		color: #ffb6bd;
		border: 1px solid rgba(255, 108, 122, 0.18);
	}

	.btn-danger:hover {
		background: rgba(255, 108, 122, 0.18);
	}

	.form-error {
		margin: 0;
		color: var(--color-danger);
	}

	.modal-close {
		position: absolute;
		right: 14px;
		top: 14px;
		border: 0;
		background: transparent;
		color: var(--color-text-secondary);
		transition:
			color 0.18s ease,
			transform 0.18s ease;
	}

	.modal-close:hover {
		color: var(--color-text-primary);
		transform: rotate(90deg);
	}

	@media (max-width: 960px) {
		.featured-grid {
			grid-template-columns: 1fr;
		}

		.table-head,
		.table-row-main {
			grid-template-columns: minmax(0, 1.6fr) 0.8fr 0.9fr 0.6fr 32px;
			font-size: 0.85rem;
		}
	}

	@media (max-width: 720px) {
		.hero-row,
		.section-head {
			flex-direction: column;
			align-items: stretch;
		}

		.table-head {
			display: none;
		}

		.table-row {
			grid-template-columns: minmax(0, 1fr) auto;
		}

		.table-row-main {
			grid-template-columns: 1fr;
			gap: 8px;
		}

		.row-action {
			justify-content: flex-start;
		}

		.manage-actions {
			flex-direction: column;
			align-items: stretch;
		}

		.manage-primary {
			display: grid;
			grid-template-columns: 1fr 1fr;
		}
	}
</style>
