<script lang="ts">
	import Icon from '$lib/components/ui/Icon.svelte';
	import { featuredNotebooks } from '$lib/data/workbench';

	const highlights = [
		'Curate candidate packets into reusable notebooks',
		'Synthesize resumes, portfolios, transcripts, and job descriptions',
		'Move from raw files to AI-assisted hiring narratives inside one Workspace'
	];
</script>

<svelte:head>
	<title>HuShi</title>
</svelte:head>

<div class="landing-shell">
	<div class="landing-noise"></div>

	<header class="landing-header">
		<a href="/" class="brand-mark">
			<span class="brand-icon"><Icon name="brain" size={18} /></span>
			<span>HuShi</span>
		</a>

		<nav class="landing-nav">
			<a href="#features">Features</a>
			<a href="#preview">Preview</a>
			<a href="/login" class="ghost-link">Login</a>
			<a href="/login" class="primary-link">Enter Workspace</a>
		</nav>
	</header>

	<main class="landing-main">
		<section class="hero-copy">
			<p class="eyebrow">AI recruiting studio</p>
			<h1>Turn scattered candidate materials into a navigable hiring Workspace.</h1>
			<p class="hero-text">
				HuShi brings resumes, portfolios, conversations, and job descriptions into a single
				synthesis flow so teams can review faster and make stronger decisions.
			</p>

			<div class="hero-actions">
				<a href="/login" class="cta-primary">Start curating</a>
				<a href="/login" class="cta-secondary">Open existing Workspace</a>
			</div>

			<ul class="hero-list">
				{#each highlights as item}
					<li><Icon name="zap" size={14} /> {item}</li>
				{/each}
			</ul>
		</section>

		<section class="hero-preview" id="preview">
			<div class="preview-window">
				<div class="preview-topbar">
					<span>Featured notebooks</span>
					<a href="/login">Open app</a>
				</div>

				<div class="preview-grid">
					{#each featuredNotebooks as notebook}
						<article class="preview-card" style={`background:${notebook.accent}`}>
							<div class="preview-tag">{notebook.tag}</div>
							<h2>{notebook.title}</h2>
							<p>{notebook.summary}</p>
							<div class="preview-meta">
								<span>{notebook.sources} sources</span>
								<span>{notebook.updatedAgo}</span>
							</div>
						</article>
					{/each}
				</div>
			</div>
		</section>
	</main>

	<section class="feature-band" id="features">
		<article>
			<span class="feature-label">01</span>
			<h2>Multi-source notebooks</h2>
			<p>Bundle candidate evidence once, then reuse it across screens, reviewers, and prompts.</p>
		</article>
		<article>
			<span class="feature-label">02</span>
			<h2>Studio actions</h2>
			<p>
				Generate role fit, audio overviews, interview prompts, and summary packs without leaving the
				flow.
			</p>
		</article>
		<article>
			<span class="feature-label">03</span>
			<h2>Decision-ready outputs</h2>
			<p>
				Keep saved outputs attached to each notebook so handoff and async review stay grounded in
				source material.
			</p>
		</article>
	</section>
</div>

<style>
	.landing-shell {
		min-height: 100vh;
		padding: 24px 24px 40px;
		background:
			radial-gradient(circle at top right, rgba(59, 158, 255, 0.13), transparent 32%),
			radial-gradient(circle at bottom left, rgba(49, 190, 145, 0.08), transparent 28%),
			linear-gradient(180deg, #090b0f 0%, #0d1117 100%);
		position: relative;
		overflow: hidden;
	}

	.landing-noise {
		position: absolute;
		inset: 0;
		background:
			linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px),
			linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px);
		background-size: 72px 72px;
		mask-image: radial-gradient(circle at center, black 35%, transparent 85%);
		pointer-events: none;
	}

	.landing-header,
	.landing-main,
	.feature-band {
		position: relative;
		z-index: 1;
	}

	.landing-header {
		max-width: 1240px;
		margin: 0 auto;
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 24px;
	}

	.brand-mark {
		display: inline-flex;
		align-items: center;
		gap: 10px;
		text-decoration: none;
		color: var(--color-text-primary);
		font-size: 1rem;
		font-weight: 700;
		transition:
			transform 0.18s ease,
			opacity 0.18s ease;
	}

	.brand-mark:hover {
		opacity: 0.92;
		transform: translateY(-1px);
	}

	.brand-icon {
		width: 34px;
		height: 34px;
		display: grid;
		place-items: center;
		border-radius: 10px;
		color: var(--color-accent);
		background: rgba(59, 158, 255, 0.12);
		border: 1px solid rgba(119, 189, 255, 0.24);
	}

	.landing-nav {
		display: flex;
		align-items: center;
		gap: 18px;
		flex-wrap: wrap;
	}

	.landing-nav a {
		color: var(--color-text-secondary);
		text-decoration: none;
		font-size: 0.95rem;
		transition:
			color 0.18s ease,
			transform 0.18s ease,
			border-color 0.18s ease;
	}

	.landing-nav a:hover {
		color: var(--color-text-primary);
		transform: translateY(-1px);
	}

	.ghost-link,
	.primary-link {
		padding: 10px 16px;
		border-radius: 999px;
		border: 1px solid var(--color-border);
	}

	.primary-link {
		background: linear-gradient(135deg, #8dd8ff, #57b6ff);
		color: #04263f;
		border-color: transparent;
		font-weight: 700;
		box-shadow: 0 12px 26px rgba(87, 182, 255, 0.18);
	}

	.landing-main {
		max-width: 1240px;
		margin: 72px auto 0;
		display: grid;
		grid-template-columns: minmax(0, 1.05fr) minmax(320px, 0.95fr);
		gap: 40px;
		align-items: center;
	}

	.hero-copy h1 {
		max-width: 10ch;
		margin: 0;
		font-size: clamp(3rem, 8vw, 5.4rem);
		line-height: 0.96;
		letter-spacing: -0.05em;
	}

	.eyebrow {
		margin: 0 0 18px;
		text-transform: uppercase;
		letter-spacing: 0.2em;
		color: #7cc9ff;
		font-size: 0.8rem;
		font-weight: 700;
	}

	.hero-text {
		max-width: 58ch;
		color: var(--color-text-secondary);
		font-size: 1.08rem;
		margin: 22px 0 0;
	}

	.hero-actions {
		display: flex;
		gap: 14px;
		margin-top: 28px;
		flex-wrap: wrap;
	}

	.cta-primary,
	.cta-secondary {
		padding: 13px 20px;
		border-radius: 999px;
		text-decoration: none;
		font-weight: 700;
		transition:
			transform 0.18s ease,
			box-shadow 0.18s ease,
			border-color 0.18s ease;
	}

	.cta-primary:hover,
	.cta-secondary:hover,
	.primary-link:hover {
		transform: translateY(-2px);
	}

	.cta-primary {
		background: linear-gradient(135deg, #8dd8ff, #57b6ff);
		color: #04263f;
		box-shadow: 0 14px 32px rgba(87, 182, 255, 0.2);
	}

	.cta-secondary {
		border: 1px solid var(--color-border);
		color: var(--color-text-primary);
		background: rgba(255, 255, 255, 0.03);
	}

	.hero-list {
		display: grid;
		gap: 12px;
		margin: 28px 0 0;
		padding: 0;
		list-style: none;
		color: var(--color-text-secondary);
	}

	.hero-list li {
		display: flex;
		align-items: center;
		gap: 10px;
	}

	.hero-list :global(svg) {
		color: #7cc9ff;
		flex: none;
	}

	.preview-window {
		border-radius: 28px;
		padding: 22px;
		border: 1px solid rgba(255, 255, 255, 0.08);
		background: rgba(10, 13, 18, 0.82);
		box-shadow: 0 36px 70px rgba(0, 0, 0, 0.45);
		backdrop-filter: blur(18px);
	}

	.preview-topbar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		color: var(--color-text-secondary);
		font-size: 0.92rem;
		margin-bottom: 18px;
	}

	.preview-topbar a {
		color: #7cc9ff;
		text-decoration: none;
	}

	.preview-grid {
		display: grid;
		gap: 14px;
	}

	.preview-card {
		min-height: 160px;
		border-radius: 20px;
		border: 1px solid rgba(255, 255, 255, 0.06);
		padding: 20px;
		display: flex;
		flex-direction: column;
		justify-content: space-between;
		transition:
			transform 0.22s ease,
			border-color 0.22s ease,
			box-shadow 0.22s ease;
	}

	.preview-card:hover {
		transform: translateY(-4px);
		border-color: rgba(140, 217, 255, 0.18);
		box-shadow: 0 18px 36px rgba(0, 0, 0, 0.24);
	}

	.preview-card h2 {
		margin: 10px 0 8px;
		font-size: 1.18rem;
	}

	.preview-card p {
		margin: 0;
		color: var(--color-text-secondary);
	}

	.preview-tag {
		color: #55bfff;
		font-size: 0.75rem;
		font-weight: 700;
		letter-spacing: 0.08em;
	}

	.preview-meta {
		display: flex;
		gap: 14px;
		flex-wrap: wrap;
		color: var(--color-text-muted);
		font-size: 0.85rem;
	}

	.feature-band {
		max-width: 1240px;
		margin: 64px auto 0;
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 18px;
	}

	.feature-band article {
		padding: 24px;
		border-radius: 22px;
		border: 1px solid rgba(255, 255, 255, 0.06);
		background: rgba(255, 255, 255, 0.03);
		transition:
			transform 0.18s ease,
			border-color 0.18s ease;
	}

	.feature-band article:hover {
		transform: translateY(-3px);
		border-color: rgba(140, 217, 255, 0.16);
	}

	.feature-label {
		color: #7cc9ff;
		font-size: 0.82rem;
		font-weight: 700;
		letter-spacing: 0.1em;
	}

	.feature-band h2 {
		margin: 12px 0 8px;
		font-size: 1.2rem;
	}

	.feature-band p {
		margin: 0;
		color: var(--color-text-secondary);
	}

	@media (max-width: 960px) {
		.landing-main,
		.feature-band {
			grid-template-columns: 1fr;
		}

		.landing-header {
			align-items: flex-start;
			flex-direction: column;
		}

		.hero-copy h1 {
			max-width: 12ch;
		}
	}
</style>
