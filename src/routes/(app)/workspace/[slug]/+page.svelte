<script lang="ts">
	import { enhance } from '$app/forms';
	import { invalidateAll } from '$app/navigation';
	import Icon from '$lib/components/ui/Icon.svelte';
	import { Streamdown } from 'svelte-streamdown';
	import type { ActionData, PageData } from './$types';

	let { data, form } = $props<{ data: PageData; form?: ActionData }>();
	let showSourceModal = $state(false);
	let showToolModal = $state(false);
	let showSettingsModal = $state(false);
	let showSourceDetailModal = $state(false);
	let showOutputModal = $state(false);
	let shareLabel = $state('Share');
	let activeTool = $state<(typeof data.notebook.studioTools)[number] | null>(null);
	let activeSource = $state<(typeof data.notebook.sources)[number] | null>(null);
	let activeOutput = $state<(typeof data.notebook.savedOutputs)[number] | null>(null);
	let selectedOutputId = $state<string | null>(null);
	let outputPreviewElement = $state<HTMLDivElement | null>(null);
	let analysisStatus = $state<'idle' | 'running' | 'done' | 'error'>('idle');
	let analysisTitle = $state('Analysis Console');
	let analysisText = $state(
		'Choose a Studio action to run a mock AI analysis. The generated text will stream into this panel and then be saved to Outputs.'
	);
	let analysisError = $state('');
	let isConsoleExpanded = $state(true);

	// 全局防崩溃异常捕获 (高优全屏弹窗)
	let globalCrashError = $state<{title: string, message: string} | null>(null);

	function handleGlobalError(event: ErrorEvent) {
		globalCrashError = {
			title: '系统发生严重异常',
			message: `前端页面发生未知异常，请立刻联系负责人排查，防止服务崩溃！\n\n报错详情：${event.message}`
		};
	}

	function handleUnhandledRejection(event: PromiseRejectionEvent) {
		const reason = event.reason instanceof Error ? event.reason.message : String(event.reason);
		globalCrashError = {
			title: '前端存在未处理的异步异常',
			message: `请立刻联系负责人排查，防止服务崩溃！\n\n报错内容：${reason}`
		};
	}

	// pipeline 阶段进度跟踪
	type StageStatus = 'pending' | 'running' | 'done' | 'failed';
	type StageInfo = {
		status: StageStatus;
		message: string;
		attempt?: number;
		startedAt?: number;  // ms timestamp
		durationMs?: number; // final ms
	};
	type ToolCall = {
		name: string;
		arguments?: any;
		result?: any;
		status: 'calling' | 'success' | 'error';
	};

	type LlmCallEntry = {
		stage: string;
		step: number;
		prompt?: string;
		thinking?: string;       // 完整 thinking（llm_call 汇总时设置）
		response?: string;       // 完整 response（llm_call 汇总时设置）
		thinkingStream: string;  // 流式累积 reasoning
		responseStream: string;  // 流式累积 text
		usage?: { input_tokens: number; output_tokens: number; total_tokens: number };
		collapsed: boolean;
		toolCalls?: ToolCall[];  // 工具调用记录（可选，默认空数组）
	};
	const STAGE_LABELS: Record<string, string> = {
		stage1a: '岗位匹配', stage1b: '聪明度', stage1c: '勤奋度',
		stage1d: '目标感',   stage1e: '皮实度', stage1f: '客户第一',
		stage2:  '综合汇总', stage3:  '报告生成'
	};
	// Progressive: only stages that have fired stage_start appear here (in order)
	let pipelineStageOrder = $state<string[]>([]);
	let pipelineStages = $state<Record<string, StageInfo>>({});
	let isPipelineMode = $state(false);
	let debugMode = $state(false);
	let llmCalls = $state<LlmCallEntry[]>([]);
	let tokenTotals = $state({ input: 0, output: 0, total: 0 });
	// Tick counter to update elapsed timers every second
	let timerTick = $state(0);
	let timerInterval: ReturnType<typeof setInterval> | null = null;
	let analysisConfig = $state({
		focus: '',
		tone: 'concise',
		length: 'standard',
		includeCitations: true,
		resumeSourceId: '',
		conversationSourceId: '',
		jdSourceId: '',
		evaluationMode: 'basic' as 'basic' | 'advanced',
		resumeCheckpoint: false
	});
	let dragOver = $state(false);
	let uploadedFile = $state<File | null>(null);
	// null = 未选，强制用户手动选卡片
	let selectedSourceType = $state<'resume' | 'conversation' | 'job-description' | null>(null);
	let sourceTypeShake = $state(false); // 晃动动画触发标志
	// 文件解析状态
	let parseStatus = $state<'idle' | 'parsing' | 'done' | 'error'>('idle');
	let parsedContent = $state('');   // 流式累积的 md 内容（用于预览）
	let parseError = $state('');
	let titleInputValue = $state(''); // 绑定 title 输入框
	const streamdownControls = { code: false, table: false, mermaid: false } as const;

	async function copyShareLink() {
		if (typeof navigator === 'undefined' || !navigator.clipboard) return;
		await navigator.clipboard.writeText(window.location.href);
		shareLabel = 'Copied';
		setTimeout(() => {
			shareLabel = 'Share';
		}, 1400);
	}

	function openToolModal(tool: (typeof data.notebook.studioTools)[number]) {
		activeTool = tool;
		analysisConfig = {
			focus: '',
			tone: 'concise',
			length: 'standard',
			includeCitations: true,
			resumeSourceId: '',
			conversationSourceId: '',
			jdSourceId: '',
			evaluationMode: 'basic',
			resumeCheckpoint: false
		};
		showToolModal = true;
	}

	function handleDragOver(event: DragEvent) {
		event.preventDefault();
		dragOver = true;
	}

	function handleDragLeave(event: DragEvent) {
		event.preventDefault();
		dragOver = false;
	}

	function setUploadedFile(file: File) {
		uploadedFile = file;
		// 自动填充 title（如果用户还没填）
		if (!titleInputValue) {
			titleInputValue = file.name.replace(/\.[^.]+$/, '');
		}
		// 重置解析状态
		parseStatus = 'idle';
		parsedContent = '';
		parseError = '';
	}

	function handleDrop(event: DragEvent) {
		event.preventDefault();
		dragOver = false;

		const files = event.dataTransfer?.files;
		if (files && files.length > 0) {
			setUploadedFile(files[0]);
		}
	}

	function handleFileSelect(event: Event) {
		const target = event.target as HTMLInputElement;
		if (target.files && target.files.length > 0) {
			setUploadedFile(target.files[0]);
		}
	}

	function openSourceModal(type: 'resume' | 'conversation' | 'job-description' | null = null) {
		selectedSourceType = type;
		showSourceModal = true;
		// 每次打开重置状态
		uploadedFile = null;
		titleInputValue = '';
		parseStatus = 'idle';
		parsedContent = '';
		parseError = '';
		sourceTypeShake = false;
	}

	function selectSourceType(type: 'resume' | 'conversation' | 'job-description') {
		selectedSourceType = type;
		sourceTypeShake = false;
	}

	/** 触发晃动动画（未选 type 时提示用户） */
	function triggerSourceTypeShake() {
		sourceTypeShake = false;
		// 强制重新触发动画
		requestAnimationFrame(() => {
			sourceTypeShake = true;
			setTimeout(() => {
				sourceTypeShake = false;
			}, 600);
		});
	}

	/** 调用 parse-source API，流式解析文件为 md */
	async function parseFile() {
		if (!uploadedFile || !selectedSourceType) return;

		parseStatus = 'parsing';
		parsedContent = '';
		parseError = '';

		const fd = new FormData();
		fd.set('file', uploadedFile, uploadedFile.name);
		fd.set('sourceType', selectedSourceType);
		fd.set('title', titleInputValue || uploadedFile.name.replace(/\.[^.]+$/, ''));

		const parseUrl = `/workspace/${data.notebook.slug}/parse-source`;

		try {
			const resp = await fetch(parseUrl, {
				method: 'POST',
				body: fd
			});

			if (!resp.ok || !resp.body) {
				let msg = '解析失败';
				try {
					const j = await resp.json();
					msg = j.error ?? msg;
				} catch { /* ignore */ }
				parseStatus = 'error';
				parseError = msg;
				return;
			}

			const reader = resp.body.getReader();
			const decoder = new TextDecoder();
			let buf = '';

			while (true) {
				const { value, done } = await reader.read();
				if (done) break;
				buf += decoder.decode(value, { stream: true });

				// 按换行拆分 JSON lines
				const lines = buf.split('\n');
				buf = lines.pop() ?? '';

				for (const line of lines) {
					if (!line.trim()) continue;
					try {
						const msg = JSON.parse(line) as { type: string; text?: string; message?: string };
						if (msg.type === 'chunk' && msg.text) {
							parsedContent += msg.text;
						} else if (msg.type === 'error') {
							parseStatus = 'error';
							parseError = msg.message ?? '解析出错';
						} else if (msg.type === 'done') {
							parseStatus = 'done';
						}
					} catch { /* ignore malformed line */ }
				}
			}

			if (parseStatus === 'parsing') parseStatus = 'done';
		} catch (e) {
			parseStatus = 'error';
			parseError = e instanceof Error ? e.message : '网络错误';
		}
	}

	function getSourceTypeFromTitle(title: string): 'resume' | 'conversation' | 'job-description' {
		if (title === 'Resume') return 'resume';
		if (title === 'Conversation') return 'conversation';
		return 'job-description';
	}

	function openSourceDetail(source: (typeof data.notebook.sources)[number]) {
		activeSource = source;
		showSourceDetailModal = true;
	}

	function isMarkdownLike(title: string, content: string) {
		const lowerTitle = title.toLowerCase();
		if (lowerTitle.endsWith('.md') || lowerTitle.endsWith('.markdown')) {
			return true;
		}

		return /(^|\n)(#{1,6}\s|[-*]\s|\d+\.\s|>\s|```|\|.+\|)/m.test(content);
	}

	function getSourceDetailContent(source: (typeof data.notebook.sources)[number]) {
		if (form?.sourceEditValues?.sourceId === source.id) {
			return form.sourceEditValues?.content ?? source.content;
		}

		return source.content;
	}

	function openSavedOutput(output: (typeof data.notebook.savedOutputs)[number]) {
		activeOutput = output;
		selectedOutputId = output.id;
		analysisStatus = 'done';
		analysisError = '';
		analysisTitle = output.title;
		analysisText = output.content;
	}

	function resetConsole() {
		activeOutput = null;
		selectedOutputId = null;
		analysisStatus = 'idle';
		analysisError = '';
		analysisTitle = 'Analysis Console';
		analysisText =
			'Choose a Studio action to run a mock AI analysis. The generated text will stream into this panel and then be saved to Outputs.';
		isConsoleExpanded = true;
		isPipelineMode = false;
		pipelineStageOrder = [];
		pipelineStages = {};
		llmCalls = [];
		tokenTotals = { input: 0, output: 0, total: 0 };
		if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
	}

	function toggleConsoleMode() {
		isConsoleExpanded = !isConsoleExpanded;
	}

	function openOutputModal(output: (typeof data.notebook.savedOutputs)[number]) {
		activeOutput = output;
		showOutputModal = true;
	}

	function downloadOutput(output: (typeof data.notebook.savedOutputs)[number]) {
		if (typeof document === 'undefined') return;
		const blob = new Blob([output.content], { type: 'text/markdown;charset=utf-8' });
		const url = URL.createObjectURL(blob);
		const link = document.createElement('a');
		link.href = url;
		link.download = `${output.title.toLowerCase().replace(/[^a-z0-9]+/g, '-') || 'output'}.md`;
		link.click();
		URL.revokeObjectURL(url);
	}

	function downloadOutputAsText(output: (typeof data.notebook.savedOutputs)[number]) {
		if (typeof document === 'undefined') return;
		const blob = new Blob([output.content], { type: 'text/plain;charset=utf-8' });
		const url = URL.createObjectURL(blob);
		const link = document.createElement('a');
		link.href = url;
		link.download = `${output.title.toLowerCase().replace(/[^a-z0-9]+/g, '-') || 'output'}.txt`;
		link.click();
		URL.revokeObjectURL(url);
	}

	async function copyOutput(output: (typeof data.notebook.savedOutputs)[number]) {
		if (typeof navigator === 'undefined' || !navigator.clipboard) return;
		await navigator.clipboard.writeText(output.content);
	}

	function printOutputAsPdf(output: (typeof data.notebook.savedOutputs)[number]) {
		if (typeof window === 'undefined' || !outputPreviewElement) return;
		const previewWindow = window.open('', '_blank', 'noopener,noreferrer,width=960,height=720');
		if (!previewWindow) return;

		const rendered = outputPreviewElement.innerHTML;
		previewWindow.document.write(
// `<!doctype html>
// <html>
// <head>
// <meta charset="utf-8" />
// <title>${output.title}</title>
// <style>
// 	body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px; color: #111827; line-height: 1.65; }
// 	h1,h2,h3,h4 { line-height: 1.2; margin: 0 0 12px; }
// 	h2,h3,h4 { margin-top: 18px; }
// 	pre { white-space: pre-wrap; word-break: break-word; padding: 14px; border-radius: 12px; background: #f3f4f6; overflow: auto; }
// 	code { font-family: "SFMono-Regular", Menlo, monospace; }
// 	blockquote { margin: 14px 0; padding: 12px 14px; border-left: 3px solid #60a5fa; background: #f8fafc; }
// 	table { width: 100%; border-collapse: collapse; margin: 12px 0; }
// 	th, td { border: 1px solid #d1d5db; padding: 10px 12px; text-align: left; }
// 	ul, ol { margin: 10px 0 14px 1.2rem; }
// </style>
// </head>
// <body>
// ${rendered}
// </body>
// </html>`
);
		previewWindow.document.close();
		previewWindow.focus();
		previewWindow.print();
	}

	async function runMockAnalysis() {
		if (!activeTool) return;

		// Validation for interview-assessment
		if (activeTool.key === 'interview-assessment') {
			if (!analysisConfig.resumeSourceId || !analysisConfig.conversationSourceId || !analysisConfig.jdSourceId) {
				analysisStatus = 'error';
				analysisError = 'Please select all required sources: Resume, Interview Conversation, and Job Description.';
				return;
			}
		}

		selectedOutputId = null;
		showToolModal = false;
		analysisStatus = 'running';
		analysisError = '';
		analysisTitle = activeTool.title;
		analysisText = '';
		isPipelineMode = activeTool.key === 'interview-assessment';
		pipelineStageOrder = [];
		pipelineStages = {};
		llmCalls = [];
		tokenTotals = { input: 0, output: 0, total: 0 };
		if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
		if (isPipelineMode) {
			timerInterval = setInterval(() => { timerTick++; }, 1000);
		}

		try {
			const response = await fetch(`/workspace/${data.notebook.slug}/analyze`, {
				method: 'POST',
				headers: {
					'content-type': 'application/json'
				},
				body: JSON.stringify({
					tool: activeTool.key,
					...analysisConfig
				})
			});

			if (!response.ok || !response.body) {
				let message = 'Analysis failed to start.';
				try {
					const payload = await response.json();
					message = payload.error ?? message;
				} catch {
					// Ignore JSON parse failures and keep the fallback message.
				}
				analysisStatus = 'error';
				analysisError = message;
				return;
			}

			const reader = response.body.getReader();
			const decoder = new TextDecoder();
			let buf = '';

			while (true) {
				const { value, done } = await reader.read();
				if (done) break;
				buf += decoder.decode(value, { stream: true });

				const lines = buf.split('\n');
				buf = lines.pop() ?? '';

				for (const line of lines) {
					if (!line.trim()) continue;
					try {
						const event = JSON.parse(line) as {
							type: string;
							stage?: string;
							message?: string;
							text?: string;
							reasoning?: string;
							name?: string;
							attempt?: number;
							durationMs?: number;
							step?: number;
							prompt?: string;
							thinking?: string;
							response?: string;
							usage?: { input_tokens: number; output_tokens: number; total_tokens: number };
							args?: any;
							result?: any;
						};
						if (event.type === 'stage_start' && event.stage) {
							if (!pipelineStages[event.stage]) pipelineStageOrder = [...pipelineStageOrder, event.stage];
							pipelineStages = { ...pipelineStages, [event.stage]: { status: 'running', message: event.message ?? '', startedAt: Date.now() } };
						} else if (event.type === 'stage_complete' && event.stage) {
							pipelineStages = { ...pipelineStages, [event.stage]: {
								...pipelineStages[event.stage],
								status: 'done',
								message: event.message ?? '',
								durationMs: event.durationMs
							}};
						} else if (event.type === 'stage_retry' && event.stage) {
							pipelineStages = { ...pipelineStages, [event.stage]: {
								...pipelineStages[event.stage],
								status: 'running',
								message: event.message ?? '',
								attempt: event.attempt,
								startedAt: Date.now()
							}};
						} else if (event.type === 'stage_error' && event.stage) {
							pipelineStages = { ...pipelineStages, [event.stage]: { ...pipelineStages[event.stage], status: 'failed', message: event.message ?? '' } };
						} else if (event.type === 'llm_chunk' && event.stage && event.step) {
							// 找到对应 entry，追加流式内容；如果还没有则创建
							const idx = llmCalls.findIndex(c => c.stage === event.stage && c.step === event.step);
							if (idx === -1) {
								llmCalls = [...llmCalls, {
									stage: event.stage,
									step: event.step,
									thinkingStream: event.reasoning ?? '',
									responseStream: event.text ?? '',
									collapsed: false,
									toolCalls: []
								}];
							} else {
								const updated = [...llmCalls];
								updated[idx] = {
									...updated[idx],
									thinkingStream: updated[idx].thinkingStream + (event.reasoning ?? ''),
									responseStream: updated[idx].responseStream + (event.text ?? '')
								};
								llmCalls = updated;
							}
						} else if (event.type === 'llm_call' && event.stage && event.step) {
							// 汇总：找到对应 entry 补全 prompt/usage，如不存在则新建
							const idx = llmCalls.findIndex(c => c.stage === event.stage && c.step === event.step);
							const base: LlmCallEntry = idx !== -1 ? llmCalls[idx] : {
								stage: event.stage,
								step: event.step,
								thinkingStream: '',
								responseStream: '',
								collapsed: false,
								toolCalls: []
							};
							const entry: LlmCallEntry = {
								...base,
								prompt: event.prompt ?? base.prompt,
								thinking: event.thinking ?? base.thinking,
								response: event.response ?? base.response,
								usage: event.usage ?? base.usage
							};
							if (idx !== -1) {
								const updated = [...llmCalls];
								updated[idx] = entry;
								llmCalls = updated;
							} else {
								llmCalls = [...llmCalls, entry];
							}
							if (event.usage) {
								tokenTotals = {
									input: tokenTotals.input + (event.usage.input_tokens ?? 0),
									output: tokenTotals.output + (event.usage.output_tokens ?? 0),
									total: tokenTotals.total + (event.usage.total_tokens ?? 0)
								};
							}
						} else if (event.type === 'status') {
							if (!isPipelineMode) analysisText += `\n> ${event.message}\n`;
						} else if (event.type === 'tool') {
							// 工具调用事件 - 添加到对应的 stage 记录中
							if (event.stage && isPipelineMode) {
								// 在 pipeline 模式下，找到对应 stage 的 LlmCallEntry
								const stageCalls = llmCalls.filter(c => c.stage === event.stage);
								if (stageCalls.length > 0) {
									// 找到该 stage 最新的 LlmCallEntry（step 最大的）
									const latestCall = stageCalls.reduce((prev, current) =>
										(current.step > prev.step) ? current : prev
									);
									const idx = llmCalls.findIndex(c =>
										c.stage === latestCall.stage && c.step === latestCall.step
									);

									if (idx !== -1) {
										const updated = [...llmCalls];
										const call = updated[idx];

										// 确保 toolCalls 数组存在
										if (!call.toolCalls) {
											call.toolCalls = [];
										}

										// 根据消息内容判断工具调用状态
										let toolStatus: 'calling' | 'success' | 'error' = 'calling';
										let toolArguments: string | undefined;
										let toolResult: string | undefined;

										if (event.message.includes('接收参数')) {
											toolStatus = 'calling';
											toolArguments = event.message.replace('接收参数: ', '');
										} else if (event.message.includes('完成') || event.message.includes('解析完成')) {
											toolStatus = 'success';
											toolResult = event.message;
										} else if (event.message.includes('错误') || event.message.includes('失败')) {
											toolStatus = 'error';
											toolResult = event.message;
										}

										// 检查是否已经有同名的进行中工具调用
										const existingIndex = call.toolCalls.findIndex(t =>
											t.name === event.name && t.status === 'calling'
										);

										if (existingIndex !== -1) {
											// 更新现有的工具调用
											const toolCall = call.toolCalls[existingIndex];
											if (event.args) {
												toolCall.arguments = event.args;
											} else if (toolArguments) {
												toolCall.arguments = (toolCall.arguments || '') + toolArguments;
											}
											if (toolStatus !== 'calling') {
												toolCall.status = toolStatus;
												if (event.result) {
													toolCall.result = event.result;
												} else if (toolResult) {
													toolCall.result = toolResult;
												}
											}
										} else {
											// 创建新的工具调用记录
											const toolCall: ToolCall = {
												name: event.name,
												status: toolStatus,
												arguments: event.args || toolArguments,
												result: event.result || toolResult
											};
											call.toolCalls.push(toolCall);
										}

										llmCalls = updated;
									}
								}
							} else {
								// 非 pipeline 模式或没有 stage 信息，保持原来的显示方式
								analysisText += `\n> ⚙️  [${event.name}] ${event.message}\n`;
							}
						} else if (event.type === 'warning') {
							analysisText += `> ⚠️ ${event.message}\n`;
						} else if (event.type === 'chunk') {
							analysisText += event.text ?? '';
						} else if (event.type === 'done') {
							// 最终文本已通过 chunk 累积，无需额外处理
						} else if (event.type === 'error') {
							analysisStatus = 'error';
							analysisError = event.message ?? 'Analysis failed.';
							return;
						}
					} catch { /* ignore malformed line */ }
				}
			}

			analysisStatus = 'done';
			if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
			await invalidateAll();
		} catch (err) {
			console.error(err);
			analysisStatus = 'error';
			analysisError = '请求发生异常，请立刻联系开发，防止服务崩溃。详细信息: ' + (err instanceof Error ? err.message : String(err));
			if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
		}
	}
	function fmtMs(ms: number): string {
		if (ms < 1000) return `${ms}ms`;
		return `${(ms / 1000).toFixed(1)}s`;
	}

	function elapsedSec(startedAt: number | undefined): string {
		if (!startedAt) return '';
		// timerTick forces reactivity each second
		void timerTick;
		return fmtMs(Date.now() - startedAt);
	}
</script>

<svelte:head>
	<title>{data.notebook.title} · {data.notebook.subject}</title>
</svelte:head>

<svelte:window on:error={handleGlobalError} on:unhandledrejection={handleUnhandledRejection} />

<div class="beta-banner">
	<Icon name="alert-triangle" size={14} />
	本项目处于开发阶段，不支持的操作可能会引起错误，即时刷新或联系负责人
</div>

<section class="detail-shell">
	<header class="detail-topbar">
		<div class="detail-title-wrap">
			<a href="/workspace" class="back-link">
				<Icon name="arrow-left" size={16} />
			</a>
			<div>
				<h1>{data.notebook.title} - {data.notebook.subject}</h1>
				<div class="tab-row">
					<span class="active-tab">{data.notebook.role}</span>
					<span>{data.notebook.tag}</span>
				</div>
			</div>
		</div>

		<div class="detail-actions">
			<button
				type="button"
				class="ghost-icon"
				aria-label="Notebook settings"
				onclick={() => (showSettingsModal = true)}
			>
				<Icon name="settings" size={15} />
			</button>
			<button
				type="button"
				class="ghost-icon"
				aria-label="Open sources"
				onclick={() => openSourceModal()}
			>
				<Icon name="plus-circle" size={15} />
			</button>
			<button type="button" class="share-button" onclick={copyShareLink}>
				<Icon name="link" size={15} />
				{shareLabel}
			</button>
		</div>
	</header>

	<div class="detail-grid">
		<aside class="sources-panel">
			<div class="panel-head">
				<h2>Sources</h2>
			</div>

<button
					type="button"
					class="add-source-button"
					onclick={() => openSourceModal()}
				>
				<Icon name="plus-circle" size={16} />
				Add sources
			</button>

			<div class="source-list">
				{#each ['resume', 'conversation', 'job-description'] as sourceType}
					{@const sources = data.notebook.sources.filter((s: { sourceType: string }) => s.sourceType === sourceType)}
					{#if sources.length > 0}
						<div class="source-group">
							<h4 class="source-group-title">
								{sourceType === 'resume' ? '📄 简历 (Resume)' :
								 sourceType === 'conversation' ? '💬 面试对话 (Conversation)' :
								 '📋 职位描述 (Job Description)'}
								<span class="source-count">({sources.length})</span>
							</h4>
							{#each sources as source (source.id)}
								<div class:selected={source.selected} class="source-row">
									<form method="POST" action="?/selectSource" use:enhance class="source-form">
										<input type="hidden" name="sourceId" value={source.id} />
										<button type="submit" class:selected={source.selected} class="source-item">
											<Icon name={source.icon} size={18} />
											<div>
												<strong>{source.title}</strong>
												<span>{source.subtitle}</span>
												<small>{source.meta}</small>
											</div>
										</button>
									</form>

									<div class="source-actions">
										<button
											type="button"
											class="mini-icon"
											aria-label="View source"
											onclick={() => openSourceDetail(source)}
										>
											<Icon name="eye" size={14} />
										</button>
										<form
											method="POST"
											action="?/deleteSource"
											use:enhance
											class="inline-delete-form"
										>
											<input type="hidden" name="sourceId" value={source.id} />
											<button type="submit" class="mini-icon danger-icon" aria-label="Delete source">
												<Icon name="trash" size={14} />
											</button>
										</form>
									</div>
								</div>
							{/each}
						</div>
					{/if}
				{/each}
			</div>
		</aside>

		<main class="document-panel">
			<div class="panel-head">
				<h2>Desktop</h2>
			</div>

			<article class="document-card">
				<h2>{data.notebook.subject}</h2>
				<p class="identity-line">
					<span>{data.notebook.location}</span>
					<span>{data.notebook.contact}</span>
					<span>{data.notebook.link}</span>
				</p>

				<section>
					<h3>Notebook Summary</h3>
					<div class="desktop-markdown">
						<Streamdown
							content={data.notebook.summary}
							class="markdown-body"
							static
							controls={streamdownControls}
						/>
					</div>
				</section>

				<section>
					<h3>Selected Source Preview</h3>
					<div class="desktop-markdown">
						<Streamdown
							content={data.notebook.selectedSourceContent}
							class="markdown-body"
							static
							controls={streamdownControls}
						/>
					</div>
				</section>
			</article>

			<div class="conversation-card analysis-console">
				<div class="conversation-head">
					<h3>{analysisTitle}</h3>
					<span>
						{#if analysisStatus === 'running'}
							运行中...
						{:else if analysisStatus === 'done'}
							已保存到输出
						{:else if analysisStatus === 'error'}
							出错
						{:else}
							就绪
						{/if}
					</span>
				</div>

				{#if isPipelineMode}
					<div class="pipeline-header">
						<span class="pipeline-title">流水线进度</span>
						<div class="pipeline-header-right">
							{#if debugMode && (tokenTotals.input > 0 || tokenTotals.output > 0)}
								<span class="token-badge">
									in {tokenTotals.input.toLocaleString()} · out {tokenTotals.output.toLocaleString()} · tot {tokenTotals.total.toLocaleString()}
								</span>
							{/if}
							{#if data.debug}
								<button type="button" class="debug-toggle {debugMode ? 'active' : ''}" onclick={() => { debugMode = !debugMode; }}>
									{debugMode ? 'DEBUG ON' : 'DEBUG'}
								</button>
							{/if}
						</div>
					</div>
					<div class="pipeline-progress">
						{#each pipelineStageOrder as stageId (stageId)}
							{@const stage = pipelineStages[stageId]}
							{@const stageCalls = llmCalls.filter(c => c.stage === stageId)}
							<div class="pipeline-stage" data-status={stage?.status ?? 'pending'}>
								<div class="stage-row">
									<span class="stage-icon">
										{#if stage?.status === 'done'}✓{:else if stage?.status === 'running'}⋯{:else if stage?.status === 'failed'}✗{:else}·{/if}
									</span>
									<span class="stage-label">{STAGE_LABELS[stageId] ?? stageId}</span>
									<span class="stage-timer">
										{#if stage?.status === 'running'}
											{elapsedSec(stage.startedAt)}
										{:else if stage?.status === 'done' && stage.durationMs !== undefined}
											{fmtMs(stage.durationMs)}
										{/if}
									</span>
								</div>
								{#if stage?.message}
									<div class="stage-message">
										{stage.message}{#if stage.attempt !== undefined} · 重试 {stage.attempt}{/if}
									</div>
								{/if}

								{#if stageCalls.length > 0}
									<div class="stage-llm-calls">
										{#each stageCalls as call, idx (call.stage + '-' + call.step)}
											<div class="llm-call-entry">
												<button
													type="button"
													class="llm-call-header"
													onclick={() => { call.collapsed = !call.collapsed; llmCalls = [...llmCalls]; }}
												>
													<span class="llm-call-step">#{call.step}</span>
													{#if !call.thinking && !call.thinkingStream && !call.prompt && call.response}
														<span class="llm-call-tag">tool_call</span>
													{/if}
													{#if call.usage}
														<span class="llm-call-usage">
															in {call.usage.input_tokens?.toLocaleString() ?? '—'} · out {call.usage.output_tokens?.toLocaleString() ?? '—'}
														</span>
													{/if}
													<span class="llm-call-chevron">{call.collapsed ? '▶' : '▼'}</span>
												</button>
												{#if !call.collapsed}
													<div class="llm-call-body">
														{#if debugMode && call.prompt}
															<details>
																<summary class="llm-section-title">Prompt</summary>
																<pre class="llm-pre">{call.prompt}</pre>
															</details>
														{/if}
														{#if call.thinkingStream || call.thinking}
															<details>
																<summary class="llm-section-title">Thinking{call.usage ? '' : ' ···'}</summary>
																<pre class="llm-pre llm-thinking">{call.thinkingStream || call.thinking}</pre>
															</details>
														{/if}
														{#if call.responseStream || call.response}
															<details open>
																<summary class="llm-section-title">Output{call.usage ? '' : ' ···'}</summary>
																<pre class="llm-pre">{call.responseStream || call.response}</pre>
															</details>
														{/if}
														{#if call.toolCalls && call.toolCalls.length > 0}
															<details>
																<summary class="llm-section-title">Tool Calls</summary>
																<div class="tool-calls">
																	{#each call.toolCalls as toolCall, i (toolCall.name + i)}
																		<div class="tool-call {toolCall.status}">
																			<div class="tool-call-header">
																				<span class="tool-call-name">⚙️  {toolCall.name}</span>
																				<span class="tool-call-status">
																					{#if toolCall.status === 'calling'}调用中...
																					{:else if toolCall.status === 'success'}✓ 完成
																					{:else}✗ 错误
																					{/if}
																				</span>
																			</div>
																			{#if toolCall.arguments}
																				<div class="tool-call-arguments">
																					<strong>参数:</strong>
																					<pre class="tool-call-pre">{typeof toolCall.arguments === 'string' ? toolCall.arguments : JSON.stringify(toolCall.arguments, null, 2)}</pre>
																				</div>
																			{/if}
																			{#if toolCall.result}
																				<div class="tool-call-result">
																					<strong>结果:</strong>
																					<pre class="tool-call-pre">{typeof toolCall.result === 'string' ? toolCall.result : JSON.stringify(toolCall.result, null, 2)}</pre>
																				</div>
																			{/if}
																		</div>
																	{/each}
																</div>
															</details>
														{/if}
													</div>
												{/if}
											</div>
										{/each}
									</div>
								{/if}
							</div>
						{/each}
					</div>
				{/if}

				<div class={`analysis-stream ${analysisStatus} ${isConsoleExpanded ? 'expanded' : 'collapsed'}`}>
					<Streamdown
						content={analysisText}
						parseIncompleteMarkdown={analysisStatus === 'running'}
						class="markdown-body"
						controls={streamdownControls}
					/>
				</div>

				<div class="console-toggle-row">
					<button type="button" class="console-toggle" onclick={toggleConsoleMode}>
						<Icon name={isConsoleExpanded ? 'chevron-up' : 'chevron-down'} size={16} />
						{isConsoleExpanded ? 'Collapse console' : 'Expand console'}
					</button>
				</div>
				<div class="console-actions">
					<button type="button" class="text-button" onclick={resetConsole}>Clear console</button>
					<div class="console-meta">
						{#if selectedOutputId}
							<span>Viewing saved output</span>
						{/if}
						{#if activeOutput}
							<button type="button" class="text-button" onclick={() => downloadOutput(activeOutput)}>
								Download markdown
							</button>
						{/if}
					</div>
				</div>

				<form method="POST" action="?/sendMessage" use:enhance class="composer composer-disabled">
					<button type="button" class="composer-icon" onclick={() => (showSourceModal = true)}>
						<Icon name="paperclip" size={16} />
					</button>
					<input
						type="text"
						name="prompt"
						placeholder="Question input is temporarily disabled. Use Studio actions instead."
						disabled
					/>
					<button type="submit" class="composer-send" disabled>
						<Icon name="send" size={18} />
					</button>
				</form>
				<p class="inline-note">
					Interactive prompting is temporarily disabled while Studio-triggered analysis is the
					primary workflow.
				</p>
				{#if analysisError}
					<p class="inline-error">{analysisError}</p>
				{/if}
			</div>
		</main>

		<aside class="studio-panel">
			<div class="panel-head">
				<h2>Studio</h2>
			</div>

			<div class="tool-grid">
				{#each data.notebook.studioTools.filter(t => t.key !== 'audio-overview' || data.debug) as tool}
					<button type="button" class="tool-card" onclick={() => openToolModal(tool)}>
						<Icon name={tool.icon} size={18} />
						<strong>{tool.title}</strong>
						<span>{tool.description}</span>
					</button>
				{/each}
			</div>

			<div class="saved-head">
				<span>Saved outputs</span>
			</div>

			<div class="output-list">
				{#each data.notebook.savedOutputs as output}
					<div class:selected={selectedOutputId === output.id} class="output-row">
						<button type="button" class="output-card" onclick={() => openSavedOutput(output)}>
							<div class="output-head">
								<Icon name="zap" size={13} />
								<span>{output.updatedAgo}</span>
							</div>
							<h3>{output.title}</h3>
							<p>{output.summary}</p>

							{#if output.progress}
								<div class="playback">
									<div class="playback-bar">
										<span style={`width:${output.progress * 100}%`}></span>
									</div>
									<small>{output.duration}</small>
								</div>
							{/if}
						</button>

						<div class="source-actions">
							<button
								type="button"
								class="mini-icon"
								aria-label="Manage output"
								onclick={() => openOutputModal(output)}
							>
								<Icon name="eye" size={14} />
							</button>
							<form
								method="POST"
								action="?/deleteOutput"
								use:enhance={() => {
									return async ({ update, result }) => {
										await update();
										if (result.type === 'failure') return;
										if (selectedOutputId === output.id) {
											resetConsole();
										}
										if (activeOutput?.id === output.id) {
											showOutputModal = false;
											activeOutput = null;
										}
									};
								}}
								class="inline-delete-form"
							>
								<input type="hidden" name="outputId" value={output.id} />
								<button type="submit" class="mini-icon danger-icon" aria-label="Delete output">
									<Icon name="trash" size={14} />
								</button>
							</form>
						</div>
					</div>
				{/each}
			</div>

		</aside>
	</div>
</section>

{#if showSourceModal}
	<div
		class="modal-overlay"
		role="presentation"
		tabindex="-1"
		onclick={() => (showSourceModal = false)}
		onkeydown={(event) => {
			if (event.key === 'Escape') showSourceModal = false;
		}}
	>
		<div
			class="modal-box source-modal"
			role="dialog"
			aria-modal="true"
			aria-labelledby="source-modal-title"
			tabindex="-1"
			onclick={(event) => event.stopPropagation()}
			onkeydown={(event) => event.stopPropagation()}
		>
			<button
				type="button"
				class="modal-close"
				aria-label="Close modal"
				onclick={() => (showSourceModal = false)}
			>
				<Icon name="x" size={16} />
			</button>

			<h2 id="source-modal-title">Add Sources</h2>
			<p>
				Paste resume excerpts, interview notes, websites, or job descriptions into this notebook.
			</p>

				<!-- 来源类型卡片（必选） -->
			<p class="type-required-hint">请先选择来源类型（必选）</p>
			<div class="source-template-grid {sourceTypeShake ? 'shake' : ''}">
				{#each data.sourceTemplates as template (template.title)}
					{@const tType = getSourceTypeFromTitle(template.title)}
					<button
						type="button"
						class="template-card {selectedSourceType === tType ? 'selected' : ''} {!selectedSourceType && sourceTypeShake ? 'shake-outline' : ''}"
						onclick={() => selectSourceType(tType)}
					>
						<Icon name={template.icon} size={20} />
						<strong>{template.title}</strong>
						<span>{template.caption}</span>
					</button>
				{/each}
			</div>

			<form
				method="POST"
				action="?/addSource"
				enctype="multipart/form-data"
				use:enhance={({ formData, cancel }) => {
					// 必须先选 sourceType
					if (!selectedSourceType) {
						triggerSourceTypeShake();
						cancel();
						return;
					}
					// 上传了文件但尚未完成 AI 解析，阻止提交
					if (uploadedFile && parseStatus !== 'done') {
						if (parseStatus === 'idle') {
							parseError = '请先点击「AI 解析为 Markdown」完成解析。';
						} else if (parseStatus === 'parsing') {
							parseError = '正在解析中，请稍候...';
						}
						// parseStatus === 'error' 时 parseError 已经有内容
						cancel();
						return;
					}
					// 注入解析好的 md 内容
					if (parseStatus === 'done' && parsedContent) {
						formData.set('content', parsedContent);
					}
					// 拖拽场景：手动注入文件
					if (uploadedFile) {
						const existing = formData.get('file');
						if (!(existing instanceof File) || existing.size === 0) {
							formData.set('file', uploadedFile, uploadedFile.name);
						}
					}
					return async ({ update, result }) => {
						if (result.type === 'failure') {
							await update({ reset: false });
							return;
						}
						showSourceModal = false;
						uploadedFile = null;
						parsedContent = '';
						parseStatus = 'idle';
						titleInputValue = '';
						await invalidateAll();
					};
				}}
				class="source-form-grid"
			>
				<input type="hidden" name="sourceType" value={selectedSourceType ?? ''} />

				<label>
					<span class="label">Title</span>
					<input
						class="input-field"
						name="title"
						placeholder="文件名或自定义标题"
						bind:value={titleInputValue}
						required
					/>
				</label>

				<label>
					<span class="label">Meta Label</span>
					<input class="input-field" name="meta" placeholder="Primary source" />
				</label>

				<!-- 文件上传区域 -->
				<label>
					<span class="label">上传文件</span>
					<div
						class="upload-zone {dragOver ? 'drag-over' : ''}"
						role="region"
						aria-label="File upload drop zone"
						ondragover={handleDragOver}
						ondragleave={handleDragLeave}
						ondrop={handleDrop}
					>
						{#if uploadedFile}
							<div class="file-preview">
								<Icon name="file" size={24} />
								<div class="file-info">
									<div class="file-name">{uploadedFile.name}</div>
									<div class="file-size">{Math.round(uploadedFile.size / 1024)} KB</div>
								</div>
								<button
									type="button"
									class="remove-file"
									onclick={() => { uploadedFile = null; parseStatus = 'idle'; parsedContent = ''; }}
									aria-label="Remove file"
								>
									<Icon name="x" size={16} />
								</button>
							</div>
						{:else}
							<div class="upload-placeholder">
								<Icon name="upload" size={32} />
								<p>拖拽文件到此处或 <span class="upload-link">点击选择</span></p>
								<p class="upload-hint">支持 PDF、TXT、MD、CSV、JSON 等格式，上传后需 AI 解析为 Markdown</p>
							</div>
						{/if}
						<input
							class="file-input"
							type="file"
							name="file"
							accept=".pdf,.doc,.docx,.txt,.md,.rtf,.csv,.json"
							onchange={handleFileSelect}
						/>
					</div>
				</label>

				<!-- 解析按钮 & 进度 -->
				{#if uploadedFile && selectedSourceType}
					<div class="parse-row">
						{#if parseStatus === 'idle'}
							<button type="button" class="btn btn-secondary parse-btn" onclick={parseFile}>
								<Icon name="zap" size={14} /> AI 解析为 Markdown
							</button>
						{:else if parseStatus === 'parsing'}
							<div class="parse-progress">
								<span class="parse-spinner"></span> 正在解析...
							</div>
						{:else if parseStatus === 'done'}
							<div class="parse-done">✓ 解析完成，内容已填入预览</div>
						{:else if parseStatus === 'error'}
							<div class="parse-error">✗ {parseError}</div>
							<button type="button" class="btn btn-secondary parse-btn" onclick={parseFile}>
								重试
							</button>
						{/if}
					</div>
				{/if}

				<!-- Markdown 预览（解析结果） -->
				{#if parseStatus === 'done' || parseStatus === 'parsing'}
					<div class="parse-preview">
						<div class="preview-label">
							{parseStatus === 'parsing' ? 'AI 正在解析...' : 'Markdown 预览'}
						</div>
						<div class="output-preview parse-preview-body" onclick={(e) => { if ((e.target as HTMLElement).closest('a')) e.preventDefault(); }}>
							<Streamdown
								content={parsedContent}
								parseIncompleteMarkdown={parseStatus === 'parsing'}
								class="markdown-body"
								controls={streamdownControls}
							/>
						</div>
					</div>
				{/if}

				<!-- 直接粘贴内容（无文件时） -->
				{#if !uploadedFile}
					<label>
						<span class="label">或直接粘贴内容</span>
						<textarea
							class="input-field source-textarea"
							name="content"
							placeholder="直接粘贴简历、面试记录、JD 文本..."
						></textarea>
					</label>
				{/if}

				{#if form?.sourceError}
					<p class="inline-error">{form.sourceError}</p>
				{/if}

				<div class="form-actions">
					<button type="button" class="btn btn-secondary" onclick={() => (showSourceModal = false)}>
						Cancel
					</button>
					<button type="submit" class="btn btn-primary">添加源</button>
				</div>
			</form>
		</div>
	</div>
{/if}

{#if showSourceDetailModal && activeSource}
	<div
		class="modal-overlay"
		role="presentation"
		tabindex="-1"
		onclick={() => (showSourceDetailModal = false)}
		onkeydown={(event) => {
			if (event.key === 'Escape') showSourceDetailModal = false;
		}}
	>
		<div
			class="modal-box source-detail-modal"
			role="dialog"
			aria-modal="true"
			aria-labelledby="source-detail-title"
			tabindex="-1"
			onclick={(event) => event.stopPropagation()}
			onkeydown={(event) => event.stopPropagation()}
		>
			<button
				type="button"
				class="modal-close"
				aria-label="Close modal"
				onclick={() => (showSourceDetailModal = false)}
			>
				<Icon name="x" size={16} />
			</button>

			<h2 id="source-detail-title">{activeSource.title}</h2>
			<p>
				{activeSource.subtitle} · {activeSource.meta}
			</p>

			{#if activeSource.editable}
				<form
					method="POST"
					action="?/updateSource"
					use:enhance={() => {
						return async ({ update, result }) => {
							await update();
							if (result.type === 'failure') return;
							showSourceDetailModal = false;
							await invalidateAll();
						};
					}}
					class="source-form-grid"
				>
					<input type="hidden" name="sourceId" value={activeSource.id} />

					<label>
						<span class="label">Title</span>
						<input
							class="input-field"
							name="title"
							value={form?.sourceEditValues?.sourceId === activeSource.id
								? form.sourceEditValues?.title
								: activeSource.title}
							required
						/>
					</label>

					<label>
						<span class="label">Meta Label</span>
						<input
							class="input-field"
							name="meta"
							value={form?.sourceEditValues?.sourceId === activeSource.id
								? form.sourceEditValues?.meta
								: activeSource.meta}
							required
						/>
					</label>

					<label>
						<span class="label">Content</span>
						<textarea class="input-field source-textarea" name="content" required
							>{form?.sourceEditValues?.sourceId === activeSource.id
								? form.sourceEditValues?.content
								: activeSource.content}</textarea
						>
					</label>

					{#if isMarkdownLike(activeSource.title, getSourceDetailContent(activeSource))}
						<div class="source-preview">
							<div class="preview-label">Markdown preview</div>
							<div class="output-preview">
								<Streamdown
									content={getSourceDetailContent(activeSource)}
									class="markdown-body"
									static
									controls={streamdownControls}
								/>
							</div>
						</div>
					{/if}

					{#if form?.sourceEditError}
						<p class="inline-error">{form.sourceEditError}</p>
					{/if}

					<div class="form-actions">
						<button type="button" class="btn btn-secondary" onclick={() => (showSourceDetailModal = false)}>
							Cancel
						</button>
						<button type="submit" class="btn btn-primary">Save source</button>
					</div>
				</form>
			{:else}
				<div class="readonly-source">
					<div class="readonly-note">
						This source is currently read-only. File-based content like PDF resumes can be viewed
						here, but not edited inline.
					</div>
					{#if isMarkdownLike(activeSource.title, activeSource.content)}
						<div class="output-preview">
							<Streamdown
								content={activeSource.content}
								class="markdown-body"
								static
								controls={streamdownControls}
							/>
						</div>
					{:else}
						<pre>{activeSource.content}</pre>
					{/if}
				</div>
			{/if}
		</div>
	</div>
{/if}

{#if showToolModal && activeTool}
	<div
		class="modal-overlay"
		role="presentation"
		tabindex="-1"
		onclick={() => (showToolModal = false)}
		onkeydown={(event) => {
			if (event.key === 'Escape') showToolModal = false;
		}}
	>
		<div
			class="modal-box tool-modal"
			role="dialog"
			aria-modal="true"
			aria-labelledby="tool-modal-title"
			tabindex="-1"
			onclick={(event) => event.stopPropagation()}
			onkeydown={(event) => event.stopPropagation()}
		>
			<button
				type="button"
				class="modal-close"
				aria-label="Close modal"
				onclick={() => (showToolModal = false)}
			>
				<Icon name="x" size={16} />
			</button>

			<h2 id="tool-modal-title">{activeTool.title}</h2>
			<p>{activeTool.description} Configure a few parameters before launching the mock AI run.</p>

			<form
				class="tool-config-grid"
				onsubmit={(event) => {
					event.preventDefault();
					runMockAnalysis();
				}}
			>
				<label>
					<span class="label">Focus</span>
					<input
						class="input-field"
						bind:value={analysisConfig.focus}
						placeholder="e.g. product judgment, delivery risk, executive summary"
					/>
				</label>


				{#if activeTool?.key === 'interview-assessment'}
					<label>
						<span class="label">Resume Source</span>
						<select class="input-field" bind:value={analysisConfig.resumeSourceId}>
							<option value="">Select resume...</option>
							{#each data.notebook.sources.filter((s) => s.sourceType === 'resume') as source (source.id)}
								<option value={source.id}>{source.title}</option>
							{/each}
						</select>
					</label>

					<label>
						<span class="label">Interview Conversation</span>
						<select class="input-field" bind:value={analysisConfig.conversationSourceId}>
							<option value="">Select conversation...</option>
							{#each data.notebook.sources.filter((s) => s.sourceType === 'conversation') as source (source.id)}
								<option value={source.id}>{source.title}</option>
							{/each}
						</select>
					</label>

					<label>
						<span class="label">Job Description</span>
						<select class="input-field" bind:value={analysisConfig.jdSourceId}>
							<option value="">Select JD...</option>
							{#each data.notebook.sources.filter((s) => s.sourceType === 'job-description') as source (source.id)}
								<option value={source.id}>{source.title}</option>
							{/each}
						</select>
					</label>

					<label>
						<span class="label">评估模式</span>
						<select class="input-field" bind:value={analysisConfig.evaluationMode}>
							<option value="basic">基础模式</option>
							<option value="advanced" disabled>进阶模式</option>
						</select>
					</label>

					<label class="checkbox-row">
						<input type="checkbox" bind:checked={analysisConfig.resumeCheckpoint} />
						<span>断点续跑（复用上次未完成的进度）</span>
					</label>
				{/if}

				<label class="checkbox-row">
					<input type="checkbox" bind:checked={analysisConfig.includeCitations} />
					<span>Include evidence anchors from the selected source</span>
				</label>

				<div class="form-actions">
					<button type="button" class="btn btn-secondary" onclick={() => (showToolModal = false)}>
						Cancel
					</button>
					<button type="submit" class="btn btn-primary">Run mock analysis</button>
				</div>
			</form>
		</div>
	</div>
{/if}

{#if showSettingsModal}
	<div
		class="modal-overlay"
		role="presentation"
		tabindex="-1"
		onclick={() => (showSettingsModal = false)}
		onkeydown={(event) => {
			if (event.key === 'Escape') showSettingsModal = false;
		}}
	>
		<div
			class="modal-box tool-modal"
			role="dialog"
			aria-modal="true"
			aria-labelledby="settings-modal-title"
			tabindex="-1"
			onclick={(event) => event.stopPropagation()}
			onkeydown={(event) => event.stopPropagation()}
		>
			<button
				type="button"
				class="modal-close"
				aria-label="Close modal"
				onclick={() => (showSettingsModal = false)}
			>
				<Icon name="x" size={16} />
			</button>

			<h2 id="settings-modal-title">Notebook settings</h2>
			<p>Update Workspace metadata, then save it back into HuShi.</p>

			<form
				method="POST"
				action="?/updateNotebook"
				use:enhance={() => {
					return async ({ update, result }) => {
						await update();
						if (result.type === 'failure') return;
						showSettingsModal = false;
					};
				}}
				class="tool-config-grid"
			>
				<label>
					<span class="label">Title</span>
					<input
						class="input-field"
						name="title"
						value={form?.notebookValues?.title ?? data.notebook.title}
						required
					/>
				</label>

				<label>
					<span class="label">Subject</span>
					<input
						class="input-field"
						name="subject"
						value={form?.notebookValues?.subject ?? data.notebook.subject}
						required
					/>
				</label>

				<label>
					<span class="label">Tag</span>
					<input
						class="input-field"
						name="tag"
						value={form?.notebookValues?.tag ?? data.notebook.tag}
						required
					/>
				</label>

				<label>
					<span class="label">Location</span>
					<input
						class="input-field"
						name="location"
						value={form?.notebookValues?.location ?? data.notebook.location}
						required
					/>
				</label>

				<label>
					<span class="label">Contact</span>
					<input
						class="input-field"
						name="contact"
						value={form?.notebookValues?.contact ?? data.notebook.contact}
						required
					/>
				</label>

				<label>
					<span class="label">Link</span>
					<input
						class="input-field"
						name="link"
						value={form?.notebookValues?.link ?? data.notebook.link}
						required
					/>
				</label>

				<label>
					<span class="label">Summary</span>
					<textarea class="input-field source-textarea" name="summary" required
						>{form?.notebookValues?.summary ?? data.notebook.summary}</textarea
					>
				</label>

				{#if form?.notebookError}
					<p class="inline-error">{form.notebookError}</p>
				{/if}

				<div class="form-actions settings-actions">
					<button
						type="submit"
						class="btn btn-danger danger-form"
						formaction="?/deleteNotebook"
						formnovalidate
					>
						Delete notebook
					</button>
					<div class="action-pair">
						<button type="button" class="btn btn-secondary" onclick={() => (showSettingsModal = false)}>
							Cancel
						</button>
						<button type="submit" class="btn btn-primary">Save changes</button>
					</div>
				</div>
			</form>
		</div>
	</div>
{/if}

{#if showOutputModal && activeOutput}
	<div
		class="modal-overlay"
		role="presentation"
		tabindex="-1"
		onclick={() => (showOutputModal = false)}
		onkeydown={(event) => {
			if (event.key === 'Escape') showOutputModal = false;
		}}
	>
		<div
			class="modal-box output-modal"
			role="dialog"
			aria-modal="true"
			aria-labelledby="output-modal-title"
			tabindex="-1"
			onclick={(event) => event.stopPropagation()}
			onkeydown={(event) => event.stopPropagation()}
		>
			<button
				type="button"
				class="modal-close"
				aria-label="Close modal"
				onclick={() => (showOutputModal = false)}
			>
				<Icon name="x" size={16} />
			</button>

			<h2 id="output-modal-title">Output details</h2>
			<p>Rename this result, review the rendered markdown, or download it as a file.</p>

			<form
				method="POST"
				action="?/updateOutput"
				use:enhance={() => {
					return async ({ update, result }) => {
						await update();
						if (result.type === 'failure') return;
						await invalidateAll();
						showOutputModal = false;
					};
				}}
				class="tool-config-grid"
			>
				<input type="hidden" name="outputId" value={activeOutput.id} />
				<label>
					<span class="label">Output title</span>
					<input
						class="input-field"
						name="title"
						value={form?.outputValues?.outputId === activeOutput.id
							? form.outputValues?.title
							: activeOutput.title}
						required
					/>
				</label>

				<div class="output-preview" bind:this={outputPreviewElement}>
					<Streamdown
						content={activeOutput.content}
						class="markdown-body"
						static
						controls={streamdownControls}
					/>
				</div>

				{#if form?.outputError}
					<p class="inline-error">{form.outputError}</p>
				{/if}

				<div class="form-actions output-actions">
					<div class="action-pair">
						<button type="button" class="btn btn-secondary" onclick={() => copyOutput(activeOutput)}>
							Copy content
						</button>
						<button type="button" class="btn btn-secondary" onclick={() => downloadOutputAsText(activeOutput)}>
							Export txt
						</button>
						<button type="button" class="btn btn-secondary" onclick={() => printOutputAsPdf(activeOutput)}>
							Export PDF
						</button>
						<button type="button" class="btn btn-secondary" onclick={() => downloadOutput(activeOutput)}>
							Download markdown
						</button>
					</div>
					<div class="action-pair">
						<button type="button" class="btn btn-secondary" onclick={() => (showOutputModal = false)}>
							Close
						</button>
						<button type="submit" class="btn btn-primary">Rename output</button>
					</div>
				</div>
			</form>
		</div>
	</div>
{/if}

{#if globalCrashError}
	<div class="modal-overlay danger-overlay" role="presentation" tabindex="-1">
		<div class="modal-box danger-modal" role="dialog" aria-modal="true">
			<div class="danger-icon-wrapper">
				<Icon name="alert-triangle" size={48} />
			</div>
			<h2>{globalCrashError.title}</h2>
			<div class="danger-message-box">
				<p>{globalCrashError.message}</p>
			</div>
			
			<div class="danger-actions">
				<button class="primary-button danger-button" onclick={() => window.location.reload()}>
					重新加载页面
				</button>
				<button class="text-button" onclick={() => (globalCrashError = null)}>
					关闭提示
				</button>
			</div>
		</div>
	</div>
{/if}

<style>
	.beta-banner {
		background-color: rgba(234, 179, 8, 0.15);
		color: #facc15;
		text-align: center;
		padding: 8px 16px;
		font-size: 0.85rem;
		font-weight: 500;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 8px;
		border-bottom: 1px solid rgba(234, 179, 8, 0.3);
	}

	.detail-shell {
		min-height: calc(100vh - 48px);
	}

	.detail-topbar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 16px;
		padding: 14px 18px;
		border-bottom: 1px solid rgba(255, 255, 255, 0.05);
	}

	.detail-title-wrap {
		display: flex;
		align-items: flex-start;
		gap: 14px;
	}

	.detail-title-wrap h1 {
		margin: 0;
		font-size: 1.4rem;
		letter-spacing: -0.03em;
	}

	.back-link {
		width: 30px;
		height: 30px;
		display: inline-grid;
		place-items: center;
		border-radius: 999px;
		color: #7bc8ff;
		text-decoration: none;
		transition:
			background 0.18s ease,
			transform 0.18s ease;
	}

	.back-link:hover {
		background: rgba(91, 182, 255, 0.08);
		transform: translateX(-2px);
	}

	.tab-row {
		display: flex;
		gap: 16px;
		margin-top: 10px;
		color: var(--color-text-secondary);
		font-size: 0.92rem;
	}

	.active-tab {
		color: #6bc5ff;
		position: relative;
		padding-bottom: 8px;
	}

	.active-tab::after {
		content: '';
		position: absolute;
		left: 0;
		right: 0;
		bottom: 0;
		height: 2px;
		border-radius: 999px;
		background: #6bc5ff;
	}

	.detail-actions {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.ghost-icon,
	.share-button {
		border: 0;
		border-radius: 8px;
		background: transparent;
		color: var(--color-text-secondary);
	}

	.ghost-icon {
		width: 34px;
		height: 34px;
		display: inline-grid;
		place-items: center;
		transition:
			background 0.18s ease,
			color 0.18s ease,
			transform 0.18s ease;
	}

	.ghost-icon:hover {
		background: rgba(255, 255, 255, 0.05);
		color: var(--color-text-primary);
		transform: translateY(-1px);
	}

	.logout-icon:hover {
		color: #ffb6bd;
		border-color: rgba(255, 108, 122, 0.3);
		background: rgba(255, 108, 122, 0.08);
	}

	.share-button {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		padding: 10px 14px;
		background: rgba(91, 182, 255, 0.14);
		color: #7ecfff;
		transition:
			transform 0.18s ease,
			background 0.18s ease,
			box-shadow 0.18s ease;
	}

	.share-button:hover {
		transform: translateY(-2px);
		background: rgba(91, 182, 255, 0.2);
		box-shadow: 0 12px 26px rgba(91, 182, 255, 0.16);
	}

	.detail-grid {
		display: grid;
		grid-template-columns: 330px minmax(0, 1fr) 380px;
		gap: 0;
		min-height: calc(100vh - 106px);
	}

	.sources-panel,
	.studio-panel {
		padding: 18px 16px;
		background: rgba(255, 255, 255, 0.02);
	}

	.document-panel {
		padding: 18px;
		border-left: 1px solid rgba(255, 255, 255, 0.04);
		border-right: 1px solid rgba(255, 255, 255, 0.04);
		display: grid;
		gap: 18px;
		align-content: start;
	}

	.panel-head h2,
	.conversation-head h3 {
		margin: 0;
		font-size: 1.2rem;
	}

	.add-source-button {
		width: 100%;
		margin-top: 16px;
		padding: 12px 14px;
		border-radius: 10px;
		border: 1px solid rgba(255, 255, 255, 0.06);
		background: rgba(255, 255, 255, 0.04);
		color: var(--color-text-primary);
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: 10px;
		transition:
			transform 0.18s ease,
			border-color 0.18s ease,
			background 0.18s ease,
			box-shadow 0.18s ease;
	}

	.add-source-button:hover {
		transform: translateY(-2px);
		border-color: rgba(140, 217, 255, 0.16);
		background: rgba(255, 255, 255, 0.055);
		box-shadow: 0 14px 26px rgba(0, 0, 0, 0.14);
	}

	.source-list,
	.output-list {
		display: grid;
		gap: 10px;
		margin-top: 18px;
	}

	.source-group {
		margin-bottom: 24px;
	}

	.source-group-title {
		font-size: 0.9rem;
		font-weight: 600;
		color: var(--color-text-secondary);
		margin-bottom: 8px;
		display: flex;
		align-items: center;
		gap: 8px;
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.source-count {
		font-size: 0.8rem;
		font-weight: 400;
		color: var(--color-text-muted);
		background: rgba(255, 255, 255, 0.05);
		padding: 2px 6px;
		border-radius: 10px;
	}

	.source-row,
	.output-row {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto;
		gap: 8px;
		align-items: start;
	}

	.source-form {
		display: block;
	}

	.source-item {
		width: 100%;
		border: 1px solid rgba(255, 255, 255, 0.06);
		background: rgba(255, 255, 255, 0.03);
		border-radius: 10px;
		padding: 12px;
		display: flex;
		align-items: flex-start;
		gap: 10px;
		color: inherit;
		text-align: left;
		transition:
			transform 0.18s ease,
			border-color 0.18s ease,
			background 0.18s ease,
			box-shadow 0.18s ease;
	}

	.source-item:hover {
		transform: translateX(4px);
		border-color: rgba(140, 217, 255, 0.14);
		background: rgba(255, 255, 255, 0.045);
		box-shadow: 0 10px 22px rgba(0, 0, 0, 0.16);
	}

	.source-row.selected .source-item,
	.source-item.selected {
		border-color: rgba(102, 198, 255, 0.5);
		background: rgba(76, 182, 255, 0.08);
	}

	.source-item div {
		display: grid;
		gap: 4px;
	}

	.source-item span,
	.source-item small,
	.identity-line,
	.output-card p,
	.inline-error,
	.source-modal > p {
		color: var(--color-text-secondary);
	}

	.storage-card,
	.document-card,
	.conversation-card,
	.output-card,
	.engine-card {
		border: 1px solid rgba(255, 255, 255, 0.06);
		background: rgba(255, 255, 255, 0.03);
		border-radius: 12px;
	}

	.storage-card {
		padding: 14px;
		margin-top: 18px;
	}

	.storage-head,
	.output-head,
	.conversation-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
	}

	.storage-track,
	.playback-bar {
		height: 8px;
		border-radius: 999px;
		background: rgba(255, 255, 255, 0.06);
		overflow: hidden;
		margin-top: 10px;
	}

	.storage-track span,
	.playback-bar span {
		display: block;
		height: 100%;
		background: linear-gradient(90deg, #6acaff, #5baeff);
	}

	.document-card,
	.conversation-card {
		padding: 22px;
	}

	.analysis-console {
		min-height: 520px;
	}

	.pipeline-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 10px 0 6px;
		border-bottom: 1px solid rgba(255,255,255,0.06);
		margin-bottom: 10px;
	}

	.pipeline-title {
		font-size: 0.82rem;
		font-weight: 600;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		color: var(--color-text-secondary);
	}

	.pipeline-header-right {
		display: flex;
		align-items: center;
		gap: 10px;
	}

	.token-badge {
		font-size: 0.75rem;
		font-family: monospace;
		color: var(--color-text-muted);
		background: rgba(255,255,255,0.05);
		border: 1px solid rgba(255,255,255,0.08);
		padding: 2px 8px;
		border-radius: 6px;
	}

	.debug-toggle {
		font-size: 0.72rem;
		font-weight: 700;
		letter-spacing: 0.06em;
		padding: 3px 10px;
		border-radius: 6px;
		border: 1px solid rgba(255,255,255,0.1);
		background: transparent;
		color: var(--color-text-muted);
		cursor: pointer;
		transition: background 0.15s, color 0.15s, border-color 0.15s;
	}

	.debug-toggle.active {
		border-color: rgba(106, 202, 255, 0.5);
		background: rgba(106, 202, 255, 0.1);
		color: #6acaff;
	}

	.debug-toggle:hover {
		background: rgba(255,255,255,0.06);
		color: var(--color-text-primary);
	}

	.pipeline-progress {
		display: flex;
		flex-direction: column;
		gap: 4px;
		padding: 0 0 10px;
		border-bottom: 1px solid rgba(255,255,255,0.06);
		margin-bottom: 10px;
	}

	.pipeline-stage {
		display: flex;
		flex-direction: column;
		gap: 1px;
		padding: 7px 10px;
		border-radius: 7px;
		border: 1px solid rgba(255,255,255,0.06);
		background: rgba(255,255,255,0.02);
		font-size: 0.82rem;
		transition: border-color 0.2s, background 0.2s;
	}

	.pipeline-stage[data-status='running'] {
		border-color: rgba(59, 130, 246, 0.4);
		background: rgba(59, 130, 246, 0.06);
	}

	.pipeline-stage[data-status='done'] {
		border-color: rgba(34, 197, 94, 0.35);
		background: rgba(34, 197, 94, 0.05);
	}

	.pipeline-stage[data-status='failed'] {
		border-color: rgba(239, 68, 68, 0.4);
		background: rgba(239, 68, 68, 0.06);
	}

	.stage-row {
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.stage-icon {
		font-size: 0.9rem;
		font-weight: 700;
		width: 14px;
		flex-shrink: 0;
		line-height: 1;
	}

	.pipeline-stage[data-status='running'] .stage-icon {
		color: #3b82f6;
		animation: pulse 1.2s ease-in-out infinite;
	}

	.pipeline-stage[data-status='done'] .stage-icon {
		color: #22c55e;
	}

	.pipeline-stage[data-status='failed'] .stage-icon {
		color: #ef4444;
	}

	@keyframes pulse {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.35; }
	}

	.stage-label {
		font-weight: 600;
		color: var(--color-text-primary);
		flex: 1;
	}

	.stage-timer {
		font-size: 0.72rem;
		font-family: monospace;
		color: var(--color-text-muted);
		flex-shrink: 0;
	}

	.pipeline-stage[data-status='running'] .stage-timer {
		color: #60a5fa;
	}

	.pipeline-stage[data-status='done'] .stage-timer {
		color: #4ade80;
	}

	.stage-message {
		color: var(--color-text-muted);
		font-size: 0.74rem;
		padding-left: 20px;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	/* LLM calls inside each stage */
	.stage-llm-calls {
		display: flex;
		flex-direction: column;
		gap: 3px;
		margin-top: 6px;
		padding-top: 6px;
		border-top: 1px solid rgba(255,255,255,0.05);
	}

	.llm-call-entry {
		border: 1px solid rgba(255,255,255,0.07);
		border-radius: 8px;
		background: rgba(255,255,255,0.02);
		overflow: hidden;
	}

	.llm-call-header {
		width: 100%;
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 7px 10px;
		background: transparent;
		border: 0;
		color: var(--color-text-secondary);
		cursor: pointer;
		text-align: left;
		font-size: 0.8rem;
		transition: background 0.15s;
	}

	.llm-call-header:hover {
		background: rgba(255,255,255,0.04);
	}

	.llm-call-step {
		font-weight: 600;
		color: var(--color-text-muted);
		font-size: 0.75rem;
		font-family: monospace;
	}

	.llm-call-tag {
		font-size: 0.68rem;
		font-family: monospace;
		background: rgba(99, 102, 241, 0.15);
		color: #a5b4fc;
		border-radius: 3px;
		padding: 1px 5px;
	}

	.llm-call-usage {
		font-family: monospace;
		font-size: 0.74rem;
		color: var(--color-text-muted);
	}

	.llm-call-chevron {
		margin-left: auto;
		font-size: 0.7rem;
		color: var(--color-text-muted);
	}

	.llm-call-body {
		padding: 8px 10px 10px;
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.llm-section-title {
		font-size: 0.75rem;
		font-weight: 700;
		letter-spacing: 0.05em;
		text-transform: uppercase;
		color: var(--color-text-muted);
		cursor: pointer;
		padding: 3px 0;
		list-style: none;
	}

	.llm-section-title::marker { display: none; }
	.llm-section-title::-webkit-details-marker { display: none; }

	.llm-pre {
		margin: 4px 0 0;
		padding: 10px 12px;
		border-radius: 6px;
		background: rgba(0,0,0,0.3);
		border: 1px solid rgba(255,255,255,0.05);
		font-size: 0.74rem;
		font-family: 'SFMono-Regular', Menlo, monospace;
		color: var(--color-text-secondary);
		white-space: pre-wrap;
		word-break: break-word;
		max-height: 320px;
		overflow-y: auto;
	}

	.llm-thinking {
		color: #a5b4fc;
		background: rgba(99, 102, 241, 0.08);
		border-color: rgba(99, 102, 241, 0.15);
	}

	.console-actions {
		margin-top: 12px;
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		color: var(--color-text-muted);
		font-size: 0.85rem;
	}

	.console-meta {
		display: flex;
		align-items: center;
		gap: 12px;
	}

	.document-card h2 {
		margin: 0;
		font-size: 2rem;
	}

	.document-card section + section {
		margin-top: 24px;
	}

	.document-card h3,
	.saved-head {
		margin: 0 0 10px;
	}

	.desktop-markdown {
		display: grid;
		gap: 8px;
	}

	.identity-line {
		display: flex;
		gap: 10px;
		flex-wrap: wrap;
		margin: 10px 0 0;
	}

	.analysis-stream {
		margin-top: 18px;
		position: relative;
		border-radius: 12px;
		border: 1px solid rgba(255, 255, 255, 0.06);
		background: rgba(7, 10, 15, 0.55);
		padding: 16px;
		min-height: 340px;
	}

	.analysis-stream.collapsed {
		min-height: 360px;
		max-height: 520px;
		overflow: auto;
	}

	.analysis-stream.collapsed::after {
		content: '';
		position: absolute;
		left: 0;
		right: 0;
		bottom: 0;
		height: 4.2rem;
		pointer-events: none;
		background: linear-gradient(180deg, rgba(7, 10, 15, 0) 0%, rgba(7, 10, 15, 0.95) 85%);
	}

	.analysis-stream.expanded {
		min-height: 360px;
		max-height: none;
		overflow: visible;
	}

	.console-toggle-row {
		margin-top: 12px;
		display: flex;
		justify-content: center;
	}

	.console-toggle {
		border: 1px solid rgba(255, 255, 255, 0.1);
		background: rgba(255, 255, 255, 0.06);
		color: var(--color-text-primary);
		padding: 8px 14px;
		border-radius: 999px;
		display: inline-flex;
		align-items: center;
		gap: 8px;
		font-size: 0.9rem;
		cursor: pointer;
		transition: transform 0.18s ease, background 0.18s ease, border-color 0.18s ease;
	}

	.console-toggle:hover {
		transform: translateY(-1px);
		background: rgba(255, 255, 255, 0.12);
	}

	.analysis-stream.running {
		border-color: rgba(106, 202, 255, 0.18);
		box-shadow: inset 0 0 0 1px rgba(106, 202, 255, 0.05);
	}

	.message-bubble {
		padding: 14px;
		border-radius: 12px;
		transition:
			transform 0.18s ease,
			border-color 0.18s ease;
	}

	.message-bubble:hover {
		transform: translateY(-2px);
	}

	.message-bubble.user {
		background: rgba(91, 182, 255, 0.08);
		border: 1px solid rgba(91, 182, 255, 0.2);
	}

	.message-bubble.assistant {
		background: rgba(255, 255, 255, 0.03);
		border: 1px solid rgba(255, 255, 255, 0.06);
	}

	.document-card p,
	.output-card p {
		margin: 0;
		line-height: 1.65;
	}

	.composer {
		margin-top: 16px;
		display: grid;
		grid-template-columns: 44px minmax(0, 1fr) 52px;
		align-items: center;
		border: 1px solid rgba(255, 255, 255, 0.07);
		background: rgba(255, 255, 255, 0.03);
		border-radius: 14px;
		padding: 8px;
		gap: 8px;
	}

	.composer-disabled {
		opacity: 0.55;
	}

	.composer input {
		background: transparent;
		border: 0;
		color: var(--color-text-primary);
		outline: none;
		font-size: 0.95rem;
	}

	.composer input:disabled {
		cursor: not-allowed;
	}

	.composer-icon,
	.composer-send {
		border: 0;
		border-radius: 10px;
		background: transparent;
		color: var(--color-text-secondary);
		height: 38px;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.composer-send {
		background: #6acaff;
		color: #06273c;
		transition:
			transform 0.18s ease,
			box-shadow 0.18s ease,
			background 0.18s ease,
			filter 0.18s ease;
		justify-self: center;
		width: 38px;
	}

	.composer-send:disabled {
		cursor: not-allowed;
		filter: grayscale(0.15);
		box-shadow: none;
	}

	.composer-send:hover {
		transform: translateY(-1px);
		box-shadow: 0 12px 22px rgba(106, 202, 255, 0.22);
		filter: brightness(1.02);
	}

	.tool-grid {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 10px;
		margin-top: 16px;
	}

	.tool-card {
		padding: 14px;
		border-radius: 12px;
		border: 1px solid rgba(255, 255, 255, 0.06);
		background: rgba(255, 255, 255, 0.03);
		color: inherit;
		display: grid;
		gap: 8px;
		text-align: left;
		transition:
			transform 0.18s ease,
			border-color 0.18s ease,
			background 0.18s ease,
			box-shadow 0.18s ease;
	}

	.tool-card:hover {
		transform: translateY(-3px);
		border-color: rgba(140, 217, 255, 0.16);
		background: rgba(255, 255, 255, 0.045);
		box-shadow: 0 14px 28px rgba(0, 0, 0, 0.18);
	}

	.output-card {
		width: 100%;
		border: 1px solid rgba(255, 255, 255, 0.06);
		background: rgba(255, 255, 255, 0.03);
		color: inherit;
		text-align: left;
		padding: 14px;
		display: grid;
		gap: 10px;
		transition:
			transform 0.18s ease,
			border-color 0.18s ease,
			box-shadow 0.18s ease;
	}

	.output-card:hover {
		transform: translateY(-3px);
		border-color: rgba(140, 217, 255, 0.14);
		box-shadow: 0 14px 26px rgba(0, 0, 0, 0.18);
	}

	.output-row.selected .output-card {
		border-color: rgba(102, 198, 255, 0.5);
		background: rgba(76, 182, 255, 0.08);
	}

	.output-head {
		color: var(--color-text-muted);
		font-size: 0.8rem;
	}

	.playback {
		display: grid;
		gap: 6px;
	}

	.playback small {
		color: var(--color-text-muted);
		text-align: right;
	}

	.engine-card {
		margin-top: 16px;
		padding: 14px;
		display: flex;
		align-items: center;
		gap: 12px;
	}

	.engine-icon {
		width: 38px;
		height: 38px;
		border-radius: 10px;
		display: grid;
		place-items: center;
		background: rgba(91, 182, 255, 0.12);
		color: #7ecfff;
	}

	.source-modal {
		max-width: 720px;
	}

	.source-template-grid {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 10px;
		margin: 18px 0;
	}

	.template-card {
		border: 1px solid rgba(255, 255, 255, 0.06);
		background: rgba(255, 255, 255, 0.03);
		border-radius: 12px;
		padding: 14px;
		display: grid;
		gap: 8px;
		transition:
			transform 0.18s ease,
			border-color 0.18s ease,
			background 0.18s ease;
	}

	.template-card:hover {
		transform: translateY(-2px);
		border-color: rgba(140, 217, 255, 0.16);
		background: rgba(255, 255, 255, 0.045);
	}

	.template-card.selected {
		border-color: rgba(59, 130, 246, 0.35);
		background: rgba(59, 130, 246, 0.14);
	}

	.selected-source-type {
		display: grid;
		gap: 4px;
		padding: 12px 14px;
		background: rgba(255, 255, 255, 0.04);
		border-radius: 10px;
		border: 1px solid rgba(255, 255, 255, 0.06);
	}

	.source-form-grid {
		display: grid;
		gap: 16px;
	}

	.source-detail-modal {
		max-width: 760px;
	}

	.output-modal {
		max-width: 860px;
	}

	.source-form-grid label {
		display: grid;
		gap: 8px;
	}

	.source-textarea {
		min-height: 160px;
		resize: vertical;
	}

	.upload-field {
		padding: 12px;
	}

	.upload-zone {
		border: 2px dashed var(--color-border);
		border-radius: 12px;
		padding: 24px;
		text-align: center;
		transition: all 0.2s ease;
		position: relative;
		background: rgba(255, 255, 255, 0.02);
		cursor: pointer;
	}

	.upload-zone:hover,
	.upload-zone.drag-over {
		border-color: var(--color-accent);
		background: rgba(59, 130, 246, 0.05);
	}

	.upload-placeholder {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 12px;
		color: var(--color-text-muted);
	}

	.upload-placeholder p {
		margin: 0;
		font-size: 0.9rem;
	}

	.upload-link {
		color: var(--color-accent);
		text-decoration: underline;
		cursor: pointer;
	}

	.upload-hint {
		font-size: 0.8rem !important;
		opacity: 0.7;
	}

	.file-input {
		position: absolute;
		inset: 0;
		opacity: 0;
		cursor: pointer;
	}

	.file-preview {
		display: flex;
		align-items: center;
		gap: 12px;
		padding: 16px;
		background: rgba(255, 255, 255, 0.04);
		border-radius: 8px;
		border: 1px solid var(--color-border);
	}

	.file-info {
		flex: 1;
		text-align: left;
	}

	.file-name {
		font-weight: 500;
		color: var(--color-text);
		margin-bottom: 4px;
	}

	.file-size {
		font-size: 0.8rem;
		color: var(--color-text-muted);
	}

	.remove-file {
		background: none;
		border: none;
		color: var(--color-text-muted);
		cursor: pointer;
		padding: 4px;
		border-radius: 4px;
		transition: all 0.2s ease;
	}

	.remove-file:hover {
		color: var(--color-error);
		background: rgba(239, 68, 68, 0.1);
	}

	.readonly-source {
		display: grid;
		gap: 14px;
	}

	.source-preview {
		display: grid;
		gap: 8px;
	}

	.preview-label {
		font-size: 0.85rem;
		color: var(--color-text-muted);
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}

	.readonly-note {
		padding: 12px 14px;
		border-radius: 10px;
		background: rgba(255, 255, 255, 0.04);
		color: var(--color-text-secondary);
	}

	.readonly-source pre {
		margin: 0;
		padding: 16px;
		border-radius: 12px;
		border: 1px solid rgba(255, 255, 255, 0.06);
		background: rgba(255, 255, 255, 0.03);
		color: var(--color-text-primary);
		white-space: pre-wrap;
		word-break: break-word;
		max-height: 56vh;
		overflow: auto;
	}

	.output-preview {
		max-height: 52vh;
		overflow: auto;
		padding: 16px;
		border-radius: 12px;
		border: 1px solid rgba(255, 255, 255, 0.06);
		background: rgba(255, 255, 255, 0.03);
	}

	.output-actions {
		justify-content: space-between;
		align-items: center;
	}

	.markdown-body :global(*) {
		box-sizing: border-box;
	}

	.markdown-body :global(h1),
	.markdown-body :global(h2),
	.markdown-body :global(h3),
	.markdown-body :global(h4) {
		margin: 0 0 12px;
		line-height: 1.2;
		letter-spacing: -0.03em;
	}

	.markdown-body :global(h1) {
		font-size: 1.9rem;
	}

	.markdown-body :global(h2) {
		font-size: 1.45rem;
		margin-top: 18px;
	}

	.markdown-body :global(h3) {
		font-size: 1.12rem;
		margin-top: 16px;
	}

	.markdown-body :global(p),
	.markdown-body :global(li) {
		color: var(--color-text-secondary);
		line-height: 1.72;
	}

	.markdown-body :global(ul),
	.markdown-body :global(ol) {
		margin: 10px 0 14px 1.2rem;
		padding: 0;
	}

	.markdown-body :global(pre) {
		margin: 14px 0;
		padding: 14px;
		border-radius: 12px;
		overflow: auto;
		background: rgba(4, 10, 16, 0.88);
		border: 1px solid rgba(255, 255, 255, 0.06);
	}

	.markdown-body :global(code) {
		font-family: 'IBM Plex Mono', 'SFMono-Regular', monospace;
	}

	.markdown-body :global(blockquote) {
		margin: 14px 0;
		padding: 12px 14px;
		border-left: 3px solid rgba(106, 202, 255, 0.55);
		background: rgba(255, 255, 255, 0.03);
		color: var(--color-text-secondary);
	}

	.markdown-body :global(table) {
		width: 100%;
		border-collapse: collapse;
		margin: 12px 0;
	}

	.markdown-body :global(th),
	.markdown-body :global(td) {
		padding: 10px 12px;
		border: 1px solid rgba(255, 255, 255, 0.08);
		text-align: left;
	}

	.tool-modal {
		max-width: 560px;
	}

	/* ===== 工具调用样式 ===== */
	.tool-calls {
		display: grid;
		gap: 12px;
		margin-top: 8px;
		margin-bottom: 12px;
	}

	.tool-call {
		background: rgba(255, 255, 255, 0.02);
		border: 1px solid rgba(255, 255, 255, 0.05);
		border-radius: 6px;
		padding: 12px;
	}

	.tool-call.calling {
		border-color: rgba(91, 182, 255, 0.3);
		background: rgba(91, 182, 255, 0.03);
	}

	.tool-call.success {
		border-color: rgba(74, 222, 128, 0.3);
		background: rgba(74, 222, 128, 0.03);
	}

	/* 防崩溃严重弹窗 styles */
	.danger-overlay {
		background: rgba(15, 5, 5, 0.85);
		backdrop-filter: blur(8px);
		z-index: 9999;
	}
	.danger-modal {
		max-width: 600px;
		border: 1px solid rgba(239, 68, 68, 0.5);
		box-shadow: 0 20px 50px rgba(239, 68, 68, 0.2);
		background: linear-gradient(to bottom, #1f0808, #111);
		text-align: center;
		align-items: center;
		animation: shake 0.5s cubic-bezier(.36,.07,.19,.97) both;
	}
	.danger-icon-wrapper {
		color: #ef4444;
		margin-bottom: 8px;
		background: rgba(239, 68, 68, 0.1);
		padding: 24px;
		border-radius: 50%;
	}
	.danger-modal h2 {
		color: #fca5a5;
		font-size: 1.5rem;
		margin: 0;
	}
	.danger-message-box {
		width: 100%;
		background: rgba(0, 0, 0, 0.3);
		border: 1px solid rgba(239, 68, 68, 0.2);
		border-radius: 8px;
		padding: 16px;
		text-align: left;
		margin-top: 8px;
	}
	.danger-message-box p {
		color: #f87171;
		white-space: pre-wrap;
		font-family: inherit;
		margin: 0;
		font-size: 0.95rem;
		line-height: 1.5;
	}
	.danger-actions {
		display: flex;
		gap: 16px;
		margin-top: 12px;
		width: 100%;
		justify-content: center;
	}
	.danger-button {
		background: #ef4444;
		color: white;
		border: none;
	}
	.danger-button:hover {
		background: #dc2626;
	}

	.tool-call.error {
		border-color: rgba(239, 68, 68, 0.3);
		background: rgba(239, 68, 68, 0.03);
	}

	.tool-call-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 8px;
	}

	.tool-call-name {
		font-weight: 500;
		font-size: 0.95rem;
		color: #7bc8ff;
	}

	.tool-call-status {
		font-size: 0.85rem;
		color: var(--color-text-secondary);
	}

	.tool-call.calling .tool-call-status {
		color: #6acaff;
	}

	.tool-call.success .tool-call-status {
		color: #4ade80;
	}

	.tool-call.error .tool-call-status {
		color: #f87171;
	}

	.tool-call-arguments,
	.tool-call-result {
		margin-top: 8px;
	}

	.tool-call-arguments strong,
	.tool-call-result strong {
		display: block;
		margin-bottom: 4px;
		font-size: 0.85rem;
		color: var(--color-text-secondary);
	}

	.tool-call-pre {
		background: rgba(0, 0, 0, 0.2);
		border-radius: 4px;
		padding: 8px 10px;
		font-family: 'JetBrains Mono', 'SF Mono', 'Monaco', 'Consolas', monospace;
		font-size: 0.85rem;
		line-height: 1.4;
		white-space: pre-wrap;
		word-break: break-all;
		max-height: 200px;
		overflow-y: auto;
		margin: 0;
	}

	.tool-config-grid {
		display: grid;
		gap: 16px;
	}

	.tool-config-grid label {
		display: grid;
		gap: 8px;
	}

	.checkbox-row {
		grid-template-columns: auto 1fr;
		align-items: center;
		gap: 10px;
		color: var(--color-text-secondary);
	}

	.inline-note {
		margin: 10px 0 0;
		color: var(--color-text-muted);
		font-size: 0.92rem;
	}

	.text-button,
	.mini-icon {
		border: 0;
		background: transparent;
		color: var(--color-text-secondary);
	}

	.text-button {
		padding: 0;
		font: inherit;
		transition: color 0.18s ease;
	}

	.text-button:hover {
		color: var(--color-text-primary);
	}

	.inline-delete-form {
		display: flex;
		align-items: stretch;
	}

	.source-actions {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.mini-icon {
		width: 34px;
		min-height: 34px;
		border-radius: 10px;
		border: 1px solid rgba(255, 255, 255, 0.06);
		background: rgba(255, 255, 255, 0.03);
		display: inline-grid;
		place-items: center;
		transition:
			transform 0.18s ease,
			border-color 0.18s ease,
			background 0.18s ease;
	}

	.mini-icon:hover {
		transform: translateY(-2px);
		border-color: rgba(255, 255, 255, 0.12);
		background: rgba(255, 255, 255, 0.05);
	}

	.danger-icon:hover {
		border-color: rgba(255, 108, 122, 0.25);
		background: rgba(255, 108, 122, 0.08);
		color: #ff9ea7;
	}

	.form-actions {
		display: flex;
		justify-content: flex-end;
		gap: 12px;
	}

	.settings-actions {
		justify-content: space-between;
		align-items: center;
	}

	.action-pair {
		display: flex;
		gap: 12px;
	}

	.danger-form {
		margin-right: auto;
	}

	.btn-danger {
		background: rgba(255, 108, 122, 0.12);
		color: #ffb6bd;
		border: 1px solid rgba(255, 108, 122, 0.18);
	}

	.btn-danger:hover {
		background: rgba(255, 108, 122, 0.18);
	}

	.inline-error {
		margin: 8px 0 0;
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

	@media (max-width: 1100px) {
		.detail-grid {
			grid-template-columns: 300px minmax(0, 1fr);
		}

		.studio-panel {
			grid-column: 1 / -1;
			border-top: 1px solid rgba(255, 255, 255, 0.04);
		}
	}

	@media (max-width: 820px) {
		.detail-topbar,
		.detail-grid {
			grid-template-columns: 1fr;
		}

		.detail-topbar {
			flex-direction: column;
			align-items: stretch;
		}

		.detail-grid {
			display: block;
		}

		.document-panel {
			border: 0;
		}

		.source-template-grid,
		.tool-grid {
			grid-template-columns: 1fr;
		}

		.source-row,
		.output-row,
		.settings-actions {
			grid-template-columns: 1fr;
			display: grid;
		}

		.action-pair {
			justify-content: stretch;
		}
	}

	/* ===== 来源类型必选提示 ===== */
	.type-required-hint {
		margin: 0 0 4px;
		font-size: 0.85rem;
		color: var(--color-text-muted);
	}

	/* ===== 晃动动画 ===== */
	@keyframes shake {
		0%   { transform: translateX(0); }
		15%  { transform: translateX(-6px); }
		30%  { transform: translateX(6px); }
		45%  { transform: translateX(-5px); }
		60%  { transform: translateX(5px); }
		75%  { transform: translateX(-3px); }
		90%  { transform: translateX(3px); }
		100% { transform: translateX(0); }
	}

	@keyframes flash-outline {
		0%   { box-shadow: none; }
		20%  { box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.8); }
		60%  { box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.8); }
		100% { box-shadow: none; }
	}

	.source-template-grid.shake {
		animation: shake 0.55s ease;
	}

	.template-card.shake-outline {
		animation: flash-outline 0.55s ease;
		border-color: rgba(239, 68, 68, 0.45) !important;
	}

	/* ===== 解析行 ===== */
	.parse-row {
		display: flex;
		align-items: center;
		gap: 12px;
		flex-wrap: wrap;
	}

	.parse-btn {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 8px 14px;
		font-size: 0.9rem;
	}

	.parse-progress {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		color: var(--color-text-secondary);
		font-size: 0.9rem;
	}

	.parse-spinner {
		display: inline-block;
		width: 14px;
		height: 14px;
		border: 2px solid rgba(106, 202, 255, 0.25);
		border-top-color: #6acaff;
		border-radius: 50%;
		animation: spin 0.7s linear infinite;
	}

	@keyframes spin {
		to { transform: rotate(360deg); }
	}

	.parse-done {
		color: #4ade80;
		font-size: 0.9rem;
	}

	.parse-error {
		color: #f87171;
		font-size: 0.9rem;
	}

	/* ===== 解析预览区域 ===== */
	.parse-preview {
		display: grid;
		gap: 8px;
	}

	.parse-preview-body {
		max-height: 320px;
	}
</style>
