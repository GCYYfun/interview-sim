import { defineTool } from '$menglong';

type NotebookDetail = NonNullable<
	Awaited<ReturnType<typeof import('$lib/server/workbench').getNotebookDetailForUser>>
>;

function compactText(value: string, limit = 100000) {
	return value.length > limit ? `${value.slice(0, limit)}\n\n[truncated]` : value;
}

export function createNotebookGeneratorTools(detail: NotebookDetail) {
	return [
		defineTool({
			name: 'get_notebook_overview',
			description: 'Read the current notebook overview, owner-facing summary, and metadata.',
			parameters: {
				type: 'object',
				properties: {},
				required: []
			},
			handler: async () => ({
				title: detail.record.title,
				subject: detail.record.subject,
				tag: detail.record.tag,
				summary: detail.record.summary,
				location: detail.record.location,
				contact: detail.record.contact,
				link: detail.record.link,
				role: detail.record.role
			})
		}),
		defineTool({
			name: 'get_selected_source',
			description: 'Read the currently selected source in the notebook, including its content.',
			parameters: {
				type: 'object',
				properties: {},
				required: []
			},
			handler: async () => {
				const source = detail.selectedSource;
				if (!source) {
					return { status: 'empty', message: 'No selected source is available yet.' };
				}

				return {
					id: source.id,
					title: source.title,
					subtitle: source.subtitle,
					meta: source.meta,
					sourceType: source.sourceType,
					content: compactText(source.content)
				};
			}
		}),
		defineTool({
			name: 'list_sources',
			description: 'List every source in this notebook with title, type, and whether it is selected.',
			parameters: {
				type: 'object',
				properties: {},
				required: []
			},
			handler: async () =>
				detail.sources.map((source) => ({
					id: source.id,
					title: source.title,
					subtitle: source.subtitle,
					meta: source.meta,
					sourceType: source.sourceType,
					selected: detail.selectedSource?.id === source.id
				}))
		}),
		defineTool({
			name: 'get_source_by_type',
			description: 'Get a source by its type (resume, conversation, job-description) or by source ID.',
			parameters: {
				type: 'object',
				properties: {
					sourceType: {
						type: 'string',
						description: 'The source type to find.',
						enum: ['resume', 'conversation', 'job-description']
					},
					sourceId: {
						type: 'string',
						description: 'The source ID to find (takes precedence over sourceType).'
					}
				},
				required: []
			},
			handler: async (params) => {
				const sourceId = String(params.sourceId ?? '').trim();
				const sourceType = String(params.sourceType ?? '').trim();

				let source;
				if (sourceId) {
					source = detail.sources.find((entry) => entry.id === sourceId);
				} else if (sourceType) {
					source = detail.sources.find((entry) => entry.sourceType === sourceType);
				}

				if (!source) {
					return {
						status: 'missing',
						message: `No source found for ${sourceId ? `ID "${sourceId}"` : `type "${sourceType}"`}.`
					};
				}

				return {
					id: source.id,
					title: source.title,
					subtitle: source.subtitle,
					sourceType: source.sourceType,
					meta: source.meta,
					content: compactText(source.content)
				};
			}
		}),
		defineTool({
			name: 'evaluate_text_by_criteria',
			description: 'Evaluate text content against specific assessment criteria.',
			parameters: {
				type: 'object',
				properties: {
					text: {
						type: 'string',
						description: 'The text to evaluate.'
					},
					criteria: {
						type: 'string',
						description: 'The evaluation criteria to apply.'
					},
					scale: {
						type: 'string',
						description: 'The scoring scale (e.g., "1-10" or "Strong/Medium/Weak").',
						default: '1-10'
					}
				},
				required: ['text', 'criteria']
			},
			handler: async (params) => {
				// This tool provides structured evaluation framework
				// The actual evaluation logic is handled by the model in the prompt
				return {
					text: String(params.text ?? ''),
					criteria: String(params.criteria ?? ''),
					scale: String(params.scale ?? '1-10'),
					evaluation_method: 'Use the provided criteria to score the text on the given scale, providing detailed reasoning.'
				};
			}
		}),
		defineTool({
			name: 'compare_texts',
			description: 'Compare two texts and identify similarities, differences, strengths, and weaknesses.',
			parameters: {
				type: 'object',
				properties: {
					text1: {
						type: 'string',
						description: 'First text to compare.'
					},
					text2: {
						type: 'string',
						description: 'Second text to compare.'
					},
					comparisonType: {
						type: 'string',
						description: 'Type of comparison (similarities, differences, pros-cons, etc.).',
						default: 'comprehensive'
					}
				},
				required: ['text1', 'text2']
			},
			handler: async (params) => {
				// Structured comparison framework
				return {
					text1: compactText(String(params.text1 ?? '')),
					text2: compactText(String(params.text2 ?? '')),
					comparisonType: String(params.comparisonType ?? 'comprehensive'),
					analysis_method: 'Compare the texts systematically, highlighting key similarities, differences, strengths, and areas for improvement.'
				};
			}
		}),
		defineTool({
			name: 'summarize_content',
			description: 'Create a concise summary of the provided content.',
			parameters: {
				type: 'object',
				properties: {
					content: {
						type: 'string',
						description: 'The content to summarize.'
					},
					focus: {
						type: 'string',
						description: 'Specific focus area for the summary.',
						default: 'key points'
					},
					length: {
						type: 'string',
						description: 'Desired summary length.',
						enum: ['brief', 'standard', 'detailed'],
						default: 'standard'
					}
				},
				required: ['content']
			},
			handler: async (params) => {
				return {
					content: compactText(String(params.content ?? '')),
					focus: String(params.focus ?? 'key points'),
					length: String(params.length ?? 'standard'),
					summary_method: 'Extract and condense the most important information, focusing on the specified area.'
				};
			}
		}),
		defineTool({
			name: 'analyze_intent',
			description: 'Analyze the intent, purpose, or underlying meaning in text.',
			parameters: {
				type: 'object',
				properties: {
					text: {
						type: 'string',
						description: 'The text to analyze for intent.'
					},
					context: {
						type: 'string',
						description: 'Additional context about the text.',
						default: ''
					}
				},
				required: ['text']
			},
			handler: async (params) => {
				return {
					text: compactText(String(params.text ?? '')),
					context: String(params.context ?? ''),
					analysis_method: 'Identify the primary intent, goals, motivations, and communication style expressed in the text.'
				};
			}
		}),
		defineTool({
			name: 'reflect_on_purpose',
			description: 'Reflect on the purpose and objectives of a process or interaction.',
			parameters: {
				type: 'object',
				properties: {
					content: {
						type: 'string',
						description: 'The content to reflect upon.'
					},
					objective: {
						type: 'string',
						description: 'The stated or implied objective.',
						default: 'clarify purpose'
					}
				},
				required: ['content']
			},
			handler: async (params) => {
				return {
					content: compactText(String(params.content ?? '')),
					objective: String(params.objective ?? 'clarify purpose'),
					reflection_method: 'Examine the content to understand the underlying purpose, goals, and whether they were effectively achieved.'
				};
			}
		})
	];
}
