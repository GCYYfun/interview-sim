export type NotebookRole = 'OWNER' | 'EDITOR' | 'VIEWER';

export interface FeaturedNotebook {
	slug: string;
	tag: string;
	title: string;
	summary: string;
	sources: number;
	updatedAgo: string;
	accent: string;
}

export interface NotebookRow {
	slug: string;
	title: string;
	sourcesLabel: string;
	createdDate: string;
	role: NotebookRole;
}

export interface SourceItem {
	id?: string;
	title: string;
	subtitle: string;
	meta: string;
	icon: 'file-text' | 'file' | 'link' | 'message-square' | 'briefcase';
	selected?: boolean;
}

export interface StudioTool {
	key?: string;
	title: string;
	description: string;
	icon: 'mic' | 'bar-chart' | 'zap' | 'message-square';
}

export interface SavedOutput {
	id?: string;
	title: string;
	summary: string;
	updatedAgo: string;
	progress?: number;
	duration?: string;
}

export interface SourceTemplateItem {
	title: string;
	icon: SourceItem['icon'];
	caption: string;
}

export interface NotebookDetail {
	slug: string;
	title: string;
	role: NotebookRole;
	subject: string;
	location: string;
	contact: string;
	link: string;
	summary: string;
	experience: Array<{
		title: string;
		period: string;
		bullets: string[];
	}>;
	sources: SourceItem[];
	studioTools: StudioTool[];
	savedOutputs: SavedOutput[];
}

export const featuredNotebooks: FeaturedNotebook[] = [
	{
		slug: 'executive-strategy-2024',
		tag: 'AI ANALYSIS',
		title: 'Executive Strategy 2024',
		summary:
			'Deep dive into C-suite resume trends for the tech sector, focusing on innovation leadership.',
		sources: 24,
		updatedAgo: '2h ago',
		accent: 'linear-gradient(135deg, rgba(96, 194, 255, 0.14), rgba(23, 25, 31, 0.05))'
	},
	{
		slug: 'frontend-engineer-benchmarks',
		tag: 'MARKET RESEARCH',
		title: 'Frontend Engineer Benchmarks',
		summary: 'Synthesizing skill distributions across senior frontend candidates in EMEA region.',
		sources: 156,
		updatedAgo: '1d ago',
		accent: 'linear-gradient(135deg, rgba(255, 184, 77, 0.12), rgba(23, 25, 31, 0.04))'
	},
	{
		slug: 'graduate-pipeline-2023',
		tag: 'LEGACY ARCHIVES',
		title: '2023 Graduate Pipeline',
		summary:
			'Comparison of academic projects vs internship experience in entry-level data science roles.',
		sources: 89,
		updatedAgo: '3d ago',
		accent: 'linear-gradient(135deg, rgba(76, 212, 167, 0.1), rgba(23, 25, 31, 0.04))'
	}
];

export const recentNotebooks: NotebookRow[] = [
	{
		slug: 'jordan-smith-resume-analysis',
		title: 'Staff Product Designer Pool',
		sourcesLabel: '12 files',
		createdDate: 'Oct 24, 2023',
		role: 'OWNER'
	},
	{
		slug: 'q4-sales-representative-search',
		title: 'Q4 Sales Representative Search',
		sourcesLabel: '42 files',
		createdDate: 'Oct 20, 2023',
		role: 'EDITOR'
	},
	{
		slug: 'backend-go-specialists',
		title: 'Backend Engineering - Go Specialists',
		sourcesLabel: '28 files',
		createdDate: 'Oct 15, 2023',
		role: 'OWNER'
	},
	{
		slug: 'marketing-director-shortlist',
		title: 'Marketing Director Shortlist',
		sourcesLabel: '8 files',
		createdDate: 'Oct 12, 2023',
		role: 'VIEWER'
	},
	{
		slug: 'project-managers-pmp',
		title: 'Project Managers (PMP Certified)',
		sourcesLabel: '65 files',
		createdDate: 'Oct 08, 2023',
		role: 'OWNER'
	}
];

export const notebookDetails: NotebookDetail[] = [
	{
		slug: 'jordan-smith-resume-analysis',
		title: 'Resume Analysis',
		role: 'VIEWER',
		subject: 'Jordan Smith',
		location: 'San Francisco, CA',
		contact: 'j.smith@email.com',
		link: 'linkedin.com/in/jsmith',
		summary:
			'Innovative Senior Product Designer with 8+ years of experience in creating intuitive digital experiences. Specialist in systems design, accessibility, and high-fidelity prototyping.',
		experience: [
			{
				title: 'Design Lead @ TechNova',
				period: '2020 - Present',
				bullets: [
					'Spearheaded redesign of flagship SaaS platform increasing engagement by 40%.',
					'Established unified design system used by 5 cross-functional teams.'
				]
			},
			{
				title: 'Senior Designer @ CreativePulse',
				period: '2017 - 2020',
				bullets: [
					'Developed mobile-first e-commerce solutions for Fortune 500 clients.',
					'Collaborated with engineering to ensure pixel-perfect implementation.'
				]
			}
		],
		sources: [
			{
				title: 'Jordan_Smith_Resume.pdf',
				subtitle: 'Selected • 1.2 MB',
				meta: 'Primary source',
				icon: 'file-text',
				selected: true
			},
			{
				title: 'Portfolio_v4.pdf',
				subtitle: '2.8 MB',
				meta: 'Supporting portfolio',
				icon: 'file'
			},
			{
				title: 'GitHub - jsmith-dev',
				subtitle: 'External Source',
				meta: 'github.com/jsmith-dev',
				icon: 'link'
			}
		],
		studioTools: [
			{
				title: 'Audio Overview',
				description: 'Turn findings into a spoken briefing.',
				icon: 'mic'
			},
			{
				title: 'Resume Analysis',
				description: 'Extract strengths, signals, and gaps.',
				icon: 'bar-chart'
			},
			{
				title: 'Skill Matching',
				description: 'Map evidence to target role requirements.',
				icon: 'zap'
			},
			{
				title: 'Interview Qs',
				description: 'Generate prompts from resume weak spots.',
				icon: 'message-square'
			}
		],
		savedOutputs: [
			{
				title: 'Career Path Projection',
				summary: "Based on Jordan's experience at TechNova, the ideal next step is...",
				updatedAgo: '2h ago'
			},
			{
				title: 'Gap Analysis Notes',
				summary: 'Identified missing proficiency in Figma Variables and design tokens.',
				updatedAgo: 'Yesterday'
			},
			{
				title: 'Podcast Summary',
				summary: 'Audio synthesis of portfolio themes and hiring-manager talking points.',
				updatedAgo: '3d ago',
				progress: 0.72,
				duration: '12:04'
			}
		]
	}
];

export const sourceTemplates: SourceTemplateItem[] = [
	{ title: 'Resume', icon: 'file-text' as const, caption: 'Upload a candidate resume or CV.' },
	{
		title: 'Conversation',
		icon: 'message-square' as const,
		caption: 'Import interviews, notes, or transcripts.'
	},
	{
		title: 'Job Description (JD)',
		icon: 'briefcase' as const,
		caption: 'Compare against a target position.'
	}
];

export function getNotebookDetail(slug: string) {
	return notebookDetails.find((entry) => entry.slug === slug) ?? notebookDetails[0];
}
