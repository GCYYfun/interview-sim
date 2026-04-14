export type StudioAgentKey =
	| 'audio-overview'
	| 'interview-assessment';

export type StudioAgentInput = {
	tool: StudioAgentKey;
	focus?: string;
	tone?: string;
	length?: string;
	includeCitations?: boolean;
	resumeSourceId?: string;
	conversationSourceId?: string;
	jdSourceId?: string;
	evaluationMode?: 'basic' | 'advanced';
	resumeCheckpoint?: boolean;
};
