import { defineTool } from '$menglong';
import type { DimensionKey } from '../types';

/**
 * submit_dimension_result —— stage1b–1f 专属提交工具。
 *
 * 参数结构对应 rubric-rater.md 的输出 JSON Schema，
 * 把子维度证据点、评分依据、置信度等复杂结构下沉到 tool schema 层。
 */
export function createSubmitDimensionResultTool(dimension: DimensionKey) {
	return defineTool({
		name: 'submit_dimension_result',
		description: `提交【${dimension}】维度的结构化评估结果。完成所有分析推理后调用此工具。提交后本阶段结束。`,
		parameters: {
			type: 'object',
			properties: {
				dimension: {
					type: 'string',
					description: '维度(Dimension)：本次评估的维度名称'
				},
				overall: {
					type: 'object',
					description: '维度总体评价(Dimension Overall)',
					properties: {
						summary: {
							type: 'string',
							description: '总体摘要(Summary)'
						},
						level: {
							type: 'string',
							enum: ['显著优势', '达到预期', '证据不足/有风险', '明显不足'],
							description: '档位(Level)'
						},
						score: {
							type: 'number',
							minimum: 0,
							maximum: 100,
							description: '分数(Score)：0–100'
						},
						confidence: {
							type: 'number',
							minimum: 0,
							maximum: 1,
							description: '置信度(Confidence)：0–1'
						}
					},
					required: ['summary', 'level', 'score', 'confidence']
				},
				subdimensions: {
					type: 'array',
					description: '子维度列表(Subdimensions)',
					items: {
						type: 'object',
						properties: {
							name: {
								type: 'string',
								description: '名称(Name)'
							},
							elicited: {
								type: 'boolean',
								description: '是否被问到(Elicited)'
							},
							count: {
								type: 'object',
								description: '计数(Count)',
								properties: {
									total_items: { type: 'number', description: '总证据点数 T' },
									elicited_count: { type: 'number', description: '被问到数 E' },
									demonstrated_count: { type: 'number', description: '被展示数 D' }
								},
								required: ['total_items', 'elicited_count', 'demonstrated_count']
							},
							evidence_points: {
								type: 'array',
								description: '证据点列表(Evidence Points)',
								items: {
									type: 'object',
									properties: {
										evidence_id: { type: 'string', description: '证据点ID' },
										type: {
											type: 'string',
											enum: ['Must', 'Nice'],
											description: '类型'
										},
										trigger_met: { type: 'boolean', description: 'Trigger满足' },
										qualifier_met: { type: 'boolean', description: 'Qualifier满足' },
										elicited: { type: 'boolean', description: '是否被问到' },
										demonstrated: { type: 'boolean', description: '是否展示' },
										strength: {
											type: 'number',
											enum: [0, 1, 2, 3],
											description: '强度分数(Strength 0-3)'
										},
										primary_quote: {
											type: 'object',
											description: '主证据(Primary Quote)',
											properties: {
												speaker: { type: 'string', enum: ['candidate', 'interviewer'] },
												quote: { type: 'string' }
											},
											required: ['speaker', 'quote']
										},
										secondary_quote: {
											type: 'object',
											description: '次证据(Secondary Quote)，可选',
											properties: {
												speaker: { type: 'string' },
												quote: { type: 'string' }
											}
										},
										scoring_rationale: {
											type: 'object',
											description: '评分依据(Scoring Rationale)',
											properties: {
												question_asked: { type: 'string', description: '面试官问了什么(Question Asked)' },
												underlying_intent: { type: 'string', description: '实际在问什么(Underlying Intent)' },
												answer_gist: { type: 'string', description: '候选人回答要点(Answer Gist)' },
												why_this_strength: { type: 'string', description: '为何得此分(Why This Strength)' }
											},
											required: ['question_asked', 'underlying_intent', 'answer_gist', 'why_this_strength']
										}
									},
									required: [
										'evidence_id', 'type', 'trigger_met', 'qualifier_met',
										'elicited', 'demonstrated', 'strength', 'primary_quote', 'scoring_rationale'
									]
								}
							},
							other_evidence: {
								type: 'array',
								description: '其他证据(Other Evidence)：不可计分的 quote',
								items: {
									type: 'object',
									properties: {
										quote: { type: 'string' },
										context: { type: 'string' },
										reason: { type: 'string', description: '为什么不计分' }
									},
									required: ['quote', 'reason']
								}
							},
							unmapped: {
								type: 'array',
								description: '未映射信号(UNMAPPED)',
								items: {
									type: 'object',
									properties: {
										quote: { type: 'string' },
										scoring_rationale: {
											type: 'object',
											properties: {
												question_asked: { type: 'string' },
												underlying_intent: { type: 'string' },
												answer_gist: { type: 'string' },
												why_this_strength: { type: 'string' }
											},
											required: ['question_asked', 'underlying_intent', 'answer_gist', 'why_this_strength']
										},
										signal: { type: 'string', description: '强信号描述' },
										why: { type: 'string', description: '为什么无法映射' },
										suggested_rubric_update: { type: 'string', description: '建议补充证据点' }
									},
									required: ['quote', 'scoring_rationale', 'signal', 'why']
								}
							},
							summary: { type: 'string', description: '总结(Summary)' },
							level: {
								type: 'string',
								enum: ['显著优势', '达到预期', '证据不足/有风险', '明显不足'],
								description: '档位(Level)'
							},
							score_range: { type: 'string', description: '分数区间(Score Range)，如 "75-89"' },
							subdimension_score: {
								type: 'number',
								minimum: 0,
								maximum: 100,
								description: '子维度分数(Subdimension Score)'
							},
							confidence: {
								type: 'number',
								minimum: 0,
								maximum: 1,
								description: '置信度(Confidence)'
							},
							confidence_breakdown: {
								type: 'object',
								description: '置信度构成(Confidence Breakdown)',
								properties: {
									elicitation_rate: { type: 'number' },
									coverage: { type: 'number' },
									coverage_outcome: { type: 'number' },
									avg_strength: { type: 'number' },
									failure_rate: { type: 'number' },
									contradiction_penalty: { type: 'number' },
									cap_by_e: { type: 'number' }
								},
								required: [
									'elicitation_rate', 'coverage', 'coverage_outcome',
									'avg_strength', 'failure_rate', 'contradiction_penalty', 'cap_by_e'
								]
							},
							follow_up_questions: {
								type: 'array',
								items: { type: 'string' },
								description: '追问问题(Follow-up Questions)'
							}
						},
						required: [
							'name', 'elicited', 'count', 'evidence_points',
							'summary', 'level', 'subdimension_score', 'confidence', 'confidence_breakdown'
						]
					}
				}
			},
			required: ['dimension', 'overall', 'subdimensions']
		},
		handler: async () => {
			return { status: 'submitted', received: true };
		}
	});
}

/** 将工具参数重组为 rubric-rater.md 输出 Schema 格式 */
export function parseDimensionResult(args: Record<string, unknown>): unknown {
	const overall = args.overall as Record<string, unknown>;
	const subdimensions = (args.subdimensions as Record<string, unknown>[]) ?? [];

	return {
		'维度(Dimension)': args.dimension,
		'维度总体评价(Dimension Overall)': {
			'总体摘要(Summary)': overall.summary,
			'档位(Level)': overall.level,
			'分数(Score)': overall.score,
			'置信度(Confidence)': overall.confidence
		},
		'子维度列表(Subdimensions)': subdimensions.map((sub) => {
			const count = sub.count as Record<string, unknown>;
			const cbd = sub.confidence_breakdown as Record<string, unknown>;
			return {
				'名称(Name)': sub.name,
				'是否被问到(Elicited)': sub.elicited,
				'计数(Count)': {
					'总证据点数(Total Items, T)': count.total_items,
					'被问到数(Elicited Count, E)': count.elicited_count,
					'被展示数(Demonstrated Count, D)': count.demonstrated_count
				},
				'证据点列表(Evidence Points)': ((sub.evidence_points as Record<string, unknown>[]) ?? []).map((ep) => {
					const pq = ep.primary_quote as Record<string, unknown>;
					const sq = ep.secondary_quote as Record<string, unknown> | undefined;
					const sr = ep.scoring_rationale as Record<string, unknown>;
					return {
						'证据点ID(Evidence ID)': ep.evidence_id,
						'类型(Type)': ep.type,
						'Trigger满足(Trigger Met)': ep.trigger_met,
						'Qualifier满足(Qualifier Met)': ep.qualifier_met,
						'是否被问到(Elicited)': ep.elicited,
						'是否展示(Demonstrated)': ep.demonstrated,
						'强度分数(Strength 0-3)': ep.strength,
						'主证据(Primary Quote)': { speaker: pq.speaker, quote: pq.quote },
						...(sq ? { '次证据(Secondary Quote, optional)': { speaker: sq.speaker, quote: sq.quote } } : {}),
						'评分依据(Scoring Rationale)': {
							'面试官问了什么(Question Asked)': sr.question_asked,
							'实际在问什么(Underlying Intent)': sr.underlying_intent,
							'候选人回答要点(Answer Gist)': sr.answer_gist,
							'为何得此分(Why This Strength)': sr.why_this_strength
						}
					};
				}),
				'其他证据(Other Evidence)': ((sub.other_evidence as Record<string, unknown>[]) ?? []).map((oe) => ({
					quote: oe.quote,
					context: oe.context ?? '',
					'为什么不计分(Reason)': oe.reason
				})),
				'未映射信号(UNMAPPED)': ((sub.unmapped as Record<string, unknown>[]) ?? []).map((u) => {
					const sr = u.scoring_rationale as Record<string, unknown>;
					return {
						quote: u.quote,
						'评分依据(Scoring Rationale)': {
							'面试官问了什么(Question Asked)': sr.question_asked,
							'实际在问什么(Underlying Intent)': sr.underlying_intent,
							'候选人回答要点(Answer Gist)': sr.answer_gist,
							'为何得此分(Why This Strength)': sr.why_this_strength
						},
						'强信号描述(Signal)': u.signal,
						'为什么无法映射(Why)': u.why,
						'建议补充证据点(Suggested Rubric Update)': u.suggested_rubric_update ?? ''
					};
				}),
				'总结(Summary)': sub.summary,
				'档位(Level)': sub.level,
				'分数区间(Score Range)': sub.score_range ?? '',
				'子维度分数(Subdimension Score)': sub.subdimension_score,
				'置信度(Confidence)': sub.confidence,
				'置信度构成(Confidence Breakdown)': {
					'问得够不够(elicitation_rate)': cbd.elicitation_rate,
					'展示得够不够(coverage)': cbd.coverage,
					'证据产出覆盖(coverage_outcome=D/T)': cbd.coverage_outcome,
					'答得好不好(avg_strength)': cbd.avg_strength,
					'问到但未展示(failure_rate)': cbd.failure_rate,
					'矛盾惩罚(contradiction_penalty)': cbd.contradiction_penalty,
					'小样本上限(cap_by_E)': cbd.cap_by_e
				},
				'追问问题(Follow-up Questions)': sub.follow_up_questions ?? []
			};
		})
	};
}
