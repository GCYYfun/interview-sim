import { integer, sqliteTable, text } from 'drizzle-orm/sqlite-core';

export const task = sqliteTable('task', {
	id: text('id')
		.primaryKey()
		.$defaultFn(() => crypto.randomUUID()),
	title: text('title').notNull(),
	priority: integer('priority').notNull().default(1)
});

export const notebook = sqliteTable('notebook', {
	id: text('id')
		.primaryKey()
		.$defaultFn(() => crypto.randomUUID()),
	userId: text('user_id').notNull(),
	slug: text('slug').notNull(),
	title: text('title').notNull(),
	tag: text('tag').notNull(),
	summary: text('summary').notNull(),
	subject: text('subject').notNull(),
	location: text('location').notNull(),
	contact: text('contact').notNull(),
	link: text('link').notNull(),
	role: text('role').notNull().default('OWNER'),
	accent: text('accent').notNull(),
	createdAt: integer('created_at').notNull(),
	updatedAt: integer('updated_at').notNull()
});

export const notebookSource = sqliteTable('notebook_source', {
	id: text('id')
		.primaryKey()
		.$defaultFn(() => crypto.randomUUID()),
	notebookId: text('notebook_id').notNull(),
	title: text('title').notNull(),
	subtitle: text('subtitle').notNull(),
	meta: text('meta').notNull(),
	icon: text('icon').notNull(),
	content: text('content').notNull(),
	sourceType: text('source_type').notNull(),
	isSelected: integer('is_selected', { mode: 'boolean' }).notNull().default(false),
	createdAt: integer('created_at').notNull()
});

export const notebookOutput = sqliteTable('notebook_output', {
	id: text('id')
		.primaryKey()
		.$defaultFn(() => crypto.randomUUID()),
	notebookId: text('notebook_id').notNull(),
	title: text('title').notNull(),
	summary: text('summary').notNull(),
	content: text('content').notNull().default(''),
	updatedAgoLabel: text('updated_ago_label').notNull(),
	progress: integer('progress'),
	duration: text('duration'),
	createdAt: integer('created_at').notNull()
});

export const notebookMessage = sqliteTable('notebook_message', {
	id: text('id')
		.primaryKey()
		.$defaultFn(() => crypto.randomUUID()),
	notebookId: text('notebook_id').notNull(),
	role: text('role').notNull(),
	content: text('content').notNull(),
	createdAt: integer('created_at').notNull()
});

export const pipelineCheckpoint = sqliteTable('pipeline_checkpoint', {
	id: text('id')
		.primaryKey()
		.$defaultFn(() => crypto.randomUUID()),
	notebookId: text('notebook_id').notNull(),
	/** JSON-serialized PipelineCheckpoint */
	state: text('state').notNull(),
	status: text('status').notNull().default('running'), // running | done | failed
	createdAt: integer('created_at').notNull(),
	updatedAt: integer('updated_at').notNull()
});

export * from './auth.schema';
