/**
 * MengLong TypeScript SDK
 * Chat Template 相关数据结构
 * 
 * 对应 Python 版 menglong/schemas/chat.py
 */

// ==================== 请求结构 ====================

/** 消息角色枚举 */
export type MessageRole = 'system' | 'user' | 'assistant' | 'tool';

/** 内容片段基类 */
export interface ContentPart {
  type: string;
}

/** 文本片段 */
export interface TextPart extends ContentPart {
  type: 'text';
  text: string;
}

/** 
 * 图片片段 
 * 
 * 支持两种输入方式:
 * 1. URL: image_url={url: "https://...", detail?: "high"|"low"|"auto"}
 * 2. Base64: data="base64_string", media_type="image/jpeg"
 */
export interface ImagePart extends ContentPart {
  type: 'image';
  /** URL 格式 */
  image_url?: { url: string; detail?: 'low' | 'high' | 'auto' };
  /** Base64 数据 */
  data?: string;
  /** 媒体类型 (e.g., "image/jpeg") */
  media_type?: string;
  /** 控制图片分析精度 (OpenAI 特有) */
  detail?: 'low' | 'high' | 'auto';
}

/** 文档片段 (如 PDF) */
export interface DocumentPart extends ContentPart {
  type: 'document';
  /** Base64 数据 */
  data: string;
  /** 媒体类型 (e.g., "application/pdf") */
  media_type?: string;
}

/** 音频片段 */
export interface AudioPart extends ContentPart {
  type: 'audio';
  /** 音频 URL */
  audio_url?: string;
  /** Base64 数据 */
  data?: string;
  /** 媒体类型 (e.g., "audio/mp3") */
  media_type?: string;
}

/** 视频片段 */
export interface VideoPart extends ContentPart {
  type: 'video';
  /** 视频 URL */
  video_url?: string;
  /** Base64 数据 */
  data?: string;
  /** 媒体类型 (e.g., "video/mp4") */
  media_type?: string;
}

/** 动作描述 (用于 Assistant 输出或输入) */
export interface Action extends ContentPart {
  type: 'action';
  /** 调用 ID */
  id?: string;
  /** 函数名 */
  name: string;
  /** 函数参数 */
  arguments?: Record<string, unknown> | string;
  /** 流式工具调用时的索引 (用于增量累积) */
  index?: number;
}

/** 工具执行结果 (用于 Tool 输入) */
export interface Outcome extends ContentPart {
  type: 'outcome';
  /** 对应 Action 的 ID */
  id: string;
  /** 函数名 (可选) */
  name?: string;
  /** 执行结果字符串 */
  result: string;
}

/** 消息内容片段联合类型 */
export type AnyContentPart = 
  | TextPart 
  | ImagePart 
  | DocumentPart 
  | AudioPart 
  | VideoPart 
  | Action 
  | Outcome;

/** 聊天消息结构 */
export interface Message {
  /** 角色 */
  role: MessageRole;
  /** 
   * 内容: 
   * - 简单聊天为 string
   * - 多模态或工具调用为 AnyContentPart 数组
   */
  content: string | AnyContentPart[] | null;
}

/** 
 * 历史对话容器 (Managed Context)
 * 
 * 提供便捷的消息添加方法与迭代支持。
 */
export class Context {
  messages: Message[] = [];

  constructor(messages?: Message[]) {
    if (messages) this.messages = [...messages];
  }

  /** 添加原始消息或字符串（默认为 user 角色） */
  add(message: Message | string): this {
    if (typeof message === 'string') {
      this.messages.push({ role: 'user', content: message });
    } else {
      this.messages.push(message);
    }
    return this;
  }

  /** 快捷添加 User 消息 */
  user(content: string | AnyContentPart[], kwargs?: UserKwargs): this {
    return this.add(User(content, kwargs));
  }

  /** 快捷添加 Assistant 消息 */
  assistant(content?: string, actions?: Array<{ id?: string; name: string; arguments?: Record<string, unknown> }>): this {
    return this.add(Assistant(content, actions));
  }

  /** 快捷添加 System 消息 */
  system(content: string): this {
    return this.add(System(content));
  }

  /** 快捷添加 Tool (Outcome) 消息 */
  tool(toolId: string, content: string, name?: string): this {
    return this.add(Tool(toolId, content, name));
  }

  /** 获取最后一条消息 */
  get last(): Message | undefined {
    return this.messages[this.messages.length - 1];
  }

  /** 获取消息条数 */
  get length(): number {
    return this.messages.length;
  }

  /** 支持 for...of 迭代 */
  [Symbol.iterator]() {
    return this.messages[Symbol.iterator]();
  }

  /** 转换为纯数组 */
  toArray(): Message[] {
    return [...this.messages];
  }
}

// ==================== 快捷构造函数 ====================

export interface UserKwargs {
  image?: string;
  document?: string;
  pdf?: string;
  audio?: string;
  video?: string;
  detail?: 'low' | 'high' | 'auto';
}

/** 
 * 快捷构造 User 消息，支持多模态输入 
 * 
 * @example
 * User("描述这张图片", { image: "https://..." })
 */
export function User(content: string | AnyContentPart[], kwargs?: UserKwargs): Message {
  if (typeof content === 'string' && !kwargs) {
    return { role: 'user', content };
  }

  const parts: AnyContentPart[] = [];

  if (typeof content === 'string') {
    parts.push({ type: 'text', text: content });
  } else {
    parts.push(...content);
  }

  if (kwargs?.image) {
    const val = kwargs.image;
    const detail = kwargs.detail;
    if (val.startsWith('http://') || val.startsWith('https://')) {
      parts.push({ type: 'image', image_url: { url: val, detail } });
    } else {
      // 假设为 Base64 或路径（TS 版目前主要期望 Base64，不处理本地文件读取逻辑，除非在 Node 环境）
      parts.push({ type: 'image', data: val, detail });
    }
  }

  if (kwargs?.document || kwargs?.pdf) {
    const val = (kwargs.document ?? kwargs.pdf)!;
    parts.push({ type: 'document', data: val });
  }

  if (kwargs?.audio) {
    const val = kwargs.audio;
    if (val.startsWith('http://') || val.startsWith('https://')) {
      parts.push({ type: 'audio', audio_url: val });
    } else {
      parts.push({ type: 'audio', data: val });
    }
  }

  if (kwargs?.video) {
    const val = kwargs.video;
    if (val.startsWith('http://') || val.startsWith('https://')) {
      parts.push({ type: 'video', video_url: val });
    } else {
      parts.push({ type: 'video', data: val });
    }
  }

  return { role: 'user', content: parts };
}

/** 
 * 快捷构造 Assistant 消息，支持工具调用 
 */
export function Assistant(
  content?: string,
  actions?: Array<{ id?: string; name: string; arguments?: Record<string, unknown> }>,
): Message {
  if (!actions || actions.length === 0) {
    return { role: 'assistant', content: content ?? null };
  }

  const parts: AnyContentPart[] = [];
  if (content) parts.push({ type: 'text', text: content });
  for (const ac of actions) {
    parts.push({ type: 'action', id: ac.id, name: ac.name, arguments: ac.arguments ?? {} });
  }
  return { role: 'assistant', content: parts };
}

/** 快捷构造 System 消息 */
export function System(content: string): Message {
  return { role: 'system', content };
}

/** 
 * 快捷构造 Tool 结果消息 
 */
export function Tool(toolId: string, content: string, name?: string): Message {
  return {
    role: 'tool',
    content: [{ type: 'outcome', id: toolId, name, result: content }],
  };
}

// ==================== 响应结构 ====================

export interface Content {
  text?: string;
  reasoning?: string;
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}

export interface Output {
  content?: Content;
  actions?: Action[];
  status?: string;
}

export interface Response {
  output?: Output;
  model?: string;
  usage?: Usage;
  /** 快捷获取文本内容 (Getter) */
  readonly text?: string;
  /** 快捷获取工具调用列表 (Getter) */
  readonly tool_calls?: Action[];
}

/** 
 * 创建 Response 对象的工厂函数
 * 通过 Object.defineProperty 注入 getter，确持续兼容性与灵活性。
 */
export function createResponse(data: Omit<Response, 'text' | 'tool_calls'>): Response {
  const resp = { ...data } as Response;
  Object.defineProperty(resp, 'text', {
    get() { return this.output?.content?.text; },
    enumerable: true,
    configurable: true,
  });
  Object.defineProperty(resp, 'tool_calls', {
    get() { return this.output?.actions; },
    enumerable: true,
    configurable: true,
  });
  return resp;
}

// ==================== 流式响应结构 ====================

export interface Delta {
  text?: string;
  reasoning?: string;
  /** 流式工具调用增量 */
  actions?: Action[];
}

export interface StreamOutput {
  delta?: Delta;
  start?: string;
  end?: string;
  /** 当流结束且 finish_reason=tool_calls 时，携带完整的工具调用列表 */
  actions?: Action[];
}

export interface StreamResponse {
  output?: StreamOutput;
  model?: string;
  usage?: Usage;
}

