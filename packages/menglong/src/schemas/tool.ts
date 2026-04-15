/**
 * MengLong TypeScript SDK
 * 工具定义类型
 *
 * 对应 Python 版 menglong/schemas/tool.py
 */

/** JSON Schema property 定义，支持嵌套 object / array 等复杂结构 */
export interface PropertySchema {
  type: string;
  description?: string;
  enum?: string[] | number[];
  default?: unknown;
  minimum?: number;
  maximum?: number;
  items?: PropertySchema;
  properties?: Record<string, PropertySchema>;
  required?: string[];
}

export interface FunctionParameters {
  type: 'object';
  properties: Record<string, PropertySchema>;
  required?: string[];
}

export interface FunctionInfo {
  name: string;
  description: string;
  parameters: FunctionParameters;
}

/** MengLong 标准工具定义（各 Provider 会根据此对象生成各自所需的特定格式） */
export interface ToolInfo {
  type: 'function';
  function: FunctionInfo;
}

/**
 * @tool 装饰器的替代：将函数标注为 MengLong 工具
 *
 * @example
 * ```typescript
 * const myTool = defineTool({
 *   name: 'get_weather',
 *   description: '获取指定城市的天气',
 *   parameters: {
 *     type: 'object',
 *     properties: {
 *       city: { type: 'string', description: '城市名称' },
 *     },
 *     required: ['city'],
 *   },
 *   handler: async ({ city }) => `${city} 今天晴天，25°C`,
 * });
 * ```
 */
export interface ToolDefinition<TParams = Record<string, unknown>> {
  name: string;
  description: string;
  parameters: FunctionParameters;
  handler: (params: TParams) => unknown | Promise<unknown>;
}

export function defineTool<TParams = Record<string, unknown>>(
  def: ToolDefinition<TParams>,
): ToolDefinition<TParams> & { schema(): ToolInfo } {
  return {
    ...def,
    schema(): ToolInfo {
      return {
        type: 'function',
        function: {
          name: def.name,
          description: def.description,
          parameters: def.parameters,
        },
      };
    },
  };
}
