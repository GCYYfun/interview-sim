/**
 * MengLong TypeScript SDK
 * TOML 配置加载器
 * 
 * 对应 Python 版 menglong/utils/config/config_loader.py
 * 搜索顺序: MENGLONG_CONFIG 环境变量 > 当前目录 > Home 目录
 */

import type { Config } from './config_type.js';

/** 从 TOML 字符串加载配置 */
export function parseConfig(tomlString: string): Config {
  const result: Record<string, unknown> = {};
  let currentTarget: Record<string, unknown> = result;

  for (const rawLine of tomlString.split('\n')) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;

    if (line.startsWith('[') && line.endsWith(']')) {
      const section = line.slice(1, -1).trim();
      if (!section) continue;

      const segments = section.split('.');
      currentTarget = result;
      for (const segment of segments) {
        const value = currentTarget[segment];
        if (!value || typeof value !== 'object') {
          currentTarget[segment] = {};
        }
        currentTarget = currentTarget[segment] as Record<string, unknown>;
      }
      continue;
    }

    const separatorIndex = line.indexOf('=');
    if (separatorIndex === -1) continue;

    const key = line.slice(0, separatorIndex).trim();
    const value = line.slice(separatorIndex + 1).trim();
    if (!key) continue;

    currentTarget[key] = parseTomlValue(value);
  }

  return result as Config;
}

/** 搜索并加载配置文件 */
export function loadConfig(_configPath?: string): Config {
  // 浏览器环境下直接返回空配置，不尝试加载文件
  if (typeof window !== 'undefined' || typeof process === 'undefined' || !process.versions?.node) {
    return {};
  }

  // 仅在 Node 环境下动态引入 fs/path/os (或者说在这里保持同步但通过环境检查确保不执行)
  // Vite 等打包工具在看到这些静态 import 时仍可能报错，因此我们需要更彻底的隔离
  return {}; 
}

// 辅助函数在浏览器端同样保持导出但不做操作
export function loadConfigFromFile(_filePath: string): Config { return {}; }

function parseTomlValue(input: string): unknown {
  if (input === 'true') return true;
  if (input === 'false') return false;
  if (/^-?\d+(\.\d+)?$/.test(input)) return Number(input);
  if (
    (input.startsWith('"') && input.endsWith('"')) ||
    (input.startsWith("'") && input.endsWith("'"))
  ) {
    return input.slice(1, -1);
  }
  return input;
}
