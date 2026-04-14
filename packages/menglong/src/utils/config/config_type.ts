/**
 * MengLong TypeScript SDK
 * 配置类型定义
 * 
 * 对应 Python 版 menglong/utils/config/config_type.py
 */

export interface ModelConfig {
  temperature?: number;
  max_tokens?: number;
  dimensions?: number; // 用于 embedding 模型
  [key: string]: unknown;
}

export interface ProviderConfig {
  api_key?: string;
  base_url?: string;
  timeout?: number;
  max_retries?: number;
  /** 额外字段（如 AWS region, Google project 等） */
  [key: string]: unknown;
  models?: Record<string, ModelConfig>;
}

export interface SystemConfig {
  debug?: boolean;
  log_level?: string;
}

export interface DefaultConfig {
  model_id?: string;
  embedding_model_id?: string;
  temperature?: number;
  max_tokens?: number;
  timeout?: number;
}

/** MengLong 主配置结构，与 TOML 文件一一对应 */
export interface Config {
  default?: DefaultConfig;
  system?: SystemConfig;
  providers?: Record<string, ProviderConfig>;
}
