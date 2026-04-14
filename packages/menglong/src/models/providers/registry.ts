/**
 * MengLong TypeScript SDK
 * Provider 注册中心
 * 
 * 对应 Python 版 menglong/models/providers/registry.py
 */

import type { BaseProvider } from './base.js';
import type { ProviderConfig, Config } from '../../utils/config/config_type.js';

type ProviderClass = new (config: ProviderConfig) => BaseProvider;

export class ProviderRegistry {
  private static readonly registry = new Map<string, ProviderClass>();

  /** 注册 Provider 类 */
  static register(name: string, providerClass: ProviderClass): void {
    ProviderRegistry.registry.set(name, providerClass);
  }

  /** 获取 Provider 类 */
  static getProviderClass(name: string): ProviderClass | undefined {
    return ProviderRegistry.registry.get(name);
  }

  /**
   * 工厂方法：根据名称创建 Provider 实例
   */
  static getInstance(providerName: string, config: Config): BaseProvider {
    const ProviderClass = ProviderRegistry.registry.get(providerName);
    if (!ProviderClass) {
      const available = ProviderRegistry.listProviders().join(', ');
      throw new Error(
        `Provider '${providerName}' not found. Available providers: [${available}]. ` +
        `Make sure to register it with ProviderRegistry.register('${providerName}', YourProvider).`
      );
    }

    // 优先从 providers[name] 读取，如果没有，则尝试使用根配置作为备选
    const providerConfig: ProviderConfig = (config.providers?.[providerName] as ProviderConfig) ?? (config as ProviderConfig);
    return new ProviderClass(providerConfig);
  }


  static listProviders(): string[] {
    return Array.from(ProviderRegistry.registry.keys());
  }
}
