/**
 * MengLong TypeScript SDK
 * Logger 接口定义
 */

export interface MengLongLogger {
  debug(message: string, ...args: unknown[]): void;
  info(message: string, ...args: unknown[]): void;
  warn(message: string, ...args: unknown[]): void;
  error(message: string, ...args: unknown[]): void;
}

/** 默认静默 Logger */
export const silentLogger: MengLongLogger = {
  debug: () => {},
  info: () => {},
  warn: () => {},
  error: () => {},
};

/** 简单的 Console Logger */
export class ConsoleLogger implements MengLongLogger {
  constructor(private readonly prefix: string = '[MengLong]') {}

  debug(message: string, ...args: unknown[]): void {
    console.debug(`${this.prefix} [DEBUG] ${message}`, ...args);
  }

  info(message: string, ...args: unknown[]): void {
    console.info(`${this.prefix} [INFO] ${message}`, ...args);
  }

  warn(message: string, ...args: unknown[]): void {
    console.warn(`${this.prefix} [WARN] ${message}`, ...args);
  }

  error(message: string, ...args: unknown[]): void {
    console.error(`${this.prefix} [ERROR] ${message}`, ...args);
  }
}
