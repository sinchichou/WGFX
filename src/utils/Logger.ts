/**
 * @file Logger.ts
 * @description Unified logging utility for WGFX
 * WGFX 的統一日誌工具
 */

/**
 * Log level enumeration
 * 日誌層級列舉
 */
export enum LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3,
    NONE = 4,
}

/**
 * Logger class for unified error output and debug mode control
 * 用於統一錯誤輸出與偵錯模式控制的日誌類別
 */
export class Logger {
    private static level: LogLevel = LogLevel.INFO;
    private static prefix: string = "[WGFX]";

    /**
     * Set the current log level
     * 設定目前的日誌層級
     * @param level - The log level to set (設定的日誌層級)
     */
    public static setLevel(level: LogLevel): void {
        this.level = level;
    }

    /**
     * Enable or disable debug mode
     * 啟用或停用偵錯模式
     * @param enabled - Whether to enable debug mode (是否啟用偵錯模式)
     */
    public static setDebug(enabled: boolean): void {
        this.level = enabled ? LogLevel.DEBUG : LogLevel.INFO;
    }

    /**
     * Log a debug message
     * 記錄偵錯訊息
     * @param message - The message to log (要記錄的訊息)
     * @param args - Additional arguments (額外參數)
     */
    public static debug(message: string, ...args: any[]): void {
        if (this.level <= LogLevel.DEBUG) {
            console.debug(`${this.prefix} [DEBUG] ${message}`, ...args);
        }
    }

    /**
     * Log an info message
     * 記錄一般訊息
     * @param message - The message to log (要記錄的訊息)
     * @param args - Additional arguments (額外參數)
     */
    public static info(message: string, ...args: any[]): void {
        if (this.level <= LogLevel.INFO) {
            console.log(`${this.prefix} [INFO] ${message}`, ...args);
        }
    }

    /**
     * Log a warning message
     * 記錄警告訊息
     * @param message - The message to log (要記錄的訊息)
     * @param args - Additional arguments (額外參數)
     */
    public static warn(message: string, ...args: any[]): void {
        if (this.level <= LogLevel.WARN) {
            console.warn(`${this.prefix} [WARN] ${message}`, ...args);
        }
    }

    /**
     * Log an error message
     * 記錄錯誤訊息
     * @param message - The message to log (要記錄的訊息)
     * @param args - Additional arguments (額外參數)
     */
    public static error(message: string, ...args: any[]): void {
        if (this.level <= LogLevel.ERROR) {
            console.error(`${this.prefix} [ERROR] ${message}`, ...args);
        }
    }
}
