/**
 * @module Logger
 * @description Unified logging utility for the WGFX engine.
 * ---
 * WGFX 引擎的統一日誌工具，支援層級過濾與自定義輸出格式。
 */

/**
 * Log level enumeration.
 * ---
 * 日誌層級列舉，用於控制日誌輸出的詳細程度。
 *
 * @category Constants
 */
export enum LogLevel {
    /** Detailed debug information / 詳細的偵錯資訊 */
    DEBUG = 0,
    /** General operational messages / 一般運作訊息 */
    INFO = 1,
    /** Warning messages for non-critical issues / 非致命問題的警告訊息 */
    WARN = 2,
    /** Critical error messages / 致命錯誤訊息 */
    ERROR = 3,
    /** Disable all logging / 停用所有日誌輸出 */
    NONE = 4,
}

/**
 * Logger class for unified error output and debug mode control.
 * ---
 * 用於統一錯誤輸出與偵錯模式控制的靜態類別。
 *
 * @group Utils
 * @category Utility
 *
 * @example
 * ```ts
 * // Enable debug mode to see all logs / 啟用偵錯模式以查看所有日誌
 * Logger.setDebug(true);
 * Logger.debug("Checking resources...");
 *
 * // Set specific level / 設定特定層級
 * Logger.setLevel(LogLevel.ERROR);
 * ```
 */
export class Logger {
    /** @internal Current log level / 目前日誌層級 */
    private static level: LogLevel = LogLevel.INFO;

    /** @internal Prefix for all console outputs / 所有控制台輸出的前綴 */
    private static prefix: string = "[WGFX]";

    /**
     * Set the current log level.
     * ---
     * 設定目前的日誌過濾層級。低於此層級的日誌將不會被輸出。
     *
     * @group Configuration
     * @param level - The target {@link LogLevel} / 目標日誌層級
     */
    public static setLevel(level: LogLevel): void {
        this.level = level;
    }

    /**
     * Quickly enable or disable debug mode.
     * ---
     * 快速啟用或停用偵錯模式。
     * 啟用後層級會設為 DEBUG，停用則恢復為 INFO。
     *
     * @group Configuration
     * @param enabled - True to see all logs / 是否啟用偵錯模式
     */
    public static setDebug(enabled: boolean): void {
        this.level = enabled ? LogLevel.DEBUG : LogLevel.INFO;
    }

    /**
     * Log a debug message.
     * ---
     * 記錄偵錯訊息。僅在層級為 DEBUG 時顯示。
     *
     * @group Logging
     * @param message - The debug message / 要記錄的偵錯訊息
     * @param args - Additional contextual data / 額外的上下文參數
     */
    public static debug(message: string, ...args: any[]): void {
        if (this.level <= LogLevel.DEBUG) {
            console.debug(`${this.prefix} [DEBUG] ${message}`, ...args);
        }
    }

    /**
     * Log an info message.
     * ---
     * 記錄一般資訊。適用於系統狀態更新。
     *
     * @group Logging
     * @param message - The info message / 要記錄的一般訊息
     * @param args - Additional data / 額外參數
     */
    public static info(message: string, ...args: any[]): void {
        if (this.level <= LogLevel.INFO) {
            console.log(`${this.prefix} [INFO] ${message}`, ...args);
        }
    }

    /**
     * Log a warning message.
     * ---
     * 記錄警告訊息。用於提醒可能存在但不會中斷執行的問題。
     *
     * @group Logging
     * @param message - The warning message / 要記錄的警告訊息
     * @param args - Additional data / 額外參數
     */
    public static warn(message: string, ...args: any[]): void {
        if (this.level <= LogLevel.WARN) {
            console.warn(`${this.prefix} [WARN] ${message}`, ...args);
        }
    }

    /**
     * Log an error message.
     * ---
     * 記錄錯誤訊息。用於記錄導致功能失效或執行中斷的嚴重問題。
     *
     * @group Logging
     * @param message - The error message / 要記錄的錯誤訊息
     * @param args - Additional data or Error object / 額外參數或 Error 物件
     */
    public static error(message: string, ...args: any[]): void {
        if (this.level <= LogLevel.ERROR) {
            console.error(`${this.prefix} [ERROR] ${message}`, ...args);
        }
    }
}