import { WGFXShaderInfo } from '@/types';

/**
 * Parses the WGFX effect source code into structured shader information.
 * ---
 * 將 WGFX 特效原始碼解析為結構化的著色器元數據（Shader Metadata）。
 * 此函式會分析 WGSL 代碼、Uniform 定義、取樣器配置與渲染通道。
 *
 * @group Parser
 * @category Functions
 *
 * @param input - The effect source code / 特效原始碼字串
 * @param options - Optional parser configurations / 選用的解析器配置選項
 * @returns Structured shader information / 結構化的著色器資訊物件
 *
 * @throws {@link SyntaxError}
 * 當原始碼語法不符合 WGFX 規範時拋出此異常。
 *
 * @example
 * ```ts
 * try {
 *   const shaderInfo = parse(wgslCode);
 *   console.log(`Parsed ${shaderInfo.passes.length} passes`);
 * } catch (err) {
 *   if (err instanceof SyntaxError) {
 *     console.error(`Error at line ${err.location.start.line}: ${err.message}`);
 *   }
 * }
 * ```
 */
export function parse(input: string, options?: any): WGFXShaderInfo;

/**
 * Error class representing syntax errors during shader parsing.
 * ---
 * 代表著色器解析過程中語法錯誤的異常類別。
 * 包含了錯誤發生的位置資訊、預期的 Token 以及實際找到的內容。
 *
 * @group Parser
 * @category Exceptions
 */
export class SyntaxError extends Error {
    /**
     * Location of the error in the source code.
     * @zh 原始碼中發生錯誤的精確位置（包含行列資訊）
     * @example `{ start: { offset: 0, line: 1, column: 1 }, end: { ... } }`
     */
    location: any;

    /**
     * Expected tokens at the error location.
     * @zh 錯誤位置預期出現的 Token 列表
     */
    expected: any;

    /**
     * The actual token found.
     * @zh 實際解析到的 Token 內容
     */
    found: any;

    /**
     * @internal
     * @param message - Error message / 錯誤訊息
     * @param expected - Expected tokens / 預期的 Token
     * @param found - Found token / 實際找到的 Token
     * @param location - Error location / 錯誤位置
     */
    constructor(message: string, expected: any, found: any, location: any) {
        super(message);
        this.name = 'SyntaxError';
        this.expected = expected;
        this.found = found;
        this.location = location;

        // Ensure the prototype is correctly set / 確保原型鏈正確（針對 TS 繼承 Error 的處理）
        Object.setPrototypeOf(this, SyntaxError.prototype);
    }
}