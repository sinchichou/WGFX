// src/utils/MacroExpander.js

/**
 * @fileoverview 用於擴展預處理器宏的工具。
 * 此類別處理將常見的 HLSL/MagpieFX 宏 (例如 `MULADD`)
 * 替換為其 WGSL 等效項。
 *
 * 注意：這是一個非常簡化的實作，不能作為完整的預處理器。
 * 它使用簡單的字串/正則表達式替換。
 */
export class MacroExpander {
    constructor() {
        /**
         * 已知宏及其替換的字典。
         * 鍵是宏名稱 (或函數式宏的簽名)，
         * 值是替換它的 WGSL 程式碼。
         * @type {Object.<string, string>}
         */
        this.macros = {
            // 常數宏範例
            'MP_PI': '3.141592653589793',

            // 函數式宏範例。
            // 注意：基於正則表達式的函數式宏方法可能很脆弱。
            // 更穩健的方法是使用適當的詞法分析器。
            'MF_LERP(a, b, t)': 'mix(a, b, t)',
            'MULADD(a, b, c)': '(a * b + c)',

            // TODO: 根據需要在此處添加更多 HLSL/MagpieFX 特定宏。
        };
    }

    /**
     * 在給定的著色器程式碼字串中擴展已知宏。
     * @param {string} shaderCode - 可能包含宏的著色器程式碼。
     * @returns {string} 擴展所有識別出的宏後的程式碼。
     */
    expand(shaderCode) {
        let expandedCode = shaderCode;

        for (const macro in this.macros) {
            const replacement = this.macros[macro];

            // 此正則表達式嘗試將宏作為一個完整的單詞進行匹配，以避免
            // 替換其他識別符的部分。
            // 例如，`\bMACRO_NAME\b`
            const regex = new RegExp(`\b${macro.replace(/[-\/\\^$*+?.()|[\\\]{}]/g, '\\$&')}\b`, 'g');
            expandedCode = expandedCode.replace(regex, replacement);
        }

        return expandedCode;
    }
}
