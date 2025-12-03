// src/utils/FunctionOverloadResolver.js

/**
 * @fileoverview
 * - EN: Utility for resolving function overloads.
 *   WGSL does not support function overloading (multiple functions with the same name but different parameters).
 *   This class helps generate unique names for different function signatures to avoid naming conflicts in the final WGSL code.
 * - TW: 用於解析函數重載的工具。
 *   WGSL 不支援函數重載 (多個函數具有相同的名稱但參數不同)。
 *   此類別有助於為不同的函數簽名生成唯一的名稱，以避免最終 WGSL 程式碼中的命名衝突。
 */
export class FunctionOverloadResolver {
    constructor() {
        /**
         * - EN: Stores the signatures found for each function name.
         *   For example, Map<'myFunc', Set<'float_vec2', 'int'>>
         * - TW: 儲存為每個函數名稱找到的簽名。
         *   例如，Map<'myFunc', Set<'float_vec2', 'int'>>
         * @type {Map<string, Set<string>>}
         */
        this.functionSignatures = new Map();
    }

    /**
     * - EN: Generates a unique, mangled function name based on the function signature.
     *   For example, `resolve('doSomething', ['float', 'vec2'])` might return 'doSomething_float_vec2'.
     * - TW: 根據函數簽名生成一個唯一的、經過混淆的函數名稱。
     *   例如，`resolve('doSomething', ['float', 'vec2'])` 可能會返回 'doSomething_float_vec2'。
     * @param {string} functionName
     * - EN: The original name of the function (e.g., "doSomething").
     * - TW: 函數的原始名稱 (例如："doSomething")。
     * @param {string[]} paramTypes
     * - EN: An array of parameter type names (e.g., ['float', 'vec2']).
     * - TW: 參數類型名稱的陣列 (例如：['float', 'vec2'])。
     * @returns {string}
     * - EN: A unique name for the function signature.
     * - TW: 函數簽名的唯一名稱。
     */
    resolve(functionName, paramTypes) {
        const signature = paramTypes.join('_');
        const uniqueName = `${functionName}_${signature}`;

        if (!this.functionSignatures.has(functionName)) {
            this.functionSignatures.set(functionName, new Set());
        }
        this.functionSignatures.get(functionName).add(signature);

        return uniqueName;
    }

    /**
     * - EN: Resets the state of the resolver, clearing all stored function signatures.
     *   This is useful when parsing new files.
     * - TW: 重置解析器的狀態，清除所有儲存的函數簽名。
     *   這在解析新檔案時很有用。
     */
    reset() {
        this.functionSignatures.clear();
    }
}
