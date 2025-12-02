// src/index.js

/**
 * @fileoverview
 * - EN: This is the main public API entry point for the WGFX runtime.
 *   It exposes a simplified, function-based interface for interacting with the underlying WGFXRuntime class,
 *   managing a singleton instance of the runtime.
 * - TW: 這是 WGFX 運行時的主要公共 API 入口點。
 *   它公開了一個簡化的、基於函數的介面，用於與底層 WGFXRuntime 類別互動，
 *   管理運行時的單例實例。
 */

import {WGFXRuntime} from './runtime/WGFXRuntime.js';

/**
 * - EN: Singleton instance of WGFXRuntime.
 * - TW: WGFXRuntime 的單例實例。
 * @type {WGFXRuntime | null}
 */
let runtimeInstance = null;

/**
 * - EN: Initializes the runtime (if needed) and compiles the WGFX effect.
 *   This function must be called before any other API functions.
 * - TW: 初始化運行時 (如果需要) 並編譯 WGFX 效果。
 *   必須在任何其他 API 函數之前呼叫此函數。
 * @param {string} effectCode
 * - EN: The string containing the entire WGFX effect code.
 * - TW: 包含整個 WGFX 效果程式碼的字串。
 * @param {GPUDevice} device
 * - EN: The active WebGPU device.
 * - TW: 作用中的 WebGPU 裝置。
 * @returns {Promise<void>}
 * - EN: A Promise that resolves when compilation is complete.
 * - TW: 編譯完成時解析的 Promise。
 */
export async function compile(effectCode, device) {
    runtimeInstance = new WGFXRuntime(device);
    await runtimeInstance.compile(effectCode);
    return runtimeInstance.shaderInfo;
}

/**
 * - EN: Encodes commands to dispatch a specific compute pass in the compiled effect.
 * - TW: 編碼調度已編譯效果中特定計算通道的命令。
 * @param {string} passName
 * - EN: The name of the pass to dispatch (e.g., 'PASS_1').
 * - TW: 要調度的通道名稱 (例如：'PASS_1')。
 * @param {GPUCommandEncoder} commandEncoder
 * - EN: The command encoder for the current frame.
 * - TW: 當前幀的命令編碼器。
 */
export function dispatchPass(passName, commandEncoder) {
    if (!runtimeInstance) {
        throw new Error("WGFX runtime not initialized. Call compile() first. / WGFX 運行時未初始化。請先呼叫 compile()。");
    }
    runtimeInstance.dispatchPass(passName, commandEncoder);
}

/**
 * - EN: Updates the value of a single uniform parameter.
 * - TW: 更新單個統一參數的值。
 * @param {string} name
 * - EN: The name of the uniform to update.
 * - TW: 要更新的統一名稱。
 * @param {number} value
 * - EN: The new numeric value for the uniform.
 * - TW: 統一的新數值。
 */
export function updateUniform(name, value) {
    if (!runtimeInstance) {
        throw new Error("WGFX runtime not initialized. Call compile() first. / WGFX 運行時未初始化。請先呼叫 compile()。");
    }
    runtimeInstance.updateUniform(name, value);
}

/**
 * - EN: Gets the GPUTextureView of the final output texture.
 *   By convention, this is the texture named 'OUTPUT' in the WGFX file.
 * - TW: 獲取最終輸出紋理的 GPUTextureView。
 *   按照慣例，這是 WGFX 檔案中名為 'OUTPUT' 的紋理。
 * @returns {GPUTextureView}
 * - EN: The texture view of the final output.
 * - TW: 最終輸出的紋理視圖。
 */
export function getOutput() {
    if (!runtimeInstance) {
        throw new Error("WGFX runtime not initialized. Call compile() first. / WGFX 運行時未初始化。請先呼叫 compile()。");
    }
    return runtimeInstance.getOutput();
}
