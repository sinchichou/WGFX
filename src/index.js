// src/index.js

/**
 * @fileoverview 這是 WGFX 運行時的主要公共 API 入口點。
 * 它公開了一個簡化的、基於函數的介面，用於與底層 WGFXRuntime 類別互動，
 * 管理運行時的單例實例。
 */

import {WGFXRuntime} from './runtime/WGFXRuntime.js';

/**
 * WGFXRuntime 的單例實例。
 * @type {WGFXRuntime | null}
 */
let runtimeInstance = null;

/**
 * 初始化運行時 (如果需要) 並編譯 WGFX 效果。
 * 必須在任何其他 API 函數之前呼叫此函數。
 * @param {string} effectCode - 包含整個 WGFX 效果程式碼的字串。
 * @param {GPUDevice} device - 作用中的 WebGPU 裝置。
 * @returns {Promise<void>} 編譯完成時解析的 Promise。
 */
export async function compile(effectCode, device) {
    if (!runtimeInstance) {
        runtimeInstance = new WGFXRuntime(device);
    }
    await runtimeInstance.compile(effectCode);
}

/**
 * 編碼調度已編譯效果中特定計算通道的命令。
 * @param {string} passName - 要調度的通道名稱 (例如：'PASS_1')。
 * @param {GPUCommandEncoder} commandEncoder - 當前幀的命令編碼器。
 */
export function dispatchPass(passName, commandEncoder) {
    if (!runtimeInstance) {
        throw new Error("WGFX 運行時未初始化。請先呼叫 compile()。");
    }
    runtimeInstance.dispatchPass(passName, commandEncoder);
}

/**
 * 更新單個統一參數的值。
 * @param {string} name - 要更新的統一名稱。
 * @param {number} value - 統一的新數值。
 */
export function updateUniform(name, value) {
    if (!runtimeInstance) {
        throw new Error("WGFX 運行時未初始化。請先呼叫 compile()。");
    }
    runtimeInstance.updateUniform(name, value);
}

/**
 * 獲取最終輸出紋理的 GPUTextureView。
 * 按照慣例，這是 WGFX 檔案中名為 'OUTPUT' 的紋理。
 * @returns {GPUTextureView} 最終輸出的紋理視圖。
 */
export function getOutput() {
    if (!runtimeInstance) {
        throw new Error("WGFX 運行時未初始化。請先呼叫 compile()。");
    }
    return runtimeInstance.getOutput();
}

// TODO: 如果專案需要統一的 Node.js 入口點來處理運行時和 CLI 操作，請在此處增加 CLI 模組整合。
