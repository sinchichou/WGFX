// src/runtime/UniformBinder.js

/**
 * @fileoverview 處理統一緩衝區的動態更新。
 * 此類別提供一個簡單的 API，用於在運行時更改統一值。
 */

let GPUDevice;

// In a Node.js environment, native WebGPU objects don't exist.
// We check for their existence on the global scope. If they don't exist,
// `GPUDevice` will remain `undefined`, which is safe for the CLI code path
// as it doesn't instantiate or use any GPU-related objects.
try {
    // This will only succeed in a browser-like environment with WebGPU support.
    if (globalThis.GPUDevice) {
        GPUDevice = globalThis.GPUDevice;
    }
    // In Node.js or a non-WebGPU environment, this will remain undefined.
} catch (e) {
    // Ignore any errors during this detection phase.
}

export class UniformBinder {
    /**
     * @param {GPUDevice} [device] - 作用中的 WebGPU 裝置。
     * @param {import('./ResourceManager.js').ResourceManager} resourceManager - 資源管理器的實例。
     */
    constructor(device, resourceManager) {
        this.device = device || new GPUDevice();
        this.resourceManager = resourceManager;
    }

    /**
     * 在共享統一緩衝區中動態更新單個統一的值。
     * @param {string} name - 要更新的統一名稱 (例如："Strength")。
     * @param {number} value - 統一的新數值。
     */
    updateUniform(name, value) {
        // 從資源管理器獲取統一的元資料 (緩衝區、偏移量、大小)。
        const uniformInfo = this.resourceManager.getUniform(name);
        if (!uniformInfo) {
            console.warn(`找不到統一 '${name}'。無法更新。`);
            return;
        }

        const {buffer, offset, size} = uniformInfo;

        // 使用新值建立一個臨時的類型化陣列。
        // 此實作簡化了事情，假設所有統一都是 4 位元組浮點數。
        // 更穩健的解決方案將檢查 IR 中的參數類型。
        const data = new Float32Array([value]);

        if (data.byteLength > size) {
            console.error(`統一 '${name}' 的資料大小不匹配。預期為 ${size}，但得到 ${data.byteLength}。`);
            return;
        }

        // 將資料寫入 GPU 緩衝區中的正確位置。
        this.device.queue.writeBuffer(
            buffer,   // 目標緩衝區
            offset,   // 開始寫入的位元組偏移量
            data,     // 來源資料
            0,        // 來源資料中的偏移量
            size      // 要寫入的位元組數
        );
    }
}