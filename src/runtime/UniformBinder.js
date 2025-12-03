// src/runtime/UniformBinder.js

/**
 * @fileoverview
 * - EN: Handles dynamic updates of uniform buffers.
 *   This class provides a simple API for changing uniform values at runtime.
 * - TW: 處理統一緩衝區的動態更新。
 *   此類別提供一個簡單的 API，用於在運行時更改統一值。
 */

let GPUDevice;

/**
 * - EN: In a Node.js environment, native WebGPU objects don't exist.
 *   We check for their existence on the global scope. If they don't exist,
 *   `GPUDevice` will remain `undefined`, which is safe for the CLI code path
 *   as it doesn't instantiate or use any GPU-related objects.
 * - TW: 在 Node.js 環境中，原生 WebGPU 物件不存在。
 *   我們檢查它們是否在全域範圍內存在。如果不存在，
 *   `GPUDevice` 將保持 `undefined`，這對於 CLI 程式碼路徑是安全的，
 *   因為它不實例化或使用任何與 GPU 相關的物件。
 */
try {
    /**
     * - EN: This will only succeed in a browser-like environment with WebGPU support.
     * - TW: 這只會在支援 WebGPU 的瀏覽器環境中成功。
     */
    if (globalThis.GPUDevice) {
        GPUDevice = globalThis.GPUDevice;
    }
    /**
     * - EN: In Node.js or a non-WebGPU environment, this will remain undefined.
     * - TW: 在 Node.js 或非 WebGPU 環境中，這將保持未定義。
     */
} catch (e) {
    /**
     * - EN: Ignore any errors during this detection phase.
     * - TW: 在此檢測階段忽略任何錯誤。
     */
}


export class UniformBinder {
    /**
     * @param {GPUDevice} [device]
     * - EN: The active WebGPU device.
     * - TW: 作用中的 WebGPU 裝置。
     * @param {import('./ResourceManager.js').ResourceManager} resourceManager
     * - EN: An instance of the resource manager.
     * - TW: 資源管理器的實例。
     */
    constructor(device, resourceManager) {
        this.device = device || new GPUDevice();
        /**
         * - EN: Use provided device or a mock device.
         * - TW: 使用提供的裝置或模擬裝置。
         */
        this.resourceManager = resourceManager;
    }

    /**
     * - EN: Dynamically updates the value of a single uniform in the shared uniform buffer.
     * - TW: 在共享統一緩衝區中動態更新單個統一的值。
     * @param {string} name
     * - EN: The name of the uniform to update (e.g., `Strength`).
     * - TW: 要更新的統一名稱 (例如：`Strength`)。
     * @param {number} value
     * - EN: The new numeric value for the uniform.
     * - TW: 統一的新數值。
     */
    updateUniform(name, value) {
        /**
         * - EN: Retrieve uniform metadata (buffer, offset, size) from the resource manager.
         * - TW: 從資源管理器獲取統一的元資料 (緩衝區、偏移量、大小)。
         */
        const uniformInfo = this.resourceManager.getUniform(name);
        if (!uniformInfo) {
            console.warn(`Uniform '${name}' not found. Cannot update. / 找不到統一 '${name}'。無法更新。`);
            return;
        }

        const {buffer, offset, size} = uniformInfo;

        /**
         * - EN: Create a temporary typed array with the new value.
         *   This implementation simplifies things, assuming all uniforms are 4-byte floats.
         *   A more robust solution would check the parameter type from the IR.
         * - TW: 使用新值建立一個臨時的類型化陣列。
         *   此實作簡化了事情，假設所有統一都是 4 位元組浮點數。
         *   更穩健的解決方案將檢查 IR 中的參數類型。
         */
        const data = new Float32Array([value]);

        if (data.byteLength > size) {
            console.error(`Data size mismatch for uniform '${name}'. Expected ${size}, but got ${data.byteLength}. / 統一 '${name}' 的資料大小不匹配。預期為 ${size}，但得到 ${data.byteLength}。`);
            return;
        }

        /**
         * - EN: Write the data to the correct position in the GPU buffer.
         * - TW: 將資料寫入 GPU 緩衝區中的正確位置。
         */
        this.device.queue.writeBuffer(
            buffer,
            /**
             * - EN: Destination buffer.
             * - TW: 目標緩衝區。
             */
            offset,
            /**
             * - EN: Byte offset to start writing from.
             * - TW: 開始寫入的位元組偏移量。
             */
            data,
            /**
             * - EN: Source data.
             * - TW: 來源資料。
             */
            0,
            /**
             * - EN: Offset within the source data.
             * - TW: 來源資料中的偏移量。
             */
            size
            /**
             * - EN: Number of bytes to write.
             * - TW: 要寫入的位元組數。
             */
        );
    }
}
