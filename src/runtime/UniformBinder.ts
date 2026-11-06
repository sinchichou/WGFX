import { ResourceManager } from './ResourceManager';
import { Logger } from '../utils/Logger';

/**
 * Handles the binding and updating of WebGPU uniform buffers.
 * ---
 * 處理 WebGPU Uniform 緩衝區的繫結與數據更新。
 * 負責將 CPU 端的數值同步至 GPU 端的緩衝區。
 *
 * @group Core
 * @category Managers
 */
export class UniformBinder {
    /** @zh WebGPU 裝置實例 */
    public device: GPUDevice;
    /** @zh 資源管理員，用於檢索 Uniform 元數據 */
    public resourceManager: ResourceManager;

    /**
     * Initialize the uniform binder.
     * ---
     * 初始化 Uniform 繫結器。
     *
     * @param device - The active WebGPU device / 有效的 WebGPU 裝置
     * @param resourceManager - The resource manager / 資源管理員
     */
    constructor(device: GPUDevice, resourceManager: ResourceManager) {
        this.device = device;
        this.resourceManager = resourceManager;
    }

    /**
     * Update a specific uniform value in the GPU buffer.
     * ---
     * 更新 GPU 緩衝區中特定 Uniform 的數值。
     * 自動處理單一數值或陣列型數據的封裝與傳輸。
     *
     * @group Update
     * @param name - The name of the uniform defined in shader / 著色器中定義的 Uniform 名稱
     * @param value - The new value(s) to upload / 要上傳的新數值（單一數值或數值陣列）
     *
     * @example
     * ```ts
     * binder.updateUniform('u_Brightness', 0.5);
     * binder.updateUniform('u_Color', [1.0, 0.0, 0.0]);
     * ```
     */
    public updateUniform(name: string, value: number | number[]): void {
        // Retrieve uniform metadata from manager / 從管理員獲取 Uniform 元數據
        const metadata = this.resourceManager.getUniform(name);

        if (!metadata) {
            Logger.warn(`Uniform "${name}" not found`);
            return;
        }

        const { buffer, offset, size, type } = metadata;
        const values = Array.isArray(value) ? value : [value];
        const isIntegerType = /^(int|i32|uint|u32|int2|uint2|int4|uint4)$/i.test(type || 'float');

        // Length validation / 長度驗證（vec 型別需要完整分量，純量只需 1 個）
        if (values.length > size / 4) {
            Logger.warn(`Uniform "${name}": ${values.length} components given but slot size is ${size} bytes; extra components ignored.`);
        }

        // Convert value to TypedArray based on declared type / 依宣告型別轉換為類型化陣列
        let data: Float32Array | Int32Array;
        if (isIntegerType) {
            data = new Int32Array(values.map(v => Math.round(Number(v))));
        } else {
            data = new Float32Array(values.map(v => Number(v)));
        }

        // Optimized GPU buffer write / 優化的 GPU 緩衝區寫入操作
        this.device.queue.writeBuffer(
            buffer,            // Target GPU buffer / 目標 GPU 緩衝區
            offset,            // Byte offset in buffer / 緩衝區中的位元組偏移量
            data.buffer,       // Source array buffer / 來源陣列緩衝區
            data.byteOffset,   // Offset in source data / 來源數據的起始偏移
            data.byteLength    // Total bytes to copy / 要拷貝的總位元組長度
        );
    }
}