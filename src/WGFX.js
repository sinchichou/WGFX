// src/WGFX.js
import {WGFXRuntime} from './runtime/WGFXRuntime.js';

/**
 * @fileoverview
 * - EN: The primary public interface for interacting with WGFX effects.
 *   It encapsulates the setup, initialization, and per-frame processing logic.
 * - TW: 用於與 WGFX 效果互動的主要公共接口。
 *   它封裝了設定、初始化和每幀處理邏輯。
 */

class WGFX {
    constructor(runtime) {
        this.runtime = runtime;
        this.initialized = false;
        this.currentInputSource = null; // 儲存當前輸入源
    }

    /**
     * 靜態工廠方法:創建並初始化 WGFX 實例
     * @param {object} options
     * @param {GPUDevice} options.device - WebGPU 裝置
     * @param {string} options.effectCode - WGFX 著色器程式碼
     * @param {number} options.width - 輸入/輸出寬度
     * @param {number} options.height - 輸入/輸出高度
     * @returns {Promise<WGFX>}
     */
    static async create({device, effectCode, width, height}) {
        if (!device) {
            throw new Error('必須提供有效的 GPUDevice');
        }
        if (!effectCode || typeof effectCode !== 'string') {
            throw new Error('必須提供有效的 effectCode 字串');
        }
        if (!width || !height || width <= 0 || height <= 0) {
            throw new Error('寬度和高度必須為正數');
        }

        // 1. 建立 Runtime 實例
        const runtime = new WGFXRuntime(device);

        // 2. 準備外部資源定義
        const externalResources = {
            // 定義常數供著色器使用
            defines: {
                INPUT_WIDTH: width,
                INPUT_HEIGHT: height
            },
            // 預先建立 INPUT 和 OUTPUT 紋理
            textures: {
                INPUT: {
                    size: [width, height],
                    format: 'rgba8unorm', // 輸入使用標準格式
                    usage: GPUTextureUsage.TEXTURE_BINDING |
                        GPUTextureUsage.COPY_DST |
                        GPUTextureUsage.RENDER_ATTACHMENT
                },
                OUTPUT: {
                    size: [width, height],
                    format: 'rgba16float', // 輸出必須使用 Storage 相容格式
                    usage: GPUTextureUsage.STORAGE_BINDING |
                        GPUTextureUsage.COPY_SRC |
                        GPUTextureUsage.TEXTURE_BINDING
                }
            }
        };

        // 3. 編譯著色器並建立管線
        try {
            await runtime.compile(effectCode, externalResources);
        } catch (error) {
            console.error('WGFX 編譯失敗:', error);
            throw new Error(`著色器編譯錯誤: ${error.message}`);
        }

        // 4. 建立並回傳實例
        const instance = new WGFX(runtime);
        instance.width = width;
        instance.height = height;
        instance.initialized = true;

        console.log(`WGFX 初始化完成: ${width}x${height}`);
        return instance;
    }

    /**
     * 方法 1: 初始化(已整合到 create 中,保留作為相容性介面)
     */
    initialize() {
        if (!this.initialized) {
            throw new Error('實例尚未初始化,請使用 WGFX.create() 建立');
        }

        const shaderInfo = this.runtime.shaderInfo;

        // 回傳元資料供 UI 使用
        return {
            width: this.width,
            height: this.height,
            uniforms: shaderInfo.parameters.map(p => ({
                name: p.name,
                type: p.type || 'f32',
                default: p.default ?? 0.0,
                min: p.min ?? 0.0,
                max: p.max ?? 1.0,
                step: p.step ?? 0.01
            })),
            passes: shaderInfo.passes.length
        };
    }

    /**
     * 方法 2: 更新著色器參數
     * @param {object} uniforms - 參數鍵值對 {參數名: 值}
     * @example
     * wgfx.updateUniforms({
     *   Strength: 0.8,
     *   Threshold: 0.5
     * });
     */
    updateUniforms(uniforms) {
        if (!this.initialized) {
            throw new Error('實例尚未初始化');
        }

        if (!uniforms || typeof uniforms !== 'object') {
            console.warn('updateUniforms: 無效的 uniforms 物件');
            return;
        }

        for (const [name, value] of Object.entries(uniforms)) {
            try {
                this.runtime.updateUniform(name, value);
            } catch (error) {
                console.warn(`更新參數 "${name}" 失敗:`, error.message);
            }
        }
    }

    /**
     * 方法 3: 處理影格(輸入並輸出)
     * @param {ImageBitmap | VideoFrame | HTMLVideoElement | HTMLCanvasElement} inputSource - 輸入影格
     * @returns {Promise<GPUTexture>} - 輸出紋理
     */
    async process(inputSource) {
        if (!this.initialized) {
            throw new Error('實例尚未初始化');
        }

        if (!inputSource) {
            throw new Error('必須提供輸入影格');
        }

        // 驗證輸入尺寸
        let sourceWidth, sourceHeight;
        if (inputSource instanceof HTMLVideoElement) {
            sourceWidth = inputSource.videoWidth;
            sourceHeight = inputSource.videoHeight;
        } else if (inputSource instanceof VideoFrame) {
            sourceWidth = inputSource.displayWidth;
            sourceHeight = inputSource.displayHeight;
        } else {
            sourceWidth = inputSource.width;
            sourceHeight = inputSource.height;
        }

        if (sourceWidth !== this.width || sourceHeight !== this.height) {
            throw new Error(
                `輸入尺寸 (${sourceWidth}x${sourceHeight}) ` +
                `與初始化尺寸 (${this.width}x${this.height}) 不符`
            );
        }

        try {
            // 1. 上傳輸入影格到 INPUT 紋理
            this.runtime.resourceManager.updateTextureFromImage('INPUT', inputSource);

            // 2. 建立命令編碼器
            const commandEncoder = this.runtime.device.createCommandEncoder({
                label: 'WGFX Frame Processing'
            });

            // 3. 依序執行所有通道
            for (const pass of this.runtime.shaderInfo.passes) {
                this.runtime.dispatchPass(`PASS_${pass.index}`, commandEncoder);
            }

            // 4. 提交 GPU 命令
            this.runtime.device.queue.submit([commandEncoder.finish()]);

            // 5. 等待 GPU 完成(可選,用於除錯)
            // await this.runtime.device.queue.onSubmittedWorkDone();

            // 6. 回傳輸出紋理
            return this.runtime.resourceManager.getTexture('OUTPUT');

        } catch (error) {
            console.error('處理影格時發生錯誤:', error);
            throw error;
        }
    }

    /**
     * 取得輸出紋理的 View(用於渲染到畫布)
     * @returns {GPUTextureView}
     */
    getOutputView() {
        return this.runtime.getOutput();
    }

    /**
     * 釋放所有 GPU 資源
     */
    dispose() {
        if (this.runtime) {
            this.runtime.resourceManager.dispose();
            this.runtime.pipelineManager.dispose();
        }
        this.initialized = false;
        console.log('WGFX 資源已釋放');
    }
}

export default WGFX;
