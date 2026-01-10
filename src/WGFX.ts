import { WGFXShaderInfo, ParameterInfo } from './types/shader';
import { WGFXRuntime } from './runtime/WGFXRuntime';
import { Logger, LogLevel } from './utils/Logger';

/**
 * Options for creating a WGFX instance.
 * ---
 * 建立 WGFX 實例的配置選項。
 *
 * @category Interfaces
 */
export interface WGFXOptions {
    /**
     * The WebGPU device to use.
     * @zh 使用的 WebGPU 裝置
     */
    device: GPUDevice;
    /**
     * The WGSL or custom effect code string.
     * @zh WGSL 或自定義特效代碼字串
     */
    effectCode: string;
    /** @zh 初始處理寬度 */
    width: number;
    /** @zh 初始處理高度 */
    height: number;
    /** @zh 選用的外部資源 (textures, samplers, etc.) */
    externalResources?: any;
}

/**
 * Information about the loaded WGFX effect metadata.
 * ---
 * 已載入的 WGFX 特效元數據資訊。
 *
 * @category Interfaces
 */
export interface WGFXInfo {
    /** @zh 處理寬度 */
    width: number;
    /** @zh 處理高度 */
    height: number;
    /**
     * List of adjustable uniform parameters.
     * @zh 可調整的 Uniform 參數列表
     */
    uniforms: {
        name: string;
        type: string;
        /** @defaultValue 0.0 */
        default: number;
        min: number;
        max: number;
        step: number;
    }[];
    /** @zh 渲染通道總數 */
    passes: number;
}

/**
 * Main WGFX class for handling WebGPU graphics effects.
 * ---
 * 處理 WebGPU 圖形特效的主要控制器。
 *
 * @example
 * ```ts
 * const wgfx = await WGFX.create({ device, effectCode, width: 1280, height: 720 });
 * const output = await wgfx.process(videoElement);
 * // 渲染完成後關閉
 * wgfx.dispose();
 * ```
 */
export class WGFX {
    /** @internal 底層運行時控制器 */
    public runtime: WGFXRuntime;
    /** @zh 實例是否已初始化完成 */
    public initialized: boolean;
    /** @zh 目前處理寬度 */
    public width: number;
    /** @zh 目前處理高度 */
    public height: number;
    /** @zh 目前輸入源物件 */
    public currentInputSource: any;

    /**
     * @internal
     * Internal constructor. Use {@link WGFX.create} instead.
     * @param runtime - WGFXRuntime instance
     */
    constructor(runtime: WGFXRuntime) {
        this.runtime = runtime;
        this.initialized = false;
        this.currentInputSource = null;
        this.width = 0;
        this.height = 0;
    }

    /**
     * Enable or disable debug mode globally.
     * ---
     * 全域啟用或停用偵錯模式。
     *
     * @group Configuration
     * @param enabled - Whether to enable debug mode / 是否啟用偵錯模式
     */
    public static setDebug(enabled: boolean): void {
        Logger.setDebug(enabled);
    }

    /**
     * Set the current log level globally.
     * ---
     * 全域設定目前日誌層級。
     *
     * @group Configuration
     * @param level - The target {@link LogLevel} / 目標日誌層級
     */
    public static setLevel(level: LogLevel): void {
        Logger.setLevel(level);
    }

    /**
     * Factory method to create and initialize a WGFX instance.
     * ---
     * 建立並初始化 WGFX 實例的工廠方法。
     *
     * @group Lifecycle
     * @param options - Configuration options / 配置選項
     * @returns A promise that resolves to a WGFX instance
     * @throws {Error} 如果 WebGPU 裝置、代碼無效或編譯失敗時拋出錯誤
     */
    public static async create({ device, effectCode, width, height }: WGFXOptions): Promise<WGFX> {
        if (!device) {
            const error = 'Must provide a valid GPUDevice';
            Logger.error(error);
            throw new Error(error);
        }
        if (!effectCode || typeof effectCode !== 'string') {
            const error = 'Must provide valid effectCode string';
            Logger.error(error);
            throw new Error(error);
        }
        if (!width || !height || width <= 0 || height <= 0) {
            const error = 'Width and height must be positive numbers';
            Logger.error(error);
            throw new Error(error);
        }

        const runtime = new WGFXRuntime(device);

        const externalResources = {
            defines: {
                INPUT_WIDTH: width,
                INPUT_HEIGHT: height
            },
            textures: {
                INPUT: {
                    size: [width, height],
                    format: 'rgba8unorm' as GPUTextureFormat,
                    usage: GPUTextureUsage.TEXTURE_BINDING |
                        GPUTextureUsage.COPY_DST |
                        GPUTextureUsage.RENDER_ATTACHMENT
                },
                OUTPUT: {
                    size: [width, height],
                    format: 'rgba16float' as GPUTextureFormat,
                    usage: GPUTextureUsage.STORAGE_BINDING |
                        GPUTextureUsage.COPY_SRC |
                        GPUTextureUsage.TEXTURE_BINDING
                }
            }
        };

        try {
            await runtime.compile(effectCode, externalResources);
        } catch (error: any) {
            Logger.error('WGFX compilation failed:', error);
            throw new Error(`Shader compilation error: ${error.message}`);
        }

        const instance = new WGFX(runtime);
        instance.width = width;
        instance.height = height;
        instance.initialized = true;

        Logger.info(`WGFX initialized: ${width}x${height}`);
        return instance;
    }

    /**
     * Get information about the initialized effect.
     * ---
     * 獲取已初始化特效的相關資訊（如參數列表、寬高）。
     *
     * @group Metadata
     * @returns Metadata about the current effect / 目前特效的元數據
     * @throws {Error} 若實例尚未初始化則拋出錯誤
     */
    public initialize(): WGFXInfo {
        if (!this.initialized) {
            const error = 'Instance not initialized, use WGFX.create()';
            Logger.error(error);
            throw new Error(error);
        }

        const shaderInfo = this.runtime.shaderInfo;
        if (!shaderInfo) {
            const error = "Shader info missing";
            Logger.error(error);
            throw new Error(error);
        }

        return {
            width: this.width,
            height: this.height,
            uniforms: shaderInfo.parameters.map((p: ParameterInfo) => ({
                name: p.name,
                type: p.type || 'f32',
                default: (Array.isArray(p.default) ? p.default[0] : p.default) ?? 0.0,
                min: p.min ?? 0.0,
                max: p.max ?? 1.0,
                step: p.step ?? 0.01
            })),
            passes: shaderInfo.passes.length
        };
    }

    /**
     * Update shader uniform values.
     * ---
     * 更新 Uniform 數值，可用於即時調整特效參數。
     *
     * @group Rendering
     * @param uniforms - Key-value pairs of uniform names and values / 名稱與數值的鍵值對
     */
    public updateUniforms(uniforms: Record<string, number | number[]>): void {
        if (!this.initialized) {
            Logger.warn('updateUniforms: Instance not initialized');
            return;
        }

        if (!uniforms || typeof uniforms !== 'object') {
            Logger.warn('updateUniforms: Invalid uniforms object');
            return;
        }

        for (const [name, value] of Object.entries(uniforms)) {
            try {
                this.runtime.updateUniform(name, value);
            } catch (error: any) {
                Logger.warn(`Failed to update uniform "${name}": ${error.message}`);
            }
        }
    }

    /**
     * Process an input source and return the output texture.
     * ---
     * 處理輸入源並回傳輸出紋理。支援多種影像來源。
     *
     * @group Rendering
     * @param inputSource - The image/video source to process / 要處理的影像來源
     * @param options - Optional parameters including target output dimensions / 選用參數，包含目標輸出尺寸
     * @returns A promise that resolves to the output {@link GPUTexture}
     * @throws {Error} 當輸入來源維度與初始化不符時拋出錯誤
     */
    public async process(
        inputSource: ImageBitmap | VideoFrame | HTMLVideoElement | HTMLCanvasElement,
        options: { outWidth?: number; outHeight?: number } = {}
    ): Promise<GPUTexture> {
        if (!this.initialized) {
            const error = 'Instance not initialized';
            Logger.error(error);
            throw new Error(error);
        }

        if (!inputSource) {
            const error = 'Must provide input source';
            Logger.error(error);
            throw new Error(error);
        }

        let sourceWidth: number = 0;
        let sourceHeight: number = 0;

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
            const error = `Input dimensions (${sourceWidth}x${sourceHeight}) do not match initialized dimensions (${this.width}x${this.height})`;
            Logger.error(error);
            throw new Error(error);
        }

        try {
            this.runtime.resourceManager.updateTextureFromImage('INPUT', inputSource, options.outWidth, options.outHeight);

            const commandEncoder = this.runtime.device.createCommandEncoder({
                label: 'WGFX Frame Processing'
            });

            if (this.runtime.shaderInfo) {
                for (const pass of this.runtime.shaderInfo.passes) {
                    this.runtime.dispatchPass(`PASS_${pass.index}`, commandEncoder);
                }
            }

            this.runtime.device.queue.submit([commandEncoder.finish()]);

            const output = this.runtime.resourceManager.getTexture('OUTPUT');
            if (!output) {
                const error = "Output texture missing";
                Logger.error(error);
                throw new Error(error);
            }
            return output;

        } catch (error) {
            Logger.error('Frame processing error:', error);
            throw error;
        }
    }

    /**
     * Get the texture view of the final output.
     * ---
     * 獲取最終輸出的紋理視圖 (TextureView)。
     *
     * @group Rendering
     * @returns The output {@link GPUTextureView}
     */
    public getOutputView(): GPUTextureView {
        return this.runtime.getOutput();
    }

    /**
     * Dispose all resources and clean up.
     * ---
     * 釋放所有 WebGPU 資源並清理內部管理器，防止記憶體洩漏。
     *
     * @group Lifecycle
     */
    public dispose(): void {
        if (this.runtime) {
            this.runtime.resourceManager.dispose();
            this.runtime.pipelineManager.dispose();
        }
        this.initialized = false;
        Logger.info('WGFX resources disposed');
    }
}

export default WGFX;