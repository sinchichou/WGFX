import { ParameterInfo } from './types/shader';
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
    /**
     * Optional output width for upscaling effects (defaults to width).
     * @zh 選用的輸出寬度，供升頻特效使用（預設等同 width）
     */
    outWidth?: number;
    /**
     * Optional output height for upscaling effects (defaults to height).
     * @zh 選用的輸出高度，供升頻特效使用（預設等同 height）
     */
    outHeight?: number;
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
        default: number | number[];
        min: number;
        max: number;
        step: number;
        /** @zh UI 顯示標籤 */
        label?: string;
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
    public static async create({ device, effectCode, width, height, outWidth, outHeight }: WGFXOptions): Promise<WGFX> {
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
        // Output dimensions default to input dimensions; upscaling effects pass 2x.
        // 輸出尺寸預設等同輸入；升頻特效傳入 2 倍尺寸。
        // Host-provided outWidth/outHeight is an explicit override with top priority;
        // when omitted, the effect's own declared OUTPUT size (e.g. `INPUT_WIDTH * 2`)
        // is honored instead of silently collapsing to 1x.
        // 宿主明確傳入的 outWidth/outHeight 具最高優先權；未傳入時採用效果檔宣告的
        // OUTPUT 尺寸（如 `INPUT_WIDTH * 2`），而非默默退化為 1x。
        const outW = (outWidth && outWidth > 0) ? Math.floor(outWidth) : width;
        const outH = (outHeight && outHeight > 0) ? Math.floor(outHeight) : height;
        const hostOverridesOutput = (outWidth !== undefined && outWidth > 0) || (outHeight !== undefined && outHeight > 0);

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
                    size: [outW, outH],
                    format: 'rgba16float' as GPUTextureFormat,
                    usage: GPUTextureUsage.STORAGE_BINDING |
                        GPUTextureUsage.COPY_SRC |
                        GPUTextureUsage.TEXTURE_BINDING
                }
            }
        };

        try {
            await runtime.compile(effectCode, externalResources, { honorDeclaredOutput: !hostOverridesOutput });
        } catch (error: any) {
            Logger.error('WGFX compilation failed:', error);
            throw new Error(`Shader compilation error: ${error.message}`);
        }

        // Write the scene buffer once so the first dispatch has valid resolution info.
        // The size comes from the ACTUAL output texture — it may have been created at
        // the effect's declared size (e.g. 2x) when the host did not override it.
        // 初始化時先寫入場景資訊，確保首次 dispatch 即有正確的解析度資料。
        // 尺寸取自「實際」OUTPUT 紋理——宿主未覆寫時可能來自效果檔宣告（如 2x）。
        const outputTexture = runtime.resourceManager.getTexture('OUTPUT');
        const finalOutW = outputTexture ? outputTexture.width : outW;
        const finalOutH = outputTexture ? outputTexture.height : outH;
        runtime.resourceManager.updateSceneBuffer(width, height, finalOutW, finalOutH);

        const instance = new WGFX(runtime);
        instance.width = width;
        instance.height = height;
        instance.initialized = true;

        Logger.info(`WGFX initialized: ${width}x${height} -> ${finalOutW}x${finalOutH}`);
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
                type: p.type || 'float',
                default: p.default ?? 0.0,
                min: p.min ?? 0.0,
                max: p.max ?? 1.0,
                step: p.step ?? 0.01,
                label: p.label
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
            // Advance the frame counter before uploading scene info so temporal effects
            // (TAA / 時域降噪) see a monotonically increasing index that matches this
            // frame's dispatch — the value rides the same scene-buffer write below.
            // 先遞增影格計數器，使其與本幀 dispatch 一致；數值隨下方的
            // scene buffer 寫入一併上傳，不產生額外的 buffer 往返。
            this.runtime.advanceFrame();

            this.runtime.resourceManager.updateTextureFromImage('INPUT', inputSource, options.outWidth, options.outHeight);

            const commandEncoder = this.runtime.device.createCommandEncoder({
                label: 'WGFX Frame Processing'
            });

            // Hot path: dispatch every pass directly from metadata — no per-pass
            // "PASS_i" string building / regex validation per frame.
            // 熱路徑：直接以 pass 元數據依序 dispatch，免除每幀每 pass 的
            // "PASS_i" 字串組建與正規式驗證。
            this.runtime.dispatchAllPasses(commandEncoder);

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
     * Set the frame counter exposed to shaders as `scene.frame`.
     * ---
     * 設定著色器可讀取的影格計數器（`scene.frame`）。
     * 供時間性特效（TAA、時域降噪等）取得穩定的影格索引。
     *
     * @group Rendering
     * @param n - Frame index / 影格索引
     */
    public setFrame(n: number): void {
        this.runtime.setFrame(n);
    }

    /**
     * Advance the frame counter by one.
     * ---
     * 將影格計數器遞增 1；{@link process} 每次呼叫亦會自動遞增。
     *
     * @group Rendering
     * @returns The new frame counter / 遞增後的計數值
     */
    public advanceFrame(): number {
        return this.runtime.advanceFrame();
    }

    /** @group Query */
    public getFrame(): number {
        return this.runtime.getFrame();
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