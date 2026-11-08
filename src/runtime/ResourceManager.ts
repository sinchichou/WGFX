/// <reference types="@webgpu/types" />
import { WGFXShaderInfo, TextureInfo } from '../types/shader';
import { logicalTextureName, paramType } from './WGSLCodeGenerator';
import { Logger } from '../utils/Logger';

/**
 * Manages WebGPU resources including textures, views, samplers, and buffers.
 * ---
 * 管理 WebGPU 資源，包括紋理、視圖、取樣器與緩衝區的建立、查詢與釋放。
 *
 * @group Core
 * @category Managers
 */
export class ResourceManager {
    /** @zh WebGPU 裝置實例 */
    public device: GPUDevice;
    /** @zh 紋理映射表：名稱 -> GPUTexture */
    public textures: Map<string, GPUTexture>;
    /** @zh 視圖映射表：名稱 -> { 取樣視圖, 儲存視圖 } */
    public views: Map<string, { sampled: GPUTextureView; storage: GPUTextureView }>;
    /** @zh 取樣器映射表：名稱 -> GPUSampler */
    public samplers: Map<string, GPUSampler>;
    /** @zh Uniform 映射表：名稱 -> 緩衝區區段資訊 */
    public uniforms: Map<string, { buffer: GPUBuffer; offset: number; size: number; type: string }>;
    /** @zh 主要參數 Uniform 緩衝區 */
    public uniformBuffer: GPUBuffer | null;
    /** @zh 場景資訊（寬高、解析度）Uniform 緩衝區 */
    public sceneBuffer: GPUBuffer | null = null;
    /** @zh storage buffer（CNN 權重與特徵圖）名稱 → GPUBuffer */
    public storageBuffers: Map<string, GPUBuffer>;
    /** @zh storage buffer 元素 stride（bytes）名稱 → 4|8|16 */
    public storageBufferStrides: Map<string, number>;
    /** @zh 外部 defines（DISPATCH 運算式求值用） */
    public defines: Record<string, number> = {};

    /** @zh 資源版本號：任何紋理/緩衝區（重）建立時遞增，供管線層偵測綁定失效 */
    private resourceVersion = 0;
    /** @zh updateSceneBuffer 暫存寫入區（重用，避免每幀配置新 ArrayBuffer；48B 含 frame） */
    private sceneScratch = new ArrayBuffer(48);
    private sceneScratchU32 = new Uint32Array(this.sceneScratch);
    private sceneScratchF32 = new Float32Array(this.sceneScratch);
    /** @zh 目前影格計數器（寫入 SceneInfo.frame，供時間性特效使用） */
    private frameCounter = 0;

    /** @zh 目前資源版本號 / Current resource revision for bind-group cache invalidation */
    public get revision(): number {
        return this.resourceVersion;
    }

    private bumpRevision(): void {
        this.resourceVersion++;
    }

    /**
     * Initialize the resource manager.
     * ---
     * 初始化資源管理員。
     *
     * @param device - The active WebGPU device / 有效的 WebGPU 裝置
     */
    constructor(device: GPUDevice) {
        if (!device) {
            throw new Error('ResourceManager requires a valid GPUDevice');
        }
        this.device = device;
        this.textures = new Map();
        this.views = new Map();
        this.samplers = new Map();
        this.uniforms = new Map();
        this.storageBuffers = new Map();
        this.storageBufferStrides = new Map();
        this.uniformBuffer = null;
    }

    /**
     * Initialize resources based on shader info and external definitions.
     * ---
     * 根據著色器元數據與外部定義初始化所有必要的 GPU 資源。
     *
     * @group Lifecycle
     * @param shaderInfo - Parsed shader metadata / 解析後的著色器資訊
     * @param externalResources - External resource definitions / 外部資源定義
     * @param options - Initialization options / 初始化選項
     * @param options.honorDeclaredOutput - When true, an effect-declared explicit
     *   WIDTH/HEIGHT for an already-provided external texture (e.g. OUTPUT declared
     *   as `INPUT_WIDTH * 2`) overrides the external size. INPUT is always
     *   host-owned (media size) and never overridden. Low-level compile() keeps the
     *   default (external wins) contract.
     *   為 true 時，效果檔對已存在外部紋理的顯式 WIDTH/HEIGHT 宣告（如 `INPUT_WIDTH * 2`）
     *   優先於外部尺寸；INPUT 恆由宿主（媒體尺寸）決定。低階 compile() 預設維持
     *   「外部資源優先」契約。
     */
    public initialize(shaderInfo: WGFXShaderInfo, externalResources: any = {}, options: { honorDeclaredOutput?: boolean } = {}): void {
        const context: Record<string, number> = {};

        // 1. Load external defines / 載入外部巨集定義
        if (externalResources.defines) {
            Object.assign(context, externalResources.defines);
            this.defines = { ...context }; // 保留供 //! DISPATCH 運算式求值
            Logger.debug('Loaded external defines:', context);
        }

        // 2. Create external textures / 建立外部指定紋理
        if (externalResources.textures) {
            for (const [name, descriptor] of Object.entries(externalResources.textures)) {
                Logger.debug(`Creating external texture: ${name}`, descriptor);
                this.createTexture(name, descriptor as GPUTextureDescriptor);
            }
        }

        // 2b. Create external storage buffers / 建立外部 storage buffer
        //     OpenNR/CNN 類特效：權重與特徵圖以 storage buffer 餵入。
        //     每項：{ size: bytes, stride: 4|8|16, usage?, init?: ArrayBuffer }
        if (externalResources.buffers) {
            for (const [name, desc] of Object.entries(externalResources.buffers)) {
                const d = desc as { size: number; stride?: number; init?: ArrayBuffer };
                this.createStorageBuffer(name, d.size, d.stride || 8);
                if (d.init) this.writeStorageBuffer(name, d.init);
                Logger.debug(`Created external storage buffer: ${name} (${d.size} bytes)`);
            }
        }

        /**
         * Recursive helper to evaluate numeric expressions from shader strings.
         * ---
         * 遞迴輔助函式：解析並計算著色器定義中的數值運算式。
         */
        const evaluate = (expr: string | number, ctx: Record<string, number>): number => {
            if (typeof expr === 'number') return expr;
            if (typeof expr !== 'string') {
                throw new Error('expression missing');
            }
            expr = expr.replace(/\s+/g, ''); // Remove spaces / 移除空白
            let pos = 0;

            // Internal parser steps / 內部解析步驟
            const parseNumber = (): number => {
                let startKey = pos;
                if (/[a-zA-Z_]/.test(expr[pos])) { // Handle variables / 處理變數
                    while (pos < expr.length && /[a-zA-Z0-9_]/.test(expr[pos])) pos++;
                    const key = expr.slice(startKey, pos);
                    if (key in ctx) return ctx[key];
                    throw new Error(`Undefined variable: ${key}`);
                } else if (/[0-9.]/.test(expr[pos])) { // Handle constants / 處理常數
                    while (pos < expr.length && /[0-9.]/.test(expr[pos])) pos++;
                    return parseFloat(expr.slice(startKey, pos));
                } else if (expr[pos] === '(') { // Handle parenthesis / 處理括號
                    pos++;
                    const val = parseExpression();
                    pos++; // skip ')'
                    return val;
                }
                throw new Error(`Unexpected character at ${pos}: ${expr[pos]}`);
            };

            const parseFactor = (): number => {
                let val = parseNumber();
                while (pos < expr.length && (expr[pos] === '*' || expr[pos] === '/')) {
                    const op = expr[pos++];
                    const right = parseNumber();
                    val = (op === '*') ? val * right : val / right;
                }
                return val;
            };

            const parseExpression = (): number => {
                let val = parseFactor();
                while (pos < expr.length && (expr[pos] === '+' || expr[pos] === '-')) {
                    const op = expr[pos++];
                    const right = parseFactor();
                    val = (op === '+') ? val + right : val - right;
                }
                return val;
            };

            try {
                return parseExpression();
            } catch (e: any) {
                Logger.warn(`Failed to evaluate expression: ${expr}`, e.message);
                return 0;
            }
        };

        // 3. Create intermediate textures / 建立中間渲染紋理
        //    Magpie 風格的 `tex1_sampled` / `tex1_storaged` 成對宣告代表**同一個**邏輯紋理，
        //    主機端只配置一塊 GPUTexture，並提供 sampled/storage 兩種 view。
        const createdLogical = new Set<string>();
        shaderInfo.textures.forEach((tex: TextureInfo) => {
            const logicalName = logicalTextureName(tex.name);
            if (this.textures.has(logicalName) || createdLogical.has(logicalName)) {
                // Effect-declared explicit size wins for non-INPUT external textures
                // when honorDeclaredOutput is set（見上方 options 說明）。
                if (
                    options.honorDeclaredOutput === true &&
                    this.textures.has(logicalName) &&
                    logicalName !== 'INPUT'
                ) {
                    const existing = this.textures.get(logicalName)!;
                    try {
                        const declaredW = evaluate(tex.width as string | number, context);
                        const declaredH = evaluate(tex.height as string | number, context);
                        if (declaredW > 0 && declaredH > 0 && (declaredW !== existing.width || declaredH !== existing.height)) {
                            Logger.info(
                                `Texture "${logicalName}": external size ${existing.width}x${existing.height} ` +
                                `overridden by effect declaration ${declaredW}x${declaredH}`
                            );
                            // Recreate with the host's usage/format (host owns those); only the
                            // declared geometry wins. 以宿主的 usage/format 重建，僅尺寸依宣告。
                            this.createTexture(logicalName, {
                                size: [declaredW, declaredH],
                                format: existing.format,
                                usage: existing.usage
                            });
                        }
                    } catch {
                        // Declaration has no/explicit-unresolvable WIDTH/HEIGHT (e.g. plain
                        // `Texture2D OUTPUT`) → keep the external size. 宣告缺漏時保留外部尺寸。
                    }
                } else {
                    Logger.debug(`Skipping duplicate texture declaration: ${tex.name} -> ${logicalName}`);
                }
                return;
            }
            createdLogical.add(logicalName);

            let width: number;
            let height: number;
            try {
                width = evaluate(tex.width as string | number, context);
                height = evaluate(tex.height as string | number, context);
            } catch (e: any) {
                throw new Error(
                    `Texture "${tex.name}": WIDTH/HEIGHT missing or unresolvable (declare //!WIDTH///!HEIGHT or provide the texture via externalResources)`
                );
            }

            if (!width || !height) {
                throw new Error(`Failed to determine size for texture ${tex.name}`);
            }

            // Format: declared FORMAT wins; INPUT defaults to rgba8unorm, others to rgba16float.
            // 格式：優先使用宣告的 FORMAT；INPUT 預設 rgba8unorm，其餘預設高精度浮點。
            const format: GPUTextureFormat = (tex.format as GPUTextureFormat)
                || (logicalName === 'INPUT' ? 'rgba8unorm' : 'rgba16float');

            const descriptor: GPUTextureDescriptor = {
                size: [width, height],
                format: format,
                usage: GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.STORAGE_BINDING |
                    GPUTextureUsage.COPY_DST |
                    GPUTextureUsage.COPY_SRC
            };

            this.createTexture(logicalName, descriptor);
            Logger.debug(`Created texture: ${tex.name} -> ${logicalName} (${width}x${height}, ${format})`);
        });

        // 4. Create samplers / 建立取樣器
        shaderInfo.samplers.forEach(samp => {
            const descriptor: GPUSamplerDescriptor = {
                magFilter: samp.filter === 'LINEAR' ? 'linear' : 'nearest',
                minFilter: samp.filter === 'LINEAR' ? 'linear' : 'nearest',
                addressModeU: samp.address === 'WRAP' ? 'repeat' : 'clamp-to-edge',
                addressModeV: samp.address === 'WRAP' ? 'repeat' : 'clamp-to-edge',
            };
            this.createSampler(samp.name, descriptor);
            Logger.debug(`Created sampler: ${samp.name}`);
        });

        // 5. Create uniform buffer for parameters / 建立參數 Uniform 緩衝區
        if (shaderInfo.parameters.length > 0) {
            this.createUniformBuffer(shaderInfo.parameters);
            Logger.debug(`Created UniformBuffer for ${shaderInfo.parameters.length} parameters`);
        }

        // 6. Create Scene Buffer / 建立場景資訊緩衝區
        this.createSceneBuffer();
    }

    /**
     * Create or update a texture and its dual views (Sampled & Storage).
     * ---
     * 建立或更新紋理及其雙重視圖。若名稱重複則先銷毀舊有資源。
     *
     * @group Management
     * @param name - Unique texture name / 紋理唯一名稱
     * @param descriptor - WebGPU texture descriptor / 紋理描述符
     * @returns The newly created {@link GPUTexture}
     */
    public createTexture(name: string, descriptor: GPUTextureDescriptor): GPUTexture {
        if (this.textures.has(name)) {
            this.textures.get(name)!.destroy(); // Clean up old texture / 清理舊紋理
        }

        const texture = this.device.createTexture(descriptor);
        texture.label = name;
        this.textures.set(name, texture);

        // Create both views for flexible pipeline usage / 建立兩種視圖供不同管線階段使用
        this.views.set(name, {
            sampled: texture.createView({ label: `${name}_sampled` }),
            storage: texture.createView({ label: `${name}_storage` })
        });

        // Old views/texture replaced → cached bind groups must be rebuilt
        // 舊視圖/紋理已被替換 → 管線層快取的 bind group 必須重建
        this.bumpRevision();

        return texture;
    }

    /**
     * Get a specific view of a texture by name and type.
     * ---
     * 獲取紋理的特定視圖（取樣用或儲存用）。
     *
     * @group Query
     * @param name - Texture name / 紋理名稱
     * @param type - View type: 'sampled' (for texture()) or 'storage' (for imageStore())
     * @returns The requested {@link GPUTextureView}
     */
    public getTextureView(name: string, type: 'sampled' | 'storage' = 'sampled'): GPUTextureView {
        // Resolve sampled/storaged alias to the logical texture / 解析成對別名至邏輯紋理
        const v = this.views.get(name) || this.views.get(logicalTextureName(name));
        if (!v) {
            throw new Error(`View "${name}" not found`);
        }
        return v[type];
    }

    /**
     * Get a GPUTexture object by name.
     * ---
     * 根據名稱獲取 GPUTexture 物件。`X_sampled`/`X_storaged` 別名會解析至邏輯紋理 `X`。
     *
     * @group Query
     * @param name - Texture name / 紋理名稱
     */
    public getTexture(name: string): GPUTexture | undefined {
        return this.textures.get(name) ?? this.textures.get(logicalTextureName(name));
    }

    /**
     * Create and register a sampler.
     * ---
     * 建立並註冊取樣器。
     *
     * @group Management
     */
    public createSampler(name: string, descriptor: GPUSamplerDescriptor): GPUSampler {
        const sampler = this.device.createSampler(descriptor);
        sampler.label = name;
        this.samplers.set(name, sampler);
        return sampler;
    }

    /**
     * Get a GPUSampler object by name.
     * ---
     * 根據名稱獲取 GPUSampler 物件。
     *
     * @group Query
     */
    public getSampler(name: string): GPUSampler | undefined {
        return this.samplers.get(name);
    }

    /**
     * Create a storage buffer（CNN 權重與特徵圖）。
     * ---
     * USAGE 固定 STORAGE_BINDING | COPY_DST | COPY_SRC（COPY_SRC 供讀回驗證）。
     *
     * @group Management
     * @param name - Buffer name / 緩衝區名稱
     * @param size - Size in bytes / 大小（bytes）
     * @param stride - Element stride in bytes (4|8|16) / 元素 stride
     */
    public createStorageBuffer(name: string, size: number, stride: number = 8): GPUBuffer {
        if (this.storageBuffers.has(name)) {
            this.storageBuffers.get(name)!.destroy();
        }
        // WebGPU 要求綁定大小為 4 的倍數
        const aligned = Math.max(4, Math.ceil(size / 4) * 4);
        const buffer = this.device.createBuffer({
            size: aligned,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            label: name
        });
        this.storageBuffers.set(name, buffer);
        this.storageBufferStrides.set(name, stride);
        this.bumpRevision();
        return buffer;
    }

    /** @group Query */
    public getStorageBuffer(name: string): GPUBuffer | undefined {
        return this.storageBuffers.get(name);
    }

    /** 寫入 storage buffer 內容（queue.writeBuffer） */
    public writeStorageBuffer(name: string, data: ArrayBuffer | ArrayBufferView): void {
        const b = this.storageBuffers.get(name);
        if (!b) throw new Error(`Storage buffer "${name}" not found`);
        this.device.queue.writeBuffer(b, 0, data as ArrayBuffer);
    }

    /** 讀回 storage buffer 內容（staging mapRead；供驗證/除錯） */
    public async readStorageBuffer(name: string): Promise<ArrayBuffer> {
        const b = this.storageBuffers.get(name);
        if (!b) throw new Error(`Storage buffer "${name}" not found`);
        const staging = this.device.createBuffer({
            size: b.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const cmd = this.device.createCommandEncoder();
        cmd.copyBufferToBuffer(b, 0, staging, 0, b.size);
        this.device.queue.submit([cmd.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const out = staging.getMappedRange().slice(0);
        staging.unmap();
        staging.destroy();
        return out;
    }

    /**
     * Get uniform buffer segment info for a specific parameter.
     * ---
     * 獲取特定 Uniform 項目的緩衝區位址資訊。
     *
     * @group Query
     */
    public getUniform(name: string): { buffer: GPUBuffer; offset: number; size: number; type: string } | undefined {
        return this.uniforms.get(name);
    }

    /**
     * Update a texture's content from an external image source.
     * ---
     * 從外部影像來源（如 Video 或 Canvas）更新紋理內容，並同步更新場景資訊。
     *
     * @group Update
     * @param name - Target texture name (usually 'INPUT') / 目標紋理名稱
     * @param source - Source visual element / 來源影像元素
     */
    public updateTextureFromImage(name: string, source: ImageBitmap | HTMLVideoElement | HTMLCanvasElement | VideoFrame | OffscreenCanvas, outWidth?: number, outHeight?: number): void {
        const texture = this.textures.get(name);
        if (!texture) {
            Logger.warn(`Texture ${name} not found`);
            return;
        }

        // Optimized GPU copy / 優化後的 GPU 內容拷貝
        this.device.queue.copyExternalImageToTexture(
            { source: source },
            { texture: texture },
            [texture.width, texture.height]
        );

        // Update scene info if INPUT changes / 若輸入源改變，更新場景解析度資訊
        if (name === 'INPUT' && (source instanceof HTMLVideoElement || source instanceof ImageBitmap || source instanceof VideoFrame)) {
            const width = (source instanceof VideoFrame) ? source.displayWidth : source.width;
            const height = (source instanceof VideoFrame) ? source.displayHeight : source.height;

            // Use provided output dimensions or fallback to OUTPUT texture dimensions / 使用提供的輸出維度或獲取輸出紋理維度
            let outW = outWidth;
            let outH = outHeight;

            if (outW === undefined || outH === undefined) {
                const output = this.textures.get('OUTPUT');
                outW = output ? output.width : width;
                outH = output ? output.height : height;
            }

            this.updateSceneBuffer(width, height, outW, outH);
        }
    }

    /** @group Query */
    public getUniformBuffer(): GPUBuffer | null {
        return this.uniformBuffer;
    }

    /**
     * Create the scene info uniform buffer (Width, Height, InvWidth, InvHeight).
     * ---
     * 建立場景資訊緩衝區（包含寬高及其倒數）。
     *
     * 大小 48 bytes：前 40 bytes 為既有欄位（向後相容），尾端新增 `frame: u32` 與
     * 一個 f32 pad，使結構大小符合 WGSL uniform 的 16-byte 倍數要求。
     *
     * @group Management
     */
    public createSceneBuffer(): void {
        if (this.sceneBuffer) return;
        this.sceneBuffer = this.device.createBuffer({
            size: 48, // SceneInfo struct: 40B legacy + frame(u32) + pad → 48B
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'SceneInfo'
        });
        this.bumpRevision();
    }

    /**
     * Update scene buffer with current source dimensions.
     * ---
     * 以目前的影像尺寸更新場景資訊緩衝區。
     *
     * `frame` 為選用參數（向後相容）：未提供時沿用目前內部計數器
     * （由 {@link setFrame} / {@link advanceFrame} 維護，預設 0），
     * 既有呼叫端無需修改即可維持原行為。
     *
     * @group Update
     * @param frame - Optional frame counter / 選用的影格計數器
     */
    public updateSceneBuffer(width: number, height: number, outWidth: number = 0, outHeight: number = 0, frame?: number): void {
        if (!this.sceneBuffer || width === 0 || height === 0) return;

        if (frame !== undefined) this.frameCounter = frame >>> 0;

        if (outWidth === 0) outWidth = width;
        if (outHeight === 0) outHeight = height;

        // Reuse the persistent scratch views（重用暫存視圖，熱路徑零配置）
        const u32 = this.sceneScratchU32;
        const f32 = this.sceneScratchF32;

        // inputSize: uint2 (8 bytes)
        u32[0] = width;
        u32[1] = height;

        // inputPt: MF2 (8 bytes)
        f32[2] = 1.0 / width;
        f32[3] = 1.0 / height;

        // outputSize: uint2 (8 bytes)
        u32[4] = outWidth;
        u32[5] = outHeight;

        // outputPt: MF2 (8 bytes)
        f32[6] = 1.0 / outWidth;
        f32[7] = 1.0 / outHeight;

        // scale: MF2 (8 bytes)
        f32[8] = outWidth / width;
        f32[9] = outHeight / height;

        // frame: u32 @40 + _pad0: f32 @44（尾端新增欄位，前 40 bytes 佈局不變）
        u32[10] = this.frameCounter;
        f32[11] = 0;

        this.device.queue.writeBuffer(this.sceneBuffer, 0, this.sceneScratch);
        Logger.debug(`Updated scene buffer: IN(${width}x${height}), OUT(${outWidth}x${outHeight}), SCALE(${f32[8].toFixed(2)}, ${f32[9].toFixed(2)})`);
    }

    /** @group Query */
    public getSceneBuffer(): GPUBuffer | null {
        return this.sceneBuffer;
    }

    /**
     * Set the frame counter written into SceneInfo.frame.
     * ---
     * 設定寫入 SceneInfo.frame 的影格計數器。
     *
     * 僅更新內部計數值，實際 GPU 寫入發生於下一次 {@link updateSceneBuffer}
     * （通常由宿主每幀觸發），避免額外一次 writeBuffer 往返。
     *
     * @group Update
     * @param n - Frame index / 影格索引
     */
    public setFrame(n: number): void {
        const v = Number(n);
        this.frameCounter = Number.isFinite(v) ? (Math.max(0, Math.floor(v)) >>> 0) : 0;
    }

    /**
     * Advance the frame counter by one.
     * ---
     * 將影格計數器遞增 1（u32 環繞，供時間性特效取用）。
     *
     * @group Update
     * @returns The new frame counter / 遞增後的計數值
     */
    public advanceFrame(): number {
        this.frameCounter = (this.frameCounter + 1) >>> 0;
        return this.frameCounter;
    }

    /** @group Query */
    public getFrame(): number {
        return this.frameCounter;
    }

    /**
     * Dispose all textures and buffers to prevent memory leaks.
     * ---
     * 釋放所有紋理與緩衝區，防止 GPU 記憶體洩漏。
     *
     * @group Lifecycle
     */
    public dispose(): void {
        this.textures.forEach(t => t.destroy());
        this.textures.clear();
        this.views.clear();
        this.samplers.clear();
        this.storageBuffers.forEach(b => b.destroy());
        this.storageBuffers.clear();
        this.storageBufferStrides.clear();

        if (this.uniformBuffer) {
            this.uniformBuffer.destroy();
            this.uniformBuffer = null;
        }
        if (this.sceneBuffer) {
            this.sceneBuffer.destroy();
            this.sceneBuffer = null;
        }
        this.uniforms.clear();
        // All GPU resources are gone → any cached bind group is stale
        // 所有 GPU 資源已釋放 → 任何快取的 bind group 皆已失效
        this.bumpRevision();
        Logger.info('ResourceManager: All resources disposed');
    }

    /**
     * Create the main uniform buffer and map parameter offsets.
     * ---
     * 建立主要的 Uniform 緩衝區，並計算各參數在緩衝區中的偏移量。
     * 每個參數佔用一個 16-byte slot，與 WGSLCodeGenerator._generateUniformStruct
     * 產生的 struct 佈局（參數欄位 + pad 補滿 16 bytes）完全一致。
     *
     * @internal
     */
    private createUniformBuffer(parameters: any[]): void {
        let offset = 0;
        parameters.forEach(p => {
            const { size } = paramType(p.type); // Actual byte size from declared type / 依宣告型別計算位元組數
            this.uniforms.set(p.name, { buffer: null as any, offset, size, type: p.type || 'float' });
            offset += 16; // Each parameter occupies a 16-byte slot / 每個參數佔用 16-byte slot
        });

        const bufferSize = Math.max(offset, 16);
        this.uniformBuffer = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'GlobalUniforms'
        });

        // Link all parameters to this single buffer / 將所有參數關聯至此單一緩衝區
        this.uniforms.forEach(u => u.buffer = this.uniformBuffer!);

        // Upload declared default values so the first dispatch uses them.
        // 上傳宣告的預設值，確保首次 dispatch 即使用預設參數。
        const initial = new ArrayBuffer(bufferSize);
        const f32 = new Float32Array(initial);
        const i32 = new Int32Array(initial);
        parameters.forEach(p => {
            const meta = this.uniforms.get(p.name);
            if (!meta) return;
            const defaults: number[] = Array.isArray(p.default) ? p.default : [p.default];
            const isIntegerType = /^(int|i32|uint|u32|int2|uint2|int4|uint4)$/i.test(meta.type);
            defaults.forEach((v: number, i: number) => {
                if (i * 4 >= meta.size) return;
                if (isIntegerType) i32[(meta.offset >> 2) + i] = Math.round(Number(v) || 0);
                else f32[(meta.offset >> 2) + i] = Number(v) || 0;
            });
        });
        this.device.queue.writeBuffer(this.uniformBuffer, 0, initial);
        this.bumpRevision();
    }
}