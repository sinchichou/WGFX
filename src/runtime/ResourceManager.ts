/// <reference types="@webgpu/types" />
import { WGFXShaderInfo, TextureInfo } from '../types/shader';
import {Logger} from '../utils/Logger';

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
    public uniforms: Map<string, { buffer: GPUBuffer; offset: number; size: number }>;
    /** @zh 主要參數 Uniform 緩衝區 */
    public uniformBuffer: GPUBuffer | null;
    /** @zh 場景資訊（寬高、解析度）Uniform 緩衝區 */
    public sceneBuffer: GPUBuffer | null = null;

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
     */
    public initialize(shaderInfo: WGFXShaderInfo, externalResources: any = {}): void {
        const context: Record<string, number> = {};

        // 1. Load external defines / 載入外部巨集定義
        if (externalResources.defines) {
            Object.assign(context, externalResources.defines);
            Logger.debug('Loaded external defines:', context);
        }

        // 2. Create external textures / 建立外部指定紋理
        if (externalResources.textures) {
            for (const [name, descriptor] of Object.entries(externalResources.textures)) {
                Logger.debug(`Creating external texture: ${name}`, descriptor);
                this.createTexture(name, descriptor as GPUTextureDescriptor);
            }
        }

        /**
         * Recursive helper to evaluate numeric expressions from shader strings.
         * ---
         * 遞迴輔助函式：解析並計算著色器定義中的數值運算式。
         */
        const evaluate = (expr: string | number, ctx: Record<string, number>): number => {
            if (typeof expr === 'number') return expr;

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
        shaderInfo.textures.forEach((tex: TextureInfo) => {
            if (this.textures.has(tex.name)) {
                Logger.debug(`Skipping existing texture: ${tex.name}`);
                return;
            }

            const width = evaluate(tex.width, context);
            const height = evaluate(tex.height, context);

            if (!width || !height) {
                throw new Error(`Failed to determine size for texture ${tex.name}`);
            }

            // Define format: INPUT uses rgba8, others use rgba16float / 定義格式：輸入用 rgba8，其餘用高精度浮點
            const format: GPUTextureFormat = (tex.name === 'INPUT' ? 'rgba8unorm' : 'rgba16float');

            const descriptor: GPUTextureDescriptor = {
                size: [width, height],
                format: format,
                usage: GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.STORAGE_BINDING |
                    GPUTextureUsage.COPY_DST |
                    GPUTextureUsage.COPY_SRC
            };

            this.createTexture(tex.name, descriptor);
            Logger.debug(`Created texture: ${tex.name} (${width}x${height}, ${format})`);
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
        const v = this.views.get(name);
        if (!v) {
            throw new Error(`View "${name}" not found`);
        }
        return v[type];
    }

    /**
     * Get a GPUTexture object by name.
     * ---
     * 根據名稱獲取 GPUTexture 物件。
     *
     * @group Query
     * @param name - Texture name / 紋理名稱
     */
    public getTexture(name: string): GPUTexture | undefined {
        return this.textures.get(name);
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
     * Get uniform buffer segment info for a specific parameter.
     * ---
     * 獲取特定 Uniform 項目的緩衝區位址資訊。
     *
     * @group Query
     */
    public getUniform(name: string): { buffer: GPUBuffer; offset: number; size: number } | undefined {
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
     * @group Management
     */
    public createSceneBuffer(): void {
        if (this.sceneBuffer) return;
        this.sceneBuffer = this.device.createBuffer({
            size: 40, // Expanded for SceneInfo struct
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'SceneInfo'
        });
    }

    /**
     * Update scene buffer with current source dimensions.
     * ---
     * 以目前的影像尺寸更新場景資訊緩衝區。
     *
     * @group Update
     */
    public updateSceneBuffer(width: number, height: number, outWidth: number = 0, outHeight: number = 0): void {
        if (!this.sceneBuffer || width === 0 || height === 0) return;

        if (outWidth === 0) outWidth = width;
        if (outHeight === 0) outHeight = height;

        const data = new ArrayBuffer(40);
        const u32 = new Uint32Array(data);
        const f32 = new Float32Array(data);

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

        this.device.queue.writeBuffer(this.sceneBuffer, 0, data);
        Logger.debug(`Updated scene buffer: IN(${width}x${height}), OUT(${outWidth}x${outHeight}), SCALE(${f32[8].toFixed(2)}, ${f32[9].toFixed(2)})`);
    }

    /** @group Query */
    public getSceneBuffer(): GPUBuffer | null {
        return this.sceneBuffer;
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

        if (this.uniformBuffer) {
            this.uniformBuffer.destroy();
            this.uniformBuffer = null;
        }
        if (this.sceneBuffer) {
            this.sceneBuffer.destroy();
            this.sceneBuffer = null;
        }
        this.uniforms.clear();
        Logger.info('ResourceManager: All resources disposed');
    }

    /**
     * Create the main uniform buffer and map parameter offsets.
     * ---
     * 建立主要的 Uniform 緩衝區，並計算各參數在緩衝區中的偏移量（採 16 字節對齊）。
     *
     * @internal
     */
    private createUniformBuffer(parameters: any[]): void {
        let offset = 0;
        parameters.forEach(p => {
            const size = 4; // Assume f32 / 假設為單精度浮點
            this.uniforms.set(p.name, { buffer: null as any, offset, size });
            offset += 16; // Maintain standard 16-byte alignment / 維持標準 16 位元組對齊
        });

        const bufferSize = Math.max(offset, 16);
        this.uniformBuffer = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'GlobalUniforms'
        });

        // Link all parameters to this single buffer / 將所有參數關聯至此單一緩衝區
        this.uniforms.forEach(u => u.buffer = this.uniformBuffer!);
    }
}