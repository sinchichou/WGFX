// src/runtime/ResourceManager.js

/**
 * @fileoverview
 * - EN: Manages the creation, storage, and retrieval of WebGPU resources.
 *   This class handles GPUTexture, GPUSampler, and GPUBuffer objects based on the shader IR,
 *   providing a central location to access these resources by name.
 * - TW: 管理 WebGPU 資源的建立、儲存和檢索。
 *   此類別根據著色器 IR 處理 GPUTexture、GPUSampler 和 GPUBuffer 物件，
 *   提供一個中心位置，可以按名稱存取這些資源。
 */

let GPUDevice, GPUTextureUsage, GPUBufferUsage;

/**
 * - EN: In a Node.js environment, native WebGPU objects don't exist.
 *   We check for their existence on the global scope. If they don't exist,
 *   they will remain `undefined`, which is safe for the CLI code path
 *   as it doesn't instantiate or use any GPU-related objects.
 * - TW: 在 Node.js 環境中，原生 WebGPU 物件不存在。
 *   我們檢查它們是否在全域範圍內存在。如果不存在，
 *   它們將保持 `undefined`，這對於 CLI 程式碼路徑是安全的，
 *   因為它不實例化或使用任何與 GPU 相關的物件。
 */
try {
    /**
     * - EN: This will only succeed in a browser-like environment with WebGPU support.
     * - TW: 這只會在支援 WebGPU 的瀏覽器環境中成功。
     */
    if (globalThis.GPUDevice) {
        GPUDevice = globalThis.GPUDevice;
        GPUTextureUsage = globalThis.GPUTextureUsage;
        GPUBufferUsage = globalThis.GPUBufferUsage;
    }
    /**
     * - EN: In Node.js or a non-WebGPU environment, these will remain undefined.
     *   This is safe because the CLI path does not use this functionality.
     * - TW: 在 Node.js 或非 WebGPU 環境中，這些將保持未定義。
     *   這很安全，因為 CLI 路徑不使用此功能。
     */
} catch (e) {
    /**
     * - EN: Ignore any errors during this detection phase.
     *   The variables will simply remain undefined.
     * - TW: 在此檢測階段忽略任何錯誤。
     *   變數將簡單地保持未定義。
     */
}


/**
 * - EN: ResourceManager Implementation
 * - TW: 資源管理器實作
 */

export class ResourceManager {
    /**
     * @param {GPUDevice} [device]
     * - EN: The active WebGPU device. If not provided, a mock device will be used.
     * - TW: 作用中的 WebGPU 裝置。如果未提供，將使用模擬裝置。
     */
    constructor(device) {
        /** @type {GPUDevice} */
        /**
         * - EN: Use provided device or a mock device.
         * - TW: 使用提供的裝置或模擬裝置。
         */
        this.device = device || new GPUDevice();
        /**
         * - EN: Stores GPUTexture objects, mapped by their unique names.
         * - TW: 儲存 GPUTexture 物件，按其唯一名稱映射。
         * @type {Map<string, import('./WebGPU-mock.js').GPUTexture>}
         */
        this.textures = new Map();
        /**
         * - EN: Stores GPUSampler objects, mapped by their unique names.
         * - TW: 儲存 GPUSampler 物件，按其唯一名稱映射。
         * @type {Map<string, import('./WebGPU-mock.js').GPUSampler>}
         */
        this.samplers = new Map();
        /**
         * - EN: Stores metadata for each uniform, including its offset and size within the main buffer.
         * - TW: 儲存每個 uniform 的元資料，包括其在主緩衝區中的偏移量和大小。
         * @type {Map<string, {buffer: import('./WebGPU-mock.js').GPUBuffer, offset: number, size: number}>}
         */
        this.uniforms = new Map();
        /**
         * - EN: A single GPU buffer used to store all uniform parameters.
         * - TW: 用於儲存所有 uniform 參數的單一 GPU 緩衝區。
         * @type {import('./WebGPU-mock.js').GPUBuffer}
         */
        this.uniformBuffer = null;
    }

    /**
     * - EN: Initializes all GPU resources based on the parsed shader IR.
     * - TW: 根據解析後的著色器 IR 初始化所有 GPU 資源。
     * @param {import('./ShaderParser.js').WGFXShaderInfo} shaderInfo
     * - EN: Parsed shader information from Parser.js.
     * - TW: 來自 Parser.js 的解析後著色器資訊。
     */
    initialize(shaderInfo, externalResources = {}) {
        const context = {};

        // 1. 優先載入外部定義
        if (externalResources.defines) {
            Object.assign(context, externalResources.defines);
            console.log('載入外部定義:', context);
        }

        // 2. 建立外部紋理
        if (externalResources.textures) {
            for (const [name, descriptor] of Object.entries(externalResources.textures)) {
                console.log(`建立外部紋理: ${name}`, descriptor);
                this.createTexture(name, descriptor);
            }
        }

        // 3. 回填尺寸常數(如果未提供)
        if ((!context['INPUT_WIDTH'] || !context['INPUT_HEIGHT']) &&
            this.textures.has('INPUT')) {
            const inputTexture = this.getTexture('INPUT');
            if (inputTexture) {
                if (!context['INPUT_WIDTH']) {
                    context['INPUT_WIDTH'] = inputTexture.width;
                }
                if (!context['INPUT_HEIGHT']) {
                    context['INPUT_HEIGHT'] = inputTexture.height;
                }
                console.log('從 INPUT 紋理回填尺寸:', context);
            }
        }

        // 安全的數學表達式求值函數(已在原程式碼中實作)
        const evaluate = (expr, ctx) => {
            if (typeof expr !== 'string') return expr;

            let evaluatedExpr = expr;
            for (const key in ctx) {
                const regex = new RegExp('\\b' + key + '\\b', 'g');
                evaluatedExpr = evaluatedExpr.replace(regex, ctx[key]);
            }

            try {
                if (!isNaN(Number(evaluatedExpr))) {
                    return Math.ceil(Number(evaluatedExpr));
                }
                return Math.ceil(this._parseMathExpression(evaluatedExpr));
            } catch (e) {
                console.error("表達式求值失敗:", {
                    original: expr,
                    evaluated: evaluatedExpr,
                    context: ctx,
                    error: e.message
                });
                throw new Error(
                    `無法計算表達式: "${expr}" -> "${evaluatedExpr}". ` +
                    `錯誤: ${e.message}`
                );
            }
        };

        // 4. 建立著色器定義的紋理
        shaderInfo.textures.forEach(tex => {
            if (this.textures.has(tex.name)) {
                console.log(`跳過已存在的紋理: ${tex.name}`);
                return;
            }

            const width = evaluate(tex.width, context);
            const height = evaluate(tex.height, context);

            if (!width || !height) {
                throw new Error(
                    `無法確定紋理 ${tex.name} 的尺寸。` +
                    `width="${tex.width}", height="${tex.height}"`
                );
            }

            // 關鍵:為可能作為 Storage 的紋理預先升級格式
            let format = (tex.format || 'rgba8unorm').toLowerCase();

            // 如果紋理可能被用作輸出,預先升級格式
            const mightBeStorage = tex.name !== 'INPUT';
            if (mightBeStorage && !this._isValidStorageFormat(format)) {
                const originalFormat = format;
                format = this._upgradeToStorageFormat(format);
                console.log(
                    `紋理 ${tex.name} 格式從 ${originalFormat} ` +
                    `升級為 ${format} (預防性)`
                );
            }

            const descriptor = {
                size: [width, height],
                format: format,
                usage: GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.STORAGE_BINDING |
                    GPUTextureUsage.COPY_DST |
                    GPUTextureUsage.COPY_SRC,
            };

            this.createTexture(tex.name, descriptor);
            console.log(`建立紋理: ${tex.name} (${width}x${height}, ${format})`);
        });

        // 5. 建立取樣器
        shaderInfo.samplers.forEach(samp => {
            const descriptor = {
                magFilter: samp.filter === 'LINEAR' ? 'linear' : 'nearest',
                minFilter: samp.filter === 'LINEAR' ? 'linear' : 'nearest',
                addressModeU: samp.address === 'WRAP' ? 'repeat' : 'clamp-to-edge',
                addressModeV: samp.address === 'WRAP' ? 'repeat' : 'clamp-to-edge',
            };
            this.createSampler(samp.name, descriptor);
            console.log(`建立取樣器: ${samp.name}`);
        });

        // 6. 建立 Uniform Buffer
        if (shaderInfo.parameters.length > 0) {
            let totalSize = 0;
            shaderInfo.parameters.forEach(param => {
                const size = 4; // f32/i32
                this.uniforms.set(param.name, {
                    buffer: null,
                    offset: totalSize,
                    size
                });
                totalSize += size;
            });

            const alignedSize = Math.ceil(totalSize / 16) * 16;

            this.uniformBuffer = this.device.createBuffer({
                size: alignedSize,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            this.uniforms.forEach(u => u.buffer = this.uniformBuffer);
            console.log(`建立 Uniform Buffer: ${alignedSize} bytes`);
        }

        console.log("ResourceManager 初始化完成:", {
            textures: [...this.textures.keys()],
            samplers: [...this.samplers.keys()],
            uniforms: [...this.uniforms.keys()],
            context: context
        });
    }

// 輔助方法:判斷格式是否為 Storage 相容
    _isValidStorageFormat(format) {
        const validFormats = [
            'r32float', 'r32sint', 'r32uint',
            'rgba16float', 'rgba16sint', 'rgba16uint',
            'rgba32float', 'rgba32sint', 'rgba32uint',
            'rg32float', 'rg32sint', 'rg32uint'
        ];
        return validFormats.includes(format.toLowerCase().replace(/_/g, ''));
    }

// 輔助方法:升級格式到 Storage 相容版本
    _upgradeToStorageFormat(format) {
        const normalized = format.toLowerCase().replace(/_/g, '');

        if (normalized === 'rgba8unorm' || normalized === 'bgra8unorm') {
            return 'rgba16float';
        }

        if (normalized.includes('rgba16') || normalized.includes('r16')) {
            return normalized;
        }

        if (normalized.includes('32')) {
            return normalized;
        }

        console.warn(`未知格式 "${format}" 升級為 rgba16float`);
        return 'rgba16float';
    }

    /**
     * - EN: Creates or replaces a GPUTexture in the manager.
     * - TW: 在管理器中建立或替換 GPUTexture。
     * @param {string} name
     * - EN: The unique name of the texture.
     * - TW: 紋理的唯一名稱。
     * @param {GPUTextureDescriptor} descriptor
     * - EN: The descriptor for the texture.
     * - TW: 紋理描述符。
     * @returns {GPUTexture}
     * - EN: The created GPUTexture.
     * - TW: 建立的 GPUTexture。
     */
    createTexture(name, descriptor) {
        /**
         * - EN: If a texture with the same name exists, destroy the old one first.
         * - TW: 如果存在同名紋理，則先銷毀舊紋理。
         */
        if (this.textures.has(name)) {
            this.textures.get(name).destroy();
        }
        const texture = this.device.createTexture(descriptor);
        /**
         * - EN: Set label for debugging.
         * - TW: 設定標籤以進行調試。
         */
        texture.label = name;
        this.textures.set(name, texture);
        return texture;
    }

    /**
     * - EN: Retrieves a texture by its name.
     * - TW: 按名稱檢索紋理。
     * @param {string} name
     * - EN: The name of the texture.
     * - TW: 紋理的名稱。
     * @returns {GPUTexture | undefined}
     * - EN: The texture object or undefined if not found.
     * - TW: 紋理物件，如果找不到則為 undefined。
     */
    getTexture(name) {
        return this.textures.get(name);
    }

    /**
     * - EN: Creates a new GPUSampler.
     * - TW: 建立新的 GPUSampler。
     * @param {string} name
     * - EN: The unique name of the sampler.
     * - TW: 取樣器的唯一名稱。
     * @param {GPUSamplerDescriptor} descriptor
     * - EN: The descriptor for the sampler.
     * - TW: 取樣器描述符。
     * @returns {GPUSampler}
     * - EN: The created GPUSampler.
     * - TW: 建立的 GPUSampler。
     */
    createSampler(name, descriptor) {
        const sampler = this.device.createSampler(descriptor);
        this.samplers.set(name, sampler);
        return sampler;
    }

    /**
     * 從圖像源更新紋理內容
     * @param {string} textureName - 紋理名稱
     * @param {ImageBitmap | HTMLVideoElement | HTMLCanvasElement | OffscreenCanvas} imageSource - 圖像源
     */
    updateTextureFromImage(textureName, imageSource) {
        const texture = this.textures.get(textureName);
        if (!texture) {
            throw new Error(`紋理 "${textureName}" 不存在於 ResourceManager 中`);
        }

        // 獲取圖像源尺寸
        let width, height;
        if (imageSource instanceof HTMLVideoElement) {
            width = imageSource.videoWidth;
            height = imageSource.videoHeight;
        } else if (imageSource instanceof VideoFrame) {
            width = imageSource.displayWidth;
            height = imageSource.displayHeight;
        } else {
            width = imageSource.width;
            height = imageSource.height;
        }

        // 驗證尺寸
        if (width === 0 || height === 0) {
            throw new Error(`圖像源尺寸無效: ${width}x${height}`);
        }

        if (texture.width !== width || texture.height !== height) {
            throw new Error(
                `紋理 "${textureName}" 尺寸 (${texture.width}x${texture.height}) ` +
                `與圖像尺寸 (${width}x${height}) 不符`
            );
        }

        // 正確的 API 方法名稱
        this.device.queue.copyExternalImageToTexture(
            {
                source: imageSource,
                flipY: false
            },
            {
                texture: texture,
                mipLevel: 0,
                origin: {x: 0, y: 0, z: 0}
            },
            {
                width: width,
                height: height,
                depthOrArrayLayers: 1
            }
        );
    }

    /**
     * - EN: Retrieves a sampler by its name.
     * - TW: 按名稱檢索取樣器。
     * @param {string} name
     * - EN: The name of the sampler.
     * - TW: 取樣器的名稱。
     * @returns {GPUSampler | undefined}
     * - EN: The sampler object or undefined if not found.
     * - TW: 取樣器物件，如果找不到則為 undefined。
     */
    getSampler(name) {
        return this.samplers.get(name);
    }

    /**
     * - EN: Retrieves the single uniform buffer for all parameters.
     * - TW: 檢索所有參數的單一 uniform 緩衝區。
     * @returns {GPUBuffer | null} - EN: The uniform buffer object, or null if no parameters exist. - TW: 統一緩衝區物件，如果不存在參數則為 null。
     */
    getUniformBuffer() {
        return this.uniformBuffer;
    }

    /**
     * - EN: Retrieves metadata for a specific uniform.
     * - TW: 檢索特定 uniform 的元資料。
     * @param {string} name
     * - EN: The name of the uniform parameter.
     * - TW: uniform 參數的名稱。
     * @returns {{buffer: GPUBuffer, offset: number, size: number} | undefined}
     * - EN: Metadata for the uniform, or undefined if not found.
     * - TW: uniform 的元資料，如果找不到則為 undefined。
     */
    getUniform(name) {
        return this.uniforms.get(name);
    }

    /**
     * - EN: Destroys all managed GPU resources to prevent memory leaks.
     * - TW: 銷毀所有管理的 GPU 資源以防止記憶體洩漏。
     */
    dispose() {
        this.textures.forEach(texture => texture.destroy());
        /**
         * - EN: Destroy all GPU textures.
         * - TW: 銷毀所有 GPU 紋理。
         */
        this.uniforms.clear();
        /**
         * - EN: Clear uniform metadata.
         * - TW: 清除 uniform 元資料。
         */
        if (this.uniformBuffer) {
            this.uniformBuffer.destroy();
            /**
             * - EN: Destroy the uniform buffer.
             * - TW: 銷毀 uniform 緩衝區。
             */
        }
        this.textures.clear();
        /**
         * - EN: Clear the texture map.
         * - TW: 清除紋理映射。
         */
        this.samplers.clear();
        /**
         * - EN: Clear the sampler map.
         * - TW: 清除取樣器映射。
         */
        /**
         * - EN: All GPU resources destroyed.
         * - TW: 所有 GPU 資源已銷毀。
         */
        console.log("ResourceManager: All GPU resources destroyed.");
    }
}
