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

        /**
         * - EN: Pre-create INPUT and OUTPUT textures with default descriptors.
         *   These textures can be replaced by external textures later.
         * - TW: 使用預設描述符預先建立 INPUT 和 OUTPUT 紋理。
         *   這些紋理稍後可以被外部紋理替換。
         */
        this.createTexture('INPUT', {
            size: [1, 1],
            /**
             * - EN: Default size, should be overridden by actual input.
             * - TW: 預設大小，應由實際輸入覆蓋。
             */
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.createTexture('OUTPUT', {
            size: [1, 1],
            /**
             * - EN: Default size, should be overridden by actual input.
             * - TW: 預設大小，應由實際輸入覆蓋。
             */
            format: 'rgba8unorm',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING,
        });
    }

    /**
     * - EN: Initializes all GPU resources based on the parsed shader IR.
     * - TW: 根據解析後的著色器 IR 初始化所有 GPU 資源。
     * @param {import('./ShaderParser.js').WGFXShaderInfo} shaderInfo
     * - EN: Parsed shader information from Parser.js.
     * - TW: 來自 Parser.js 的解析後著色器資訊。
     */
    initialize(shaderInfo) {
        /**
         * - EN: Create textures defined in the shader.
         * - TW: 建立著色器中定義的紋理。
         */
        shaderInfo.textures.forEach(tex => {
            /**
             * - EN: INPUT and OUTPUT are special cases, assumed to be managed externally or already created.
             * - TW: INPUT 和 OUTPUT 是特殊情況，假定由外部管理或已建立。
             */
            if (tex.name === 'INPUT' || tex.name === 'OUTPUT') return;

            /**
             * - EN: TODO: A more robust implementation would parse width/height expressions (e.g., "INPUT_WIDTH * 0.5").
             * - TW: TODO: 更穩健的實作將解析寬度/高度表達式 (例如："INPUT_WIDTH * 0.5")。
             */
            const descriptor = {
                size: [1920, 1080],
                /**
                 * - EN: Current placeholder size.
                 * - TW: 目前的佔位符大小。
                 */
                format: tex.format?.toLowerCase() || 'rgba8unorm',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
            };
            this.createTexture(tex.name, descriptor);
        });

        /**
         * - EN: Create samplers defined in the shader.
         * - TW: 建立著色器中定義的取樣器。
         */
        shaderInfo.samplers.forEach(samp => {
            const descriptor = {
                magFilter: samp.filter === 'LINEAR' ? 'linear' : 'nearest',
                minFilter: samp.filter === 'LINEAR' ? 'linear' : 'nearest',
                addressModeU: samp.address === 'WRAP' ? 'repeat' : 'clamp-to-edge',
                addressModeV: samp.address === 'WRAP' ? 'repeat' : 'clamp-to-edge',
            };
            this.createSampler(samp.name, descriptor);
        });

        /**
         * - EN: Create a single uniform buffer to store all parameters.
         * - TW: 建立一個單一的 uniform 緩衝區來儲存所有參數。
         */
        if (shaderInfo.parameters.length > 0) {
            /**
             * - EN: Calculate offset and size for each parameter.
             *   Note: This is a simplified packing strategy. Real implementations must adhere to std140/std430 layout rules.
             * - TW: 計算每個參數的偏移量和大小。
             *   注意：這是一個簡化的打包策略。實際實作必須遵守 std140/std430 佈局規則。
             */
            let totalSize = 0;
            shaderInfo.parameters.forEach(param => {
                const size = param.type === 'int' ? 4 : 4;
                /**
                 * - EN: i32 and f32 are both 4 bytes.
                 * - TW: i32 和 f32 都是 4 位元組。
                 */
                this.uniforms.set(param.name, {buffer: null, offset: totalSize, size});
                totalSize += size;
            });

            /**
             * - EN: Uniform buffer offsets must be aligned to multiples of 16.
             * - TW: uniform 緩衝區偏移量必須對齊 16 的倍數。
             */
            const alignedSize = Math.ceil(totalSize / 16) * 16;

            this.uniformBuffer = this.device.createBuffer({
                size: alignedSize,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            /**
             * - EN: Update metadata for each uniform to point to the created buffer.
             * - TW: 更新每個 uniform 的元資料以指向建立的緩衝區。
             */
            this.uniforms.forEach(u => u.buffer = this.uniformBuffer);
        }

        /**
         * - EN: Resources initialized.
         * - TW: 資源已初始化。
         */
        console.log("ResourceManager: Resources initialized.", {
            textures: [...this.textures.keys()],
            samplers: [...this.samplers.keys()],
            uniforms: [...this.uniforms.keys()],
        });
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
