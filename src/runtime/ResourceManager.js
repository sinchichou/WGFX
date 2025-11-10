// src/runtime/ResourceManager.js

/**
 * @fileoverview 管理 WebGPU 資源的建立、儲存和檢索。
 * 此類別根據著色器 IR 處理 GPUTexture、GPUSampler 和 GPUBuffer 物件，
 * 提供一個中心位置，可以按名稱存取這些資源。
 */
export class ResourceManager {
    /**
     * @param {GPUDevice} device - 作用中的 WebGPU 裝置。
     */
    constructor(device) {
        /** @type {GPUDevice} */
        this.device = device;
        /** @type {Map<string, GPUTexture>} */
        this.textures = new Map();
        /** @type {Map<string, GPUSampler>} */
        this.samplers = new Map();
        /**
         * 儲存每個 uniform 的元資料，包括其在主緩衝區中的偏移量和大小。
         * @type {Map<string, {buffer: GPUBuffer, offset: number, size: number}>}
         */
        this.uniforms = new Map();
        /**
         * 用於儲存所有 uniform 參數的單一 GPU 緩衝區。
         * @type {GPUBuffer}
         */
        this.uniformBuffer = null;

        // 使用預設描述符預先建立 INPUT 和 OUTPUT 紋理。
        // 這些紋理稍後可以被外部紋理替換。
        this.createTexture('INPUT', {
            size: [1, 1], // 預設大小，應由實際輸入覆蓋
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.createTexture('OUTPUT', {
            size: [1, 1], // 預設大小，應由實際輸出覆蓋
            format: 'rgba8unorm',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING,
        });
    }

    /**
     * 根據解析後的著色器 IR 初始化所有 GPU 資源。
     * @param {import('./Parser.js').WGFXShaderInfo} shaderInfo - 來自 Parser.js 的解析後著色器資訊。
     */
    initialize(shaderInfo) {
        // 建立著色器中定義的紋理。
        shaderInfo.textures.forEach(tex => {
            // INPUT 和 OUTPUT 是特殊情況，假定由外部管理或已建立。
            if (tex.name === 'INPUT' || tex.name === 'OUTPUT') return;

            // TODO: 更穩健的實作將解析寬度/高度表達式 (例如："INPUT_WIDTH * 0.5")。
            const descriptor = {
                size: [1920, 1080], // 目前的佔位符大小。
                format: tex.format?.toLowerCase() || 'rgba8unorm',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
            };
            this.createTexture(tex.name, descriptor);
        });

        // 建立著色器中定義的取樣器。
        shaderInfo.samplers.forEach(samp => {
            const descriptor = {
                magFilter: samp.filter === 'LINEAR' ? 'linear' : 'nearest',
                minFilter: samp.filter === 'LINEAR' ? 'linear' : 'nearest',
                addressModeU: samp.address === 'WRAP' ? 'repeat' : 'clamp-to-edge',
                addressModeV: samp.address === 'WRAP' ? 'repeat' : 'clamp-to-edge',
            };
            this.createSampler(samp.name, descriptor);
        });

        // 建立一個單一的 uniform 緩衝區來儲存所有參數。
        if (shaderInfo.parameters.length > 0) {
            // 計算每個參數的偏移量和大小。
            // 注意：這是一個簡化的打包策略。實際實作必須遵守 std140/std430 佈局規則。
            let totalSize = 0;
            shaderInfo.parameters.forEach(param => {
                const size = param.type === 'int' ? 4 : 4; // i32 和 f32 都是 4 位元組。
                this.uniforms.set(param.name, {buffer: null, offset: totalSize, size});
                totalSize += size;
            });

            // uniform 緩衝區偏移量必須對齊 16 的倍數。
            const alignedSize = Math.ceil(totalSize / 16) * 16;

            this.uniformBuffer = this.device.createBuffer({
                size: alignedSize,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            // 更新每個 uniform 的元資料以指向建立的緩衝區。
            this.uniforms.forEach(u => u.buffer = this.uniformBuffer);
        }

        console.log("ResourceManager: 資源已初始化。", {
            textures: [...this.textures.keys()],
            samplers: [...this.samplers.keys()],
            uniforms: [...this.uniforms.keys()],
        });
    }

    /**
     * 在管理器中建立或替換 GPUTexture。
     * @param {string} name - 紋理的唯一名稱。
     * @param {GPUTextureDescriptor} descriptor - 紋理描述符。
     * @returns {GPUTexture} 建立的 GPUTexture。
     */
    createTexture(name, descriptor) {
        // 如果存在同名紋理，則先銷毀舊紋理。
        if (this.textures.has(name)) {
            this.textures.get(name).destroy();
        }
        const texture = this.device.createTexture(descriptor);
        texture.label = name;
        this.textures.set(name, texture);
        return texture;
    }

    /**
     * 按名稱檢索紋理。
     * @param {string} name - 紋理的名稱。
     * @returns {GPUTexture | undefined}
     */
    getTexture(name) {
        return this.textures.get(name);
    }

    /**
     * 建立新的 GPUSampler。
     * @param {string} name - 取樣器的唯一名稱。
     * @param {GPUSamplerDescriptor} descriptor - 取樣器描述符。
     * @returns {GPUSampler}
     */
    createSampler(name, descriptor) {
        const sampler = this.device.createSampler(descriptor);
        this.samplers.set(name, sampler);
        return sampler;
    }

    /**
     * 按名稱檢索取樣器。
     * @param {string} name - 取樣器的名稱。
     * @returns {GPUSampler | undefined}
     */
    getSampler(name) {
        return this.samplers.get(name);
    }

    /**
     * 檢索所有參數的單一 uniform 緩衝區。
     * @returns {GPUBuffer | null}
     */
    getUniformBuffer() {
        return this.uniformBuffer;
    }

    /**
     * 檢索特定 uniform 的元資料。
     * @param {string} name - uniform 參數的名稱。
     * @returns {{buffer: GPUBuffer, offset: number, size: number} | undefined}
     */
    getUniform(name) {
        return this.uniforms.get(name);
    }

    /**
     * 銷毀所有管理的 GPU 資源以防止記憶體洩漏。
     */
    dispose() {
        this.textures.forEach(texture => texture.destroy());
        this.uniforms.clear();
        if (this.uniformBuffer) {
            this.uniformBuffer.destroy();
        }
        this.textures.clear();
        this.samplers.clear();
        console.log("ResourceManager: 所有 GPU 資源已銷毀。");
    }
}