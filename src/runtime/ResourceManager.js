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

    /**
     * - EN: Initializes all GPU resources based on the parsed shader IR.
     * - TW: 根據解析後的著色器 IR 初始化所有 GPU 資源。
     * @param {import('./ShaderParser.js').WGFXShaderInfo} shaderInfo
     * - EN: Parsed shader information from Parser.js.
     * - TW: 來自 Parser.js 的解析後著色器資訊。
     */
    /**
     * - EN: Initializes all GPU resources based on the parsed shader IR.
     * - TW: 根據解析後的著色器 IR 初始化所有 GPU 資源。
     * @param {import('./ShaderParser.js').WGFXShaderInfo} shaderInfo
     * - EN: Parsed shader information from Parser.js.
     * - TW: 來自 Parser.js 的解析後著色器資訊。
     */
    initialize(shaderInfo, externalResources = {}) {
        const context = {};

        // 1. [優先] 載入外部定義的常數 (如 INPUT_WIDTH/HEIGHT)
        if (externalResources.defines) {
            Object.assign(context, externalResources.defines);
        }

        // 2. 建立外部紋理 (INPUT, OUTPUT 等)
        if (externalResources.textures) {
            for (const [name, descriptor] of Object.entries(externalResources.textures)) {
                this.createTexture(name, descriptor);
            }
        }

        // 3. [後備] 如果 defines 沒給尺寸，嘗試從剛剛建立的 INPUT 紋理反查
        if ((!context['INPUT_WIDTH'] || !context['INPUT_HEIGHT']) && this.textures.has('INPUT')) {
            const inputTexture = this.getTexture('INPUT');
            if (inputTexture) {
                if (!context['INPUT_WIDTH']) context['INPUT_WIDTH'] = inputTexture.width;
                if (!context['INPUT_HEIGHT']) context['INPUT_HEIGHT'] = inputTexture.height;
            }
        }

        // --- Safe Math Parser (CSP Compliant: No eval/new Function) ---
        // 實作一個簡單的遞迴下降解析器，支援 +, -, *, /, %, () 和小數
        const parseMathExpression = (str) => {
            let pos = 0;
            // 移除所有空白
            str = str.replace(/\s+/g, '');

            const peek = () => str[pos];
            const consume = () => str[pos++];

            const parseFactor = () => {
                if (peek() === '(') {
                    consume(); // 吃掉 '('
                    const result = parseExpr();
                    if (peek() !== ')') throw new Error("Expected ')'");
                    consume(); // 吃掉 ')'
                    return result;
                }

                // 解析數字 (含負號和小數點)
                let numStr = '';
                if (peek() === '-') {
                    numStr += consume();
                }
                while (pos < str.length && (/[0-9.]/).test(peek())) {
                    numStr += consume();
                }
                if (numStr === '') throw new Error(`Unexpected char: '${peek()}' at pos ${pos}`);
                return parseFloat(numStr);
            };

            const parseTerm = () => {
                let left = parseFactor();
                while (pos < str.length) {
                    const op = peek();
                    if (op === '*' || op === '/' || op === '%') {
                        consume();
                        const right = parseFactor();
                        if (op === '*') left *= right;
                        else if (op === '/') left /= right;
                        else if (op === '%') left %= right;
                    } else {
                        break;
                    }
                }
                return left;
            };

            const parseExpr = () => {
                let left = parseTerm();
                while (pos < str.length) {
                    const op = peek();
                    if (op === '+' || op === '-') {
                        consume();
                        const right = parseTerm();
                        if (op === '+') left += right;
                        else if (op === '-') left -= right;
                    } else {
                        break;
                    }
                }
                return left;
            };

            const result = parseExpr();
            return result;
        };

        const evaluate = (expr, ctx) => {
            if (typeof expr !== 'string') return expr;

            // 替換變數
            let evaluatedExpr = expr;
            for (const key in ctx) {
                const regex = new RegExp('\\b' + key + '\\b', 'g');
                evaluatedExpr = evaluatedExpr.replace(regex, ctx[key]);
            }

            try {
                // 如果替換後只是單純的數字字串，直接轉換 (最快)
                if (!isNaN(Number(evaluatedExpr))) {
                    return Math.ceil(Number(evaluatedExpr));
                }
                // 否則使用安全的解析器計算
                return Math.ceil(parseMathExpression(evaluatedExpr));
            } catch (e) {
                console.error("Evaluation failed. Context:", ctx);
                throw new Error(`Cannot evaluate expression: "${expr}". Resulted in: "${evaluatedExpr}". Error: ${e.message}`);
            }
        };

        /**
         * - EN: Create textures defined in the shader.
         * - TW: 建立著色器中定義的紋理。
         */
        shaderInfo.textures.forEach(tex => {
            if (this.textures.has(tex.name)) return; // Already created externally

            const width = evaluate(tex.width, context);
            const height = evaluate(tex.height, context);

            if (!width || !height) {
                throw new Error(`Could not determine size for texture ${tex.name}. Width or height expression is invalid.`);
            }

            const descriptor = {
                size: [width, height],
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
            let totalSize = 0;
            shaderInfo.parameters.forEach(param => {
                const size = 4; // Assuming 4 bytes (f32/i32) for simplicity
                this.uniforms.set(param.name, {buffer: null, offset: totalSize, size});
                totalSize += size;
            });

            const alignedSize = Math.ceil(totalSize / 16) * 16;

            this.uniformBuffer = this.device.createBuffer({
                size: alignedSize,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            this.uniforms.forEach(u => u.buffer = this.uniformBuffer);
        }

        console.log("ResourceManager: Resources initialized.", {
            textures: [...this.textures.keys()],
            samplers: [...this.samplers.keys()],
            uniforms: [...this.uniforms.keys()],
            context: context
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
     * - EN: Uploads image data (ImageBitmap, VideoFrame, etc.) to a specified GPUTexture.
     * - TW: 將圖像資料 (ImageBitmap, VideoFrame 等) 上傳到指定的 GPUTexture。
     * @param {string} name - The name of the texture (e.g., 'INPUT').
     * @param {ImageBitmap | VideoFrame | HTMLCanvasElement} image - The source image data.
     */
    updateTextureFromImage(name, image) {
        const texture = this.getTexture(name);

        if (!texture) {
            throw new Error(`Texture '${name}' not found for update.`);
        }

        // Ensure the texture has COPY_DST usage, which is usually included for dynamic textures.
        // We assume 'INPUT' is correctly set up with the right dimensions/format during compile/external setup.

        if (texture.width !== image.width || texture.height !== image.height) {
            // In a real implementation, you might resize the texture or throw an error.
            // For simplicity, we assume the input size matches the texture size (e.g., INPUT is pre-sized).
            console.warn(`Input image size (${image.width}x${image.height}) does not match texture size (${texture.width}x${texture.height}).`);
        }

        this.device.queue.copyExternalImageToBufferOrTexture(
            {source: image},
            {texture: texture},
            [image.width, image.height]
        );
        console.log(`Texture '${name}' updated from image source.`);
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
