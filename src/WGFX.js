// src/WGFX.js
import {WGFXRuntime} from './runtime/WGFXRuntime.js';

/**
 * @fileoverview
 * - EN: The primary public interface for interacting with WGFX effects.
 *   It encapsulates the setup, initialization, and per-frame processing logic.
 * - TW: 用於與 WGFX 效果互動的主要公共接口。
 *   它封裝了設定、初始化和每幀處理邏輯。
 */

export class WGFX {
    /**
     * @param {WGFXRuntime} runtime
     * - EN: The internal runtime instance managing GPU resources and pipelines.
     * - TW: 管理 GPU 資源和管線的內部運行時實例。
     */
    constructor(runtime) {
        this.runtime = runtime;
        this.initialized = false;
        // WGFXSession has been renamed to WGFX for the public interface
    }

    /**
     * - EN: Asynchronously creates and compiles a new WGFX effect instance.
     * - TW: 異步建立並編譯一個新的 WGFX 效果實例。
     * @param {object} options
     * @param {GPUDevice} options.device - The active WebGPU device.
     * @param {string} options.effectCode - The source code of the WGFX effect.
     * @param {object} [options.externalResources={}] - Predefined external resources like 'INPUT' or 'OUTPUT' textures.
     * @returns {Promise<WGFX>} - The initialized WGFX instance.
     */
    static async create({device, effectCode, externalResources = {}}) {
        // 1. Initialize Runtime
        const runtime = new WGFXRuntime(device);

        // 2. Compile and set up all resources and pipelines
        // NOTE: externalResources are passed to the resourceManager during compile.
        await runtime.compile(effectCode, externalResources);

        // 3. Return the public interface instance
        return new WGFX(runtime);
    }

    /**
     * - EN: First pass: Initializes the session and returns UI metadata for controls.
     * - TW: 第一輪：初始化會話並回傳用於 UI 控制的元資料。
     * @returns {object} - Metadata describing uniforms, input textures, and output format.
     */
    initialize() {
        if (this.initialized) {
            throw new Error("WGFX instance already initialized.");
        }

        const shaderInfo = this.runtime.shaderInfo;

        if (!shaderInfo) {
            throw new Error("Shader not compiled. Call create() first.");
        }

        // Determine output format (simplified: assuming 'OUTPUT' texture is available)
        const outputTexture = this.runtime.resourceManager.getTexture('OUTPUT');
        const outputFormat = outputTexture?.format; // Assuming format is available on mock/real texture

        const metadata = {
            // Uniforms are referred to as 'parameters' in the internal IR
            uniforms: shaderInfo.parameters.map(u => ({
                name: u.name,
                type: u.type,
                default: u.defaultValue,
                min: u.min ?? null,
                max: u.max ?? null
            })),
            inputTextures: shaderInfo.textures
                .filter(t => t.role === 'input')
                .map(t => ({
                    name: t.name,
                    type: 'videoFrame' // Specific role hint for external binding
                })),
            output: {
                name: 'OUTPUT',
                format: outputFormat
            }
        };

        this.initialized = true;
        console.log("WGFX: Initialization complete. Metadata returned.");
        return metadata;
    }

    /**
     * - EN: Second pass: Processes a single frame using provided uniforms.
     * - TW: 第二輪：使用提供的 uniform 處理單個影格。
     * @param {object} options
     * @param {ImageBitmap | VideoFrame | HTMLCanvasElement} options.frameBitmap - The input video frame or image data.
     * @param {object<string, number>} options.uniforms - Key-value pairs of uniform names and their current values.
     * @returns {GPUTextureView} - The texture view of the final processed output ('OUTPUT').
     */
    async processFrame({frameBitmap, uniforms}) {
        if (!this.initialized) {
            throw new Error("WGFX instance not initialized. Call initialize() first.");
        }

        // 1. Upload video frame to the 'INPUT' texture
        // Note: We need to assume ResourceManager has a method to handle this upload.
        // I will add a placeholder method to ResourceManager for this logic.
        this.runtime.resourceManager.updateTextureFromImage(
            'INPUT',
            frameBitmap
        );

        // 2. Update uniforms using the UniformBinder
        for (const [name, value] of Object.entries(uniforms)) {
            this.runtime.updateUniform(name, value);
        }

        // 3. Execute all passes
        const commandEncoder = this.runtime.device.createCommandEncoder();

        // Loop through passes defined in the shader IR
        for (const pass of this.runtime.shaderInfo.passes) {
            // Pass names are constructed as 'PASS_index' (e.g., 'PASS_1')
            this.runtime.dispatchPass(`PASS_${pass.index}`, commandEncoder);
        }

        // Submit commands to the GPU queue
        this.runtime.device.queue.submit([commandEncoder.finish()]);

        // 4. Return the final output view
        return this.runtime.getOutput();
    }

    /**
     * - EN: Cleans up all managed GPU resources.
     * - TW: 清理所有管理的 GPU 資源。
     */
    dispose() {
        this.runtime.resourceManager.dispose();
    }
}