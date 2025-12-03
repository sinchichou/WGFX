// src/runtime/WGFXRuntime.js

/**
 * @fileoverview
 * - EN: The main coordinator for the WGFX runtime.
 *   This class brings together the parser, resource manager, code generator, and pipeline manager
 *   to provide a high-level API for compiling and running WGFX effects.
 * - TW: WGFX 運行時的主要協調器。
 *   此類別將解析器、資源管理器、程式碼生成器和管線管理器結合在一起，
 *   提供用於編譯和運行 WGFX 效果的高級 API。
 */

import {parse} from './ShaderParser.js';
import {ResourceManager} from './ResourceManager.js';
import {PipelineManager} from './PipelineManager.js';
import {WGSLCodeGenerator} from './WGSLCodeGenerator.js';
import {UniformBinder} from './UniformBinder.js';

export class WGFXRuntime {
    /**
     * @param {GPUDevice} device
     * - EN: The active WebGPU device.
     * - TW: 作用中的 WebGPU 裝置。
     */
    constructor(device) {
        if (!device) {
            /**
             * - EN: WGFXRuntime requires a valid WebGPU device. No device provided.
             * - TW: WGFXRuntime 需要一個有效的 WebGPU 裝置。未提供裝置。
             */
            throw new Error("WGFXRuntime requires a valid WebGPU device. No device provided.");
        }
        this.device = device;

        /**
         * - EN: Instantiate all necessary sub-modules.
         * - TW: 實例化所有必要的子模組。
         */
        // this.parser = new ShaderParser(); // No longer needed
        this.resourceManager = new ResourceManager(this.device);
        this.pipelineManager = new PipelineManager(this.device, this.resourceManager);
        this.wgslCodeGenerator = new WGSLCodeGenerator();
        this.uniformBinder = new UniformBinder(this.device, this.resourceManager);

        /**
         * - EN: Intermediate representation of the currently compiled shader.
         * - TW: 當前編譯著色器的中介表示。
         * @type {import('./ShaderParser.js').WGFXShaderInfo | null}
         */
        this.shaderInfo = null;
    }

    /**
     * - EN: Compiles a WGFX effect from a code string. This is the main entry point for setting up a new effect.
     *   The process involves parsing, code generation, resource allocation, and pipeline creation.
     * - TW: 從程式碼字串編譯 WGFX 效果。這是設定新效果的主要入口點。
     *   該過程涉及解析、程式碼生成、資源分配和管線建立。
     * @param {string} effectCode
     * - EN: The string containing the entire WGFX effect code.
     * - TW: 包含整個 WGFX 效果程式碼的字串。
     * @returns {Promise<void>}
     * - EN: A Promise that resolves when compilation is complete.
     * - TW: 編譯完成時解析的 Promise。
     */
    async compile(effectCode) {
        /**
         * - EN: Starting effect compilation.
         * - TW: 開始效果編譯。
         */
        console.log("WGFXRuntime: Starting effect compilation.");
        /**
         * - EN: 1. Parse the FX file to get the Intermediate Representation (IR).
         * - TW: 1. 解析 FX 檔案以獲取中介表示 (IR)。
         */
        this.shaderInfo = parse(effectCode);
        /**
         * - EN: ShaderInfo (IR) parsed.
         * - TW: 已解析 ShaderInfo (IR)。
         */
        console.log("WGFXRuntime: ShaderInfo (IR) parsed:", this.shaderInfo);
        /**
         * - EN: 2. Generate a single WGSL shader module from the IR.
         * - TW: 2. 從 IR 生成單個 WGSL 著色器模組。
         */
        const generatedModules = this.wgslCodeGenerator.generate(this.shaderInfo);
        // if (this.parser.debug) {
        //     /**
        //      * - EN: Old debug check removed
        //      * - TW: 舊的調試檢查已移除
        //      */
        //     console.log("WGFXRuntime: Generated WGSL module:", generatedModules);
        // }

        /**
         * - EN: 3. Initialize all GPU resources (textures, samplers, buffers) based on the IR.
         * - TW: 3. 根據 IR 初始化所有 GPU 資源 (紋理、取樣器、緩衝區)。
         */
        this.resourceManager.initialize(this.shaderInfo);
        /**
         * - EN: Resources initialized.
         * - TW: 資源已初始化。
         */
        console.log("WGFXRuntime: Resources initialized.");
        /**
         * - EN: 4. Create compute pipelines for each pass.
         * - TW: 4. 為每個通道建立計算管線。
         */
        await this.pipelineManager.createPipelines(this.shaderInfo, generatedModules);
        /**
         * - EN: Pipelines created.
         * - TW: 管線已建立。
         */
        console.log("WGFXRuntime: Pipelines created.");
        /**
         * - EN: Compilation complete.
         * - TW: 編譯完成。
         */
        console.log("WGFXRuntime: Compilation complete.");
    }

    /**
     * - EN: Encodes commands to dispatch a specific compute pass.
     * - TW: 編碼調度特定計算通道的命令。
     * @param {string} passName
     * - EN: The name of the pass to dispatch (e.g., `PASS_1`).
     * - TW: 要調度的通道名稱 (例如：`PASS_1`)。
     * @param {GPUCommandEncoder} commandEncoder
     * - EN: The command encoder for the current frame.
     * - TW: 當前幀的命令編碼器。
     */
    dispatchPass(passName, commandEncoder) {
        if (!this.shaderInfo) {
            /**
             * - EN: Effect not compiled. Call compile() first.
             * - TW: 效果未編譯。請先呼叫 compile()。
             */
            throw new Error("Effect not compiled. Call compile() first.");
        }

        /**
         * - EN: Look up the corresponding pass information from the IR.
         * - TW: 從 IR 中查找對應的通道資訊。
         */
        const passIndex = parseInt(passName.split('_')[1], 10);
        const passInfo = this.shaderInfo.passes.find(p => p.index === passIndex);
        if (!passInfo) {
            /**
             * - EN: Pass `passName` not found in shader information.
             * - TW: 在著色器資訊中找不到通道 `passName`。
             */
            throw new Error(`Pass "${passName}" not found in shader information.`);
        }

        /**
         * - EN: Delegate the actual dispatch logic to the pipeline manager.
         * - TW: 將實際的調度邏輯委託給管線管理器。
         */
        this.pipelineManager.dispatchPass(passInfo, commandEncoder);
    }

    /**
     * - EN: Updates the value of a uniform parameter in the GPU buffer.
     * - TW: 更新 GPU 緩衝區中 uniform 參數的值。
     * @param {string} name
     * - EN: The name of the uniform to update.
     * - TW: 要更新的 uniform 名稱。
     * @param {number} value
     * - EN: The new value for the uniform.
     * - TW: uniform 的新值。
     */
    updateUniform(name, value) {
        if (!this.shaderInfo) {
            /**
             * - EN: Effect not compiled. Call compile() first.
             * - TW: 效果未編譯。請先呼叫 compile()。
             */
            throw new Error("Effect not compiled. Call compile() first.");
        }
        this.uniformBinder.updateUniform(name, value);
    }

    /**
     * - EN: Gets the GPUTextureView of the final output texture, usually named 'OUTPUT'.
     * - TW: 獲取最終輸出紋理的 GPUTextureView，通常命名為 'OUTPUT'。
     * @returns {GPUTextureView}
     * - EN: The texture view of the final output.
     * - TW: 最終輸出的紋理視圖。
     */
    getOutput() {
        const outputTexture = this.resourceManager.getTexture('OUTPUT');
        if (!outputTexture) {
            /**
             * - EN: Output texture `OUTPUT` not found in ResourceManager. Ensure it is correctly defined or managed.
             * - TW: 在 ResourceManager 中找不到輸出紋理 `OUTPUT`。請確保已正確定義或管理。
             */
            throw new Error("Output texture 'OUTPUT' not found in ResourceManager. Ensure it is correctly defined or managed.");
        }
        return outputTexture.createView();
    }

    /**
     * - EN: Utility function to get raw GPU resources.
     * - TW: 獲取原始 GPU 資源的實用函數。
     * @param {string} name
     * - EN: The name of the resource.
     * - TW: 資源的名稱。
     * @returns {GPUTexture | GPUSampler | GPUBuffer | undefined}
     * - EN: The underlying GPU resource.
     * - TW: 底層 GPU 資源。
     */
    getResource(name) {
        return this.resourceManager.getTexture(name)
            || this.resourceManager.getSampler(name)
            || this.resourceManager.getUniform(name)?.buffer;
    }
}
