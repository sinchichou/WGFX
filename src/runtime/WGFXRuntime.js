// src/runtime/WGFXRuntime.js

/**
 * @fileoverview WGFX 運行時的主要協調器。
 * 此類別將解析器、資源管理器、程式碼生成器和管線管理器結合在一起，
 * 提供用於編譯和運行 WGFX 效果的高級 API。
 */

import {ShaderParser} from './Parser.js';
import {ResourceManager} from './ResourceManager.js';
import {PipelineManager} from './PipelineManager.js';
import {WGSLCodeGenerator} from './WGSLCodeGenerator.js';
import {UniformBinder} from './UniformBinder.js';

export class WGFXRuntime {
    /**
     * @param {GPUDevice} device - 作用中的 WebGPU 裝置。
     */
    constructor(device) {
        if (!device) {
            throw new Error("WGFXRuntime 需要一個有效的 WebGPU 裝置。未提供裝置。");
        }
        this.device = device;

        // 實例化所有必要的子模組。
        this.parser = new ShaderParser();
        this.resourceManager = new ResourceManager(this.device);
        this.pipelineManager = new PipelineManager(this.device, this.resourceManager);
        this.wgslCodeGenerator = new WGSLCodeGenerator();
        this.uniformBinder = new UniformBinder(this.device, this.resourceManager);

        /**
         * 當前編譯著色器的中介表示。
         * @type {import('./Parser.js').WGFXShaderInfo | null}
         */
        this.shaderInfo = null;
    }

    /**
     * 從程式碼字串編譯 WGFX 效果。這是設定新效果的主要入口點。
     * 該過程涉及解析、程式碼生成、資源分配和管線建立。
     * @param {string} effectCode - 包含整個 WGFX 效果程式碼的字串。
     * @returns {Promise<void>} 編譯完成時解析的 Promise。
     */
    async compile(effectCode) {
        console.log("WGFXRuntime: 開始效果編譯...");

        // 1. 解析 FX 檔案以獲取中介表示 (IR)。
        this.shaderInfo = this.parser.parse(effectCode);
        console.log("WGFXRuntime: 已解析 ShaderInfo (IR):", this.shaderInfo);

        // 2. 從 IR 生成單個 WGSL 著色器模組。
        const generatedModules = this.wgslCodeGenerator.generate(this.shaderInfo);
        if (this.parser.debug) {
            console.log("WGFXRuntime: 已生成 WGSL 模組:", generatedModules);
        }

        // 3. 根據 IR 初始化所有 GPU 資源 (紋理、取樣器、緩衝區)。
        this.resourceManager.initialize(this.shaderInfo);
        console.log("WGFXRuntime: 資源已初始化。");

        // 4. 為每個通道建立計算管線。
        await this.pipelineManager.createPipelines(this.shaderInfo, generatedModules);
        console.log("WGFXRuntime: 管線已建立。");

        console.log("WGFXRuntime: 編譯完成。");
    }

    /**
     * 編碼調度特定計算通道的命令。
     * @param {string} passName - 要調度的通道名稱 (例如：'PASS_1')。
     * @param {GPUCommandEncoder} commandEncoder - 當前幀的命令編碼器。
     */
    dispatchPass(passName, commandEncoder) {
        if (!this.shaderInfo) {
            throw new Error("效果未編譯。請先呼叫 compile()。");
        }

        // 從 IR 中查找對應的通道資訊。
        const passIndex = parseInt(passName.split('_')[1], 10);
        const passInfo = this.shaderInfo.passes.find(p => p.index === passIndex);
        if (!passInfo) {
            throw new Error(`在著色器資訊中找不到通道 "${passName}"。`);
        }

        // 將實際的調度邏輯委託給管線管理器。
        this.pipelineManager.dispatchPass(passInfo, commandEncoder);
    }

    /**
     * 更新 GPU 緩衝區中 uniform 參數的值。
     * @param {string} name - 要更新的 uniform 名稱。
     * @param {number} value - uniform 的新值。
     */
    updateUniform(name, value) {
        if (!this.shaderInfo) {
            throw new Error("效果未編譯。請先呼叫 compile()。");
        }
        this.uniformBinder.updateUniform(name, value);
    }

    /**
     * 獲取最終輸出紋理的 GPUTextureView，通常命名為 'OUTPUT'。
     * @returns {GPUTextureView} 最終輸出的紋理視圖。
     */
    getOutput() {
        const outputTexture = this.resourceManager.getTexture('OUTPUT');
        if (!outputTexture) {
            throw new Error("在 ResourceManager 中找不到輸出紋理 'OUTPUT'。請確保已正確定義或管理。");
        }
        return outputTexture.createView();
    }

    /**
     * 獲取原始 GPU 資源的實用函數。
     * @param {string} name - 資源的名稱。
     * @returns {GPUTexture | GPUSampler | GPUBuffer | undefined} 底層 GPU 資源。
     */
    getResource(name) {
        return this.resourceManager.getTexture(name)
            || this.resourceManager.getSampler(name)
            || this.resourceManager.getUniform(name)?.buffer;
    }
}