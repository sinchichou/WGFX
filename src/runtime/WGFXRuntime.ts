/// <reference types="@webgpu/types" />
import { parse } from './ShaderParser';
import { ResourceManager } from './ResourceManager';
import { PipelineManager } from './PipelineManager';
import { WGSLCodeGenerator } from './WGSLCodeGenerator';
import { UniformBinder } from './UniformBinder';
import {WGFXShaderInfo} from '@/types';
import {Logger} from '@/utils/Logger';

/**
 * Core runtime for WGFX, managing resources, pipelines, and execution.
 * ---
 * WGFX 的核心運行時環境，負責協調資源管理、管線編譯與指令調度執行。
 * 它是連接高層 API 與底層 WebGPU 操作的橋樑。
 *
 * @group Core
 * @category Main
 */
export class WGFXRuntime {
    /** @zh WebGPU 裝置實例 */
    public device: GPUDevice;
    /** @zh 資源管理員：處理紋理、緩衝區與取樣器 */
    public resourceManager: ResourceManager;
    /** @zh 管線管理員：負責計算與渲染管線的建立 */
    public pipelineManager: PipelineManager;
    /** @zh WGSL 代碼產生器：將解析後的資訊轉換為著色器代碼 */
    public wgslCodeGenerator: WGSLCodeGenerator;
    /** @zh Uniform 更新繫結器：處理參數同步 */
    public uniformBinder: UniformBinder;
    /** @zh 目前載入的特效元數據資訊，未編譯前為 null */
    public shaderInfo: WGFXShaderInfo | null;
    /** @zh 生成的WGSL模組，用於除錯顯示 */
    public generatedModules: { wgslCode: string; passIndex: number; resources: any }[] | null;

    /**
     * Initialize the WGFX runtime environment.
     * ---
     * 初始化 WGFX 運行時。
     *
     * @param device - The active WebGPU device / 有效的 WebGPU 裝置
     * @throws {Error} 如果 WebGPU 裝置無效則拋出錯誤
     */
    constructor(device: GPUDevice) {
        if (!device) {
            const error = "WGFXRuntime requires a valid WebGPU device.";
            Logger.error(error);
            throw new Error(error);
        }
        this.device = device;
        this.resourceManager = new ResourceManager(this.device);
        this.pipelineManager = new PipelineManager(this.device, this.resourceManager);
        this.wgslCodeGenerator = new WGSLCodeGenerator();
        this.uniformBinder = new UniformBinder(this.device, this.resourceManager);
        this.shaderInfo = null;
        this.generatedModules = null;
    }

    /**
     * Compile effect code and initialize all GPU pipelines.
     * ---
     * 編譯特效原始碼並初始化管線。這是 WGFX 啟動過程中最耗時的操作。
     *
     * @group Lifecycle
     * @param effectCode - The source code of the effect / 特效原始碼
     * @param externalResources - Optional external resource definitions / 選用的外部資源定義
     * @returns A promise that resolves when compilation is successful
     * @throws {Error} 當語法解析、代碼產生或管線編譯失敗時拋出異常
     */
    public async compile(effectCode: string, externalResources: any = {}): Promise<void> {
        Logger.debug("WGFXRuntime: Starting effect compilation.");

        try {
            // 1. Parse source code into metadata / 步驟 1：將原始碼解析為元數據
            const shaderInfo = parse(effectCode);
            this.shaderInfo = shaderInfo;
            Logger.debug("WGFXRuntime: ShaderInfo parsed");

            // 2. Generate optimized WGSL modules / 步驟 2：產生優化後的 WGSL 模組
            const generatedModules = this.wgslCodeGenerator.generate(shaderInfo);
            this.generatedModules = generatedModules;

            // 3. Prepare GPU textures and buffers / 步驟 3：準備 GPU 紋理與緩衝區
            this.resourceManager.initialize(shaderInfo, externalResources);
            Logger.debug("WGFXRuntime: Resources initialized.");

            // 4. Create and compile GPU pipelines / 步驟 4：建立並編譯 GPU 管線
            await this.pipelineManager.createPipelines(shaderInfo, generatedModules);

            Logger.debug("WGFXRuntime: Pipelines created.");
            Logger.info("WGFXRuntime: Compilation complete.");
        } catch (e: any) {
            Logger.error("WGFXRuntime: Parse/Compile error", e);
            throw e;
        }
    }

    /**
     * Dispatch a specific rendering pass to the GPU commands.
     * ---
     * 執行特定的渲染通道（Pass）。
     *
     * @group Execution
     * @param passName - Format "PASS_i" (e.g., "PASS_0") / 格式為 "PASS_i" 的通道名稱
     * @param commandEncoder - The active GPU command encoder / 目前的 GPU 指令編碼器
     * @throws {Error} 如果特效尚未編譯或找不到指定通道則拋出錯誤
     */
    public dispatchPass(passName: string, commandEncoder: GPUCommandEncoder): void {
        if (!this.shaderInfo) {
            throw new Error("Effect not compiled. Call compile() first.");
        }

        // Extract index from name (e.g., "PASS_0" -> 0) / 從名稱提取索引
        const passIndex = parseInt(passName.split('_')[1], 10);
        const passInfo = this.shaderInfo.passes.find(p => p.index === passIndex);

        if (!passInfo) {
            throw new Error(`Pass "${passName}" not found.`);
        }

        // Delegate to pipeline manager / 委託給管線管理員執行
        this.pipelineManager.dispatchPass(passInfo, commandEncoder);
    }

    /**
     * Update a uniform value by its identifier name.
     * ---
     * 根據名稱更新 Uniform 參數數值。
     *
     * @group Update
     * @param name - The uniform name defined in shader / 著色器中定義的名稱
     * @param value - New value(s) to upload / 要上傳的新數值
     */
    public updateUniform(name: string, value: number | number[]): void {
        if (!this.shaderInfo) {
            throw new Error("Effect not compiled. Call compile() first.");
        }
        this.uniformBinder.updateUniform(name, value);
    }

    /**
     * Get the final output texture view for presentation or further processing.
     * ---
     * 獲取最終輸出的紋理視圖（Texture View）。
     *
     * @group Output
     * @returns The {@link GPUTextureView} of the 'OUTPUT' texture
     * @throws {Error} 若找不到輸出紋理則拋出錯誤
     */
    public getOutput(): GPUTextureView {
        const outputTexture = this.resourceManager.getTexture('OUTPUT');
        if (!outputTexture) {
            throw new Error("Output texture 'OUTPUT' not found.");
        }
        return outputTexture.createView();
    }

    /**
     * Retrieve a resource (Texture, Sampler, or Buffer) by its name.
     * ---
     * 根據名稱檢索資源實體。會依序從紋理、取樣器、緩衝區中尋找。
     *
     * @group Query
     * @param name - The resource identifier / 資源標識符名稱
     * @returns The generic resource object or undefined if not found
     */
    public getResource(name: string): GPUTexture | GPUSampler | GPUBuffer | undefined {
        // Search across different resource maps / 在不同的資源映射表中搜尋
        return this.resourceManager.getTexture(name)
            || this.resourceManager.getSampler(name)
            || this.resourceManager.getUniform(name)?.buffer;
    }
}