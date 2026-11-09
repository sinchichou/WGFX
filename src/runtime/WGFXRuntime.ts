/// <reference types="@webgpu/types" />
import { parse } from './ShaderParser';
import { ResourceManager } from './ResourceManager';
import { PipelineManager } from './PipelineManager';
import { WGSLCodeGenerator } from './WGSLCodeGenerator';
import { UniformBinder } from './UniformBinder';
import { WGFXShaderInfo, PassInfo } from '../types';
import { Logger } from '../utils/Logger';

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

    /** @zh pass 索引 → 元數據查找表（compile 時建立，dispatch 熱路徑免字串解析與線性搜尋） */
    private passInfoByIndex: Map<number, PassInfo> = new Map();

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
     * @param options - Compilation options / 編譯選項
     * @param options.honorDeclaredOutput - Let the effect's declared texture sizes
     *   override same-named external resources (see
     *   {@link ResourceManager.initialize}). Used by {@link WGFX.create} when the
     *   host does not pass explicit outWidth/outHeight.
     *   讓效果檔宣告的紋理尺寸覆寫同名外部資源（見 {@link ResourceManager.initialize}）。
     *   {@link WGFX.create} 在宿主未明確傳入 outWidth/outHeight 時啟用。
     * @returns A promise that resolves when compilation is successful
     * @throws {Error} 當語法解析、代碼產生或管線編譯失敗時拋出異常
     */
    public async compile(effectCode: string, externalResources: any = {}, options: { honorDeclaredOutput?: boolean } = {}): Promise<void> {
        Logger.debug("WGFXRuntime: Starting effect compilation.");

        try {
            // 1. Parse source code into metadata / 步驟 1：將原始碼解析為元數據
            const shaderInfo = parse(effectCode);
            this.shaderInfo = shaderInfo;
            // Pre-index passes for the dispatch hot path / 為 dispatch 熱路徑預建 pass 索引
            this.passInfoByIndex = new Map(shaderInfo.passes.map(p => [p.index, p]));
            Logger.debug("WGFXRuntime: ShaderInfo parsed");

            // 2. Generate optimized WGSL modules / 步驟 2：產生優化後的 WGSL 模組
            //    外部 defines 需在產生期就注入為 WGSL 常數（NW/NH/CIN/COUT 等），
            //    故此處直接取自 externalResources —— 早於 ResourceManager.initialize
            //    的資源建立時序（產生器必須先產出宣告，管線才有正確的 WGSL）。
            const generatedModules = this.wgslCodeGenerator.generate(
                shaderInfo,
                (externalResources && externalResources.defines) || {},
            );
            this.generatedModules = generatedModules;

            // 3. Prepare GPU textures and buffers / 步驟 3：準備 GPU 紋理與緩衝區
            this.resourceManager.initialize(shaderInfo, externalResources, options);
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

        // Validate format once, then O(1) lookup（格式驗證一次 + O(1) 查表）
        const m = /^PASS_(\d+)$/.exec(passName);
        if (!m) {
            throw new Error(`Invalid pass name "${passName}" (expected "PASS_<index>", e.g. "PASS_0").`);
        }
        const passInfo = this.passInfoByIndex.get(Number(m[1]));

        if (!passInfo) {
            throw new Error(`Pass "${passName}" not found.`);
        }

        // Delegate to pipeline manager / 委託給管線管理員執行
        this.pipelineManager.dispatchPass(passInfo, commandEncoder);
    }

    /**
     * Dispatch every pass in declaration order with zero per-frame overhead.
     * ---
     * 依宣告順序執行全部渲染通道。跳過 "PASS_i" 字串解析與正規式驗證，
     * 直接以 pass 元數據驅動管線，是 {@link WGFX.process} 的每幀熱路徑。
     *
     * @group Execution
     * @param commandEncoder - The active GPU command encoder / 目前的 GPU 指令編碼器
     * @throws {Error} 如果特效尚未編譯則拋出錯誤
     */
    public dispatchAllPasses(commandEncoder: GPUCommandEncoder): void {
        if (!this.shaderInfo) {
            throw new Error("Effect not compiled. Call compile() first.");
        }
        const pipelineManager = this.pipelineManager;
        for (const pass of this.shaderInfo.passes) {
            pipelineManager.dispatchPass(pass, commandEncoder);
        }
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
     * Set the frame counter exposed to shaders as `scene.frame`.
     * ---
     * 設定著色器可讀取的影格計數器（`scene.frame`）。
     *
     * 值僅記錄於資源層，於下一次 {@link ResourceManager.updateSceneBuffer}
     * （宿主每幀更新場景資訊時）寫入 GPU。
     *
     * @group Update
     * @param n - Frame index / 影格索引
     */
    public setFrame(n: number): void {
        this.resourceManager.setFrame(n);
    }

    /**
     * Advance the frame counter by one.
     * ---
     * 將影格計數器遞增 1。
     *
     * @group Update
     * @returns The new frame counter / 遞增後的計數值
     */
    public advanceFrame(): number {
        return this.resourceManager.advanceFrame();
    }

    /** @group Query */
    public getFrame(): number {
        return this.resourceManager.getFrame();
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
        // P2：重用 ResourceManager 在 createTexture 時建立、且隨紋理重建一併更新的
        // sampled view，避免每幀 createView() 產生新的原生物件（消費端每幀取輸出）。
        // 快取失效由 ResourceManager 的視圖表本身承擔（紋理重建即換新視圖）。
        const cached = this.resourceManager.views.get('OUTPUT')?.sampled;
        return cached ?? outputTexture.createView();
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